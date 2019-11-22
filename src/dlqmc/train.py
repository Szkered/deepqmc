from copy import deepcopy
from functools import partial
from importlib import resources
from pathlib import Path

import click
import toml
import torch
from pyscf import gto, mcscf, scf
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from .ansatz import OmniSchnet
from .fit import LossWeightedLogProb, batched_sampler, fit_wfnet
from .geom import get_system
from .nn import PauliNet
from .sampling import LangevinSampler, rand_from_mf


def merge_into(self, other):
    for key, val in other.items():
        if isinstance(val, dict):
            if not isinstance(self.get(key), dict):
                self[key] = {}
            merge_into(self[key], val)
        elif self.get(key) != val:
            self[key] = val


class Parametrization:
    DEFAULTS = toml.loads(resources.read_text('dlqmc', 'default-params.toml'))

    def __init__(self, dct=None):
        self._dct = dct or deepcopy(Parametrization.DEFAULTS)

    def __getitem__(self, key):
        x = self._dct
        for k in key.split('.'):
            x = x[k]
        return x

    def __setitem__(self, key, val):
        x = self._dct
        keys = key.split('.')
        for k in keys[:-1]:
            x = x[k]
        x[keys[-1]] = val

    def update(self, other):
        merge_into(self._dct, other)

    def update_with_system(self, name, **kwargs):
        system = get_system(name, **kwargs)
        for key in ['geom', 'charge', 'spin']:
            self[f'model_kwargs.{key}'] = system[key]
        self['train_kwargs.sampler_kwargs.tau'] = system['tau']


def model(*, geom, basis, charge, spin, pauli_kwargs, omni_kwargs, cas=None):
    mol = gto.M(
        atom=geom.as_pyscf(),
        unit='bohr',
        basis=basis,
        charge=charge,
        spin=spin,
        cart=True,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    if cas:
        mc = mcscf.CASSCF(mf, *cas)
        mc.kernel()
    wfnet = PauliNet.from_pyscf(
        mc if cas else mf,
        omni_factory=partial(OmniSchnet, **omni_kwargs),
        cusp_correction=True,
        cusp_electrons=True,
        **pauli_kwargs,
    )
    return wfnet, mf


def train(
    wfnet,
    mf,
    *,
    cwd=None,
    state=None,
    save_every=None,
    cuda,
    learning_rate,
    n_steps,
    sampler_size,
    sampler_kwargs,
    lr_scheduler,
    decay_rate,
    optimizer,
    fit_kwargs,
):
    batched_sampler_kwargs = sampler_kwargs.copy()
    tau = batched_sampler_kwargs.pop('tau')
    rs = rand_from_mf(mf, sampler_size)
    if cuda:
        rs = rs.cuda()
        wfnet.cuda()
    opt = getattr(torch.optim, optimizer)(wfnet.parameters(), lr=learning_rate)
    if lr_scheduler == 'inverse':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda t: 1 / (1 + t / decay_rate)
        )
    else:
        scheduler = None
    if state:
        init_step = state['step'] + 1
        opt.load_state_dict(state['opt'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler'])
    else:
        init_step = 0
    with SummaryWriter(log_dir=cwd, flush_secs=15, purge_step=init_step - 1) as writer:
        for step in fit_wfnet(
            wfnet,
            LossWeightedLogProb(),
            opt,
            batched_sampler(
                LangevinSampler(wfnet, rs, tau=tau, n_first_certain=3),
                range_sampling=partial(trange, desc='sampling', leave=False),
                **batched_sampler_kwargs,
            ),
            trange(
                init_step, n_steps, initial=init_step, total=n_steps, desc='training'
            ),
            writer=writer,
            **fit_kwargs,
        ):
            if scheduler:
                scheduler.step()
            if cwd and save_every and (step + 1) % save_every == 0:
                state = {
                    'step': step,
                    'wfnet': wfnet.state_dict(),
                    'opt': opt.state_dict(),
                }
                if scheduler:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, Path(cwd) / f'state-{step:05d}.pt')


def state_from_file(path):
    return torch.load(path) if path and Path(path).is_file() else None


def model_from_file(path, state=None):
    param = Parametrization()
    param_file = toml.loads(Path(path).read_text())
    system = param_file.pop('system', None)
    if system:
        if isinstance(system, str):
            system = {'name': system}
        param.update_with_system(**system)
    param.update(param_file)
    wfnet, mf = model(**param['model_kwargs'])
    if state:
        wfnet.load_state_dict(state['wfnet'])
    return wfnet, mf, param


@click.command('train')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.option('--state', type=click.Path(dir_okay=False))
@click.option('--save-every', default=100, show_default=True)
def train_from_file(path, state, save_every):
    state = state_from_file(state)
    wfnet, mf, param = model_from_file(path, state)
    train(
        wfnet,
        mf,
        cwd=Path(path).parent,
        state=state,
        save_every=save_every,
        **param['train_kwargs'],
    )