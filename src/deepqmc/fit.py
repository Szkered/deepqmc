from collections import namedtuple
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax
import optax

from .kfacext import make_graph_patterns
from .utils import (
    exp_normalize_mean,
    masked_mean,
    per_mol_stats,
    segment_nanmean,
    tree_norm,
)
from .wf.base import init_wf_params

__all__ = ()

TrainState = namedtuple('TrainState', 'sampler params opt')


def median_clip_and_mask(E_loc, clip_width, median_center, exclude_width=jnp.inf):
    clip_center = jnp.nanmedian(E_loc) if median_center else jnp.nanmean(E_loc)
    deviation = jnp.abs(E_loc - clip_center)
    sigma = jnp.nanmean(deviation)
    E_loc_s = jnp.clip(
        E_loc, clip_center - clip_width * sigma, clip_center + clip_width * sigma
    )
    gradient_mask = deviation < exclude_width
    return E_loc_s, gradient_mask


def log_squeeze(x):
    sgn, x = jnp.sign(x), jnp.abs(x)
    return sgn * jnp.log1p((x + 1 / 2 * x**2 + x**3) / (1 + x**2))


def median_log_squeeze(x, width, quantile):
    x_median = jnp.nanmedian(x)
    x_diff = x - x_median
    quantile = jnp.nanquantile(jnp.abs(x_diff), quantile)
    width = width * quantile
    return (
        x_median + 2 * width * log_squeeze(x_diff / (2 * width)),
        jnp.abs(x_diff) / quantile,
    )


def median_log_squeeze_and_mask(
    x, clip_width=1.0, quantile=0.95, exclude_width=jnp.inf
):
    clipped_x, sigma = median_log_squeeze(x, clip_width, quantile)
    gradient_mask = sigma < exclude_width
    return clipped_x, gradient_mask


def init_fit(rng, hamil, ansatz, sampler, sample_size):
    params = init_wf_params(rng, hamil, ansatz)
    smpl_state = sampler.init(rng, partial(ansatz.apply, params), sample_size)
    return params, smpl_state


def fit_wf(  # noqa: C901
    rng,
    hamil,
    ansatz,
    opt,
    sampler,
    sample_size,
    steps,
    train_state=None,
    *,
    clip_mask_fn=None,
    clip_mask_kwargs=None,
):
    stats_fn = partial(per_mol_stats, len(sampler))

    @partial(jax.custom_jvp, nondiff_argnums=(1, 2))
    def loss_fn(params, rng, batch):
        phys_conf, weight = batch
        rng_batch = jax.random.split(rng, len(weight))
        E_loc, hamil_stats = jax.vmap(
            hamil.local_energy(partial(ansatz.apply, params))
        )(rng_batch, phys_conf)
        loss = jnp.nanmean(E_loc * weight)
        stats = {
            **stats_fn(E_loc, phys_conf.mol_idx, 'E_loc'),
            **{
                k_hamil: stats_fn(v_hamil, phys_conf.mol_idx, k_hamil, mean_only=True)
                for k_hamil, v_hamil in hamil_stats.items()
            },
        }
        return loss, (E_loc, stats)

    @loss_fn.defjvp
    def loss_jvp(rng, batch, primals, tangents):
        phys_conf, weight = batch
        loss, aux = loss_fn(*primals, rng, batch)
        E_loc, _ = aux
        E_loc_s, gradient_mask = (clip_mask_fn or median_log_squeeze_and_mask)(
            E_loc, **(clip_mask_kwargs or {})
        )
        E_mean = segment_nanmean(E_loc_s * weight, phys_conf.mol_idx, len(sampler))[
            phys_conf.mol_idx
        ]
        assert E_loc_s.shape == E_loc.shape, (
            f'Error with clipping function: shape of E_loc {E_loc.shape} '
            f'must equal shape of clipped E_loc {E_loc_s.shape}.'
        )
        assert gradient_mask.shape == E_loc.shape, (
            f'Error with masking function: shape of E_loc {E_loc.shape} '
            f'must equal shape of mask {gradient_mask.shape}.'
        )

        def log_likelihood(params):  # log(psi(theta))
            return jax.vmap(ansatz.apply, (None, 0))(params, phys_conf).log

        log_psi, log_psi_tangent = jax.jvp(log_likelihood, primals, tangents)
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
        loss_tangent = (E_loc_s - E_mean) * log_psi_tangent * weight
        loss_tangent = masked_mean(loss_tangent, gradient_mask)
        return (loss, aux), (loss_tangent, aux)
        # jax.custom_jvp has actually no official support for auxiliary output.
        # the second aux in the tangent output should be in fact aux_tangent.
        # we just output the same thing to satisfy jax's API requirement with
        # the understanding that we'll never need aux_tangent

    energy_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    if opt is None:

        @jax.jit
        def _step(_rng_opt, params, _opt_state, batch):
            loss, (E_loc, stats) = loss_fn(params, _rng_opt, batch)

            return params, None, E_loc, {'per_mol': stats}

    elif isinstance(opt, optax.GradientTransformation):

        @jax.jit
        def _step(rng, params, opt_state, batch):
            (loss, (E_loc, per_mol_stats)), grads = energy_and_grad_fn(
                params, rng, batch
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            param_norm, update_norm, grad_norm = map(
                tree_norm, [params, updates, grads]
            )
            params = optax.apply_updates(params, updates)
            stats = {
                'opt/param_norm': param_norm,
                'opt/grad_norm': grad_norm,
                'opt/update_norm': update_norm,
                'per_mol': per_mol_stats,
            }
            return params, opt_state, E_loc, stats

        def init_opt(rng, params, batch):
            opt_state = opt.init(params)
            return opt_state

    else:

        def _step(rng, params, opt_state, batch):
            params, opt_state, opt_stats = opt.step(
                params,
                opt_state,
                rng,
                batch=batch,
                momentum=0,
            )
            stats = {
                'opt/param_norm': opt_stats['param_norm'],
                'opt/grad_norm': opt_stats['precon_grad_norm'],
                'opt/update_norm': opt_stats['update_norm'],
                'per_mol': opt_stats['aux'][1],
            }
            return (
                params,
                opt_state,
                opt_stats['aux'][0],
                stats,
            )

        def init_opt(rng, params, batch):
            opt_state = opt.init(
                params,
                rng,
                batch,
            )
            return opt_state

        opt = opt(
            value_and_grad_func=energy_and_grad_fn,
            l2_reg=0.0,
            value_func_has_aux=True,
            value_func_has_rng=True,
            auto_register_kwargs={'graph_patterns': make_graph_patterns()},
            include_norms_in_stats=True,
            estimation_mode='fisher_exact',
            num_burnin_steps=0,
            min_damping=1e-4,
            inverse_update_period=1,
        )

    @jax.jit
    def sample_wf(state, rng, params, select_idxs):
        return sampler.sample(rng, state, partial(ansatz.apply, params), select_idxs)

    @jax.jit
    def update_sampler(state, params):
        return sampler.update(state, partial(ansatz.apply, params))

    def train_step(rng, step, smpl_state, params, opt_state):
        rng_sample, rng_kfac = jax.random.split(rng)
        select_idxs = sampler.select_idxs(sample_size, step)
        smpl_state, phys_conf, smpl_stats = sample_wf(
            smpl_state, rng_sample, params, select_idxs
        )
        weight = exp_normalize_mean(
            sampler.get_state(
                'log_weight',
                smpl_state,
                select_idxs,
                default=jnp.zeros(sample_size),
            )
        )
        params, opt_state, E_loc, stats = _step(
            rng_kfac,
            params,
            opt_state,
            (phys_conf, weight),
        )
        if opt is not None:
            # WF was changed in _step, update psi values stored in smpl_state
            smpl_state = update_sampler(smpl_state, params)
        stats['per_mol'] = {**stats['per_mol'], **smpl_stats['per_mol']}
        return smpl_state, params, opt_state, E_loc, stats

    if train_state:
        smpl_state, params, opt_state = train_state
    else:
        rng, rng_init_fit = jax.random.split(rng)
        params, smpl_state = init_fit(rng_init_fit, hamil, ansatz, sampler, sample_size)
        opt_state = None
    if opt is not None and opt_state is None:
        rng, rng_opt = jax.random.split(rng)
        init_select_idxs = sampler.select_idxs(sample_size, 0)
        opt_state = init_opt(
            rng_opt,
            params,
            (
                sampler.phys_conf(smpl_state, init_select_idxs),
                jnp.ones(sample_size),
            ),
        )
    train_state = smpl_state, params, opt_state

    for step, rng in zip(steps, hk.PRNGSequence(rng)):
        *train_state, E_loc, stats = train_step(rng, step, *train_state)
        yield step, TrainState(*train_state), E_loc, stats
