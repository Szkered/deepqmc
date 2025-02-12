import pytest

from deepqmc.physics import local_potential, nonlocal_potential


@pytest.mark.parametrize('pp_type', [None, 'bfd', 'ccECP'])
@pytest.mark.parametrize('name', ['LiH', 'C'])
class TestPhysics:
    def test_pseudo_potentials(self, helpers, name, pp_type, ndarrays_regression):
        mol = helpers.mol(name, pp_type)
        hamil = helpers.hamil(mol)
        phys_conf = helpers.phys_conf(hamil)
        _wf, params = helpers.create_ansatz(hamil)
        wf = lambda phys_conf: _wf.apply(params, phys_conf)
        ndarrays_regression.check(
            {
                'local_potential': local_potential(phys_conf, mol),
                'nonlocal_potential': (
                    nonlocal_potential(helpers.rng(), phys_conf, mol, wf)
                    if pp_type
                    else 0
                ),
            },
            default_tolerance={'rtol': 2e-2, 'atol': 1e-8},
        )
        # note: nonlocal_potential is not particularly
        # numerically stable, hence the large tolerance
