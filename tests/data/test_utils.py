from tspfn.data.utils import gamma
import torch
from torch.testing import assert_close


def test_gamma():
    """make sure that global seed with the Gamma() distribution gives the same result as my hacky workaround that allows using a Generator object"""
    torch_generator1 = torch.Generator().manual_seed(42)
    torch_generator2 = torch.Generator().manual_seed(42)
    gamma_samples1 = gamma(torch.ones(1), torch.ones(1), (20,), torch_generator1)

    old_state = torch.get_rng_state()
    torch.set_rng_state(torch_generator2.get_state())
    gamma_samples2 = torch.distributions.gamma.Gamma(torch.ones(1), torch.ones(1)).sample((20,))
    torch.set_rng_state(old_state)
    assert_close(gamma_samples1, gamma_samples2)
