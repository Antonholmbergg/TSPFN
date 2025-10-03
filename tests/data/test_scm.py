import networkx as nx
import torch
from torch.testing import assert_close

from tspfn.data.prior import PriorConfig
from tspfn.data.scm import SCM

from . import get_test_config_path
import pytest

@pytest.mark.parametrize("seed", [37842, 42837])
def test_scm_is_dag(seed):
    prior_config = PriorConfig.from_yaml_config(get_test_config_path())
    prior1 = prior_config.sample_prior(seed)
    assert nx.is_directed_acyclic_graph(SCM.from_prior(prior1).graph)


@pytest.mark.parametrize("seed", [37842, 42837])
def test_seeding(seed):
    prior_config = PriorConfig.from_yaml_config(get_test_config_path())
    prior1 = prior_config.sample_prior(seed)
    dataset1 = SCM.from_prior(prior1).get_dataset()
    second_dataset1 = SCM.from_prior(prior1).get_dataset()
    prior2 = prior_config.sample_prior(seed*2)
    assert_close(dataset1, second_dataset1)

    dataset2 = SCM.from_prior(prior2).get_dataset()

    if dataset1.shape == dataset2.shape:
        assert dataset1 != dataset2


@pytest.mark.parametrize("seed", [37842, 42837])
def test_no_nans(seed):
    prior_config = PriorConfig.from_yaml_config(get_test_config_path())
    for _ in range(10):
        prior = prior_config.sample_prior(seed)
        dataset = SCM.from_prior(prior).get_dataset()
        assert not torch.any(torch.isnan(dataset))
