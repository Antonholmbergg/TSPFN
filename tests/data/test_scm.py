import networkx as nx
import torch
from tspfn.data.prior import PriorConfig
from . import get_test_config_path
from tspfn.data.scm import SCM
from torch.testing import assert_close

def test_scm_is_dag():
    prior_config = PriorConfig.from_yaml_config(get_test_config_path())
    prior1 = prior_config.sample_prior()
    assert nx.is_directed_acyclic_graph(SCM.from_prior(prior1).graph)


def test_seeding():
    prior_config = PriorConfig.from_yaml_config(get_test_config_path())
    prior1 = prior_config.sample_prior()
    dataset1  = SCM.from_prior(prior1).get_dataset()
    second_dataset1  = SCM.from_prior(prior1).get_dataset()
    prior2 = prior_config.sample_prior()
    assert_close(dataset1, second_dataset1)
    
    dataset2  = SCM.from_prior(prior2).get_dataset()
    
    if dataset1.shape == dataset2.shape:
        assert dataset1 != dataset2
        
def test_no_nans():
    prior_config = PriorConfig.from_yaml_config(get_test_config_path())
    for _ in range(10):
        prior = prior_config.sample_prior()
        dataset  = SCM.from_prior(prior).get_dataset()
        assert not torch.any(torch.isnan(dataset))
    