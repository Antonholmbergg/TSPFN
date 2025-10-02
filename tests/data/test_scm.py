import networkx as nx
from tspfn.data.prior import PriorConfig
from . import get_test_config_path
from tspfn.data.scm import SCM


def test_scm_is_dag():
    prior_config = PriorConfig.from_yaml_config(get_test_config_path())
    prior1 = prior_config.sample_prior()
    assert nx.is_directed_acyclic_graph(SCM.from_prior(prior1).graph)
