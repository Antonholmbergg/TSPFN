import networkx as nx

from tspfn.data.scm import get_scm


def test_scm_is_dag():
    assert nx.is_directed_acyclic_graph(get_scm(1).get_graph())
