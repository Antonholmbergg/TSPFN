import networkx as nx
import numpy as np
from tspfn.data.edge_functions import EdgeFunctionSampler
from tspfn.data.prior import PriorHyperParameters


class SCM:
    """
    :TODO improve the graph -> the paper hints at the real graph being a merge between gnr_graphs
    """

    def __init__(
        self,
        n_nodes_total: int,
        redirection_probablility: float,
        random_state: int,
        n_feature_nodes: int,
    ):
        self.rng = np.random.default_rng(random_state)
        self.graph = nx.gnr_graph(
            n=n_nodes_total,
            p=redirection_probablility,
            seed=random_state,
        )
        self.n_feature_nodes = n_feature_nodes
        print("-" * 10, "graph populated", "-" * 10)
        self.__populate_edge_functions()
        print("-" * 10, "edge functions populated", "-" * 10)
        self.__select_feature_nodes()
        print("-" * 10, "feature nodes selected", "-" * 10)

    def get_graph(self):
        return self.graph

    def __select_feature_nodes(self) -> None:
        print(self.graph.nodes)
        feature_nodes = self.rng.choice(self.graph.nodes, self.n_feature_nodes, replace=False)
        node_attributes = {k: {"feature_node": k in feature_nodes} for k in self.graph.nodes}
        nx.set_node_attributes(self.graph, node_attributes)
        print(self.graph.nodes(data=True))

    def __populate_edge_functions(self) -> None:
        self.efs = EdgeFunctionSampler()
        edge_attributes = {}
        for edge in self.graph.edges:
            edge_attributes[edge] = {"function": self.efs.sample()}
        nx.set_edge_attributes(self.graph, edge_attributes)


    def proppagate(self, input):
        pass


def get_scm(prior_hp: PriorHyperParameters) -> SCM:
    return SCM(25, 0.3, 42, 10)


if __name__ == "__main__":
    gnr_graph = get_scm(1)

    print(f"\nIs the graph a Directed Acyclic Graph (DAG)? {nx.is_directed_acyclic_graph(gnr_graph.graph)}")

    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(8, 6))
    # pos = nx.spring_layout(gnr_graph, seed=42)
    # nx.draw(
    #    gnr_graph,
    #    pos,
    #    with_labels=True,
    #   node_color="lightblue",
    #    edge_color="gray",
    #    arrows=True,
    #    arrowsize=10,
    #    node_size=700,
    # )
    # plt.title("Generated GNR Graph")
    # fig.savefig("gnr_graph.png")

# if __name__ == "__main__":
#     scm = SCM(PriorHyperParameters())
#     print(scm.forward(torch.Tensor([1] * scm.layers[0].in_features)))
