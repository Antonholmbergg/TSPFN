import networkx as nx

from tspfn.data.edge_functions import EdgeFunctionSampler
from tspfn.data.prior import PriorHyperParameters


class SCM:
    """
    :TODO improve the graph -> the paper hints at the real graph being a merge between gnr_graphs
    """

    def __init__(self, n_nodes_total: int, redirection_probablility: float, random_state: int):
        self.graph = nx.gnr_graph(
            n=n_nodes_total,
            p=redirection_probablility,
            seed=random_state,
        )
        self.__populate_edge_functions()

    def get_graph(self):
        return self.graph

    def __populate_edge_functions(self):
        self.efs = EdgeFunctionSampler()
        print(self.graph.number_of_nodes())
        print(self.graph.number_of_edges())
        # self.graph.nodes[0] = "root"
        print(self.graph.nodes[0])
        for edge in self.graph.edges:
            print("edge", edge)
        edge_attributes = {}
        for edge in self.graph.edges:
            edge_attributes[edge] = {"function": self.efs.sample()}
        nx.set_edge_attributes(self.graph, edge_attributes)
        for edge in self.graph.edges:
            print("edge", edge)
        for node in self.graph:
            print("node", node)

    def proppagate(self, input):
        pass


def get_scm(prior_hp: PriorHyperParameters) -> SCM:
    return SCM(25, 0.3, 42)


if __name__ == "__main__":
    gnr_graph = get_scm(1)

    print(
        f"\nIs the graph a Directed Acyclic Graph (DAG)? {nx.is_directed_acyclic_graph(gnr_graph.graph)}"
    )

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
