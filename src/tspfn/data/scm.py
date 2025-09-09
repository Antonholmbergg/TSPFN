import networkx as nx
import numpy as np
import numpy.typing as npt
import polars as pl
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
        feature_node_fraction: float,
        edge_function_sampler: EdgeFunctionSampler,
    ):
        self.rng = np.random.default_rng(random_state)
        self.graph = nx.gnr_graph(
            n=n_nodes_total,
            p=redirection_probablility,
            seed=random_state,
        )
        self.root_nodes = [v for v, d in self.graph.in_degree() if d == 0]
        self.efs = edge_function_sampler
        self.feature_node_fraction = feature_node_fraction
        self.__populate_edge_functions()
        self.__sample_feature_nodes()
        print(self.graph.nodes(data=True))
        print(self.graph.edges(data=True))

    def get_graph(self):
        return self.graph

    def __sample_feature_nodes(self) -> None:
        """Samples a number of feature nodes from the SCM. These are the features in our dataset.
        :TODO Should leaf nodes really be sampled as feature nodes?

        Returns:
            None
        """

        non_root_nodes = [node for node in self.graph.nodes if node not in self.root_nodes]
        self.n_feature_nodes = int(len(non_root_nodes) * self.feature_node_fraction)
        self.feature_nodes = self.rng.choice(non_root_nodes, self.n_feature_nodes, replace=False)
        node_attributes = {k: {"feature_node": k in self.feature_nodes} for k in self.graph.nodes}
        nx.set_node_attributes(self.graph, node_attributes)

    def __populate_edge_functions(self) -> None:
        """Sets the edge functions for each edge in the graph.
        At the moment the functions are bs but that chould be changed in the EdgeFunctionSampler class, not here.

        Returns:
            None
        """

        edge_attributes = {}
        for edge in self.graph.edges:
            edge_attributes[edge] = {"function": self.efs.sample()}
        nx.set_edge_attributes(self.graph, edge_attributes)

    def proppagate(self, n_rows: int, node_dim: int) -> pl.DataFrame:
        """:TODO sampler argument here or setup in the init, probably best in init if possible

        Parameters
        ----------
        n_rows : int
            the number of rows to generate

        Returns
        -------
        pl.DataFrame
            _description_
        """
        latent_variable_attr = {k: {"latent_variables": np.zeros((n_rows, node_dim))} for k in self.graph.nodes}
        nx.set_node_attributes(self.graph, latent_variable_attr)
        for root_node in self.root_nodes:
            self.graph.nodes[root_node]["latent_variables"] += self.rng.normal(0, 1, (n_rows, node_dim))

        for generation in nx.topological_generations(self.graph):
            print(generation)
            for node in generation:
                print(self.graph[node])
                edge_mappings = self.graph[node]
                for (
                    successor_node,
                    mapping,
                ) in edge_mappings.items():  # This should actally always have one and only one value except for node 0
                    self.graph.nodes[successor_node]["latent_variables"] += mapping["function"](
                        self.graph.nodes[node]["latent_variables"]
                    )
        
        dataset = {}
        for node in self.feature_nodes:
            continuos_feature_mapping = self.rng.multinomial(10, np.ones(node_dim)/node_dim)/node_dim
            continuos_feature = np.dot(self.graph.nodes[node]["latent_variables"], continuos_feature_mapping)
            print(continuos_feature, continuos_feature.shape)
            dataset[node] = continuos_feature
        return pl.DataFrame(dataset)





def get_scm(prior_hp: PriorHyperParameters) -> SCM:
    return SCM(25, 0.1, 42, 0.3, EdgeFunctionSampler())


if __name__ == "__main__":
    gnr_graph = get_scm(1)
    rng = np.random.default_rng(101)
    n_rows = 100
    node_dim = 8
    dataset = gnr_graph.proppagate(n_rows, node_dim)
    print(dataset)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(gnr_graph.get_graph(), seed=42)
    nx.draw(
        gnr_graph.get_graph(),
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        arrows=True,
        arrowsize=10,
        node_size=700,
    )
    plt.title("Generated GNR Graph")
    fig.savefig("gnr_graph.png")
