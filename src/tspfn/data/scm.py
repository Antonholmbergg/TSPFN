import networkx as nx
import polars as pl
import torch

from tspfn.data.edge_functions import EdgeFunctionSampler, EdgeMappingOutput
from tspfn.data.prior import PriorHyperParameters


class SCM:
    """
    ::TODO improve the graph -> the paper hints at the real graph sometimes
    being a merge between disjoint gnr_graphs.
    """

    def __init__(
        self,
        n_nodes_total: int,
        redirection_probablility: float,
        random_state: int,
        feature_node_fraction: float,
        edge_function_sampler: EdgeFunctionSampler,
        n_rows: int,
        node_dim: int,
    ):
        self.generator = torch.Generator().manual_seed(random_state)
        self.n_rows = n_rows
        self.node_dim = node_dim
        self.graph = nx.gnr_graph(
            n=n_nodes_total,
            p=redirection_probablility,
            seed=random_state,
        )
        self.root_nodes = [v for v, d in self.graph.in_degree() if d == 0]
        self.efs = edge_function_sampler
        self.feature_node_fraction = feature_node_fraction
        self.__populate_edge_functions()
        self.__set_node_features()

    def __set_node_features(self) -> None:
        """Samples a number of feature nodes from the SCM. These are the features in our dataset.
        Also sents the

        Returns:
            None
        """
        non_root_nodes = [node for node in self.graph.nodes if node not in self.root_nodes]
        self.n_feature_nodes = int(len(non_root_nodes) * self.feature_node_fraction)
        equal_probability_weigths = torch.ones(len(non_root_nodes))
        feature_node_indicies = torch.multinomial(
            equal_probability_weigths, self.n_feature_nodes, replacement=False, generator=self.generator
        )
        self.feature_nodes = [non_root_nodes[i] for i in feature_node_indicies]

        node_attributes = {}
        for node in self.graph.nodes:
            node_attributes[node] = {
                "feature_node": node in self.feature_nodes,
                "latent_variables": torch.zeros((self.n_rows, self.node_dim)),
                "categorical_feature": None,
            }
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

    def __initialize_root_nodes(self) -> None:
        for root_node in self.root_nodes:
            self.graph.nodes[root_node]["latent_variables"] += torch.normal(
                0, 1, size=(self.n_rows, self.node_dim), generator=self.generator
            )

    def __proppagate(
        self,
    ) -> None:
        for generation in nx.topological_generations(self.graph):
            for node in generation:
                edge_mappings = self.graph[node]
                # This should actally always have one and only one value except for node 0
                for successor_node, mapping in edge_mappings.items():
                    mapping_output : EdgeMappingOutput = mapping["function"](
                        self.graph.nodes[node]["latent_variables"], generator=self.generator
                    )
                    latent_variables = mapping_output.get("latent_variable")
                    cat_feature = mapping_output.get("categorical_feature")
                    self.graph.nodes[successor_node]["latent_variables"] += latent_variables
                    if cat_feature is not None:
                        if self.graph.nodes[successor_node]["categorical_feature"] is None:
                            self.graph.nodes[successor_node]["categorical_feature"] = cat_feature
                        else:
                            # This is questionable but I'm not sure what to do in this situation yet. ignore it? concatenate it?
                            self.graph.nodes[successor_node]["categorical_feature"] += cat_feature


    def get_dataset(self, n_draws_feature_mapping: int = 10) -> pl.DataFrame:
        self.__initialize_root_nodes()
        self.__proppagate()
        dataset = {}
        for node in self.feature_nodes:
            continuos_feature_mapping = torch.zeros(self.node_dim)
            for ind in torch.multinomial(
                torch.ones(self.node_dim), n_draws_feature_mapping, replacement=True, generator=self.generator
            ):
                continuos_feature_mapping[ind] += 1 / n_draws_feature_mapping
            categorical_feature = self.graph.nodes[node]["categorical_feature"]
            if categorical_feature is not None:
                dataset[f"cat_feature_{int(node)}"] = categorical_feature    
            else:
                continuos_feature = torch.matmul(self.graph.nodes[node]["latent_variables"], continuos_feature_mapping)
                dataset[f"feature_{int(node)}"] = continuos_feature
        return pl.DataFrame(dataset)


def get_scm(prior_hp: PriorHyperParameters) -> SCM:
    generator = torch.Generator().manual_seed(845)
    categorical_feature_mapping_kwargs = {
        "gamma_shape": 2.,
        "gamma_rate": 1.,
        "min_categories": 2,
        "max_categories": 20,
        }
    efs = EdgeFunctionSampler(generator, categorical_feature_mapping_kwargs)
    return SCM(
        45,
        0.1,
        42,
        0.3,
        efs,
        100,
        8,
    )


if __name__ == "__main__":
    gnr_graph = get_scm(1)
    # rng = np.random.default_rng(101)
    n_rows = 100
    node_dim = 8
    dataset = gnr_graph.get_dataset()
    print(dataset)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(gnr_graph.graph, seed=42)
    nx.draw(
        gnr_graph.graph,
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
