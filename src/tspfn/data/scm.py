from typing import Literal

import networkx as nx
import torch

from tspfn.data.edge_functions import (
    EdgeFunctionConfig,
    EdgeFunctionSampler,
    EdgeMappingOutput,
    add_noise,
    categorical_feature_mapping,
    normalize,
    small_nn,
    tree_mapping,
)
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
        n_sample_rows: int,
        node_dim: int,
        edge_normalization_dim: Literal[0, 1] | None,
        edge_noise_std: float,
        n_draws_feature_mapping: int,
    ):
        self.generator = torch.Generator().manual_seed(random_state)
        self.edge_normalization_dim = edge_normalization_dim
        self.edge_noise_std = edge_noise_std
        self.n_draws_feature_mapping = n_draws_feature_mapping
        self.n_sample_rows = n_sample_rows
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
                "latent_variables": torch.zeros((self.n_sample_rows, self.node_dim)),
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
                0, 1, size=(self.n_sample_rows, self.node_dim), generator=self.generator
            )

    def __map_edges_from_node(self, node: int) -> None:
        edge_mappings = self.graph[node]
        # This should actally always have one and only one value
        # unless I start mergin different graphs to form the SCM.
        for successor_node, mapping in edge_mappings.items():
            latent_variables_current_node = self.graph.nodes[node]["latent_variables"].clone()
            latent_variables_current_node = normalize(
                latent_variables_current_node, generator=self.generator, dim=self.edge_normalization_dim
            )
            latent_variables_current_node = add_noise(
                latent_variables_current_node, noise_std=self.edge_noise_std, generator=self.generator
            )
            mapping_output: EdgeMappingOutput = mapping["function"](
                latent_variables_current_node, generator=self.generator
            )
            latent_variables = mapping_output["latent_variable"]

            cat_feature = mapping_output.get("categorical_feature")
            self.graph.nodes[successor_node]["latent_variables"] += latent_variables
            if cat_feature is not None:
                if self.graph.nodes[successor_node]["categorical_feature"] is None:
                    self.graph.nodes[successor_node]["categorical_feature"] = cat_feature
                else:
                    # This is questionable but I'm not sure what to do in this situation yet.
                    # ignore it? concatenate it?
                    # Add it (thats what I do to the embeddigns that represent the catagories too)?
                    self.graph.nodes[successor_node]["categorical_feature"] += cat_feature

    def __proppagate(
        self,
    ) -> None:
        for generation in nx.topological_generations(self.graph):
            for node in generation:
                self.__map_edges_from_node(node)

    def get_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.__initialize_root_nodes()
        self.__proppagate()
        continuous_data = []
        categorical_data = []
        for node in self.feature_nodes:
            continuos_feature_mapping = torch.zeros(self.node_dim)
            for ind in torch.multinomial(
                torch.ones(self.node_dim), self.n_draws_feature_mapping, replacement=True, generator=self.generator
            ):
                continuos_feature_mapping[ind] += 1 / self.n_draws_feature_mapping
            categorical_feature = self.graph.nodes[node]["categorical_feature"]
            if categorical_feature is not None:
                categorical_data.append(categorical_feature.reshape(-1, 1))
            else:
                continuos_feature = torch.matmul(self.graph.nodes[node]["latent_variables"], continuos_feature_mapping)
                continuous_data.append(continuos_feature.reshape(-1, 1))
        return torch.hstack(continuous_data), torch.hstack(categorical_data)


def get_scm(prior_hp: PriorHyperParameters) -> SCM:
    random_state = 84395
    function_configs: list[EdgeFunctionConfig] = [
        {
            "function": small_nn,
            "kwargs": {},
            "weight": 3.0,
        },
        {
            "function": categorical_feature_mapping,
            "kwargs": {
                "gamma_shape": 2.0,
                "gamma_rate": 1.0,
                "min_categories": 2,
                "max_categories": 20,
            },
            "weight": 1,
        },
        {
            "function": tree_mapping,
            "kwargs": {
                "max_depth": 6,
            },
            "weight": 1,
        },
    ]
    efs = EdgeFunctionSampler(random_state, function_configs)
    return SCM(
        45,
        0.1,
        42,
        0.3,
        efs,
        1000,
        12,
        edge_normalization_dim=0,
        edge_noise_std=0.05,
        n_draws_feature_mapping=10,
    )


if __name__ == "__main__":
    gnr_graph = get_scm(1)
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
        arrowsize=10,
        node_size=700,
    )
    plt.title("Generated GNR Graph")
    fig.savefig("gnr_graph.png")
