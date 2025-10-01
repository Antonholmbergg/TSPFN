from typing import Self

import networkx as nx
import torch

from tspfn.data.edge_functions import (
    EdgeMappingOutput,
    add_noise,
    normalize,
)
from tspfn.data.prior import Prior
from tspfn.data.utils import FunctionSampler


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
        edge_function_sampler: FunctionSampler,
        noise_function_sampler: FunctionSampler,
        n_sample_rows: int,
        node_dim: int,
        edge_noise_std: float,
        n_draws_feature_mapping: int,
    ):
        self.generator = torch.Generator().manual_seed(random_state)
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
        self.edge_function_sampler = edge_function_sampler
        self.noise_function_sampler = noise_function_sampler
        self.feature_node_fraction = feature_node_fraction
        self.__populate_edge_functions()
        self.__set_node_features()

    @classmethod
    def from_prior(cls, prior: Prior) -> Self:
        cls(**prior.model_dump())

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
            edge_attributes[edge] = {"function": self.edge_function_sampler.sample(self.generator)}
        nx.set_edge_attributes(self.graph, edge_attributes)

    def __initialize_root_nodes(self) -> None:
        for root_node in self.root_nodes:
            noise_func = self.noise_function_sampler.sample(self.generator)
            self.graph.nodes[root_node]["latent_variables"] += noise_func(
                nrows=self.n_sample_rows, ncols=self.node_dim, generator=self.generator
            )

    def __map_edges_from_node(self, node: int) -> None:
        edge_mappings = self.graph[node]
        # This should actally always have one and only one value
        # unless I start merging different graphs to form the SCM.
        # But it's easies to write as a loop anyway.
        for successor_node, mapping in edge_mappings.items():
            latent_variables_current_node = self.graph.nodes[node]["latent_variables"].clone()
            latent_variables_current_node = normalize(latent_variables_current_node, generator=self.generator)
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
                    # Add it? (thats what I do to the embeddigns that represent the catagories too).
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


if __name__ == "__main__":
    pass
