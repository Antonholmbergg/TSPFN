from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, NotRequired, Required, TypedDict, cast

import torch
from torch import nn

from tspfn.data.utils import gamma

logger = logging.getLogger()


class EdgeMappingOutput(TypedDict):
    latent_variable: Required[torch.Tensor]
    categorical_feature: NotRequired[torch.Tensor]


def normalize(latent_variables: torch.Tensor, generator: torch.Generator):
    norm_options = ["minmax", "z-score"]
    norm_option = int(
        torch.randint(
            0,
            len(norm_options),
            size=(1,),
            generator=generator,
        ).item()
    )
    match norm_options[norm_option]:
        case "minmax":
            dim_max = torch.max(latent_variables)
            dim_min = torch.min(latent_variables)
            normalized_latent_variables = (latent_variables - dim_min) / (dim_max - dim_min)
        case "z-score":
            dim_mean = torch.mean(latent_variables)
            dim_std = torch.std(latent_variables)
            normalized_latent_variables = (latent_variables - dim_mean) / dim_std
        case _:
            msg = f"method {norm_option} is not implemnted"
            raise NotImplementedError(msg)
    return normalized_latent_variables


def add_noise(
    latent_variables: torch.Tensor,
    noise_std: float,
    generator: torch.Generator,
):
    noise = torch.normal(0, noise_std, size=latent_variables.shape, generator=generator)
    return latent_variables + noise


def categorical_feature_mapping(
    latent_variables: torch.Tensor,
    generator: torch.Generator,
    *,
    gamma_shape: float,
    gamma_rate: float,
    min_categories: int,
    max_categories: int,
) -> EdgeMappingOutput:
    """Apply categorical feature discretization mapping as an edge mapping in the SCM.

        Process:
        1. Find nearest prototype vector for each input
        2. Map to categorical index
        3. Embed categorical index back to continuous space

    Parameters
    ----------
    latent_variables : torch.Tensor
        _description_
    gamma_shape : float, optional
        _description_, by default 2.0
    gamma_rate : float, optional
        _description_, by default 1.0
    min_categories : int, optional
        _description_, by default 2
    max_categories : int, optional
        _description_, by default 20
    generator : torch.Generator | None, optional
        _description_, by default None

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        _description_
    """
    dim = latent_variables.shape[1]
    gamma_shape_tensor = torch.Tensor([gamma_shape])
    gamma_rate_tensor = torch.Tensor([gamma_rate])

    gamma_sample = gamma(gamma_shape_tensor, gamma_rate_tensor, (1,), generator)
    n_categories = max(
        min_categories, min(max_categories, int(torch.round(gamma_sample, decimals=0).item()) + min_categories)
    )
    # Initialize the prototype vectors and embeddings. For now only sampled from a standard normal distribution.
    # ::TODO Look into if these embeddings should be more diverse, other distributions, non-independence etc.
    # Also could it make sense to orthogonalize them?
    # Also should definitely fix such that they look more like the latent variales,
    # or maybe the normalization of variables will fix it anyway
    prototypes = torch.normal(0, 1, (n_categories, dim), generator=generator)
    embeddings = torch.normal(0, 1, (n_categories, dim), generator=generator)

    distances = torch.hstack(
        [
            torch.unsqueeze(
                torch.sum(torch.square(latent_variables - torch.unsqueeze(prototypes[i, :], dim=0)), dim=1), dim=1
            )
            for i in range(prototypes.shape[0])
        ],
    )
    category_indicies = torch.argmin(distances, dim=-1).to(torch.int32)
    output_embeddings = embeddings[category_indicies]
    output: EdgeMappingOutput = {"latent_variable": output_embeddings, "categorical_feature": category_indicies}
    return output


def __float_argsort(latent_variables: torch.Tensor):
    return torch.argsort(latent_variables, dim=0).to(torch.float32)


def __sample_activation_function(generator: torch.Generator) -> Callable:
    abailable_activation_functions = [
        nn.ReLU(),
        nn.LeakyReLU(),
        nn.Hardswish(),
        nn.SiLU(),
        nn.Tanh(),
        nn.Sigmoid(),
        torch.square,
        torch.exp,
        partial(torch.pow, 2),
        __float_argsort,
    ]
    choice = torch.randint(0, len(abailable_activation_functions), (1,), generator=generator)
    return cast(Callable[..., Any], abailable_activation_functions[int(choice.item())])


def small_nn(
    latent_variables: torch.Tensor,
    generator: torch.Generator,
):
    """
    ::TODO step function
    and modulo operation as activation functions.
    """
    output_function = __sample_activation_function(generator)
    _, columns = latent_variables.shape
    w = torch.empty(columns, columns)
    b = torch.empty(1, columns)
    nn.init.xavier_normal_(w, generator=generator)
    nn.init.xavier_normal_(b, generator=generator)
    projected_variables = output_function(latent_variables @ w.T + b)
    output: EdgeMappingOutput = {"latent_variable": projected_variables}
    return output


@dataclass
class DecisionNode:
    """A node in the decision tree representing a split condition."""

    feature_idx: int
    threshold: float
    output: torch.Tensor | None = None
    left_child: DecisionNode | None = None
    right_child: DecisionNode | None = None
    is_leaf: bool = False


def tree_mapping(latent_variables: torch.Tensor, generator: torch.Generator, max_depth: int) -> EdgeMappingOutput:
    """_summary_

    Parameters
    ----------
    latent_variables : torch.Tensor
        _description_
    generator : torch.Generator
        _description_
    max_depth : int
        _description_

    Returns
    -------
    EdgeMappingOutput
        _description_
    """
    latent_dim = int(latent_variables.shape[1])
    latent_med = torch.median(latent_variables, dim=0).values
    latent_std = torch.std(latent_variables, dim=0)

    dt = DecisionTreeMapping(
        max_depth=max_depth, generator=generator, latent_dim=latent_dim, latent_med=latent_med, latend_std=latent_std
    )
    outputs: EdgeMappingOutput = {"latent_variable": dt.forward(latent_variables)}
    return outputs


class DecisionTreeMapping:
    def __init__(
        self,
        max_depth: int,
        generator: torch.Generator,
        latent_dim: int,
        latent_med: torch.Tensor,
        latend_std: torch.Tensor,
    ):
        """_summary_

        Parameters
        ----------
        max_depth : int
            _description_
        generator : torch.Generator
            _description_
        """
        self.generator = generator
        self.latent_dim = latent_dim
        self.latent_med = latent_med
        self.latent_std = latend_std
        self.max_depth = max_depth
        self._sample_tree_structure()
        self.root = self._build_random_tree(depth=0)

    def _sample_tree_structure(self) -> None:
        """Sample the decision tree structure with random parameters."""
        n_features_to_use = int(torch.randint(1, self.latent_dim, size=(1,), generator=self.generator).item())
        equal_probability_weigths = torch.ones(self.latent_dim)
        self.selected_features = torch.multinomial(
            equal_probability_weigths, n_features_to_use, replacement=False, generator=self.generator
        )

    def _build_random_tree(self, depth: int, random_stop_threshold: float = 0.2) -> DecisionNode:
        """Recursively build a random decision tree."""
        randomly_stop = depth > 0 and torch.rand((1,), generator=self.generator) < random_stop_threshold
        if depth >= self.max_depth or randomly_stop:
            return DecisionNode(
                feature_idx=-1,  # Not used for leaf
                threshold=0.0,  # Not used for leaf
                output=self._sample_output_vector(),
                is_leaf=True,
            )

        feature_idx = int(torch.randint(0, self.selected_features.shape[0], (1,), generator=self.generator).item())
        feature_med = self.latent_med[self.selected_features[feature_idx]]
        feature_std = self.latent_std[self.selected_features[feature_idx]]
        threshold = float(torch.normal(feature_med, feature_std, generator=self.generator).item())

        node = DecisionNode(feature_idx=feature_idx, threshold=threshold, is_leaf=False)
        node.left_child = self._build_random_tree(depth + 1)
        node.right_child = self._build_random_tree(depth + 1)

        return node

    def _sample_output_vector(self) -> torch.Tensor:
        """Sample a random output vector for leaf nodes."""
        distribution_type = torch.randint(0, 2, (1,), generator=self.generator).item()
        match distribution_type:
            case 0:
                return torch.normal(0, 1, size=(1, self.latent_dim), generator=self.generator)
            case 1:
                return torch.rand(size=(1, self.latent_dim), generator=self.generator) * 2 - 1
            case _:
                msg = "Invalid distribution type, fix the randint to be at most the number of choices in the match statement"
                raise ValueError(msg)

    def _traverse_tree(self, latent_variables: torch.Tensor, node: DecisionNode) -> torch.Tensor:
        """Traverse the decision tree to get output for input vector latent_variables."""
        if node.is_leaf and node.output is not None:
            return node.output.clone()
        if node.left_child is None or node.right_child is None:
            msg = "One of the child nodes is None despite the node not being a leaf node."
            raise RuntimeError(msg)
        if latent_variables[node.feature_idx] <= node.threshold:
            return self._traverse_tree(latent_variables, node.left_child)
        return self._traverse_tree(latent_variables, node.right_child)

    def forward(self, latent_variables: torch.Tensor) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        latent_variables : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """

        outputs = [self._traverse_tree(latent_variables[i], self.root) for i in range(latent_variables.shape[0])]
        return torch.vstack(outputs)
