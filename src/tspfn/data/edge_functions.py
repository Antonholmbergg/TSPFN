import logging
from collections.abc import Callable
from typing import Literal
from functools import partial
import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from tspfn.data.utils import gamma
from scipy.spatial.distance import cdist

logger = logging.getLogger()

non_linearity_mapping = {
    "relu": nn.ReLU(),
}


class EdgeFunctionSampler:
    def __init__(
        self,
        torch_generator: torch.Generator | None = None,
    ) -> None:
        logger.warning("This constructor is not actually iplermnted yet")
        self.function_prob = [1]
        self.functions = [small_nn]
        self.rng = np.random.default_rng(1)
        self.torch_generator = torch_generator

        self.available_mappings = [
            small_nn,
        ]

    def sample(self) -> Callable:
        return self.functions[self.rng.choice(len(self.function_prob), p=self.function_prob)]


def __normalize():
    pass


def __add_noise():
    """Noise injection: at each edge, we add random normal noise from the
    normal distribution N σ I(0, )2 ."""
    pass


def decision_tree():
    """Decision trees: to incorporate structured, rule-based dependencies,
    we implement decision trees in the SCMs. At certain edges, we select
    a subset of features and apply decision boundaries on their values
    to determine the output60
    . The decision tree parameters (feature
    splits, thresholds) are randomly sampled per edge."""
    pass


def cat_feature_mapping():
    """feature discretization: to generate categorical features
    from the numerical vectors at each node, we map the vector to the
    index of the nearest neighbour in a set of per node randomly sampled
    vectors {p1, …, pK} for a feature with K categories. This discrete index
    will be observed in the feature set as a categorical feature. We sample
    the number of categories K from a rounded gamma distribution with
    an offset of 2 to yield a minimum number of classes of 2. To further
    use these discrete class assignments in the computational graph,
    they need to be embedded as continuous values. We sample a second
    set of embedding vectors p p{ , …,  }K1 for each class and transform
    the classes to these embeddings."""
    pass


def __sample_activation_function(torch_generator: torch.Generator) -> Callable:
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
        partial(torch.argsort, dim=0),
    ]
    choice = torch.randint(0, len(abailable_activation_functions), (1,), generator=torch_generator)
    return abailable_activation_functions[choice.item()]


def small_nn(
    latent_variables: torch.Tensor,
    torch_generator: torch.Generator,
):
    """
    ::TODO step function
    and modulo operation as activation functions.
    """
    output_function = __sample_activation_function(torch_generator)
    _, columns = latent_variables.shape
    w = torch.empty(columns, columns)
    b = torch.empty(1, columns)
    nn.init.xavier_normal_(w, generator=torch_generator)
    nn.init.xavier_normal_(b, generator=torch_generator)
    return output_function(latent_variables @ w.T + b)


@dataclass
class DecisionNode:
    """A node in the decision tree representing a split condition."""

    feature_idx: int
    threshold: float
    left_output: torch.Tensor | None = None
    right_output: torch.Tensor | None = None
    left_child: Literal["DecisionNode"] | None = None
    right_child: Literal["DecisionNode"] | None = None  # not sure type is correct
    is_leaf: bool = False


class DecisionTreeEdgeMapping:
    def __init__(
        self,
        latent_dim: int,
        max_depth: int,
        min_samples_split: int,
        torch_generator: torch.Generator | None,
    ):
        """
        Initialize decision tree edge mapping.

        Args:
            latent_dim: Dimension of latent space vectors
            max_depth: Maximum depth of the decision tree
            min_samples_split: Minimum samples required to split a node
            random_state: Random seed for reproducibility
        """
        self.latent_dim = latent_dim
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.torch_generator = torch_generator
        self.root = None
        self.selected_features = None

    def _sample_tree_structure(self) -> None:
        """Sample the decision tree structure with random parameters."""
        # Select a random subset of features for splits
        # ::TODO change to pytorch
        n_features_to_use = torch.randint(1, self.latent_dim)
        equal_probability_weigths = torch.ones(self.latent_dim)
        self.selected_features = torch.multinomial(
            equal_probability_weigths, n_features_to_use, replacement=False, generator=self.torch_generator
        )
        self.root = self._build_random_tree(depth=0)

    def _build_random_tree(self, depth: int) -> DecisionNode:
        """Recursively build a random decision tree."""
        # Create leaf node if max depth reached or randomly decide to terminate
        # ::TODO change to pytorch
        if depth >= self.max_depth or (depth > 0 and self.rng.random() < 0.3):
            return DecisionNode(
                feature_idx=-1,  # Not used for leaf
                threshold=0.0,  # Not used for leaf
                left_output=self._sample_output_vector(),
                is_leaf=True,
            )

        # Sample feature and threshold for split
        feature_idx = self.rng.choice(self.selected_features)
        # Sample threshold from a reasonable range
        threshold = self.rng.normal(0, 1)  # Could be adapted based on expected input range

        node = DecisionNode(feature_idx=feature_idx, threshold=threshold, is_leaf=False)

        # Recursively create children
        node.left_child = self._build_random_tree(depth + 1)
        node.right_child = self._build_random_tree(depth + 1)

        return node

    def _sample_output_vector(self) -> np.ndarray:
        """Sample a random output vector for leaf nodes."""
        # Sample from various distributions to create diverse outputs
        distribution_type = self.rng.choice(["normal", "uniform", "sparse"])

        if distribution_type == "normal":
            return self.rng.normal(0, 1, self.output_dim)
        elif distribution_type == "uniform":
            return self.rng.uniform(-1, 1, self.output_dim)
        else:  # sparse
            output = np.zeros(self.output_dim)
            n_nonzero = self.rng.randint(1, max(2, self.output_dim // 2))
            nonzero_idx = self.rng.choice(self.output_dim, size=n_nonzero, replace=False)
            output[nonzero_idx] = self.rng.normal(0, 1, n_nonzero)
            return output

    def _traverse_tree(self, latent_variables: np.ndarray, node: DecisionNode) -> np.ndarray:
        """Traverse the decision tree to get output for input vector latent_variables."""
        if node.is_leaf:
            return node.left_output.copy()

        # Apply decision boundary
        if latent_variables[node.feature_idx] <= node.threshold:
            return self._traverse_tree(latent_variables, node.left_child)
        else:
            return self._traverse_tree(latent_variables, node.right_child)

    def forward(self, latent_variables: np.ndarray) -> np.ndarray:
        """
        Apply decision tree mapping to input vector(s).

        Args:
            latent_variables: Input vector of shape (input_dim,) or batch (batch_size, input_dim)

        Returns:
            Output vector(s) after applying decision tree mapping
        """
        # Initialize tree structure if not already done
        if self.root is None:
            self._sample_tree_structure()

        # Handle single vector or batch
        if latent_variables.ndim == 1:
            return self._traverse_tree(latent_variables, self.root)
        else:
            # Batch processing
            outputs = []
            for i in range(latent_variables.shape[0]):
                outputs.append(self._traverse_tree(latent_variables[i], self.root))
            return np.array(outputs)


def categorical_feature_mapping(
    latent_variables: torch.Tensor,
    gamma_shape: float = 2.0,
    gamma_rate: float = 1.0,
    min_categories: int = 2,
    max_categories: int = 20,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    gamma_shape = torch.Tensor([gamma_shape])
    gamma_rate = torch.Tensor([gamma_rate])

    gamma_sample = gamma(gamma_shape, gamma_rate, (1,), generator)
    n_categories = max(
        min_categories, min(max_categories, int(torch.round(gamma_sample, decimals=0).item()) + min_categories)
    )
    # Initialize the prototype vectors and embeddings. For now only sampled from a standard normal distribution.
    # ::TODO Look into if these embeddings should be more diverse, other distributions, non-independence etc.
    # Also could it make sense to orthogonalize them?
    prototypes = torch.normal(0, 1, (n_categories, dim), generator=generator)
    embeddings = torch.normal(0, 1, (n_categories, dim), generator=generator)

    distances = torch.hstack(
        [torch.unsqueeze(torch.sum(torch.square(latent_variables - torch.unsqueeze(prototypes[i, :], dim=0)), dim=1), dim=1) for i in range(prototypes.shape[0])],
    )
    category_indicies = torch.argmin(distances, dim=-1)
    output_embeddings = embeddings[category_indicies]
    return output_embeddings, category_indicies.reshape(-1, 1)


if __name__ == "__main__":
    generator = torch.Generator().manual_seed(33485289)
    latent_variables = torch.normal(1, 4, size=(100, 6), generator=generator)
    cont_features, category = categorical_feature_mapping(latent_variables, generator=generator)
    print(latent_variables, cont_features, category)
    print(latent_variables.shape, cont_features.shape, category.shape)
