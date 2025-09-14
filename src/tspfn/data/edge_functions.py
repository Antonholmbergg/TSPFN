import logging
from collections.abc import Callable
from typing import Literal
from functools import partial
import numpy as np
import torch
from torch import nn

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
    latent_variable: torch.Tensor,
    torch_generator: torch.Generator,
):
    """we apply element-wise nonlinear activation func-
    tions R Rσ : →d d , randomly sampled from a set, including identity,
    logarithm, sigmoid, absolute value, sine, hyperbolic tangent, rank
    operation, squaring, power functions, smooth ReLU 59
    , step function
    and modulo operation."""
    output_function = __sample_activation_function(torch_generator)
    _, columns = latent_variable.shape
    w = torch.empty(columns, columns)
    b = torch.empty(1, columns)
    nn.init.xavier_normal_(w, generator=torch_generator)
    nn.init.xavier_normal_(b, generator=torch_generator)
    return output_function(latent_variable @ w.T + b)


from dataclasses import dataclass


@dataclass
class DecisionNode:
    """A node in the decision tree representing a split condition."""

    feature_idx: int
    threshold: float
    left_output: np.ndarray | None = None
    right_output: np.ndarray | None = None
    left_child: Literal["DecisionNode"] | None = None
    right_child: Literal["DecisionNode"] | None = None
    is_leaf: bool = False


class DecisionTreeEdgeMapping:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: int | None = None,
        torch_generator: torch.Generator | None = None,
    ):
        """
        Initialize decision tree edge mapping.

        Args:
            input_dim: Dimension of input vectors
            output_dim: Dimension of output vectors
            max_depth: Maximum depth of the decision tree
            min_samples_split: Minimum samples required to split a node
            random_state: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = np.random.RandomState(random_state)
        self.root = None
        self.selected_features = None
        self.torch_generator = torch_generator

    def _sample_tree_structure(self) -> None:
        """Sample the decision tree structure with random parameters."""
        # Select a random subset of features for splits
        # ::TODO change to pytorch
        n_features_to_use = self.rng.randint(1, min(self.input_dim + 1, 6))  # Use 1-5 features
        self.selected_features = self.rng.choice(self.input_dim, size=n_features_to_use, replace=False)

        # Build tree with random splits and outputs
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

    def _traverse_tree(self, x: np.ndarray, node: DecisionNode) -> np.ndarray:
        """Traverse the decision tree to get output for input vector x."""
        if node.is_leaf:
            return node.left_output.copy()

        # Apply decision boundary
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left_child)
        else:
            return self._traverse_tree(x, node.right_child)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply decision tree mapping to input vector(s).

        Args:
            x: Input vector of shape (input_dim,) or batch (batch_size, input_dim)

        Returns:
            Output vector(s) after applying decision tree mapping
        """
        # Initialize tree structure if not already done
        if self.root is None:
            self._sample_tree_structure()

        # Handle single vector or batch
        if x.ndim == 1:
            return self._traverse_tree(x, self.root)
        else:
            # Batch processing
            outputs = []
            for i in range(x.shape[0]):
                outputs.append(self._traverse_tree(x[i], self.root))
            return np.array(outputs)


if __name__ == "__main__":
    pass
