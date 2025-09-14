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
        partial(torch.argsort, dim=0)
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


if __name__ == "__main__":
    pass
