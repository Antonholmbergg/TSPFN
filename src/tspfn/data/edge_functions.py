import logging
from collections.abc import Callable
from typing import Literal

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
    ) -> None:
        logger.warning("This constructor is not actually iplermnted yet")
        self.function_prob = [0.3, 0.5, 0.2]
        self.functions = [nn.ReLU(), nn.LeakyReLU(), nn.Hardswish()]
        self.rng = np.random.default_rng(1)

    def sample(self) -> Callable:
        return self.functions[self.rng.choice(len(self.function_prob), p=self.function_prob)]


def normalize():
    pass


def small_nn(
    latent_variable: torch.Tensor,
    non_linearity: Literal["relu", "hardswish"],
    torch_generator: torch.Generator | None = None,
):
    output_function = non_linearity_mapping[non_linearity]
    _, columns = latent_variable.shape
    w = torch.empty(columns, columns)
    b = torch.empty(1, columns)
    nn.init.xavier_normal_(w, generator=torch_generator)
    nn.init.xavier_normal_(b, generator=torch_generator)
    return output_function(latent_variable @ w.T + b)


if __name__ == "__main__":
    pass
    # efs = EdgeFunctionSampler()
    # res = {0: 1, 1: 0, 2: 0}
    # for i in range(10_000):
#        res[efs.sample()] += 1
#    print(res)
