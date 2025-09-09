import logging
from collections.abc import Callable

import numpy as np
import torch

from tspfn.data.prior import PriorHyperParameters

logger = logging.getLogger()


class EdgeFunctionSampler:
    def __init__(
        self,
    ) -> None:
        logger.warning("This constructor is not actually iplermnted yet")
        self.function_prob = [0.3, 0.5, 0.2]
        self.functions = [np.exp, np.cos, np.square]
        self.rng = np.random.default_rng(1)

    def sample(self) -> Callable:
        return self.functions[self.rng.choice(len(self.function_prob), p=self.function_prob)]


class EdgeNN(torch.nn.Module):
    def __init__(self, prior_hp: PriorHyperParameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layers = prior_hp.nn_depth
        self.n_nodes = prior_hp.nn_width
        self.layers = [
            torch.nn.Linear(in_features=self.n_nodes, out_features=self.n_nodes) for _ in range(self.n_layers)
        ]
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    efs = EdgeFunctionSampler()
    # res = {0: 1, 1: 0, 2: 0}
    # for i in range(10_000):
#        res[efs.sample()] += 1
#    print(res)
