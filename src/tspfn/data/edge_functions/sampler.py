import logging
from typing import Callable

import numpy as np

logger = logging.getLogger()
# logging.basicConfig(encoding='utf-8', level=logging.INFO)


class EdgeFunctionSampler:
    def __init__(
        self,
    ) -> None:
        logger.warning("This constructor is not actually iplermnted yet")
        self.function_prob = [0.3, 0.5, 0.2]
        self.functions = [np.exp, np.cos, np.pow]
        self.rng = np.random.default_rng(1)

    def sample(self) -> Callable:
        return self.functions[self.rng.choice(len(self.function_prob), p=self.function_prob)]


if __name__ == "__main__":
    efs = EdgeFunctionSampler()
    # res = {0: 1, 1: 0, 2: 0}
    # for i in range(10_000):
#        res[efs.sample()] += 1
#    print(res)
