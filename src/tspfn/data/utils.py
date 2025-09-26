from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, TypedDict

import torch

FUNCTION_REGISTRY: dict[str, Callable] = {}


def gamma(
    concentration: torch.Tensor, rate: torch.Tensor, sample_shape: tuple[int], generator: torch.Generator
) -> torch.Tensor:
    """This is the only way I found of sampling from a gamma distribution with a torch.Generator. Since using the torch.distribution.gamma.Gamma() does not permit the use of a generator for some unexplained reason.

    Parameters
    ----------
    concentration : torch.Tensor
        _description_
    rate : torch.Tensor
        _description_
    sample_shape : tuple[int]
        _description_
    generator : torch.Generator
        _description_

    Returns
    -------
    torch.Tensor
        _description_
    """
    if isinstance(sample_shape, int):
        sample_shape = (sample_shape,)

    shape = sample_shape + concentration.shape
    standard_samples = torch._standard_gamma(  # noqa:SLF001 the whole point of this function, I know it's not ideal
        concentration.expand(shape), generator=generator
    )
    samples = standard_samples / rate.expand(shape)
    samples.clamp_(min=torch.finfo(samples.dtype).tiny)
    return samples


class FunctionSamplingConfig(TypedDict):
    function: Callable
    kwargs: dict[str, Any]
    weight: float


class FunctionSampler:
    def __init__(
        self,
        function_sampling_configs: Iterable[FunctionSamplingConfig],
    ) -> None:
        self.functions: list[Callable] = []
        weights: list[float] = []
        for config in function_sampling_configs:
            partial_func = partial(config["function"], **config["kwargs"])
            self.functions.append(partial_func)
            weights.append(config["weight"])
        self.weights = torch.Tensor(weights)

    def sample(self, generator: torch.Generator) -> Callable:
        function_index = int(torch.multinomial(self.weights, 1, replacement=False, generator=generator).item())
        return self.functions[function_index]


def register_function(name: str):
    def decorator(func: Callable) -> Callable:
        FUNCTION_REGISTRY[name] = func
        return func

    return decorator


def get_function(name: str) -> Callable:
    if name not in FUNCTION_REGISTRY:
        msg = f"Function '{name}' not found in registry"
        raise ValueError(msg)
    return FUNCTION_REGISTRY[name]
