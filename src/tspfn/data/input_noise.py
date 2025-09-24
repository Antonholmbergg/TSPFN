from collections.abc import Callable
from functools import partial
from typing import Any, TypedDict

import torch


class NoiseFunctionConfig(TypedDict):
    function: Callable
    kwargs: dict[str, Any]
    weight: float


def generate_coloured_noise(
    nrows: int, ncols: int, generator: torch.Generator, slope_max: float = 3, slope_min: float = 0.5
) -> torch.tensor:
    slope = torch.rand(1, generator=generator) * (slope_max - slope_min) + slope_min
    white_noise = torch.randn(size=(nrows, ncols), generator=generator)
    return dye_noise(white_noise, slope=slope)


def dye_noise(noise: torch.Tensor, slope: float) -> torch.Tensor:
    noise_spectra = torch.fft.rfft(noise, dim=0)
    noise_dc = noise_spectra[:0, :]
    filter_response = 1 / (torch.arange(1, noise.shape[0] // 2 + 1) ** slope)
    coloured_noise_spectra = noise_spectra[1:, :] * filter_response.reshape(-1, 1)
    coloured_noise_spectra = torch.vstack((noise_dc, coloured_noise_spectra))
    coloured_noise = torch.fft.irfft(coloured_noise_spectra, dim=0)
    return coloured_noise / torch.std(coloured_noise, dim=0)


def generate_dynamic_noise(
    nrows: int, ncols: int, generator: torch.Generator, slope_max: float = 3, dyn_noise_mean: float = 0
) -> torch.tensor:
    coloured_noise = generate_coloured_noise(nrows, ncols, generator=generator, slope_max=slope_max)
    dynamic_noise = torch.normal(dyn_noise_mean, torch.abs(coloured_noise), generator=generator)
    slope = torch.rand(1, generator=generator) * slope_max
    return dye_noise(dynamic_noise, slope=slope)


def get_input_noise_function(
    nrows: int, ncols: int, generator: torch.Generator, noise_function_configs: list[NoiseFunctionConfig]
):
    functions: list[Callable] = []
    weights: list[float] = []
    for config in noise_function_configs:
        partial_func = partial(config["function"], **config["kwargs"])
        functions.append(partial_func)
        weights.append(config["weight"])
    weights = torch.Tensor(weights)
    function_index = int(torch.multinomial(weights, 1, replacement=False, generator=generator).item())
    noise_function = functions[function_index]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    generator = torch.Generator().manual_seed(8742328)

    white_noise = torch.randn(size=(2000, 4), generator=generator)

    coloured_noise = generate_coloured_noise(2000, 4, generator, 3)
    fig, axes = plt.subplots(3, 1, figsize=(12, 24))

    dyn_noise = generate_dynamic_noise(2000, 4, generator, 3)
    dyn_noise_spectra = torch.fft.rfft(dyn_noise, dim=0)

    axes[0].set_title("White noise")
    axes[0].plot(white_noise)

    axes[1].set_title("Coloured noise")
    axes[1].plot(coloured_noise)

    axes[2].set_title("Dynamic noise")
    axes[2].plot(dyn_noise)

    fig.savefig("input_noise.png", dpi=150)
