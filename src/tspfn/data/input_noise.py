import torch

from tspfn.data.utils import register_function


@register_function("generate_coloured_noise")
def generate_coloured_noise(
    nrows: int, ncols: int, generator: torch.Generator, slope_max: float = 3, slope_min: float = 0.5
) -> torch.tensor:
    slope = torch.rand(1, generator=generator) * (slope_max - slope_min) + slope_min
    white_noise = generate_white_noise(nrows, ncols, generator=generator)
    return _dye_noise(white_noise, slope=slope)


def _dye_noise(noise: torch.Tensor, slope: float) -> torch.Tensor:
    noise_spectra = torch.fft.rfft(noise, dim=0)
    noise_dc = noise_spectra[:1, :]
    filter_response = 1 / (torch.arange(1, noise.shape[0] // 2 + 1) ** slope)
    coloured_noise_spectra = noise_spectra[1:, :] * filter_response.reshape(-1, 1)
    coloured_noise_spectra = torch.vstack((noise_dc, coloured_noise_spectra))
    coloured_noise = torch.fft.irfft(coloured_noise_spectra, dim=0, n=noise.shape[0])
    return coloured_noise / torch.std(coloured_noise, dim=0)


@register_function("generate_dynamic_noise")
def generate_dynamic_noise(
    nrows: int,
    ncols: int,
    generator: torch.Generator,
    slope_max: float = 3,
    slope_min: float = 0.5,
    dyn_noise_mean: float = 0,
) -> torch.tensor:
    coloured_noise = generate_coloured_noise(
        nrows, ncols, generator=generator, slope_max=slope_max, slope_min=slope_min
    )
    dynamic_noise = torch.normal(dyn_noise_mean, torch.abs(coloured_noise), generator=generator)
    slope = torch.rand(1, generator=generator) * slope_max
    return _dye_noise(dynamic_noise, slope=slope)


@register_function("generate_white_noise")
def generate_white_noise(nrows: int, ncols: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(size=(nrows, ncols), generator=generator)


@register_function("generate_uniform_noise")
def generate_uniform_noise(nrows: int, ncols: int, generator: torch.Generator) -> torch.Tensor:
    return torch.rand(size=(nrows, ncols), generator=generator)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    generator = torch.Generator().manual_seed(8742328)

    white_noise = generate_white_noise(2000, 4, generator)
    uniform_noise = generate_uniform_noise(2000, 4, generator)

    coloured_noise = generate_coloured_noise(2000, 4, generator, 3)
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))

    dyn_noise = generate_dynamic_noise(2000, 4, generator, 3)
    dyn_noise_spectra = torch.fft.rfft(dyn_noise, dim=0)

    axes[0, 0].set_title("White noise")
    axes[0, 0].plot(white_noise)

    axes[0, 1].set_title("Uniform noise")
    axes[0, 1].plot(uniform_noise)

    axes[1, 0].set_title("Coloured noise")
    axes[1, 0].plot(coloured_noise)

    axes[1, 1].set_title("Dynamic noise")
    axes[1, 1].plot(dyn_noise)

    fig.savefig("input_noise.png", dpi=150)
