import torch


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
