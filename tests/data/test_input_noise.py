import pytest
import torch

from tspfn.data.input_noise import generate_coloured_noise, generate_dynamic_noise, generate_white_noise


@pytest.mark.parametrize(
    "noise_generator_function", [generate_coloured_noise, generate_white_noise, generate_dynamic_noise]
)
@pytest.mark.parametrize("nrows", [9, 10])  # odd, even number of rows
@pytest.mark.parametrize("ncols", [3, 4])
def test_correct_shape(noise_generator_function, nrows, ncols):
    # getting the fft + ifft to not change the dim is not completely trivial so adding a test for it
    generator = torch.Generator()
    noise_tensor = noise_generator_function(nrows, ncols, generator)
    assert nrows == noise_tensor.shape[0]
    assert ncols == noise_tensor.shape[1]
