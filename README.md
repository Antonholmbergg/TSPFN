# tspfn
## The purpose of this repository is to train a [TabPFN](https://doi.org/10.1038/) style model, but with a time series prior.

This means recreating the data generation procedure described in the TabPFN paper but replacing the time independent input noise to the SCM with either coloured noise or dynamic noise. *This is very much a work in progress*. The data generation is implemented and working but needs some of the parameters tuned to give reasonable data. The data postprocessing steps are not implemented yet. I have started experiemnting with the pytorch lightning CLI for the dataset. The model is not defined yet so no training has been done on it but this is the next step.

## License

`tspfn` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.


Setup with ROCm recuires TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
