import torch

from tspfn.data.prior import Prior


class SyntheticDataset(torch.utils.data.IterableDataset):
    def __init__(self, prior: Prior, num_samples: int):
        self.prior_hp = prior
        self.num_samples = num_samples

    def __iter__(self):
        return self

    def __next__(self):
        # scm = get_scm(self.prior_hp)
        # continuous_data, categorical_covariates = scm.get_dataset()
        # select one feature to be the label
        # postprocess the features
        # ts = continuous_data[:, 0]
        # continuous_covariates = continuous_data[:, 1:]

        # return ts, continuous_covariates, categorical_covariates
        pass
