import torch

from tspfn.data.prior import PriorHyperParameters, get_scm


class SyntheticDataset(torch.utils.data.IterableDataset):
    def __init__(self, prior_hp: PriorHyperParameters, num_samples: int):
        self.prior_hp = prior_hp
        self.num_samples = num_samples

    def __iter__(self):
        return self

    def __next__(self):
        scm = get_scm(self.prior_hp)
        input_data = torch.randn(scm.n_nodes)
        features, labels = scm(input_data)
        return features, labels
