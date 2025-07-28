import torch

from tspfn.data import SCM, PriorHyperParameters


class SyntheticDataset(torch.utils.data.IterableDataset):
    def __init__(self, prior_hp: PriorHyperParameters, num_samples: int):
        self.prior_hp = prior_hp
        self.num_samples = num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            self.scm = SCM(self.prior_hp)
            # For now, just yield random data.
            # In a real implementation, you would use self.scm to generate samples.
            input_data = torch.randn(self.scm.n_nodes)
            target_data = self.scm(input_data)
            yield input_data, target_data
