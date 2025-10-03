import torch

from tspfn.data.prior import PriorConfig
from tspfn.data.scm import SCM


class SyntheticDataset(torch.utils.data.IterableDataset):
    def __init__(self, prior_config: PriorConfig, seed: int = 0):
        super().__init__()
        self.prior_config = prior_config
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        return self

    def __next__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = torch.randint(0, 1_000_000, size=(1,), generator=self.generator).item()
        if worker_info is not None:
            seed *= worker_info.id + 1
        prior = self.prior_config.sample_prior(seed=seed)
        return SCM.from_prior(prior).get_dataset()
