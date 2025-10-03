import torch
import math
from tspfn.data.prior import Prior, PriorConfig

class SyntheticDataset(torch.utils.data.IterableDataset):
    def __init__(self, prior_config: PriorConfig):
        super(SyntheticDataset).__init__()
        self.prior_config = prior_config

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))

class SyntheticDataset(torch.utils.data.IterableDataset):
    def __init__(self, prior: Prior, num_samples: int):
        self.prior_hp = prior
        self.num_samples = num_samples

    def __iter__(self):
        return self

    def __next__(self):
        pass
