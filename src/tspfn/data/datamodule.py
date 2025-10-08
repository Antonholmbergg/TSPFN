import os

import lightning
from torch.utils.data import DataLoader

from tspfn.data.dataset import SyntheticDataset
from tspfn.data.prior import Prior


class TSPFNDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        prior_config_path: os.PathLike,
        train_batch_size: int,
        train_max_steps: int,
        train_seed: int,
        val_batch_size: int,
        n_val_steps: int,
        val_seed: int,
        test_batch_size: int,
        n_test_steps: int,
        test_seed: int,
    ):
        super().__init__()
        self.prior_config_path = prior_config_path
        self.prior = Prior.from_yaml_config(self.prior_config_path)

        self.train_batch_size = train_batch_size
        self.train_max_steps = train_max_steps
        self.train_seed = train_seed

        self.val_batch_size = val_batch_size
        self.val_max_steps = n_val_steps
        self.val_seed = val_seed

        self.test_batch_size = test_batch_size
        self.test_max_steps = n_test_steps
        self.test_seed = test_seed

    def prepare_data(self):
        # Don't think I need to do anything here
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_synthetic = SyntheticDataset(self.prior, self.train_seed)
            self.val_synthetic = SyntheticDataset(self.prior, self.val_seed)
            self.val_real = None

        if stage == "test":
            self.test_synthetic = SyntheticDataset(self.prior, self.test_seed)
            self.test_real = None

    def train_dataloader(self):
        return DataLoader(self.train_synthetic, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_synthetic, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.synthetic, batch_size=self.batch_size)
