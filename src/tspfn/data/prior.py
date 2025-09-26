from typing import Self

import yaml
from pydantic import BaseModel

from tspfn.data.utils import FunctionSampler


class Prior(BaseModel):
    n_nodes_total: int
    redirection_probablility: float
    seed: int
    feature_node_fraction: float
    # edge_function_sampler: FunctionSampler
    # input_noise_function_sampler: FunctionSampler
    n_sample_rows: int
    node_dim: int
    edge_noise_std: float
    n_draws_feature_mapping: int


class PriorConfig:  # (BaseModel):
    # seed: int
    # n_datasets: int
    # log_normal_a: float
    # log_normal_b: float
    # gamma_alpha: float
    # gamma_beta: float
    # lognorm_s: float

    @classmethod
    def from_yaml_config(cls, file_path: str) -> Self:
        with open(file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            print(config)
        # return cls(**config)

    def sample_prior(self) -> Prior:
        pass

    def __sample_function_configs(
        self,
    ) -> FunctionSampler:
        pass


if __name__ == "__main__":
    PriorConfig.from_yaml_config("configs/data/prior_testing.yaml")
