from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict

from tspfn.data.utils import FunctionSampler, FunctionSamplingConfig, get_function


class Prior(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    n_nodes_total: int
    redirection_probablility: float
    seed: int
    feature_node_fraction: float
    edge_function_sampler: FunctionSampler
    input_noise_function_sampler: FunctionSampler
    n_sample_rows: int
    node_dim: int
    edge_noise_std: float
    n_draws_feature_mapping: int


class PriorConfig(BaseModel):
    seed: int
    n_datasets: int
    n_node_lognorm_params: dict[str, float]
    redirection_gamma_params: dict[str, float]
    feature_fraction_gamma_params: dict[str, float]
    node_dim_poisson_params: dict[str, float]
    edge_noise_std: float
    n_draws_feature_mapping: int
    n_rows_poisson_params: dict[str, float]
    edge_function_configs: list[FunctionSamplingConfig]
    noise_function_configs: list[FunctionSamplingConfig]

    @classmethod
    def from_yaml_config(cls, file_path: str) -> Self:
        with open(file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # need to instansiate the registerd functions
        for function_sampler_group in ["noise_function_configs", "edge_function_configs"]:
            sampler_group_config = config[function_sampler_group]
            for function_sampling_config in sampler_group_config:
                function_sampling_config["function"] = get_function(function_sampling_config["function"])
        return cls(**config)

    def sample_prior(self) -> Prior:
        pass

    def __sample_function_configs(
        self,
    ) -> FunctionSampler:
        pass


if __name__ == "__main__":
    prior_config = PriorConfig.from_yaml_config("configs/data/prior_testing.yaml")
