from typing import Self, Any
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, PrivateAttr
from scipy.stats import loguniform, gamma, poisson
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
    n_node_loguniform_params: dict[str, float]
    redirection_gamma_params: dict[str, float]
    feature_fraction_gamma_params: dict[str, float]
    node_dim_poisson_params: dict[str, float]
    edge_noise_std: float
    n_draws_feature_mapping: int
    n_rows_poisson_params: dict[str, float]
    edge_function_configs: list[FunctionSamplingConfig]
    noise_function_configs: list[FunctionSamplingConfig]
    n_draws_function_config_weights: int
    _rng: np.random.RandomState = PrivateAttr()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._rng = np.random.RandomState(seed=self.seed)

    @classmethod
    def from_yaml_config(cls, file_path: str) -> Self:
        with open(file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # need to instansiate the registered functions
        for function_sampler_group in ["noise_function_configs", "edge_function_configs"]:
            sampler_group_config = config[function_sampler_group]
            for function_sampling_config in sampler_group_config:
                function_sampling_config["function"] = get_function(function_sampling_config["function"])
        return cls(**config)

    def sample_prior(self) -> Prior:
        n_nodes = int(loguniform.rvs(**self.n_node_loguniform_params, random_state=self._rng))
        redirection_probability = gamma.rvs(**self.redirection_gamma_params, random_state=self._rng)
        feature_fraction_gamma_params = gamma.rvs(**self.feature_fraction_gamma_params, random_state=self._rng)
        n_sample_rows = poisson.rvs(**self.n_rows_poisson_params, random_state=self._rng)
        node_dim = poisson.rvs(**self.node_dim_poisson_params, random_state=self._rng)
        edge_function_sampler = self.__sample_function_configs(self.edge_function_configs)
        input_noise_function_sampler = self.__sample_function_configs(self.noise_function_configs)
        seed = self._rng.randint(0, 1_000_000_000)
        return Prior(
            n_nodes_total=n_nodes,
            redirection_probablility=redirection_probability,
            seed=seed,
            feature_node_fraction=feature_fraction_gamma_params,
            edge_function_sampler=edge_function_sampler,
            input_noise_function_sampler=input_noise_function_sampler,
            n_sample_rows=n_sample_rows,
            node_dim=node_dim,
            edge_noise_std=self.edge_noise_std,
            n_draws_feature_mapping=self.n_draws_feature_mapping,
        )

    def __sample_function_configs(self, func_sampling_configs: list[FunctionSamplingConfig]) -> FunctionSampler:
        """::TODO try to make the assosiation between weight, new weight and their corresponding config a bit more explicit"""
        weights = []
        for conf in func_sampling_configs:
            weights.append(conf["weight"])
        weights = np.array(weights, dtype="float64")
        weights /= weights.sum()
        
        new_weights = self._rng.multinomial(self.n_draws_function_config_weights, weights)

        for i, conf in enumerate(func_sampling_configs):
            conf["weight"] = new_weights[i]
        return FunctionSampler(func_sampling_configs)


if __name__ == "__main__":
    prior_config = PriorConfig.from_yaml_config("configs/data/prior_testing.yaml")
    prior1 = prior_config.sample_prior()
    print(prior1)
    prior2 = prior_config.sample_prior()
    print(prior2)
