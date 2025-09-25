import numpy as np
import torch
from pydantic import BaseModel
from scipy.stats import lognorm
from tspfn.data.utils import FunctionSamplingConfig, FunctionSampler
from tspfn.data.scm import SCM
# from attrs import define


class PriorConfig(BaseModel):
    n_datasets: int
    log_normal_a: float
    log_normal_b: float
    gamma_alpha: float
    gamma_beta: float
    lognorm_s: float

    redirection_probablility: float

    random_state: int

    nn_width: int
    nn_depth: int
    nn_activations: list[str]

    edge_probability_weights: dict[str, float]


class PriorSampler:
    __in

@dataclass
class Prior:
    n_nodes_total: int
    redirection_probablility: float
    random_state: int
    feature_node_fraction: float
    edge_function_sampler: FunctionSampler
    noise_function_sampler: FunctionSampler
    n_sample_rows: int
    node_dim: int
    edge_noise_std: float
    n_draws_feature_mapping: int
    edge_function_configs: list[FunctionSamplingConfig] = [
        {
            "function": small_nn,
            "kwargs": {},
            "weight": 3.0,
        },
        {
            "function": categorical_feature_mapping,
            "kwargs": {
                "gamma_shape": 2.0,
                "gamma_rate": 1.0,
                "min_categories": 2,
                "max_categories": 20,
            },
            "weight": 1,
        },
        {
            "function": tree_mapping,
            "kwargs": {
                "max_depth": 6,
            },
            "weight": 1,
        },
    ]
    edge_function_sampler = FunctionSampler(edge_function_configs)
    noise_function_configs: list[FunctionSamplingConfig] = [
        {
            "function": generate_white_noise,
            "kwargs": {},
            "weight": 2.0,
        },
        {
            "function": generate_coloured_noise,
            "kwargs": {
                "slope_min": 0.3,
                "slope_max": 4.0,
            },
            "weight": 5,
        },
        {
            "function": generate_dynamic_noise,
            "kwargs": {
                "slope_min": 0.5,
                "slope_max": 4.0,
                "dyn_noise_mean": 0.,
            },
            "weight": 3,
        },
        {
            "function": generate_uniform_noise,
            "kwargs": {
            },
            "weight": 2.0,
        },
    ]
    noise_function_sampler = FunctionSampler(noise_function_configs)
    SCM(
        50,
        0.1,
        42,
        0.3,
        edge_function_sampler,
        noise_function_sampler,
        1000,
        12,
        edge_noise_std=0.05,
        n_draws_feature_mapping=10,
    )

class PriorHyperParameters:
    def __init__(self, conf: PriorConfig):
        self.random_state = conf.random_state
        self.rng = np.random.default_rng(self.random_state)

        self.lognorm = lognorm(s=conf.lognorm_s, loc=conf.log_normal_a, scale=conf.log_normal_b)
        self.n_nodes_total = self.lognorm.rvs(size=1, random_state=self.rng.integers(10_000, 100_000))
        self.n_features = lognorm.rvs(
            conf.log_normal_a,
            conf.log_normal_b,
            size=1,
            random_state=self.rng.integers(10_000, 100_000),
        )
        self.redirection_probablility = conf.redirection_probablility
        self.gnr_hidden_dim = 10  # lognorm.rvs(conf.log_normal_a, conf.log_normal_b, size=1, random_state=self.rng.integers(10_000, 100_000))
        self.node_noise_scale = 1  # lognorm.rvs(conf.log_normal_a, conf.log_normal_b, size=1, random_state=self.rng.integers(10_000, 100_000))
        self.edge_probability_weights = conf.edge_probability_weights

        self.nn_width = conf.nn_width
        self.nn_depth = conf.nn_depth
        activation_ind = self.rng.choice(len(conf.nn_activations))
        self.nn_activation = getattr(torch.nn, conf.nn_activations[activation_ind])


if __name__ == "__main__":
    import tomllib

    conf = "/home/anton/projects/tspfn/config.toml"
    with open(conf, "rb") as f:
        conf_file = tomllib.load(f)

    print(conf_file)
    prior_conf = PriorConfig.model_validate(conf_file["prior"])
    print(prior_conf)
    prior_hp = PriorHyperParameters(conf=prior_conf)
    print(prior_hp)

    # s = 1
    # x = np.linspace(lognorm.ppf(0.01, s, loc=15, scale=10),
    #             lognorm.ppf(0.99, s, loc=15, scale=10), 100)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x, lognorm.pdf(x, s, loc=15, scale=10),
    #    'r', lw=5, alpha=0.6, label='lognorm pdf')
    # fig.savefig("lognorm.png")
