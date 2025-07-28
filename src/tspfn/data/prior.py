# from attrs import define
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
from scipy.stats import lognorm
import torch

class PriorConfig(BaseModel):
    n_datasets : int
    log_normal_a : float
    log_normal_b : float
    gamma_alpha : float
    gamma_beta : float
    lognorm_s : float
    
    redirection_probablility : float
    
    random_state : int
    
    nn_width : int
    nn_depth : int
    nn_activations : list[str]
    
    edge_probability_weights : dict[str, float]




class PriorHyperParameters:
    def __init__(self, conf : PriorConfig | None = None):
        self.random_state = conf.random_state
        self.rng = np.random.default_rng(self.random_state)

        self.lognorm = lognorm(s=conf.lognorm_s, loc=conf.log_normal_a, scale=conf.log_normal_b)
        self.n_nodes_total = self.lognorm.rvs(size=1, random_state=self.rng.integers(10_000, 100_000))
        self.n_features = lognorm.rvs(conf.log_normal_a, conf.log_normal_b, size=1, random_state=self.rng.integers(10_000, 100_000))
        self.redirection_probablility = conf.redirection_probablility
        self.gnr_hidden_dim = 10 # lognorm.rvs(conf.log_normal_a, conf.log_normal_b, size=1, random_state=self.rng.integers(10_000, 100_000))
        self.node_noise_scale = 1 #lognorm.rvs(conf.log_normal_a, conf.log_normal_b, size=1, random_state=self.rng.integers(10_000, 100_000))
        self.edge_probability_weights = conf.edge_probability_weights#{"nn":conf.edge_probability_weights.nn, "tree":conf.edge_probability_weights.tree, conf.edge_probability_weights.categorical:0.2}
        
        self.nn_width = conf.nn_width
        self.nn_depth = conf.nn_depth
        activation_ind = self.rng.choice(len(conf.nn_activations))
        self.nn_activation =  getattr(torch.nn, conf.nn_activations[activation_ind])



if __name__ == "__main__":
    import tomllib
    conf = "/home/anton/projects/tspfn/config.toml"
    with open(conf, "rb") as f:
        conf = tomllib.load(f)

    print(conf)
    prior_conf = PriorConfig.model_validate(conf["prior"])
    print(prior_conf)
    prior_hp = PriorHyperParameters(conf=prior_conf)
    print(prior_hp)
    from tspfn.data.edge_functions import nn_edge
    model = nn_edge(prior_hp)
    print(model)

    
    # s = 1
    # x = np.linspace(lognorm.ppf(0.01, s, loc=15, scale=10),
    #             lognorm.ppf(0.99, s, loc=15, scale=10), 100)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x, lognorm.pdf(x, s, loc=15, scale=10),
    #    'r', lw=5, alpha=0.6, label='lognorm pdf')
    # fig.savefig("lognorm.png")
