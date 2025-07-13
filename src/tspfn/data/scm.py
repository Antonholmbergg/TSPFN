"""The sampling algorithm:
Start with an MLP architecture and dropp weights from it. 

Sample dataest with k features and n samples (n, k) do:

1. sample a number of MLP layers l from p(l) and nodes per layer h from p(h).
2. Sample weights for the "edges" E_i,j from p_w(.)
3. drop a random set of edges. Set weights to 0?  # torch.nn.utils.prune.random_structured
4. sample a set of k feature nodes and a label node
5. sample noise distributions from a p(eps) from p(p(eps)). Is this the biases?
6. sample activation functions
fix SCM
for each member of the dataset (n times):
1. sample noise variables from eps_i from the sampled distributions
2. compute the value 
"""
import torch
from tspfn.data import PriorHyperParameters
import numpy as np
from torch.nn.utils import prune


class SCM(torch.nn.Module):
    def __init__(self, prior_hp : PriorHyperParameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layers = prior_hp.n_layers
        self.n_nodes = prior_hp.n_nodes
        rng = np.random.default_rng()
        
        self.layers = [
            prune.random_structured(torch.nn.Linear(in_features=self.n_nodes, out_features=self.n_nodes,), "weight", amount=rng.random(1)[0]*0.5, dim=1) for _ in range(self.n_layers)
            ]

    def forward(self, x: torch.Tensor, *args, **kwargs):
        for layer in self.layers:
            print(layer.weight)
            x = layer(x)
        return x


if __name__ == "__main__":
    scm = SCM(PriorHyperParameters())
    print(scm.forward(torch.Tensor([1] * scm.layers[0].in_features)))