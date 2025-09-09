import torch

from tspfn.data.prior import PriorHyperParameters


class nn_edge(torch.nn.Module):
    def __init__(self, prior_hp: PriorHyperParameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layers = prior_hp.nn_depth
        self.n_nodes = prior_hp.nn_width
        self.layers = [
            torch.nn.Linear(in_features=self.n_nodes, out_features=self.n_nodes) for _ in range(self.n_layers)
        ]
        self.activation = torch.nn.ReLU()

        # rng = np.random.default_rng()
        # self.layers = [
        #     prune.random_structured(
        #         torch.nn.Linear(in_features=self.n_nodes, out_features=self.n_nodes,),
        #         "weight",
        #         amount=rng.random(1)[0]*0.5, dim=1) for _ in range(self.n_layers)
        #     ]

    def forward(self, x: torch.Tensor, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x
