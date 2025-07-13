import torch
from dataclasses import dataclass


@dataclass
class PriorConfig:
    pass


class PriorHyperParameters:
    def __init__(self, conf : PriorConfig | None = None):
        self.n_layers = 3
        self.n_nodes = 5