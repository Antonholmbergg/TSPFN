import torch.nn.functional as F  # noqa
from torch import Tensor, nn


class SwiGLUNet(nn.Module):
    def __init__(self, external_dim: int, internal_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(external_dim, internal_dim, bias=False)
        self.w3 = nn.Linear(external_dim, internal_dim, bias=False)
        self.w2 = nn.Linear(internal_dim, external_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
