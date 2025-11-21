import torch
from torch import nn

from tspfn.model.simple_model import SimpleModel

device = "cuda"


def test_runs_with_nested_jagged_tensors():
    model = SimpleModel(
        input_dim=64,
        model_dim=64,
        n_attention_heads=8,
        n_transformer_blocks=4,
        loss=nn.MSELoss(),
        dim_feedforward=512,
        output_dim=64,
        is_causal=False,
    )
    model.configure_model()
    model.to(device)
    inp = torch.nested.nested_tensor(
        tensor_list=[torch.rand(100, 64), torch.rand(10, 64), torch.rand(50, 64)], layout=torch.jagged, device=device
    )
    model(inp)


def test_runs_with_normal_tensors():
    model = SimpleModel(
        input_dim=64,
        model_dim=64,
        n_attention_heads=8,
        n_transformer_blocks=4,
        loss=nn.MSELoss(),
        dim_feedforward=512,
        output_dim=64,
        is_causal=True,
    )
    model.configure_model()
    model.to(device)
    inp = torch.rand(2, 100, 64, device=device)
    model(inp)
