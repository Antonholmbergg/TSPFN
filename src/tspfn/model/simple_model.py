from collections.abc import Callable

import lightning
import torch
from torch import nn

from tspfn.model.blocks import RMSNorm, TransformerBlock


class SimpleModel(lightning.LightningModule):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        n_attention_heads: int,
        n_transformer_blocks: int,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        output_dim: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        lr: float = 1e-3,
        norm_eps: float = 1e-5,
        *,
        bias: bool | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        torch_compile: bool = False,
        is_causal: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])
        self.linear_embedding_layer: nn.Module = nn.Linear(input_dim, model_dim)
        transformer_kwargs = {
            "embedding_dim_q": input_dim,
            "embedding_dim_k": input_dim,
            "embedding_dim_v": input_dim,
            "total_embedding_dim": model_dim * n_attention_heads,
            "n_heads": n_attention_heads,
            "ff_hidden_dim": dim_feedforward,
            "dropout": dropout,
            "norm_eps": norm_eps,
            "bias": bias,
            "device": device,
            "dtype": dtype,
            "is_causal": is_causal,
        }
        self.transformer_layers = nn.ModuleList(
            TransformerBlock(**transformer_kwargs) for _ in range(n_transformer_blocks)
        )
        self.norm = RMSNorm(model_dim, eps=norm_eps)
        self.forecast_head: nn.Module = nn.Linear(model_dim, output_dim)
        self.loss = loss
        self.lr = lr
        self.torch_compile = torch_compile

    def configure_model(self):
        self.model = nn.Sequential(self.linear_embedding_layer, *self.transformer_layers, self.norm, self.forecast_head)
        if self.torch_compile is True:
            self.model = torch.compile(self.model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], *args):  # noqa
        x, y = batch
        y_hat = self(x, y)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    device = "cuda"
    model = SimpleModel(
        input_dim=64,
        model_dim=64,
        n_attention_heads=8,
        n_transformer_blocks=3,
        loss=nn.MSELoss(),
        dim_feedforward=512,
        output_dim=64,
        torch_compile=False,
        is_causal=False,
    )
    model.configure_model()
    model.to(device)
    inp = torch.nested.nested_tensor(
        tensor_list=[
            torch.rand(100, 64, dtype=torch.float32),
            torch.rand(10, 64, dtype=torch.float32),
            torch.rand(1000, 64, dtype=torch.float32),
        ],
        layout=torch.jagged,
        dtype=torch.float32,
        device=device,
    )
    # inp = torch.rand(2, 100, 64, dtype=torch.float32).to(device)
    # print(inp.size)
    out = model(inp)
    print(out.shape)
