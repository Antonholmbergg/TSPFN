from collections.abc import Callable

import lightning
import torch
from torch import nn


class SimpleModel(lightning.LightningModule):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        n_atention_heads: int,
        n_encoder_layers: int,
        loss: Callable[[torch.Tensor], torch.Tensor],
        output_dim: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = "gelu",
        lr: float = 1e-3,
        *,
        batch_first: bool = False,
        norm_first: bool = False,
        encoder_norm: None | nn.Module = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.linear_embedding_layer: nn.Module = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            model_dim,
            n_atention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.transformer: nn.Module = nn.TransformerEncoder(
            encoder_layer,
            n_encoder_layers,
            norm=encoder_norm,
            enable_nested_tensor=enable_nested_tensor,
            mask_check=mask_check,
        )
        self.forecast_head: nn.Module = nn.Linear(model_dim, output_dim)
        self.loss = loss
        self.lr = lr

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x_embedded = self.linear_embedding_layer(inputs)
        x_latent = self.transformer(x_embedded)
        return self.forecast_head(x_latent)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], *args):
        x, y = batch
        y_hat = self(x, y)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
    forecast_head = nn.Linear(64, 1)
    src = torch.rand(100, 2, 64)
    latent_var = encoder_layer(src)
    print(latent_var.shape)
    out = forecast_head(latent_var)
    print(out.shape)
