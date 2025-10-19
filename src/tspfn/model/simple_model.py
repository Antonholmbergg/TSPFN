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
        output_dim: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable = "gelu",
        *,
        batch_first: bool = False,
        norm_first: bool = False,
        encoder_norm: None | nn.Module = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ):
        super().__init__()
        self.linear_embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            model_dim,
            n_atention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            n_encoder_layers,
            norm=encoder_norm,
            enable_nested_tensor=enable_nested_tensor,
            mask_check=mask_check,
        )
        self.forecast_head = nn.Linear(model_dim, output_dim)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self.encoder(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    forecast_head = nn.Linear(512, 1)
    src = torch.rand(100, 1, 512)
    out = forecast_head(encoder_layer(src).flatten())
    print(out.shape)
