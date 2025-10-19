from typing import Self

import lightning
import torch
from torch import nn


class SimpleModel(lightning.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class TimeSeriesPFN(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, n_covariates: int) -> Self:
        self.value_embedding = nn.Linear(1 + n_covariates, d_model)
        self.time_encoding = 1
        self.transformer = nn.TransformerEncoder(...)
        self.forecast_head = nn.Linear(d_model, 1)
