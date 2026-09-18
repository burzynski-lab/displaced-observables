"""The autoencoder LightningModule.

Architecture and loss mirror the author's ej-vae framework (``vae_net.VAENet``,
``losses.vae_loss``) so the checkpoints and feature files stay interchangeable:
a plain fully connected encoder/decoder, an ELBO of per-feature-mean MSE plus
KLD, and a z-score input normalization carried inside the checkpoint.
"""
from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


class VAENet(nn.Module):
    """Plain FC (V)AE. ``variational=False`` gives the deterministic AE."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int,
                 variational: bool = True):
        super().__init__()
        self.variational = variational
        dims = [input_dim, *hidden_dims]
        enc = []
        for i in range(len(dims) - 1):
            enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        dec = []
        ddims = [latent_dim, *hidden_dims[::-1]]
        for i in range(len(ddims) - 1):
            dec += [nn.Linear(ddims[i], ddims[i + 1]), nn.ReLU()]
        dec.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # sample only while training: evaluation is deterministic at z = mu, so
        # the anomaly score of a given jet does not move between calls
        if self.variational and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return self.decoder(z), mu, logvar


def kld_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=-1)


def sample_loss(net, x):
    """Per-jet loss: this *is* the anomaly score, so it is never reduced here."""
    recon, mu, logvar = net(x)
    mse = F.mse_loss(recon, x, reduction="none").mean(dim=-1)
    return mse + kld_loss(mu, logvar) if net.variational else mse


class AnomalyAutoencoder(L.LightningModule):
    def __init__(self, input_dim: int = 27, hidden_dims: list[int] | None = None,
                 latent_dim: int = 8, variational: bool = True, lr: float = 1e-3,
                 name: str = "vae", basis: str = "S+D",
                 preprocessor: dict | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.net = VAENet(input_dim, hidden_dims or [64, 32], latent_dim, variational)

    def setup(self, stage: str | None = None) -> None:
        # freeze the datamodule's normalization into the checkpoint, so a loaded
        # model scores raw features without needing the training data
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and getattr(dm, "preprocessor", None) is not None:
            self.hparams["preprocessor"] = dm.preprocessor

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, prefix: str):
        x = batch if torch.is_tensor(batch) else batch[0]
        loss = sample_loss(self.net, x).mean()
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite {prefix} loss")
        self.log(f"{prefix}/loss", loss, prog_bar=(prefix == "val"), sync_dist=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
