import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class TabularDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TabularVAE(nn.Module):
    """Variational autoencoder for tabular data.

    Encodes a flat feature vector into a Gaussian latent space and decodes it
    back. The KL term is weighted to control how tightly the posterior hugs N(0,I),
    which affects how well the diffusion model can learn the latent distribution.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: List[int] = None,
        kl_weight: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.encoder = TabularEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = TabularDecoder(latent_dim, hidden_dims, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.kl_weight * kl
