import math
from typing import List, Optional

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal position encoding repurposed for diffusion timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cond_proj = nn.Linear(cond_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x + self.cond_proj(cond)))


class DenoisingMLP(nn.Module):
    """MLP denoising network for 1-D latent vectors.

    Accepts an optional class label y for conditional generation. The conditioning
    signal (time embedding + optional class embedding) is injected at every
    residual block so the network can modulate its predictions at all depths.
    """

    def __init__(
        self,
        latent_dim: int,
        time_embed_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 256, 256]

        self.num_classes = num_classes

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        cond_dim = time_embed_dim
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes + 1, time_embed_dim)
            cond_dim += time_embed_dim

        self.input_proj = nn.Linear(latent_dim, hidden_dims[0])
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dims[i], cond_dim) for i in range(len(hidden_dims))]
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dims[-1], latent_dim)

        # Upsample/downsample between hidden layers of different sizes
        self.transitions = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] != hidden_dims[i + 1]:
                self.transitions.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            else:
                self.transitions.append(nn.Identity())

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self.time_embed(t)
        if self.num_classes > 0:
            label = y if y is not None else torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            cond = torch.cat([cond, self.class_embed(label)], dim=-1)

        h = self.input_proj(z)
        for i, block in enumerate(self.blocks):
            h = block(h, cond)
            h = self.dropout(h)
            if i < len(self.transitions):
                h = self.transitions[i](h)
        return self.out(h)
