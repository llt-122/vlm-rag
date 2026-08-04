from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualProjection(nn.Module):
    """A small trainable residual head placed after a frozen encoder."""

    def __init__(self, embedding_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(embedding_dim, rank, bias=False)
        self.up = nn.Linear(rank, embedding_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        adapted = values + self.up(F.gelu(self.down(values)))
        return F.normalize(adapted, dim=-1)


class DualTowerAdapter(nn.Module):
    """Separate lightweight heads for frozen query and page embeddings."""

    def __init__(self, embedding_dim: int, rank: int = 16) -> None:
        super().__init__()
        self.query_projection = ResidualProjection(embedding_dim, rank)
        self.page_projection = ResidualProjection(embedding_dim, rank)

    def encode_queries(self, values: torch.Tensor) -> torch.Tensor:
        return self.query_projection(values)

    def encode_pages(self, values: torch.Tensor) -> torch.Tensor:
        return self.page_projection(values)


class SharedTowerAdapter(nn.Module):
    """Apply one learned metric-preserving head to both modalities."""

    def __init__(self, embedding_dim: int, rank: int = 16) -> None:
        super().__init__()
        self.projection = ResidualProjection(embedding_dim, rank)

    def encode_queries(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values)

    def encode_pages(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values)
