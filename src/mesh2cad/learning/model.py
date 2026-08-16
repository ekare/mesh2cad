"""A dependency-light PyTorch message-passing baseline for triangle graphs."""

from __future__ import annotations

import torch
from torch import nn

from mesh2cad.learning.features import FEATURE_NAMES, SURFACE_CLASSES


class MessagePassingBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.normalization = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        left, right = adjacency[:, 0], adjacency[:, 1]
        sources = torch.cat((left, right))
        destinations = torch.cat((right, left))
        aggregated = torch.zeros_like(hidden)
        aggregated.index_add_(0, destinations, hidden[sources])
        degree = torch.zeros(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
        degree.index_add_(0, destinations, torch.ones_like(destinations, dtype=hidden.dtype))
        aggregated /= degree.clamp_min(1.0).unsqueeze(1)
        update = self.update(torch.cat((hidden, aggregated), dim=1))
        return torch.relu(self.normalization(hidden + update))


class TriangleGraphNet(nn.Module):
    """Predict per-triangle semantics and per-adjacency patch boundaries."""

    def __init__(self, hidden_size: int = 96, layers: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(MessagePassingBlock(hidden_size) for _ in range(layers))
        self.surface_head = nn.Linear(hidden_size, len(SURFACE_CLASSES))
        self.boundary_head = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        for block in self.blocks:
            hidden = block(hidden, adjacency)
        left_hidden = hidden[adjacency[:, 0]]
        right_hidden = hidden[adjacency[:, 1]]
        edge_features = torch.cat(
            (left_hidden, right_hidden, torch.abs(left_hidden - right_hidden)), dim=1
        )
        return self.surface_head(hidden), self.boundary_head(edge_features).squeeze(1)
