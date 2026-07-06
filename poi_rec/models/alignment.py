from __future__ import annotations

import torch
from torch import nn


class AlignmentModule(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.topology_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.semantic_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, topology: torch.Tensor, semantic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.topology_projection(topology), self.semantic_projection(semantic)

    @staticmethod
    def discrepancy(topology: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.nn.functional.cosine_similarity(topology, semantic, dim=-1)

