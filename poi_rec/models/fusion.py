from __future__ import annotations

import torch
from torch import nn


class DynamicFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        topology: torch.Tensor,
        semantic: torch.Tensor,
        spatial: torch.Tensor,
        temporal: torch.Tensor,
        user: torch.Tensor,
        profile: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.gate(torch.cat([topology, semantic, temporal, user, profile], dim=-1))
        static = gate * topology + (1.0 - gate) * semantic
        return self.output(torch.cat([static, spatial, temporal, user, profile], dim=-1))
