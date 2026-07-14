from __future__ import annotations

import torch
from torch import nn


class DynamicFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, mode: str = "context_gate") -> None:
        super().__init__()
        if mode not in {"context_gate", "concat", "fixed_mean"}:
            raise ValueError("fusion_mode must be one of: context_gate, concat, fixed_mean.")
        self.mode = mode
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
        self.concat_projection = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.last_gate: torch.Tensor | None = None

    def forward(
        self,
        topology: torch.Tensor,
        semantic: torch.Tensor,
        spatial: torch.Tensor,
        temporal: torch.Tensor,
        user: torch.Tensor,
        profile: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "concat":
            self.last_gate = None
            return self.concat_projection(torch.cat([topology, semantic, spatial, temporal, user, profile], dim=-1))
        if self.mode == "fixed_mean":
            gate = torch.full_like(topology, 0.5)
        else:
            gate = self.gate(torch.cat([topology, semantic, temporal, user, profile], dim=-1))
        self.last_gate = gate.detach()
        static = gate * topology + (1.0 - gate) * semantic
        return self.output(torch.cat([static, spatial, temporal, user, profile], dim=-1))
