from __future__ import annotations

import torch
from torch import nn


class TopologyEncoder(nn.Module):
    def __init__(self, num_pois: int, transition_features: torch.Tensor, hidden_dim: int) -> None:
        super().__init__()
        self.poi_embedding = nn.Embedding(num_pois, hidden_dim)
        self.feature_projection = nn.Linear(transition_features.shape[1], hidden_dim)
        self.register_buffer("transition_features", transition_features.float())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, poi_idx: torch.Tensor) -> torch.Tensor:
        features = self.transition_features[poi_idx]
        return self.norm(self.poi_embedding(poi_idx) + self.feature_projection(features))

    def all_embeddings(self) -> torch.Tensor:
        poi_idx = torch.arange(self.transition_features.shape[0], device=self.transition_features.device)
        return self.forward(poi_idx)


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        num_categories: int,
        poi_category: torch.Tensor,
        poi_coords: torch.Tensor,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.category_embedding = nn.Embedding(num_categories, hidden_dim)
        self.coord_projection = nn.Linear(poi_coords.shape[1], hidden_dim)
        self.register_buffer("poi_category", poi_category.long())
        self.register_buffer("poi_coords", poi_coords.float())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, poi_idx: torch.Tensor) -> torch.Tensor:
        category = self.poi_category[poi_idx]
        coords = self.poi_coords[poi_idx]
        return self.norm(self.category_embedding(category) + self.coord_projection(coords))

    def all_embeddings(self) -> torch.Tensor:
        poi_idx = torch.arange(self.poi_category.shape[0], device=self.poi_category.device)
        return self.forward(poi_idx)


class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hour_embedding = nn.Embedding(24, hidden_dim)
        self.weekday_embedding = nn.Embedding(7, hidden_dim)
        self.delta_embedding = nn.Embedding(8, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hour: torch.Tensor, weekday: torch.Tensor, delta_bucket: torch.Tensor) -> torch.Tensor:
        return self.norm(
            self.hour_embedding(hour.clamp(0, 23))
            + self.weekday_embedding(weekday.clamp(0, 6))
            + self.delta_embedding(delta_bucket.clamp(0, 7))
        )

