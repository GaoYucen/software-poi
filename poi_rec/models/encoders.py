from __future__ import annotations

import torch
from torch import nn


class TopologyEncoder(nn.Module):
    def __init__(
        self,
        num_pois: int,
        transition_features: torch.Tensor,
        node2vec_embeddings: torch.Tensor,
        topology_available: torch.Tensor,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_pois = num_pois
        self.feature_projection = nn.Linear(transition_features.shape[1], hidden_dim)
        self.node2vec_projection = nn.Linear(node2vec_embeddings.shape[1], hidden_dim)
        self.register_buffer("transition_features", transition_features.float())
        self.register_buffer("node2vec_embeddings", node2vec_embeddings.float())
        self.register_buffer("topology_available", topology_available.float())
        self.norm = nn.LayerNorm(hidden_dim)

    def _encode_raw(self, raw_poi_idx: torch.Tensor) -> torch.Tensor:
        features = self.transition_features[raw_poi_idx]
        node2vec = self.node2vec_embeddings[raw_poi_idx]
        available = self.topology_available[raw_poi_idx].unsqueeze(-1)
        encoded = self.norm(self.feature_projection(features) + self.node2vec_projection(node2vec))
        return encoded * available

    def forward(self, shifted_poi_idx: torch.Tensor) -> torch.Tensor:
        mask = shifted_poi_idx.gt(0).unsqueeze(-1)
        raw = (shifted_poi_idx - 1).clamp(min=0, max=self.num_pois - 1)
        encoded = self._encode_raw(raw)
        return encoded * mask

    def all_embeddings(self) -> torch.Tensor:
        poi_idx = torch.arange(self.transition_features.shape[0], device=self.transition_features.device)
        return self._encode_raw(poi_idx)


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        num_categories: int,
        poi_category: torch.Tensor,
        text_embeddings: torch.Tensor,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_pois = poi_category.shape[0]
        self.category_embedding = nn.Embedding(num_categories, hidden_dim)
        self.text_projection = nn.Linear(text_embeddings.shape[1], hidden_dim)
        self.register_buffer("poi_category", poi_category.long())
        self.register_buffer("text_embeddings", text_embeddings.float())
        self.norm = nn.LayerNorm(hidden_dim)

    def _encode_raw(self, raw_poi_idx: torch.Tensor) -> torch.Tensor:
        category = self.poi_category[raw_poi_idx]
        text = self.text_embeddings[raw_poi_idx]
        return self.norm(self.category_embedding(category) + self.text_projection(text))

    def forward(self, shifted_poi_idx: torch.Tensor) -> torch.Tensor:
        mask = shifted_poi_idx.gt(0).unsqueeze(-1)
        raw = (shifted_poi_idx - 1).clamp(min=0, max=self.num_pois - 1)
        encoded = self._encode_raw(raw)
        return encoded * mask

    def all_embeddings(self) -> torch.Tensor:
        poi_idx = torch.arange(self.poi_category.shape[0], device=self.poi_category.device)
        return self._encode_raw(poi_idx)


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


class SpatialEncoder(nn.Module):
    def __init__(self, poi_coords: torch.Tensor, hidden_dim: int) -> None:
        super().__init__()
        self.num_pois = poi_coords.shape[0]
        self.register_buffer("poi_coords", poi_coords.float())
        self.coord_projection = nn.Sequential(
            nn.Linear(poi_coords.shape[1], hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def _encode_raw(self, raw_poi_idx: torch.Tensor) -> torch.Tensor:
        return self.coord_projection(self.poi_coords[raw_poi_idx])

    def forward(self, shifted_poi_idx: torch.Tensor) -> torch.Tensor:
        mask = shifted_poi_idx.gt(0).unsqueeze(-1)
        raw = (shifted_poi_idx - 1).clamp(min=0, max=self.num_pois - 1)
        return self._encode_raw(raw) * mask

    def all_embeddings(self) -> torch.Tensor:
        poi_idx = torch.arange(self.poi_coords.shape[0], device=self.poi_coords.device)
        return self._encode_raw(poi_idx)
