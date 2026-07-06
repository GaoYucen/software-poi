from __future__ import annotations

from typing import Any

import torch
from torch import nn

from poi_rec.models.alignment import AlignmentModule
from poi_rec.models.encoders import SemanticEncoder, TemporalEncoder, TopologyEncoder
from poi_rec.models.fusion import DynamicFusion
from poi_rec.models.gpt_backbone import GPTBackbone, last_valid_state


class POIRecommendationModel(nn.Module):
    def __init__(self, metadata: dict[str, Any], arrays: dict[str, torch.Tensor], config: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        self.num_pois = int(metadata["num_pois"])
        self.topology = TopologyEncoder(self.num_pois, arrays["transition_features"], hidden_dim)
        self.semantic = SemanticEncoder(
            int(metadata["num_categories"]),
            arrays["poi_category"],
            arrays["poi_coords"],
            hidden_dim,
        )
        self.temporal = TemporalEncoder(hidden_dim)
        self.user_embedding = nn.Embedding(int(metadata["num_users"]), hidden_dim)
        self.alignment = AlignmentModule(hidden_dim)
        self.fusion = DynamicFusion(hidden_dim, float(config.get("dropout", 0.1)))
        self.backbone = GPTBackbone(
            hidden_dim=hidden_dim,
            model_name=str(config.get("gpt_model_name", "gpt2")),
            use_pretrained=bool(config.get("use_pretrained_gpt", False)),
            freeze=bool(config.get("freeze_gpt", False)),
            unfreeze_last_n=int(config.get("unfreeze_last_n", 0)),
            layers=int(config.get("gpt_layers", 2)),
            heads=int(config.get("gpt_heads", 4)),
            max_seq_len=int(config["max_seq_len"]),
            dropout=float(config.get("dropout", 0.1)),
        )
        self.score_temperature = nn.Parameter(torch.tensor(1.0))

    def encode_sequence(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        poi = batch["poi"]
        topology = self.topology(poi)
        semantic = self.semantic(poi)
        aligned_topology, aligned_semantic = self.alignment(topology, semantic)
        temporal = self.temporal(batch["hour"], batch["weekday"], batch["delta_bucket"])
        user = self.user_embedding(batch["user_idx"]).unsqueeze(1).expand_as(temporal)
        tokens = self.fusion(aligned_topology, aligned_semantic, temporal, user)
        hidden = self.backbone(tokens, batch["attention_mask"])
        query = last_valid_state(hidden, batch["attention_mask"])
        discrepancy = self.alignment.discrepancy(aligned_topology, aligned_semantic)
        return query, aligned_topology, aligned_semantic, discrepancy

    def candidate_embeddings(self) -> torch.Tensor:
        topology = self.topology.all_embeddings()
        semantic = self.semantic.all_embeddings()
        aligned_topology, aligned_semantic = self.alignment(topology, semantic)
        return 0.5 * (aligned_topology + aligned_semantic)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        query, aligned_topology, aligned_semantic, discrepancy = self.encode_sequence(batch)
        candidates = self.candidate_embeddings()
        scale = self.score_temperature.clamp(min=0.05, max=20.0)
        scores = query @ candidates.t() / scale
        return {
            "scores": scores,
            "aligned_topology": aligned_topology,
            "aligned_semantic": aligned_semantic,
            "discrepancy": discrepancy,
        }

