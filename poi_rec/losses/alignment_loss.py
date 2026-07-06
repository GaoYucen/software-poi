from __future__ import annotations

import torch
from torch.nn import functional as F


def info_nce_alignment_loss(
    topology: torch.Tensor,
    semantic: torch.Tensor,
    attention_mask: torch.Tensor,
    poi_idx: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    valid = attention_mask.bool().reshape(-1)
    topology_flat = topology.reshape(-1, topology.shape[-1])[valid]
    semantic_flat = semantic.reshape(-1, semantic.shape[-1])[valid]
    if poi_idx is not None:
        poi_flat = poi_idx.reshape(-1)[valid]
        keep = poi_flat.gt(0)
        topology_flat = topology_flat[keep]
        semantic_flat = semantic_flat[keep]
        poi_flat = poi_flat[keep]
        if poi_flat.numel() > 0:
            _, inverse = torch.unique(poi_flat, sorted=True, return_inverse=True)
            first_positions = []
            for unique_idx in range(int(inverse.max().item()) + 1):
                first_positions.append(torch.nonzero(inverse == unique_idx, as_tuple=False)[0, 0])
            first_positions_tensor = torch.stack(first_positions)
            topology_flat = topology_flat[first_positions_tensor]
            semantic_flat = semantic_flat[first_positions_tensor]
    if topology_flat.shape[0] <= 1:
        return topology.sum() * 0.0
    topology_flat = F.normalize(topology_flat, dim=-1)
    semantic_flat = F.normalize(semantic_flat, dim=-1)
    logits = topology_flat @ semantic_flat.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def feature_level_alignment_loss(
    topology: torch.Tensor,
    semantic: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    valid = attention_mask.bool().reshape(-1)
    topology_flat = topology.reshape(-1, topology.shape[-1])[valid]
    semantic_flat = semantic.reshape(-1, semantic.shape[-1])[valid]
    if topology_flat.shape[0] <= 1:
        return topology.sum() * 0.0
    topology_features = F.normalize(topology_flat.t(), dim=-1)
    semantic_features = F.normalize(semantic_flat.t(), dim=-1)
    logits = topology_features @ semantic_features.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def combined_alignment_loss(
    topology: torch.Tensor,
    semantic: torch.Tensor,
    attention_mask: torch.Tensor,
    poi_idx: torch.Tensor,
    instance_weight: float,
    feature_weight: float,
) -> torch.Tensor:
    instance = info_nce_alignment_loss(topology, semantic, attention_mask, poi_idx=poi_idx)
    feature = feature_level_alignment_loss(topology, semantic, attention_mask)
    total_weight = instance_weight + feature_weight
    if total_weight <= 0:
        return topology.sum() * 0.0
    return (instance_weight * instance + feature_weight * feature) / total_weight
