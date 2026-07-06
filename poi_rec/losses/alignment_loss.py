from __future__ import annotations

import torch
from torch.nn import functional as F


def info_nce_alignment_loss(
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
    topology_flat = F.normalize(topology_flat, dim=-1)
    semantic_flat = F.normalize(semantic_flat, dim=-1)
    logits = topology_flat @ semantic_flat.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

