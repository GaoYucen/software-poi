from __future__ import annotations

import math

import torch


@torch.no_grad()
def ranking_metrics(scores: torch.Tensor, target: torch.Tensor, ks: list[int]) -> dict[str, float]:
    max_k = min(max(ks), scores.shape[1])
    _, top_idx = torch.topk(scores, k=max_k, dim=1)
    target_col = target.view(-1, 1)
    matches = top_idx.eq(target_col)
    metrics: dict[str, float] = {}
    for k in ks:
        kk = min(k, scores.shape[1])
        hit = matches[:, :kk].any(dim=1).float()
        metrics[f"Recall@{k}"] = hit.mean().item()
        ndcg_values = []
        for row in matches[:, :kk]:
            positions = torch.nonzero(row, as_tuple=False)
            if positions.numel() == 0:
                ndcg_values.append(0.0)
            else:
                rank = int(positions[0].item()) + 1
                ndcg_values.append(1.0 / math.log2(rank + 1))
        metrics[f"NDCG@{k}"] = float(sum(ndcg_values) / max(1, len(ndcg_values)))

    full_rank = scores.argsort(dim=1, descending=True)
    reciprocal = []
    for row, label in zip(full_rank, target):
        pos = torch.nonzero(row.eq(label), as_tuple=False)
        reciprocal.append(1.0 / (int(pos[0].item()) + 1) if pos.numel() else 0.0)
    metrics["MRR"] = float(sum(reciprocal) / max(1, len(reciprocal)))
    return metrics


def average_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {key: sum(item[key] for item in items) / len(items) for key in keys}

