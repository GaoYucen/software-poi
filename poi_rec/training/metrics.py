from __future__ import annotations

import math

import torch


@torch.no_grad()
def ranking_metric_sums(scores: torch.Tensor, target: torch.Tensor, ks: list[int]) -> tuple[dict[str, float], int]:
    max_k = min(max(ks), scores.shape[1])
    _, top_idx = torch.topk(scores, k=max_k, dim=1)
    target_col = target.view(-1, 1)
    matches = top_idx.eq(target_col)
    sums: dict[str, float] = {}
    batch_size = int(target.shape[0])
    for k in ks:
        kk = min(k, scores.shape[1])
        hit = matches[:, :kk].any(dim=1).float()
        hit_sum = hit.sum().item()
        sums[f"HR@{k}"] = hit_sum
        sums[f"Recall@{k}"] = hit_sum
        ndcg_values = []
        for row in matches[:, :kk]:
            positions = torch.nonzero(row, as_tuple=False)
            if positions.numel() == 0:
                ndcg_values.append(0.0)
            else:
                rank = int(positions[0].item()) + 1
                ndcg_values.append(1.0 / math.log2(rank + 1))
        sums[f"NDCG@{k}"] = float(sum(ndcg_values))

    target_scores = scores.gather(1, target_col)
    ranks = scores.gt(target_scores).sum(dim=1) + 1
    sums["MRR"] = torch.reciprocal(ranks.float()).sum().item()
    return sums, batch_size


@torch.no_grad()
def ranking_metrics(scores: torch.Tensor, target: torch.Tensor, ks: list[int]) -> dict[str, float]:
    sums, count = ranking_metric_sums(scores, target, ks)
    metrics = {key: value / max(1, count) for key, value in sums.items()}
    return metrics


def average_metric_dicts(items: list[tuple[dict[str, float], int]] | list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    if isinstance(items[0], tuple):
        total_count = sum(count for _, count in items)  # type: ignore[misc]
        keys = items[0][0].keys()  # type: ignore[index]
        return {
            key: sum(metric_sums[key] for metric_sums, _ in items) / max(1, total_count)  # type: ignore[misc]
            for key in keys
        }
    keys = items[0].keys()  # type: ignore[union-attr]
    return {key: sum(item[key] for item in items) / len(items) for key in keys}  # type: ignore[index]
