#!/usr/bin/env python
"""Extract auditable efficiency, gate, and alignment statistics from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from poi_rec.data.dataset import POISequenceDataset, load_processed_arrays
from poi_rec.models.poi_model import POIRecommendationModel
from poi_rec.training.checkpoint import load_checkpoint
from poi_rec.utils.random import resolve_device


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def retrieval_metrics(left: torch.Tensor, right: torch.Tensor, ks: tuple[int, ...] = (1, 10)) -> dict[str, float]:
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    scores = left @ right.t()
    target = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
    result: dict[str, float] = {}
    for k in ks:
        top = scores.topk(min(k, scores.shape[1]), dim=1).indices
        result[f"R@{k}"] = float(top.eq(target).any(dim=1).float().mean().item())
    result["paired_cosine"] = float((left * right).sum(dim=-1).mean().item())
    return result


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--retrieval_sample", type=int, default=2000)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = dict(checkpoint["config"])
    device = resolve_device(str(config.get("device", "auto")))
    arrays = {key: value.to(device) for key, value in load_processed_arrays(config["processed_dir"]).items()}
    model = POIRecommendationModel(checkpoint["metadata"], arrays, config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    dataset = POISequenceDataset(
        config["processed_dir"],
        "test",
        max_seq_len=int(config["max_seq_len"]),
        candidate_protocol=str(config.get("candidate_protocol", "closed_world")),
    )
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 64)), shuffle=False, num_workers=0)

    gates: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    elapsed = 0.0
    examples = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index, batch in enumerate(loader):
        if index >= args.max_batches:
            break
        batch = move(batch, device)
        start = time.perf_counter()
        _ = model(batch, include_priors=False, need_alignment_outputs=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start
        examples += int(batch["target"].shape[0])
        if model.fusion.last_gate is not None:
            gates.append(model.fusion.last_gate.cpu())
            masks.append(batch["attention_mask"].bool().cpu())

    gate_stats: dict[str, object] = {}
    if gates:
        gate_tensor = torch.cat(gates, dim=0)
        mask_tensor = torch.cat(masks, dim=0)
        valid = gate_tensor[mask_tensor]
        position_means = []
        for position in range(gate_tensor.shape[1]):
            position_valid = gate_tensor[:, position][mask_tensor[:, position]]
            position_means.append(float(position_valid.mean().item()) if position_valid.numel() else None)
        lengths = mask_tensor.sum(dim=1)
        short_values = gate_tensor[lengths.le(5)][mask_tensor[lengths.le(5)]]
        long_values = gate_tensor[lengths.ge(15)][mask_tensor[lengths.ge(15)]]
        gate_stats = {
            "mean": float(valid.mean().item()),
            "std": float(valid.std().item()),
            "saturation_below_0.05": float(valid.lt(0.05).float().mean().item()),
            "saturation_above_0.95": float(valid.gt(0.95).float().mean().item()),
            "position_means": position_means,
            "short_history_mean": float(short_values.mean().item()) if short_values.numel() else None,
            "long_history_mean": float(long_values.mean().item()) if long_values.numel() else None,
        }

    visible = torch.nonzero(arrays["train_seen_poi"].gt(0), as_tuple=False).flatten()
    generator = torch.Generator(device="cpu").manual_seed(42)
    if visible.numel() > args.retrieval_sample:
        visible = visible[torch.randperm(visible.numel(), generator=generator)[: args.retrieval_sample]]
    visible = visible.to(device)
    topology = model.topology.all_embeddings()[visible]
    semantic = model.semantic.all_embeddings()[visible]
    aligned_topology, aligned_semantic = model.alignment(topology, semantic)
    alignment_stats = {
        "before": retrieval_metrics(topology, semantic),
        "after": retrieval_metrics(aligned_topology, aligned_semantic),
        "sample_size": int(visible.numel()),
    }

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "city": config.get("city"),
        "seed": config.get("seed"),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": trainable_parameters / max(1, total_parameters),
        "full_ranking_seconds": elapsed,
        "full_ranking_examples": examples,
        "milliseconds_per_example": 1000.0 * elapsed / max(1, examples),
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
        "gate": gate_stats,
        "alignment": alignment_stats,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
