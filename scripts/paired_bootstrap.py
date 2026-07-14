#!/usr/bin/env python
"""Paired bootstrap confidence intervals for two full-ranking checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from poi_rec.data.dataset import POISequenceDataset, load_processed_arrays
from poi_rec.models.poi_model import POIRecommendationModel
from poi_rec.training.checkpoint import load_checkpoint
from poi_rec.utils.random import resolve_device


def load_model(path: str, device: torch.device) -> tuple[POIRecommendationModel, dict]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    config = dict(checkpoint["config"])
    arrays = {key: value.to(device) for key, value in load_processed_arrays(config["processed_dir"]).items()}
    model = POIRecommendationModel(checkpoint["metadata"], arrays, config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, config


def per_sample(scores: torch.Tensor, target: torch.Tensor, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    top = scores.topk(k=min(k, scores.shape[1]), dim=1).indices
    matches = top.eq(target.unsqueeze(1))
    hits = matches.any(dim=1).float()
    ndcg = torch.zeros_like(hits)
    for row in range(matches.shape[0]):
        position = torch.nonzero(matches[row], as_tuple=False)
        if position.numel():
            ndcg[row] = 1.0 / math.log2(int(position[0].item()) + 2)
    return hits.cpu().numpy(), ndcg.cpu().numpy()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    reference_checkpoint = load_checkpoint(args.reference, map_location="cpu")
    device = resolve_device(str(reference_checkpoint["config"].get("device", "auto")))
    reference, config = load_model(args.reference, device)
    comparison, comparison_config = load_model(args.comparison, device)
    if config["processed_dir"] != comparison_config["processed_dir"]:
        raise ValueError("Paired bootstrap requires checkpoints evaluated on the same processed dataset.")
    dataset = POISequenceDataset(config["processed_dir"], "test", config["max_seq_len"], config["candidate_protocol"])
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 256)), shuffle=False, num_workers=0)
    ref_hr, ref_ndcg, cmp_hr, cmp_ndcg = [], [], [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        a_hr, a_ndcg = per_sample(reference(batch, include_priors=False)["scores"], batch["target"])
        b_hr, b_ndcg = per_sample(comparison(batch, include_priors=False)["scores"], batch["target"])
        ref_hr.append(a_hr)
        ref_ndcg.append(a_ndcg)
        cmp_hr.append(b_hr)
        cmp_ndcg.append(b_ndcg)
    differences = {
        "HR@10": np.concatenate(ref_hr) - np.concatenate(cmp_hr),
        "NDCG@10": np.concatenate(ref_ndcg) - np.concatenate(cmp_ndcg),
    }
    rng = np.random.default_rng(42)
    result = {}
    for metric, values in differences.items():
        bootstrap = np.empty(args.samples, dtype=np.float64)
        for index in range(args.samples):
            sample = rng.integers(0, len(values), size=len(values))
            bootstrap[index] = values[sample].mean()
        result[metric] = {
            "mean_delta": float(values.mean()),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "num_test_samples": int(len(values)),
            "bootstrap_samples": args.samples,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
