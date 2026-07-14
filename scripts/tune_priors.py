#!/usr/bin/env python
"""Tune grouped statistical-prior weights on validation data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from poi_rec.data.dataset import POISequenceDataset, load_processed_arrays
from poi_rec.models.poi_model import POIRecommendationModel
from poi_rec.training.checkpoint import load_checkpoint
from poi_rec.training.train import evaluate_model
from poi_rec.utils.random import resolve_device


BASE_GROUPS = {
    "transition": {"transition_prior_weight": 8.0, "history_transition_prior_weight": 4.0},
    "user_repeat": {
        "user_poi_prior_weight": 4.0,
        "co_visit_prior_weight": 0.5,
        "user_category_prior_weight": 1.0,
        "history_repeat_prior_weight": 2.0,
    },
    "popularity_spatial": {"popularity_prior_weight": 0.3, "spatial_prior_weight": 0.5},
    "category": {"category_transition_prior_weight": 1.0},
}


def flattened(scales: dict[str, float]) -> dict[str, float]:
    values = {
        "transition_prior_weight": 0.0,
        "history_transition_prior_weight": 0.0,
        "user_poi_prior_weight": 0.0,
        "co_visit_prior_weight": 0.0,
        "category_transition_prior_weight": 0.0,
        "user_category_prior_weight": 0.0,
        "history_repeat_prior_weight": 0.0,
        "popularity_prior_weight": 0.0,
        "spatial_prior_weight": 0.0,
    }
    for group, base in BASE_GROUPS.items():
        for key, value in base.items():
            values[key] = value * scales[group]
    return values


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = dict(checkpoint["config"])
    device = resolve_device(str(config.get("device", "auto")))
    arrays = {key: value.to(device) for key, value in load_processed_arrays(config["processed_dir"]).items()}
    model = POIRecommendationModel(checkpoint["metadata"], arrays, config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    dataset = POISequenceDataset(config["processed_dir"], "val", config["max_seq_len"], "closed_world")
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 512)), shuffle=False, num_workers=0)

    scales = {group: 0.0 for group in BASE_GROUPS}
    trials = []
    candidates = (0.0, 0.5, 1.0, 1.5, 2.0)
    for _ in range(args.rounds):
        for group in BASE_GROUPS:
            best_scale = scales[group]
            best_score = float("-inf")
            for scale in candidates:
                proposal = dict(scales)
                proposal[group] = scale
                weights = flattened(proposal)
                model.configure_priors(**weights, spatial_prior_temperature=0.05)
                metrics = evaluate_model(model, loader, device, [5, 10, 20])
                trials.append({"group": group, "scale": scale, "scales": proposal, "metrics": metrics})
                if metrics["NDCG@10"] > best_score:
                    best_score = metrics["NDCG@10"]
                    best_scale = scale
            scales[group] = best_scale

    weights = flattened(scales)
    model.configure_priors(**weights, spatial_prior_temperature=0.05)
    final_metrics = evaluate_model(model, loader, device, [5, 10, 20])
    result = {
        "selection_split": "validation",
        "objective": "NDCG@10",
        "group_scales": scales,
        "weights": weights,
        "validation_metrics": final_metrics,
        "trials": trials,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
