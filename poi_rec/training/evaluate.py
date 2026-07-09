from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from poi_rec.data.candidates import DynamicCandidateGenerator
from poi_rec.data.dataset import POISequenceDataset, load_processed_arrays
from poi_rec.models.poi_model import POIRecommendationModel
from poi_rec.training.checkpoint import load_checkpoint
from poi_rec.training.train import evaluate_model
from poi_rec.utils.random import resolve_device


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split: str = "test",
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, float]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = dict(checkpoint["config"])
    if config_overrides:
        config.update(config_overrides)
    if any(key.startswith("cf_") for key in checkpoint["model_state"]):
        config["enable_collaborative"] = True
    metadata = checkpoint["metadata"]
    device = resolve_device(str(config.get("device", "auto")))
    arrays = load_processed_arrays(config["processed_dir"])
    arrays = {key: value.to(device) for key, value in arrays.items()}
    model = POIRecommendationModel(metadata, arrays, config).to(device)
    missing_keys, _ = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing_keys:
        print(f"Note: {len(missing_keys)} missing keys in checkpoint (new modules initialized from scratch): {missing_keys[:5]}...")
    candidate_generator = None
    if bool(dict(config.get("dynamic_candidates", {})).get("enabled", False)):
        candidate_generator = DynamicCandidateGenerator(config["processed_dir"], arrays, config)
    dataset = POISequenceDataset(
        config["processed_dir"],
        split,
        max_seq_len=int(config["max_seq_len"]),
        candidate_protocol=str(config.get("candidate_protocol", "all_poi")),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    metrics_k = [int(k) for k in config.get("metrics_k", [5, 10, 20])]
    if len(dataset) == 0:
        metrics: dict[str, float] = {"MRR": 0.0}
        for k in metrics_k:
            metrics[f"HR@{k}"] = 0.0
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"NDCG@{k}"] = 0.0
        return metrics
    return evaluate_model(model, loader, device, metrics_k, candidate_generator=candidate_generator)
