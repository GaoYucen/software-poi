from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from poi_rec.data.dataset import POISequenceDataset, load_processed_arrays
from poi_rec.models.poi_model import POIRecommendationModel
from poi_rec.training.checkpoint import load_checkpoint
from poi_rec.training.train import evaluate_model
from poi_rec.utils.random import resolve_device


def evaluate_checkpoint(checkpoint_path: str | Path, split: str = "test") -> dict[str, float]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    metadata = checkpoint["metadata"]
    device = resolve_device(str(config.get("device", "auto")))
    arrays = load_processed_arrays(config["processed_dir"])
    arrays = {key: value.to(device) for key, value in arrays.items()}
    model = POIRecommendationModel(metadata, arrays, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = POISequenceDataset(config["processed_dir"], split, max_seq_len=int(config["max_seq_len"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    metrics_k = [int(k) for k in config.get("metrics_k", [5, 10, 20])]
    if len(dataset) == 0:
        return {f"Recall@{k}": 0.0 for k in metrics_k} | {f"NDCG@{k}": 0.0 for k in metrics_k} | {"MRR": 0.0}
    return evaluate_model(model, loader, device, metrics_k)

