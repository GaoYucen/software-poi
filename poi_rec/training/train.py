from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from poi_rec.data.dataset import POISequenceDataset, load_metadata, load_processed_arrays
from poi_rec.data.preprocess import preprocess_tsmc2014
from poi_rec.losses.alignment_loss import info_nce_alignment_loss
from poi_rec.losses.recommendation_loss import next_poi_loss
from poi_rec.models.poi_model import POIRecommendationModel
from poi_rec.training.checkpoint import save_checkpoint
from poi_rec.training.metrics import average_metric_dicts, ranking_metrics
from poi_rec.utils.random import resolve_device, set_seed


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _ensure_processed(config: dict[str, Any]) -> None:
    processed_dir = Path(config["processed_dir"])
    if (processed_dir / "metadata.json").exists():
        return
    preprocess_tsmc2014(
        raw_path=Path(config["raw_path"]),
        out_dir=processed_dir,
        city=str(config.get("city", "NYC")),
        max_seq_len=int(config["max_seq_len"]),
        min_user_checkins=int(config.get("min_user_checkins", 2)),
    )


@torch.no_grad()
def evaluate_model(
    model: POIRecommendationModel,
    loader: DataLoader,
    device: torch.device,
    metrics_k: list[int],
) -> dict[str, float]:
    model.eval()
    metric_batches = []
    for batch in loader:
        batch = _move_batch(batch, device)
        output = model(batch)
        metric_batches.append(ranking_metrics(output["scores"], batch["target"], metrics_k))
    return average_metric_dicts(metric_batches)


def train_from_config(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    _ensure_processed(config)
    processed_dir = Path(config["processed_dir"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metadata = load_metadata(processed_dir)
    arrays = load_processed_arrays(processed_dir)
    train_ds = POISequenceDataset(processed_dir, "train", max_seq_len=int(config["max_seq_len"]))
    val_ds = POISequenceDataset(processed_dir, "val", max_seq_len=int(config["max_seq_len"]))
    if len(train_ds) == 0:
        raise ValueError("No training samples found. Try lowering min_user_checkins or checking the raw data.")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )

    device = resolve_device(str(config.get("device", "auto")))
    arrays = {key: value.to(device) for key, value in arrays.items()}
    model = POIRecommendationModel(metadata, arrays, config).to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    align_weight = float(config.get("align_loss_weight", 0.0))
    metrics_k = [int(k) for k in config.get("metrics_k", [5, 10, 20])]
    best_recall = float("-inf")
    best_metrics: dict[str, float] = {}

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_next = 0.0
        total_align = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            rec_loss = next_poi_loss(output["scores"], batch["target"])
            align_loss = info_nce_alignment_loss(
                output["aligned_topology"],
                output["aligned_semantic"],
                batch["attention_mask"],
            )
            loss = rec_loss + align_weight * align_loss
            loss.backward()
            if float(config.get("gradient_clip", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            total_loss += loss.item()
            total_next += rec_loss.item()
            total_align += align_loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_summary = {
            "loss": total_loss / len(train_loader),
            "next_loss": total_next / len(train_loader),
            "align_loss": total_align / len(train_loader),
        }
        val_metrics = evaluate_model(model, val_loader, device, metrics_k) if len(val_ds) else {}
        print(f"epoch {epoch}: train={train_summary} val={val_metrics}")
        monitor_key = f"Recall@{metrics_k[0]}"
        monitor = val_metrics.get(monitor_key, -train_summary["loss"])
        if monitor > best_recall:
            best_recall = monitor
            best_metrics = val_metrics
            save_checkpoint(run_dir / "best.pt", model, config, metadata, val_metrics)

    if not (run_dir / "best.pt").exists():
        save_checkpoint(run_dir / "best.pt", model, config, metadata, best_metrics)
    print(f"best checkpoint: {run_dir / 'best.pt'}")

