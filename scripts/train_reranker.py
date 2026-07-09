#!/usr/bin/env python3
"""Independent CandidateReranker Training Script.

This script trains only the lightweight CandidateReranker (a tiny MLP) on top
of a frozen base model. It:
1. Loads a pre-trained checkpoint (e.g., runs/nyc_p0_optim/best.pt)
2. Freezes all base model parameters
3. For each sample, generates dynamic candidates (K=200-500) offline
4. Extracts 10 features per candidate
5. Trains only the CandidateReranker to predict the correct POI among candidates

This avoids the slow end-to-end sampled_cf training and enables fast iteration
on reranker features and hyperparameters.

Usage:
  /opt/conda/envs/py11/bin/python scripts/train_reranker.py \
    --checkpoint runs/nyc_p0_optim/best.pt \
    --config configs/nyc_p0_optim.yaml \
    --num-candidates 200 \
    --epochs 5 \
    --lr 0.01

Evaluation:
  /opt/conda/envs/py11/bin/python scripts/evaluate.py \
    --checkpoint runs/nyc_reranker/reranker_best.pt \
    --split test \
    --override dynamic_candidates.enabled=true \
    --override dynamic_candidates.num_candidates=200 \
    --override use_candidate_reranker=true \
    --override candidate_reranker_hidden_dim=32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poi_rec.data.candidates import DynamicCandidateGenerator
from poi_rec.data.dataset import POISequenceDataset, load_metadata, load_processed_arrays
from poi_rec.models.poi_model import POIRecommendationModel
from poi_rec.training.checkpoint import load_checkpoint, save_checkpoint
from poi_rec.training.metrics import ranking_metric_sums, average_metric_dicts
from poi_rec.training.train import _move_batch, _dataloader_kwargs, _configure_torch_runtime
from poi_rec.utils.random import resolve_device, set_seed


class RerankerFeatureDataset(Dataset):
    """Dataset that pre-extracts CandidateReranker features from a frozen model.

    For each sample in the base dataset, generates candidates via
    DynamicCandidateGenerator, then encodes the sequence once and caches
    the extracted 10 features per candidate.
    """

    def __init__(
        self,
        model: POIRecommendationModel,
        dataset: POISequenceDataset,
        candidate_generator: DynamicCandidateGenerator,
        device: torch.device,
        keep_on_device: bool = False,
    ) -> None:
        self.model = model
        self.base_dataset = dataset
        self.candidate_generator = candidate_generator
        self.device = device
        self.keep_on_device = keep_on_device

        # Pre-extract all features
        self.features_list: list[torch.Tensor] = []
        self.target_indices: list[int] = []
        self.coverage_mask: list[bool] = []

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )

        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch(batch, device)
                candidate_ids, candidate_mask = candidate_generator.build(batch)
                # Clamp to valid range
                candidate_ids = candidate_ids.clamp(min=0, max=model.num_pois - 1)

                # Get neural + prior scores from frozen model
                candidate_bank = model.candidate_embeddings()
                neural = model.neural_scores_for_candidates(batch, candidate_ids, candidate_bank)
                prior = model.prior_scores_for_candidates(batch, candidate_ids)
                base = neural + prior

                # Get reranker features using the frozen model's method
                rerank_scores = model.reranker_scores_for_candidates(batch, candidate_ids, base, prior)

                # Extract the 10 features as the input to reranker
                lengths = batch["attention_mask"].sum(dim=1).clamp(min=1) - 1
                last_shifted = batch["poi"][torch.arange(batch["poi"].shape[0], device=device), lengths]
                last_raw = (last_shifted - 1).clamp(min=0)
                candidate_category = model.poi_category[candidate_ids]
                last_category = model.poi_category[last_raw]
                category_match = candidate_category.eq(last_category.unsqueeze(1)).float()
                popularity = model.popularity_prior[candidate_ids].float()

                # Future shape: (B, K, 10)
                for row_idx in range(batch["target"].shape[0]):
                    valid_mask = candidate_mask[row_idx]
                    num_valid = valid_mask.sum().item()
                    if num_valid == 0:
                        continue

                    row_candidates = candidate_ids[row_idx, :num_valid]
                    row_target = batch["target"][row_idx]

                    # Is target in candidates?
                    target_idx = (row_candidates == row_target).nonzero(as_tuple=True)[0]
                    if target_idx.numel() == 0:
                        self.coverage_mask.append(False)
                        continue
                    self.coverage_mask.append(True)

                    # Extract features for this row
                    row_features = torch.zeros((num_valid, 10), dtype=torch.float32, device=device)
                    row_features[:, 0] = neural[row_idx, :num_valid].float()
                    row_features[:, 1] = prior[row_idx, :num_valid].float()
                    row_features[:, 2] = popularity[row_idx, :num_valid] if num_valid > 0 else torch.tensor([])
                    row_features[:, 3] = category_match[row_idx, :num_valid].float()

                    # Feature 4: repeat
                    poi_history = batch["poi"].detach().cpu()
                    mask_history = batch["attention_mask"].detach().cpu()
                    valid_positions = torch.nonzero(mask_history[row_idx].gt(0), as_tuple=False).flatten().tolist()
                    history_pois = set()
                    for pos in valid_positions:
                        raw = int(poi_history[row_idx, pos].item()) - 1
                        if raw >= 0:
                            history_pois.add(raw)
                    for ci in range(num_valid):
                        if int(row_candidates[ci].item()) in history_pois:
                            row_features[ci, 4] = 1.0

                    # Feature 5: spatial distance
                    last_coords = model.spatial.poi_coords[last_raw[row_idx]]
                    cand_coords = model.spatial.poi_coords[row_candidates]
                    spatial_dist = torch.sqrt(((cand_coords - last_coords.unsqueeze(0)) ** 2).sum(dim=-1).clamp(min=1e-8))
                    row_features[:, 5] = torch.exp(-spatial_dist * 10.0)

                    # Feature 6: hour match
                    hour = int(batch["hour"][row_idx, lengths[row_idx]].item())
                    row_features[:, 6] = category_match[row_idx, :num_valid].float() * 0.5 + 0.5 * (1.0 - abs(float(hour) - 12.0) / 12.0)

                    # Feature 7: last distance score
                    last_dist = torch.clamp(spatial_dist / max(model.spatial_prior_temperature, 1e-6), max=50.0)
                    row_features[:, 7] = torch.exp(-last_dist * 0.5)

                    # Features 8, 9: using candidate_category already available
                    row_cats = candidate_category[row_idx, :num_valid]
                    last_cat = last_category[row_idx]
                    row_features[:, 8] = model.category_transition_prior_matrix[last_cat.unsqueeze(0), row_cats].float()
                    row_features[:, 9] = model.user_category_prior_matrix[batch["user_idx"][row_idx].unsqueeze(0), row_cats].float()

                    # Target index in candidates
                    target_idx_val = int(target_idx[0].item())

                    if self.keep_on_device:
                        self.features_list.append(row_features.cpu())
                        self.target_indices.append(target_idx_val)
                    else:
                        self.features_list.append(row_features.cpu())
                        self.target_indices.append(target_idx_val)

        # Convert to tensors
        if self.features_list:
            self.features = torch.cat([f.unsqueeze(0) for f in self.features_list], dim=0)
            self.targets = torch.tensor(self.target_indices, dtype=torch.long)
        else:
            self.features = torch.empty((0, 0, 10), dtype=torch.float32)
            self.targets = torch.empty((0,), dtype=torch.long)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.features[idx], self.targets[idx]


def train_reranker(config: dict[str, Any], args: argparse.Namespace) -> None:
    set_seed(int(config.get("seed", 42)))

    processed_dir = Path(config["processed_dir"])
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(config.get("device", "auto")))
    _configure_torch_runtime(config, device)
    print(f"Using device: {device}")

    # Load checkpoint and model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    base_config = dict(checkpoint["config"])
    base_config.update(config)  # override with new config
    metadata = checkpoint["metadata"]

    arrays = load_processed_arrays(processed_dir)
    arrays = {key: value.to(device) for key, value in arrays.items()}

    # Enable candidate reranker in config
    base_config["use_candidate_reranker"] = True
    base_config["candidate_reranker_hidden_dim"] = args.reranker_hidden_dim
    base_config["candidate_reranker_weight"] = args.reranker_weight

    model = POIRecommendationModel(metadata, arrays, base_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)

    # Freeze all base model parameters
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if "candidate_reranker" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    print(f"Frozen: {frozen_count:,} params, Trainable (reranker only): {trainable_count:,} params")

    # Create datasets
    print("Creating candidate generators and pre-extracting features...")
    dyn_cfg = {
        "enabled": True,
        "num_candidates": args.num_candidates,
        "include_target": True,  # Always include target for training
    }
    config_with_candidates = dict(base_config)
    config_with_candidates["dynamic_candidates"] = dyn_cfg
    arrays_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in arrays.items()}

    train_dataset = POISequenceDataset(
        processed_dir,
        "train",
        max_seq_len=int(base_config["max_seq_len"]),
        candidate_protocol=str(base_config.get("candidate_protocol", "closed_world")),
    )
    val_dataset = POISequenceDataset(
        processed_dir,
        "val",
        max_seq_len=int(base_config["max_seq_len"]),
        candidate_protocol=str(base_config.get("candidate_protocol", "closed_world")),
    )

    # We need arrays on CPU for DynamicCandidateGenerator init
    arrays_for_candidates = {}
    for k, v in arrays.items():
        if torch.is_tensor(v):
            arrays_for_candidates[k] = v.cpu() if v.device.type == "cuda" else v
        else:
            arrays_for_candidates[k] = v

    candidate_gen = DynamicCandidateGenerator(processed_dir, arrays_for_candidates, config_with_candidates)

    # Small subset for quick testing
    max_train = min(args.max_train_samples, len(train_dataset)) if args.max_train_samples > 0 else len(train_dataset)
    max_val = min(args.max_val_samples, len(val_dataset)) if args.max_val_samples > 0 else len(val_dataset)

    class SubsetDataset:
        """A wrapper that returns only the first n samples."""
        def __init__(self, ds, n):
            self.ds = ds
            self.n = n
        def __len__(self):
            return min(self.n, len(self.ds))
        def __getitem__(self, idx):
            return self.ds[idx]

    train_subset = SubsetDataset(train_dataset, max_train)
    val_subset = SubsetDataset(val_dataset, max_val)

    print(f"Extracting train features from {len(train_subset)} samples (K={args.num_candidates})...")
    t0 = time.time()
    train_feat_ds = RerankerFeatureDataset(model, train_subset, candidate_gen, device)
    t1 = time.time()
    print(f"  Done in {t1 - t0:.1f}s, {len(train_feat_ds)} usable samples (target in candidates)")

    print(f"Extracting val features from {len(val_subset)} samples...")
    val_feat_ds = RerankerFeatureDataset(model, val_subset, candidate_gen, device)
    t2 = time.time()
    print(f"  Done in {t2 - t1:.1f}s, {len(val_feat_ds)} usable samples")

    if len(train_feat_ds) == 0:
        raise RuntimeError("No training samples with target in candidates. Try larger num_candidates or include_target=true.")

    train_loader = DataLoader(
        train_feat_ds,
        batch_size=min(256, len(train_feat_ds)),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_feat_ds,
        batch_size=min(256, len(val_feat_ds)),
        shuffle=False,
    )

    # Optimizer for reranker only
    optimizer = torch.optim.AdamW(
        model.candidate_reranker.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    best_accuracy = 0.0
    metrics_k = [5, 10, 20]

    for epoch in range(1, args.epochs + 1):
        model.candidate_reranker.train()
        total_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            # Reranker forward: features (B, K, 10) -> scores (B, K)
            reranker_scores = model.candidate_reranker(features)
            loss = nn.functional.cross_entropy(reranker_scores, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.shape[0]

        avg_loss = total_loss / len(train_feat_ds)

        # Validation
        model.candidate_reranker.eval()
        correct = 0
        total = 0
        metric_sums_list = []
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                reranker_scores = model.candidate_reranker(features)
                preds = reranker_scores.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.shape[0]
                metric_sums_list.append(ranking_metric_sums(reranker_scores, targets, metrics_k))

        accuracy = correct / max(1, total)
        val_metrics = average_metric_dicts(metric_sums_list)
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | Acc@1: {accuracy:.4f} | NDCG@10: {val_metrics.get('NDCG@10', 0):.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(run_dir / "reranker_best.pt", model, base_config, metadata, val_metrics)
            print(f"  -> Saved best reranker (Acc@1={accuracy:.4f})")

    print(f"Training complete. Best val Acc@1: {best_accuracy:.4f}")
    print(f"Checkpoint saved to: {run_dir / 'reranker_best.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CandidateReranker independently")
    parser.add_argument("--checkpoint", required=True, help="Path to base model checkpoint (e.g., runs/nyc_p0_optim/best.pt)")
    parser.add_argument("--config", default="configs/nyc_p0_optim.yaml", help="Configuration file path")
    parser.add_argument("--run-dir", default="runs/nyc_reranker", help="Output directory")
    parser.add_argument("--num-candidates", type=int, default=200, help="Number of dynamic candidates")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--reranker-hidden-dim", type=int, default=32, help="Reranker hidden dimension")
    parser.add_argument("--reranker-weight", type=float, default=1.0, help="Reranker output weight")
    parser.add_argument("--max-train-samples", type=int, default=5000, help="Max training samples (0=all)")
    parser.add_argument("--max-val-samples", type=int, default=1000, help="Max validation samples (0=all)")
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["run_dir"] = args.run_dir

    train_reranker(config, args)


if __name__ == "__main__":
    main()