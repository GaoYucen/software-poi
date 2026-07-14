#!/usr/bin/env python
"""Run a baseline under the paper's chronological closed-world protocol."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader

from poi_rec.data.dataset import POISequenceDataset, load_metadata, load_processed_arrays
from poi_rec.training.metrics import average_metric_dicts, ranking_metric_sums
from poi_rec.utils.config import load_config
from poi_rec.utils.random import resolve_device, set_seed


class SequenceBaseline(nn.Module):
    def __init__(self, kind: str, num_pois: int, hidden_dim: int, max_seq_len: int) -> None:
        super().__init__()
        if kind not in {"gru4rec", "sasrec", "bert4rec"}:
            raise ValueError(f"Unsupported neural baseline: {kind}")
        self.kind = kind
        self.num_pois = num_pois
        self.mask_id = num_pois + 1
        self.embedding = nn.Embedding(num_pois + 2, hidden_dim, padding_idx=0)
        self.position = nn.Embedding(max_seq_len, hidden_dim)
        if kind == "gru4rec":
            self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def _encode(self, poi: torch.Tensor, attention_mask: torch.Tensor, causal: bool) -> torch.Tensor:
        positions = torch.arange(poi.shape[1], device=poi.device).unsqueeze(0)
        tokens = self.embedding(poi) + self.position(positions)
        if self.kind == "gru4rec":
            hidden, _ = self.encoder(tokens)
        else:
            causal_mask = None
            if causal:
                causal_mask = torch.triu(
                    torch.ones((poi.shape[1], poi.shape[1]), dtype=torch.bool, device=poi.device), diagonal=1
                )
            hidden = self.encoder(tokens, mask=causal_mask, src_key_padding_mask=attention_mask.eq(0))
        return hidden

    def forward(self, poi: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.kind == "bert4rec":
            poi = poi.clone()
            attention_mask = attention_mask.clone()
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            for row, length in enumerate(lengths.tolist()):
                if length < poi.shape[1]:
                    poi[row, length] = self.mask_id
                    attention_mask[row, length] = 1
                else:
                    poi[row, :-1] = poi[row, 1:].clone()
                    poi[row, -1] = self.mask_id
            query_positions = attention_mask.sum(dim=1) - 1
            hidden = self._encode(poi, attention_mask, causal=False)
            query = self.norm(hidden[torch.arange(poi.shape[0], device=poi.device), query_positions])
            return query @ self.embedding.weight[1 : self.num_pois + 1].t()
        hidden = self._encode(poi, attention_mask, causal=self.kind == "sasrec")
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        query = self.norm(hidden[torch.arange(poi.shape[0], device=poi.device), lengths])
        return query @ self.embedding.weight[1 : self.num_pois + 1].t()

    def masked_item_loss(self, poi: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.kind != "bert4rec":
            raise RuntimeError("masked_item_loss is only defined for bert4rec")
        selected = torch.rand(poi.shape, device=poi.device).lt(0.15).logical_and(attention_mask.bool())
        for row in range(selected.shape[0]):
            if not bool(selected[row].any()):
                length = int(attention_mask[row].sum().item())
                selected[row, max(0, length - 1)] = True
        targets = poi[selected] - 1
        masked = poi.clone()
        masked[selected] = self.mask_id
        hidden = self.norm(self._encode(masked, attention_mask, causal=False)[selected])
        scores = hidden @ self.embedding.weight[1 : self.num_pois + 1].t()
        return nn.functional.cross_entropy(scores, targets)


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_neural(model: nn.Module, loader: DataLoader, device: torch.device, train_seen: torch.Tensor) -> dict[str, float]:
    model.eval()
    batches = []
    for batch in loader:
        batch = move(batch, device)
        scores = model(batch["poi"], batch["attention_mask"])
        scores = scores.masked_fill(train_seen.unsqueeze(0).eq(0), -1e9)
        batches.append(ranking_metric_sums(scores, batch["target"], [5, 10, 20]))
    return average_metric_dicts(batches)


@torch.no_grad()
def evaluate_statistical(
    kind: str,
    loader: DataLoader,
    arrays: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    popularity = torch.log1p(arrays["train_visit_count"].to(device))
    popularity = popularity / popularity.max().clamp_min(1.0)
    train_seen = arrays["train_seen_poi"].to(device)
    transitions: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    if kind == "markov":
        grouped: dict[int, list[tuple[int, float]]] = {}
        for src, dst, count in arrays["transition_edges"].cpu().tolist():
            grouped.setdefault(int(src), []).append((int(dst), float(count)))
        for src, values in grouped.items():
            ids = torch.tensor([item[0] for item in values], dtype=torch.long, device=device)
            weights = torch.tensor([item[1] for item in values], dtype=torch.float32, device=device)
            transitions[src] = (ids, weights / weights.sum().clamp_min(1.0))
    batches = []
    for batch in loader:
        batch = move(batch, device)
        scores = popularity.unsqueeze(0).expand(batch["target"].shape[0], -1).clone()
        if kind == "markov":
            lengths = batch["attention_mask"].sum(dim=1).clamp(min=1) - 1
            last = batch["poi"][torch.arange(batch["poi"].shape[0], device=device), lengths] - 1
            for row, src in enumerate(last.tolist()):
                if src in transitions:
                    ids, values = transitions[src]
                    scores[row, ids] += 8.0 * values
        scores = scores.masked_fill(train_seen.unsqueeze(0).eq(0), -1e9)
        batches.append(ranking_metric_sums(scores, batch["target"], [5, 10, 20]))
    return average_metric_dicts(batches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["mostpopular", "markov", "gru4rec", "sasrec", "bert4rec"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(args.seed)
    device = resolve_device(str(config.get("device", "auto")))
    metadata = load_metadata(config["processed_dir"])
    arrays = load_processed_arrays(config["processed_dir"])
    train = POISequenceDataset(config["processed_dir"], "train", config["max_seq_len"], "closed_world")
    val = POISequenceDataset(config["processed_dir"], "val", config["max_seq_len"], "closed_world")
    test = POISequenceDataset(config["processed_dir"], "test", config["max_seq_len"], "closed_world")
    kwargs = {"batch_size": int(config.get("batch_size", 512)), "num_workers": 0}
    train_loader = DataLoader(train, shuffle=True, **kwargs)
    val_loader = DataLoader(val, shuffle=False, **kwargs)
    test_loader = DataLoader(test, shuffle=False, **kwargs)
    started = time.perf_counter()
    if args.model in {"mostpopular", "markov"}:
        metrics = evaluate_statistical(args.model, test_loader, arrays, device)
        trainable = 0
        total = 0
        best_val = None
    else:
        model = SequenceBaseline(args.model, int(metadata["num_pois"]), 128, int(config["max_seq_len"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        train_seen = arrays["train_seen_poi"].to(device)
        best_score = float("-inf")
        best_state = None
        best_val = None
        for _ in range(args.epochs):
            model.train()
            for batch in train_loader:
                batch = move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                if args.model == "bert4rec":
                    loss = model.masked_item_loss(batch["poi"], batch["attention_mask"])
                else:
                    scores = model(batch["poi"], batch["attention_mask"])
                    scores = scores.masked_fill(train_seen.unsqueeze(0).eq(0), -1e9)
                    loss = nn.functional.cross_entropy(scores, batch["target"])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            current = evaluate_neural(model, val_loader, device, train_seen)
            if current["NDCG@10"] > best_score:
                best_score = current["NDCG@10"]
                best_val = current
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        assert best_state is not None
        model.load_state_dict(best_state)
        metrics = evaluate_neural(model, test_loader, device, train_seen)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = {
        "city": config["city"],
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs if args.model not in {"mostpopular", "markov"} else 0,
        "protocol": "chronological-80/10/10-closed-world-full-ranking",
        "validation": best_val,
        "test": metrics,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
