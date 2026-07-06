from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

PAD_ID = 0


class POISequenceDataset(Dataset):
    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        max_seq_len: int | None = None,
        candidate_protocol: str = "all_poi",
    ) -> None:
        self.processed_dir = Path(processed_dir)
        with (self.processed_dir / "metadata.json").open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        with (self.processed_dir / f"{split}.json").open("r", encoding="utf-8") as f:
            self.samples: list[dict[str, Any]] = json.load(f)
        if candidate_protocol == "closed_world" and split in {"val", "test"}:
            arrays = np.load(self.processed_dir / "arrays.npz")
            train_seen = arrays["train_seen_poi"].astype(bool)
            self.samples = [sample for sample in self.samples if train_seen[int(sample["target_poi"])]]
        elif candidate_protocol not in {"all_poi", "closed_world"}:
            raise ValueError(f"Unknown candidate_protocol: {candidate_protocol}")
        self.max_seq_len = int(max_seq_len or self.metadata["max_seq_len"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        length = min(len(sample["history_poi"]), self.max_seq_len)
        offset = len(sample["history_poi"]) - length
        pad = self.max_seq_len - length

        def padded(values: list[int], shift_poi: bool = False) -> list[int]:
            clipped = values[offset:]
            if shift_poi:
                clipped = [value + 1 for value in clipped]
            return clipped + [0] * pad

        return {
            "user_idx": torch.tensor(sample["user_idx"], dtype=torch.long),
            "poi": torch.tensor(padded(sample["history_poi"], shift_poi=True), dtype=torch.long),
            "hour": torch.tensor(padded(sample["history_hour"]), dtype=torch.long),
            "weekday": torch.tensor(padded(sample["history_weekday"]), dtype=torch.long),
            "delta_bucket": torch.tensor(padded(sample["history_delta_bucket"]), dtype=torch.long),
            "attention_mask": torch.tensor([1] * length + [0] * pad, dtype=torch.long),
            "target": torch.tensor(sample["target_poi"], dtype=torch.long),
        }


def load_processed_arrays(processed_dir: str | Path) -> dict[str, torch.Tensor]:
    arrays = np.load(Path(processed_dir) / "arrays.npz")
    user_poi_edges = (
        torch.tensor(arrays["user_poi_edges"], dtype=torch.float32)
        if "user_poi_edges" in arrays
        else torch.zeros((0, 3), dtype=torch.float32)
    )
    return {
        "poi_category": torch.tensor(arrays["poi_category"], dtype=torch.long),
        "poi_coords": torch.tensor(arrays["poi_coords"], dtype=torch.float32),
        "transition_features": torch.tensor(arrays["transition_features"], dtype=torch.float32),
        "transition_edges": torch.tensor(arrays["transition_edges"], dtype=torch.float32),
        "node2vec_embeddings": torch.tensor(arrays["node2vec_embeddings"], dtype=torch.float32),
        "topology_available": torch.tensor(arrays["topology_available"], dtype=torch.float32),
        "train_seen_poi": torch.tensor(arrays["train_seen_poi"], dtype=torch.float32),
        "train_visit_count": torch.tensor(arrays["train_visit_count"], dtype=torch.float32),
        "user_poi_edges": user_poi_edges,
        "text_embeddings": torch.tensor(arrays["text_embeddings"], dtype=torch.float32),
    }


def load_metadata(processed_dir: str | Path) -> dict[str, Any]:
    with (Path(processed_dir) / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)
