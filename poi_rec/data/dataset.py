from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class POISequenceDataset(Dataset):
    def __init__(self, processed_dir: str | Path, split: str, max_seq_len: int | None = None) -> None:
        self.processed_dir = Path(processed_dir)
        with (self.processed_dir / "metadata.json").open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        with (self.processed_dir / f"{split}.json").open("r", encoding="utf-8") as f:
            self.samples: list[dict[str, Any]] = json.load(f)
        self.max_seq_len = int(max_seq_len or self.metadata["max_seq_len"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        length = min(len(sample["history_poi"]), self.max_seq_len)
        offset = len(sample["history_poi"]) - length
        pad = self.max_seq_len - length

        def padded(values: list[int]) -> list[int]:
            return [0] * pad + values[offset:]

        return {
            "user_idx": torch.tensor(sample["user_idx"], dtype=torch.long),
            "poi": torch.tensor(padded(sample["history_poi"]), dtype=torch.long),
            "hour": torch.tensor(padded(sample["history_hour"]), dtype=torch.long),
            "weekday": torch.tensor(padded(sample["history_weekday"]), dtype=torch.long),
            "delta_bucket": torch.tensor(padded(sample["history_delta_bucket"]), dtype=torch.long),
            "attention_mask": torch.tensor([0] * pad + [1] * length, dtype=torch.long),
            "target": torch.tensor(sample["target_poi"], dtype=torch.long),
        }


def load_processed_arrays(processed_dir: str | Path) -> dict[str, torch.Tensor]:
    arrays = np.load(Path(processed_dir) / "arrays.npz")
    return {
        "poi_category": torch.tensor(arrays["poi_category"], dtype=torch.long),
        "poi_coords": torch.tensor(arrays["poi_coords"], dtype=torch.float32),
        "transition_features": torch.tensor(arrays["transition_features"], dtype=torch.float32),
    }


def load_metadata(processed_dir: str | Path) -> dict[str, Any]:
    with (Path(processed_dir) / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)

