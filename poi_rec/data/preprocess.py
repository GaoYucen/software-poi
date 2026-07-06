from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COLUMNS = [
    "user_id",
    "venue_id",
    "category_id",
    "category_name",
    "latitude",
    "longitude",
    "timezone_offset",
    "utc_time",
]


@dataclass(frozen=True)
class Sample:
    user_idx: int
    history_poi: list[int]
    history_hour: list[int]
    history_weekday: list[int]
    history_delta_bucket: list[int]
    target_poi: int


def read_tsmc2014(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, sep="\t", header=None, names=COLUMNS, encoding="latin-1")
    df["utc_time"] = pd.to_datetime(
        df["utc_time"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["utc_time", "venue_id", "user_id"]).copy()
    df["local_time"] = df["utc_time"] + pd.to_timedelta(df["timezone_offset"], unit="m")
    df["hour"] = df["local_time"].dt.hour.astype(int)
    df["weekday"] = df["local_time"].dt.weekday.astype(int)
    return df


def _make_mapping(values: pd.Series) -> dict[str, int]:
    return {str(value): idx for idx, value in enumerate(sorted(values.astype(str).unique()))}


def _delta_bucket(minutes: float) -> int:
    if minutes < 0:
        return 0
    if minutes < 30:
        return 1
    if minutes < 60:
        return 2
    if minutes < 180:
        return 3
    if minutes < 360:
        return 4
    if minutes < 720:
        return 5
    if minutes < 1440:
        return 6
    return 7


def _split_indices(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    test_count = max(1, int(round(n * test_ratio))) if n >= 3 else 0
    val_count = max(1, int(round(n * val_ratio))) if n - test_count >= 3 else 0
    train_end = max(1, n - val_count - test_count)
    val_end = max(train_end, n - test_count)
    return train_end, val_end


def _build_samples(
    df: pd.DataFrame,
    max_seq_len: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for _, group in df.groupby("user_idx", sort=False):
        group = group.sort_values("utc_time")
        if len(group) < 2:
            continue
        poi = group["poi_idx"].to_numpy(dtype=np.int64)
        hours = group["hour"].to_numpy(dtype=np.int64)
        weekdays = group["weekday"].to_numpy(dtype=np.int64)
        times = group["utc_time"].astype("int64").to_numpy()
        deltas = np.zeros(len(group), dtype=np.int64)
        for i in range(1, len(group)):
            minutes = (times[i] - times[i - 1]) / 1e9 / 60.0
            deltas[i] = _delta_bucket(minutes)

        train_end, val_end = _split_indices(len(group), val_ratio, test_ratio)
        for target_pos in range(1, len(group)):
            start = max(0, target_pos - max_seq_len)
            sample = Sample(
                user_idx=int(group["user_idx"].iloc[0]),
                history_poi=poi[start:target_pos].astype(int).tolist(),
                history_hour=hours[start:target_pos].astype(int).tolist(),
                history_weekday=weekdays[start:target_pos].astype(int).tolist(),
                history_delta_bucket=deltas[start:target_pos].astype(int).tolist(),
                target_poi=int(poi[target_pos]),
            )
            if target_pos < train_end:
                split = "train"
            elif target_pos < val_end:
                split = "val"
            else:
                split = "test"
            splits[split].append(asdict(sample))
    return splits


def _build_transition_counts(train_samples: list[dict[str, Any]], num_pois: int) -> np.ndarray:
    counts = np.zeros((num_pois, 4), dtype=np.float32)
    for sample in train_samples:
        seq = sample["history_poi"] + [sample["target_poi"]]
        for src, dst in zip(seq[:-1], seq[1:]):
            counts[src, 0] += 1.0
            counts[dst, 1] += 1.0
            if src != dst:
                counts[src, 2] += 1.0
                counts[dst, 3] += 1.0
    return np.log1p(counts)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def preprocess_tsmc2014(
    raw_path: Path,
    out_dir: Path,
    city: str,
    max_seq_len: int = 20,
    min_user_checkins: int = 2,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    limit_users: int | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_tsmc2014(raw_path)
    user_counts = df["user_id"].value_counts()
    keep_users = user_counts[user_counts >= min_user_checkins].index.astype(str).tolist()
    keep_users = sorted(keep_users)
    if limit_users is not None:
        keep_users = keep_users[:limit_users]
    df = df[df["user_id"].astype(str).isin(keep_users)].copy()

    user_to_idx = _make_mapping(df["user_id"])
    poi_to_idx = _make_mapping(df["venue_id"])
    category_to_idx = _make_mapping(df["category_id"])
    df["user_idx"] = df["user_id"].astype(str).map(user_to_idx).astype(int)
    df["poi_idx"] = df["venue_id"].astype(str).map(poi_to_idx).astype(int)
    df["category_idx"] = df["category_id"].astype(str).map(category_to_idx).astype(int)
    df = df.sort_values(["user_idx", "utc_time"]).reset_index(drop=True)

    poi_meta = (
        df.groupby("poi_idx")
        .agg(
            venue_id=("venue_id", "first"),
            category_id=("category_id", "first"),
            category_name=("category_name", "first"),
            category_idx=("category_idx", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            visit_count=("poi_idx", "size"),
        )
        .sort_index()
    )
    coords = poi_meta[["latitude", "longitude"]].to_numpy(dtype=np.float32)
    coord_mean = coords.mean(axis=0)
    coord_std = coords.std(axis=0)
    coord_std[coord_std == 0.0] = 1.0
    coords_norm = (coords - coord_mean) / coord_std

    splits = _build_samples(df, max_seq_len=max_seq_len, val_ratio=val_ratio, test_ratio=test_ratio)
    transition_features = _build_transition_counts(splits["train"], num_pois=len(poi_to_idx))

    np.savez_compressed(
        out_dir / "arrays.npz",
        poi_category=poi_meta["category_idx"].to_numpy(dtype=np.int64),
        poi_coords=coords_norm.astype(np.float32),
        transition_features=transition_features,
    )
    for split, samples in splits.items():
        _write_json(out_dir / f"{split}.json", samples)
    _write_json(
        out_dir / "metadata.json",
        {
            "city": city,
            "raw_path": str(raw_path),
            "max_seq_len": max_seq_len,
            "num_users": len(user_to_idx),
            "num_pois": len(poi_to_idx),
            "num_categories": len(category_to_idx),
            "coord_mean": coord_mean.astype(float).tolist(),
            "coord_std": coord_std.astype(float).tolist(),
            "split_sizes": {split: len(samples) for split, samples in splits.items()},
        },
    )
    _write_json(out_dir / "user_to_idx.json", user_to_idx)
    _write_json(out_dir / "poi_to_idx.json", poi_to_idx)
    _write_json(out_dir / "category_to_idx.json", category_to_idx)
    poi_meta.to_json(out_dir / "poi_metadata.json", orient="index", force_ascii=False, indent=2)
    return {
        "city": city,
        "num_users": len(user_to_idx),
        "num_pois": len(poi_to_idx),
        "num_categories": len(category_to_idx),
        **{f"{split}_samples": len(samples) for split, samples in splits.items()},
    }
