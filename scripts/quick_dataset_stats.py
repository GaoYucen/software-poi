#!/usr/bin/env python
"""Compute split and closed-world counts without materializing model features."""

from __future__ import annotations

import argparse
import json

import pandas as pd


def split_indices(n: int) -> tuple[int, int]:
    test_count = max(1, int(round(n * 0.1))) if n >= 3 else 0
    val_count = max(1, int(round(n * 0.1))) if n - test_count >= 3 else 0
    train_end = max(1, n - val_count - test_count)
    return train_end, max(train_end, n - test_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw")
    args = parser.parse_args()
    columns = ["user", "venue", "category", "category_name", "latitude", "longitude", "timezone", "time"]
    frame = pd.read_csv(args.raw, sep="\t", header=None, names=columns, encoding="latin-1")
    counts = frame["user"].value_counts()
    frame = frame[frame["user"].isin(counts[counts >= 2].index)].copy()
    frame["timestamp"] = pd.to_datetime(
        frame["time"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce"
    )
    frame = frame.dropna(subset=["timestamp", "venue", "user"])
    frame = frame.sort_values(["user", "timestamp"])
    split = {"train": 0, "val": 0, "test": 0}
    train_seen = set()
    transition_edges = set()
    groups = []
    for _, group in frame.groupby("user", sort=True):
        venues = group["venue"].tolist()
        train_end, val_end = split_indices(len(venues))
        split["train"] += max(0, train_end - 1)
        split["val"] += val_end - train_end
        split["test"] += len(venues) - val_end
        train_seen.update(venues[:train_end])
        transition_edges.update(zip(venues[: train_end - 1], venues[1:train_end]))
        groups.append((venues, train_end, val_end))
    closed = {"val": 0, "test": 0}
    for venues, train_end, val_end in groups:
        closed["val"] += sum(venue in train_seen for venue in venues[train_end:val_end])
        closed["test"] += sum(venue in train_seen for venue in venues[val_end:])
    print(
        json.dumps(
            {
                "rows": len(frame),
                "users": int(frame["user"].nunique()),
                "pois": int(frame["venue"].nunique()),
                "categories": int(frame["category"].nunique()),
                "split": split,
                "train_seen_pois": len(train_seen),
                "transition_edges": len(transition_edges),
                "closed_world": closed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
