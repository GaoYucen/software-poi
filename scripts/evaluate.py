#!/usr/bin/env python
"""Evaluate a trained POI recommendation checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poi_rec.training.evaluate import evaluate_checkpoint
from poi_rec.utils.config import apply_config_overrides, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path produced by train.py.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", help="Optional YAML config whose values override the checkpoint config for evaluation.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a checkpoint config value for evaluation, e.g. --override transition_prior_weight=8.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = load_config(args.config) if args.config else {}
    overrides = apply_config_overrides(overrides, args.override)
    metrics = evaluate_checkpoint(args.checkpoint, split=args.split, config_overrides=overrides)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
