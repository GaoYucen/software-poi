#!/usr/bin/env python
"""Pretrain topology-semantic alignment before recommendation training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poi_rec.training.pretrain_alignment import pretrain_alignment_from_config
from poi_rec.utils.config import apply_config_overrides, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value, e.g. --override run_dir=runs/debug_v4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_config_overrides(load_config(args.config), args.override)
    pretrain_alignment_from_config(config)


if __name__ == "__main__":
    main()
