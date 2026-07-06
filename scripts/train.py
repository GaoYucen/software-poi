#!/usr/bin/env python
"""Train the POI next recommendation model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poi_rec.training.train import train_from_config
from poi_rec.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path.")
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    train_from_config(config)


if __name__ == "__main__":
    main()
