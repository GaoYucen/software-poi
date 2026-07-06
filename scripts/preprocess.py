#!/usr/bin/env python
"""Preprocess TSMC2014/Foursquare check-ins for next POI recommendation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poi_rec.data.preprocess import preprocess_tsmc2014
from poi_rec.utils.config import apply_config_overrides, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML config; CLI flags override only when set.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value, e.g. --override processed_dir=processed/NYC_tfidf_v4.",
    )
    parser.add_argument("--city", default="NYC", help="City label stored in metadata.")
    parser.add_argument("--raw", default=None, help="Path to raw TSMC2014 TSV file.")
    parser.add_argument("--out", default=None, help="Output directory for processed artifacts.")
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--min_user_checkins", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--limit_users", type=int, default=None)
    parser.add_argument("--node2vec_dim", type=int, default=64)
    parser.add_argument("--node2vec_walk_length", type=int, default=10)
    parser.add_argument("--node2vec_num_walks", type=int, default=5)
    parser.add_argument("--node2vec_p", type=float, default=1.0)
    parser.add_argument("--node2vec_q", type=float, default=1.0)
    parser.add_argument("--text_encoder", default="tfidf_svd", choices=["tfidf_svd", "hf_transformer"])
    parser.add_argument("--text_embedding_dim", type=int, default=64)
    parser.add_argument("--text_model_name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_config_overrides(load_config(args.config) if args.config else {}, args.override)
    raw_path = args.raw or config.get("raw_path")
    out_dir = args.out or config.get("processed_dir")
    if raw_path is None or out_dir is None:
        raise ValueError("--raw/--out or --config with raw_path/processed_dir is required")
    summary = preprocess_tsmc2014(
        raw_path=Path(raw_path),
        out_dir=Path(out_dir),
        city=str(config.get("city", args.city)),
        max_seq_len=int(config.get("max_seq_len", args.max_seq_len)),
        min_user_checkins=int(config.get("min_user_checkins", args.min_user_checkins)),
        val_ratio=float(config.get("val_ratio", args.val_ratio)),
        test_ratio=float(config.get("test_ratio", args.test_ratio)),
        limit_users=args.limit_users if args.limit_users is not None else config.get("limit_users"),
        node2vec_dim=int(config.get("node2vec_dim", args.node2vec_dim)),
        node2vec_walk_length=int(config.get("node2vec_walk_length", args.node2vec_walk_length)),
        node2vec_num_walks=int(config.get("node2vec_num_walks", args.node2vec_num_walks)),
        node2vec_p=float(config.get("node2vec_p", args.node2vec_p)),
        node2vec_q=float(config.get("node2vec_q", args.node2vec_q)),
        text_encoder=str(config.get("semantic_encoder", config.get("text_encoder", args.text_encoder))),
        text_embedding_dim=int(config.get("text_embedding_dim", args.text_embedding_dim)),
        text_model_name=config.get("text_model_name", args.text_model_name),
        seed=int(config.get("seed", args.seed)),
    )
    print("Preprocessing complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
