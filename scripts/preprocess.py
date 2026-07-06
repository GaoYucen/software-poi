#!/usr/bin/env python
"""Preprocess TSMC2014/Foursquare check-ins for next POI recommendation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poi_rec.data.preprocess import preprocess_tsmc2014


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="NYC", help="City label stored in metadata.")
    parser.add_argument("--raw", required=True, help="Path to raw TSMC2014 TSV file.")
    parser.add_argument("--out", required=True, help="Output directory for processed artifacts.")
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--min_user_checkins", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--limit_users", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = preprocess_tsmc2014(
        raw_path=Path(args.raw),
        out_dir=Path(args.out),
        city=args.city,
        max_seq_len=args.max_seq_len,
        min_user_checkins=args.min_user_checkins,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        limit_users=args.limit_users,
    )
    print("Preprocessing complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
