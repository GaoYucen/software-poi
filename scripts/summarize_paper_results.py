#!/usr/bin/env python
"""Collect paper experiment artifacts into a traceable CSV/JSON summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics


METRIC_PATTERN = re.compile(r"^(HR@5|HR@10|HR@20|NDCG@5|NDCG@10|NDCG@20|MRR):\s*([0-9.]+)$")


def parse_metric_text(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = METRIC_PATTERN.match(line.strip())
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/paper_revision")
    parser.add_argument("--output", default="runs/paper_revision/summary")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.rglob("test_metrics.txt")):
        parts = path.relative_to(root).parts
        city = parts[0] if parts else "unknown"
        run = "/".join(parts[1:-1])
        seed_match = re.search(r"seed(\d+)", run)
        row: dict[str, object] = {
            "city": city,
            "run": run,
            "seed": int(seed_match.group(1)) if seed_match else 42,
            "source": str(path),
        }
        row.update(parse_metric_text(path))
        rows.append(row)
    for path in sorted(root.rglob("prior_ablation/*.txt")):
        city = path.relative_to(root).parts[0]
        row = {"city": city, "run": f"prior/{path.stem}", "seed": 42, "source": str(path)}
        row.update(parse_metric_text(path))
        rows.append(row)
    for path in sorted(root.rglob("baselines/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "city": data["city"],
            "run": f"baseline/{data['model']}",
            "seed": data["seed"],
            "source": str(path),
        }
        row.update(data["test"])
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["city", "run", "seed", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR", "source"]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for city in sorted({str(row["city"]) for row in rows}):
        grouped[city] = {}
        city_rows = [row for row in rows if row["city"] == city]
        for run in sorted({str(row["run"]) for row in city_rows}):
            run_rows = [row for row in city_rows if row["run"] == run]
            metrics: dict[str, float] = {}
            for metric in fields[3:10]:
                values = [float(row[metric]) for row in run_rows if metric in row]
                if values:
                    metrics[f"{metric}_mean"] = statistics.mean(values)
                    metrics[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            grouped[city][run] = metrics
    output.with_suffix(".json").write_text(json.dumps(grouped, indent=2), encoding="utf-8")
    print(json.dumps(grouped, indent=2))


if __name__ == "__main__":
    main()
