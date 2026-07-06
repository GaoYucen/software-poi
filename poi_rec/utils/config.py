from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def parse_config_value(raw: str) -> Any:
    value = yaml.safe_load(raw)
    return raw if value is None and raw.lower() not in {"null", "none", "~"} else value


def apply_config_overrides(config: dict[str, Any], items: list[str]) -> dict[str, Any]:
    updated = dict(config)
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must use key=value format: {item}")
        key, raw_value = item.split("=", 1)
        parts = [part for part in key.split(".") if part]
        if not parts:
            raise ValueError(f"Override key cannot be empty: {item}")
        current: dict[str, Any] = updated
        for part in parts[:-1]:
            child = current.get(part)
            if child is None:
                child = {}
                current[part] = child
            if not isinstance(child, dict):
                raise ValueError(f"Cannot apply nested override through non-mapping key: {part}")
            current = child
        current[parts[-1]] = parse_config_value(raw_value)
    return updated
