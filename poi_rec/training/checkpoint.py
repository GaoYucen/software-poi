from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "metadata": metadata,
            "metrics": metrics,
            "gpt_model_name": getattr(getattr(model, "backbone", None), "gpt_model_name", None),
            "gpt_init_mode": getattr(getattr(model, "backbone", None), "gpt_init_mode", None),
            "pretrained_gpt_loaded": getattr(getattr(model, "backbone", None), "pretrained_gpt_loaded", None),
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
