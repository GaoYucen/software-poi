from __future__ import annotations

import torch
from torch.nn import functional as F


def next_poi_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(scores, target)

