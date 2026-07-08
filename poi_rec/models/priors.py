from __future__ import annotations

import torch
from torch import nn


class PriorEncoder(nn.Module):
    """Learnable non-linear combination of statistical prior scores.

    Takes a set of raw prior scores (popularity, transition, spatial, etc.)
    and learns a weighted non-linear combination to produce a final prior score
    per candidate POI.

    This replaces the hand-tuned linear weighted sum in _apply_candidate_priors.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_priors: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_priors = num_priors
        self.projector = nn.Sequential(
            nn.Linear(num_priors, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, prior_scores: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            prior_scores: shape (batch_size, num_candidates, num_priors)
                          raw prior scores for each candidate.

        Returns:
            shape (batch_size, num_candidates) learned prior scores.
        """
        out = self.projector(prior_scores)  # (B, C, 1)
        return out.squeeze(-1)  # (B, C)