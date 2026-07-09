from __future__ import annotations

import torch
from torch import nn


class PriorEncoder(nn.Module):
    """Memory-efficient learnable prior score combiner.

    Takes a set of raw prior scores (popularity, category_transition,
    user_category, spatial) and learns a per-prior weighted combination
    to produce a final prior score per candidate POI.

    Uses a single linear layer (num_priors -> 1) insteaf of an MLP
    to avoid OOM from intermediate (B, N, hidden_dim) tensors when N is
    large (34k candidates). The single layer is O(B * N * num_priors)
    which is acceptable.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_priors: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_priors = num_priors
        self.prior_weights = nn.Linear(num_priors, 1, bias=False)
        nn.init.xavier_uniform_(self.prior_weights.weight, gain=0.5)

    def forward(self, prior_scores: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            prior_scores: shape (batch_size, num_candidates, num_priors)
                          raw prior scores for each candidate.

        Returns:
            shape (batch_size, num_candidates) learned prior scores.
        """
        out = self.prior_weights(prior_scores)  # (B, C, 1)
        return out.squeeze(-1)  # (B, C)


class CandidateReranker(nn.Module):
    """Lightweight candidate-level reranker for small dynamic candidate sets.

    The reranker is intentionally tiny so it can be trained on sampled/dynamic
    candidates without materializing full ``B x num_pois x hidden`` tensors. It
    consumes scalar matching/prior features for each candidate and returns an
    additive score used on top of the base neural score.

    Features (10 dimensions):
      0: neural_score         — base neural matching score
      1: prior_score           — aggregated prior statistics score
      2: popularity            — global log-visit popularity
      3: category_match        — whether candidate category matches last POI category
      4: repeat_flag           — whether candidate was visited in user history
      5: spatial_distance      — normalized GPS distance to last POI
      6: hour_match_score      — visit frequency at this hour slot
      7: last_distance_score   — distance from the last POI (alternative encoding)
      8: category_transition   — category-level transition prior
      9: user_category_bias    — user's category-level preference
    """

    def __init__(self, num_features: int = 10, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.net = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} reranker features, got {features.shape[-1]}")
        return self.net(features.float()).squeeze(-1)