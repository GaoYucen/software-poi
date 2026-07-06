from __future__ import annotations

import unittest

import torch

from poi_rec.training.metrics import ranking_metrics


class RankingMetricsTest(unittest.TestCase):
    def test_ranking_metrics_known_example(self) -> None:
        scores = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.7, 0.1]])
        target = torch.tensor([1, 1])
        metrics = ranking_metrics(scores, target, [1, 2])
        self.assertEqual(metrics["Recall@1"], 0.5)
        self.assertEqual(metrics["Recall@2"], 1.0)
        self.assertEqual(round(metrics["MRR"], 6), 0.75)
