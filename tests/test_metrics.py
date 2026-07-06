from __future__ import annotations

import unittest

import torch

from poi_rec.training.metrics import average_metric_dicts, ranking_metric_sums, ranking_metrics


class RankingMetricsTest(unittest.TestCase):
    def test_ranking_metrics_known_example(self) -> None:
        scores = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.7, 0.1]])
        target = torch.tensor([1, 1])
        metrics = ranking_metrics(scores, target, [1, 2])
        self.assertEqual(metrics["Recall@1"], 0.5)
        self.assertEqual(metrics["HR@1"], 0.5)
        self.assertEqual(metrics["Recall@2"], 1.0)
        self.assertEqual(round(metrics["MRR"], 6), 0.75)

    def test_metric_weighting(self) -> None:
        batch1 = ranking_metric_sums(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([0, 0]), [1])
        batch2 = ranking_metric_sums(torch.tensor([[1.0, 0.0]]), torch.tensor([1]), [1])
        metrics = average_metric_dicts([batch1, batch2])
        self.assertEqual(round(metrics["HR@1"], 6), round(2 / 3, 6))
