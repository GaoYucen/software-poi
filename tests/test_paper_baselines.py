from __future__ import annotations

import unittest

import torch

from scripts.run_baseline import SequenceBaseline


class PaperBaselineTest(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        poi = torch.tensor([[1, 2, 0], [2, 3, 1]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        for kind in ("gru4rec", "sasrec", "bert4rec"):
            model = SequenceBaseline(kind, num_pois=4, hidden_dim=8, max_seq_len=3)
            self.assertEqual(model(poi, mask).shape, torch.Size([2, 4]))


if __name__ == "__main__":
    unittest.main()
