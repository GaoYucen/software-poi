from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from poi_rec.data.dataset import POISequenceDataset, load_metadata, load_processed_arrays
from poi_rec.data.preprocess import preprocess_tsmc2014
from poi_rec.models.poi_model import POIRecommendationModel


class ModelForwardTest(unittest.TestCase):
    def test_model_forward_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "tiny.tsv"
            raw.write_text(
                "\n".join(
                    [
                        "u1\tp1\tc1\tCafe\t10.0\t20.0\t0\tTue Apr 03 08:00:00 +0000 2012",
                        "u1\tp2\tc2\tOffice\t10.1\t20.1\t0\tTue Apr 03 09:00:00 +0000 2012",
                        "u1\tp3\tc1\tCafe\t10.2\t20.2\t0\tTue Apr 03 10:00:00 +0000 2012",
                        "u2\tp2\tc2\tOffice\t10.1\t20.1\t0\tTue Apr 03 08:30:00 +0000 2012",
                        "u2\tp3\tc1\tCafe\t10.2\t20.2\t0\tTue Apr 03 09:30:00 +0000 2012",
                        "u2\tp1\tc1\tCafe\t10.0\t20.0\t0\tTue Apr 03 10:30:00 +0000 2012",
                    ]
                ),
                encoding="utf-8",
            )
            out = tmp_path / "processed"
            preprocess_tsmc2014(raw, out, city="TEST", max_seq_len=3)
            dataset = POISequenceDataset(out, "train", max_seq_len=3)
            batch = {key: value.unsqueeze(0) for key, value in dataset[0].items()}
            config = {
                "hidden_dim": 32,
                "max_seq_len": 3,
                "gpt_model_name": "gpt2",
                "use_pretrained_gpt": False,
                "freeze_gpt": False,
                "unfreeze_last_n": 0,
                "gpt_layers": 1,
                "gpt_heads": 2,
                "dropout": 0.0,
            }
            model = POIRecommendationModel(load_metadata(out), load_processed_arrays(out), config)
            output = model(batch)
            self.assertEqual(output["scores"].shape, torch.Size([1, 3]))
            self.assertEqual(output["aligned_topology"].shape[-1], 32)
