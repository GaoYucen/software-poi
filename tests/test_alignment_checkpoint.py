from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from poi_rec.data.dataset import load_metadata, load_processed_arrays
from poi_rec.data.preprocess import preprocess_tsmc2014
from poi_rec.models.poi_model import POIRecommendationModel
from poi_rec.training.pretrain_alignment import _alignment_state_dict


class AlignmentCheckpointTest(unittest.TestCase):
    def test_alignment_checkpoint_has_no_data_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "tiny.tsv"
            raw.write_text(
                "\n".join(
                    [
                        "u1\tp1\tc1\tCafe\t10.0\t20.0\t0\tTue Apr 03 08:00:00 +0000 2012",
                        "u1\tp2\tc2\tOffice\t10.1\t20.1\t0\tTue Apr 03 09:00:00 +0000 2012",
                        "u1\tp3\tc1\tCafe\t10.2\t20.2\t0\tTue Apr 03 10:00:00 +0000 2012",
                    ]
                ),
                encoding="utf-8",
            )
            out = tmp_path / "processed"
            preprocess_tsmc2014(raw, out, city="TEST", max_seq_len=3, node2vec_dim=8, text_embedding_dim=8)
            config = {
                "hidden_dim": 8,
                "max_seq_len": 3,
                "topology_encoder": "node2vec",
                "semantic_encoder": "tfidf_svd",
                "gpt_model_name": "gpt2",
                "use_pretrained_gpt": False,
                "fallback_to_random_gpt": True,
                "freeze_gpt": False,
                "unfreeze_last_n": 0,
                "gpt_freeze_policy": "random",
                "gpt_layers": 1,
                "gpt_heads": 1,
                "dropout": 0.0,
            }
            model = POIRecommendationModel(load_metadata(out), load_processed_arrays(out), config)
            state = _alignment_state_dict(model)
            forbidden = [
                "transition_features",
                "node2vec_embeddings",
                "topology_available",
                "poi_category",
                "poi_coords",
                "text_embeddings",
                "user_poi_edges",
            ]
            self.assertTrue(state)
            self.assertFalse(any(any(token in key for token in forbidden) for key in state))
