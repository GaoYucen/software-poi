from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from poi_rec.data.dataset import POISequenceDataset, load_metadata, load_processed_arrays
from poi_rec.data.preprocess import preprocess_tsmc2014
from poi_rec.losses.alignment_loss import feature_level_alignment_loss, info_nce_alignment_loss
from poi_rec.models.gpt_backbone import GPTBackbone, last_valid_state
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
                "topology_encoder": "node2vec",
                "semantic_encoder": "tfidf_svd",
                "gpt_model_name": "gpt2",
                "use_pretrained_gpt": False,
                "fallback_to_random_gpt": True,
                "freeze_gpt": False,
                "unfreeze_last_n": 0,
                "gpt_freeze_policy": "random",
                "gpt_layers": 1,
                "gpt_heads": 2,
                "dropout": 0.0,
                "matching_normalize": True,
                "collaborative_score_weight": 1.0,
                "collaborative_dim": 16,
            }
            model = POIRecommendationModel(load_metadata(out), load_processed_arrays(out), config)
            output = model(batch)
            self.assertEqual(output["scores"].shape, torch.Size([1, 3]))
            self.assertEqual(model._collaborative_scores(batch).shape, torch.Size([1, 3]))
            self.assertEqual(output["aligned_topology"].shape[-1], 32)
            self.assertEqual(model.candidate_embeddings().shape, torch.Size([3, 32]))
            self.assertFalse(hasattr(model.topology, "poi_embedding"))
            no_prior_scores = model(batch, include_priors=False)["scores"]
            model.configure_priors(user_poi_prior_weight=2.0, user_poi_prior_mode="binary")
            prior_scores = model(batch, include_priors=True)["scores"]
            self.assertFalse(torch.allclose(no_prior_scores, prior_scores))

            bad_config = dict(config)
            bad_config["topology_encoder"] = "none"
            with self.assertRaises(ValueError):
                POIRecommendationModel(load_metadata(out), load_processed_arrays(out), bad_config)

            bad_config = dict(config)
            bad_config["transition_prior_mode"] = "future_oracle"
            with self.assertRaises(ValueError):
                POIRecommendationModel(load_metadata(out), load_processed_arrays(out), bad_config)

    def test_padding_last_state(self) -> None:
        hidden = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
        mask = torch.tensor([[1, 0, 0], [1, 1, 0]])
        state = last_valid_state(hidden, mask)
        self.assertEqual(state.squeeze(-1).tolist(), [1.0, 5.0])

    def test_pretrained_gpt_no_silent_fallback(self) -> None:
        with self.assertRaises(RuntimeError):
            GPTBackbone(
                hidden_dim=8,
                model_name="/definitely/missing/gpt2",
                use_pretrained=True,
                freeze=False,
                unfreeze_last_n=0,
                freeze_policy="full",
                fallback_to_random=False,
                layers=1,
                heads=1,
                max_seq_len=4,
                dropout=0.0,
            )

    def test_gpt_freeze_policy(self) -> None:
        backbone = GPTBackbone(
            hidden_dim=8,
            model_name="gpt2",
            use_pretrained=False,
            freeze=False,
            unfreeze_last_n=0,
            freeze_policy="pathllm_selective",
            fallback_to_random=True,
            layers=1,
            heads=1,
            max_seq_len=4,
            dropout=0.0,
        )
        params = dict(backbone.gpt.named_parameters())
        self.assertTrue(params["wpe.weight"].requires_grad)
        self.assertFalse(params["h.0.attn.c_attn.weight"].requires_grad)
        self.assertFalse(params["h.0.mlp.c_fc.weight"].requires_grad)
        self.assertTrue(params["h.0.ln_1.weight"].requires_grad)

    def test_no_false_negative_duplicate_poi(self) -> None:
        topology = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
        semantic = topology.clone()
        mask = torch.tensor([[1, 1, 1]])
        poi = torch.tensor([[1, 1, 2]])
        duplicate_loss = info_nce_alignment_loss(topology, semantic, mask, poi_idx=poi)
        unique_loss = info_nce_alignment_loss(topology[:, [0, 2]], semantic[:, [0, 2]], torch.tensor([[1, 1]]))
        self.assertAlmostEqual(float(duplicate_loss), float(unique_loss), places=6)

    def test_feature_level_infonce(self) -> None:
        topology = torch.eye(3).unsqueeze(0)
        semantic = torch.eye(3).unsqueeze(0)
        mask = torch.tensor([[1, 1, 1]])
        loss = feature_level_alignment_loss(topology, semantic, mask, temperature=0.05)
        self.assertLess(float(loss), 0.01)
