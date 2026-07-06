from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from poi_rec.data.dataset import POISequenceDataset, load_metadata, load_processed_arrays
from poi_rec.data.preprocess import preprocess_fingerprint, preprocess_tsmc2014, read_tsmc2014


def _write_raw(path: Path) -> None:
    lines = [
        "u1\tp1\tc1\tCafe\t10.0\t20.0\t60\tTue Apr 03 08:00:00 +0000 2012",
        "u1\tp2\tc2\tOffice\t10.1\t20.1\t60\tTue Apr 03 09:00:00 +0000 2012",
        "u1\tp3\tc1\tCafe\t10.2\t20.2\t60\tTue Apr 03 10:00:00 +0000 2012",
        "u2\tp2\tc2\tOffice\t10.1\t20.1\t0\tTue Apr 03 07:00:00 +0000 2012",
        "u2\tp4\tc3\tPark\t10.3\t20.3\t0\tTue Apr 03 08:00:00 +0000 2012",
        "u2\tp1\tc1\tCafe\t10.0\t20.0\t0\tTue Apr 03 09:00:00 +0000 2012",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


class PreprocessDatasetTest(unittest.TestCase):
    def test_preprocess_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "tiny.tsv"
            out = tmp_path / "processed"
            _write_raw(raw)
            df = read_tsmc2014(raw)
            self.assertEqual(int(df.iloc[0]["hour"]), 9)

            summary = preprocess_tsmc2014(raw, out, city="TEST", max_seq_len=2)
            self.assertEqual(summary["num_users"], 2)
            self.assertEqual(summary["num_pois"], 4)
            metadata = load_metadata(out)
            arrays = load_processed_arrays(out)
            self.assertEqual(metadata["schema_version"], 4)
            self.assertEqual(
                metadata["preprocess_fingerprint"],
                preprocess_fingerprint(metadata["preprocess_config"]),
            )
            self.assertEqual(metadata["num_categories"], 3)
            self.assertEqual(tuple(arrays["transition_features"].shape), (4, 4))
            self.assertEqual(tuple(arrays["node2vec_embeddings"].shape), (4, 64))
            self.assertEqual(tuple(arrays["text_embeddings"].shape), (4, 64))
            self.assertEqual(tuple(arrays["topology_available"].shape), (4,))
            self.assertEqual(tuple(arrays["user_poi_edges"].shape[1:]), (3,))
            self.assertGreater(metadata["num_user_poi_edges"], 0)
            self.assertEqual(metadata["split_protocol"], "user_chronological_ratio")
            poi_text = (out / "poi_text.json").read_text(encoding="utf-8")
            self.assertNotIn("Visit count", poi_text)
            self.assertNotIn("Latitude", poi_text)
            self.assertNotIn("Category ID", poi_text)

            dataset = POISequenceDataset(out, "train", max_seq_len=2)
            item = dataset[0]
            self.assertEqual(item["poi"].shape[0], 2)
            self.assertEqual(item["attention_mask"].shape[0], 2)
            self.assertEqual(item["target"].ndim, 0)
            self.assertGreaterEqual(int(item["poi"][0]), 1)

    def test_no_transition_overcount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "tiny.tsv"
            out = tmp_path / "processed"
            _write_raw(raw)
            preprocess_tsmc2014(raw, out, city="TEST", max_seq_len=3)
            metadata = load_metadata(out)
            arrays = load_processed_arrays(out)
            self.assertEqual(metadata["train_transition_weight_sum"], metadata["train_adjacent_transition_count"])
            self.assertEqual(float(arrays["transition_edges"][:, 2].sum()), metadata["train_adjacent_transition_count"])

    def test_isolated_poi_has_no_fake_topology(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "tiny.tsv"
            out = tmp_path / "processed"
            _write_raw(raw)
            preprocess_tsmc2014(raw, out, city="TEST", max_seq_len=3)
            arrays = load_processed_arrays(out)
            isolated = arrays["topology_available"].eq(0)
            if isolated.any():
                self.assertTrue(arrays["node2vec_embeddings"][isolated].eq(0).all())

    def test_preprocess_fingerprint_changes_with_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "tiny.tsv"
            out = tmp_path / "processed"
            _write_raw(raw)
            preprocess_tsmc2014(raw, out, city="TEST", max_seq_len=2)
            metadata = load_metadata(out)
            changed = dict(metadata["preprocess_config"])
            changed["max_seq_len"] = 3
            self.assertNotEqual(metadata["preprocess_fingerprint"], preprocess_fingerprint(changed))

    def test_closed_world_target_validity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "tiny.tsv"
            out = tmp_path / "processed"
            _write_raw(raw)
            preprocess_tsmc2014(raw, out, city="TEST", max_seq_len=2)
            arrays = load_processed_arrays(out)
            val_dataset = POISequenceDataset(out, "val", max_seq_len=2, candidate_protocol="closed_world")
            for item in val_dataset:
                self.assertEqual(float(arrays["train_seen_poi"][int(item["target"])]), 1.0)
