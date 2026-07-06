from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from poi_rec.data.dataset import POISequenceDataset, load_metadata, load_processed_arrays
from poi_rec.data.preprocess import preprocess_tsmc2014, read_tsmc2014


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
            self.assertEqual(metadata["num_categories"], 3)
            self.assertEqual(tuple(arrays["transition_features"].shape), (4, 4))

            dataset = POISequenceDataset(out, "train", max_seq_len=2)
            item = dataset[0]
            self.assertEqual(item["poi"].shape[0], 2)
            self.assertEqual(item["attention_mask"].shape[0], 2)
            self.assertEqual(item["target"].ndim, 0)
