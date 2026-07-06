from __future__ import annotations

import unittest

from poi_rec.utils.config import apply_config_overrides


class ConfigOverrideTest(unittest.TestCase):
    def test_apply_config_overrides(self) -> None:
        config = {"epochs": 1, "curriculum": {"enabled": False}}
        updated = apply_config_overrides(
            config,
            ["epochs=5", "lr=0.001", "curriculum.enabled=true", "run_dir=runs/debug_e5"],
        )
        self.assertEqual(updated["epochs"], 5)
        self.assertAlmostEqual(updated["lr"], 0.001)
        self.assertTrue(updated["curriculum"]["enabled"])
        self.assertEqual(updated["run_dir"], "runs/debug_e5")
        self.assertEqual(config["epochs"], 1)

