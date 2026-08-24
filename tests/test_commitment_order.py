import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from value_leakage_experiment.commitment_order import create_commitment_manifest, post_commit_prompt
from value_leakage_experiment.donation_positive_control import donation_note


class CommitmentOrderTests(unittest.TestCase):
    def test_post_prompt_contains_the_identical_donation_note(self):
        for side in ("above", "below"):
            self.assertIn(donation_note(23_400_000, side), post_commit_prompt(23_400_000, side))

    def test_manifest_is_balanced_and_post_only_has_phase_two(self):
        config = {
            "protocol_version": "test-order",
            "model": "claude-test",
            "max_output_tokens": 100,
            "threshold": 23_400_000,
            "trials_per_cell": 3,
            "seed": 9,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = create_commitment_manifest(config_path, root / "manifest.json")
        self.assertEqual(len(manifest["trials"]), 12)
        self.assertEqual(
            Counter(f"{t['order_condition']}_{t['good_side']}" for t in manifest["trials"]),
            Counter({"pre_above": 3, "pre_below": 3, "post_above": 3, "post_below": 3}),
        )
        self.assertTrue(all((t["phase2_prompt"] is not None) == (t["order_condition"] == "post") for t in manifest["trials"]))


if __name__ == "__main__":
    unittest.main()
