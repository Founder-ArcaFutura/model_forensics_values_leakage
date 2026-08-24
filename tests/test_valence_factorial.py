import json
import tempfile
import unittest
from pathlib import Path

from value_leakage_experiment.valence_factorial import create_valence_manifest, outcome_note


class ValenceFactorialTests(unittest.TestCase):
    def test_factor_presence(self):
        semantic_labeled = outcome_note(23_400_000, "semantic_labeled", "above")
        semantic_unlabeled = outcome_note(23_400_000, "semantic_unlabeled", "above")
        abstract_labeled = outcome_note(23_400_000, "abstract_labeled", "above")
        abstract_unlabeled = outcome_note(23_400_000, "abstract_unlabeled", "above")
        self.assertIn("paid vacation", semantic_labeled)
        self.assertIn("good outcome", semantic_labeled)
        self.assertIn("paid vacation", semantic_unlabeled)
        self.assertNotIn("good outcome", semantic_unlabeled)
        self.assertIn("Outcome A", abstract_labeled)
        self.assertIn("good outcome", abstract_labeled)
        self.assertIn("Outcome A", abstract_unlabeled)
        self.assertNotIn("good outcome", abstract_unlabeled)

    def test_mapping_reversal_swaps_consequences(self):
        above = outcome_note(23_400_000, "semantic_unlabeled", "above")
        below = outcome_note(23_400_000, "semantic_unlabeled", "below")
        self.assertIn("exceeds the threshold, a randomly selected employee receives", above)
        self.assertIn("exceeds the threshold, that employee loses", below)

    def test_manifest_is_balanced(self):
        config = {
            "protocol_version": "test-valence",
            "model": "test-model",
            "threshold": 23_400_000,
            "threshold_provenance": {"method": "test"},
            "trials_per_cell": 2,
            "seed": 17,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = create_valence_manifest(config_path, root / "manifest.json")
        self.assertEqual(len(manifest["trials"]), 16)
        self.assertEqual(len(manifest["cell_counts"]), 8)
        self.assertEqual(set(manifest["cell_counts"].values()), {2})


if __name__ == "__main__":
    unittest.main()
