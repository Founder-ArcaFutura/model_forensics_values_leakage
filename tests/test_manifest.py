import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from value_leakage_experiment.manifest import create_manifest


class ManifestTests(unittest.TestCase):
    def test_balanced_and_reproducible_order(self):
        config = {
            "protocol_version": "test-v1",
            "model": "test-model",
            "max_output_tokens": 100,
            "trials_per_arm": 3,
            "seed": 7,
            "neutral_recipient": "Neutral Org",
            "disliked_recipient": "Disliked Org",
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            first = create_manifest(config_path, root / "first.json")
            second = create_manifest(config_path, root / "second.json")
        first_order = [(t["arm_id"], t["replicate"]) for t in first["trials"]]
        second_order = [(t["arm_id"], t["replicate"]) for t in second["trials"]]
        self.assertEqual(first_order, second_order)
        self.assertEqual(Counter(t["arm_id"] for t in first["trials"]), Counter({
            "neutral_no_direction": 3,
            "disliked_no_direction": 3,
            "disliked_lower_escape": 3,
            "disliked_higher_escape": 3,
        }))

    def test_multiple_disliked_recipients_create_separate_balanced_arms(self):
        config = {
            "protocol_version": "test-v2",
            "model": "test-model",
            "max_output_tokens": 100,
            "trials_per_arm": 2,
            "seed": 11,
            "neutral_recipient": "Neutral Org",
            "disliked_recipients": [
                {"recipient_id": "westboro", "recipient_name": "Westboro Baptist Church"},
                {"recipient_id": "patriot_front", "recipient_name": "Patriot Front"},
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = create_manifest(config_path, root / "manifest.json")
        self.assertEqual(len(manifest["trials"]), 14)
        self.assertEqual(
            Counter(trial["arm_id"] for trial in manifest["trials"]),
            Counter({
                "neutral_no_direction": 2,
                "westboro_no_direction": 2,
                "westboro_lower_escape": 2,
                "westboro_higher_escape": 2,
                "patriot_front_no_direction": 2,
                "patriot_front_lower_escape": 2,
                "patriot_front_higher_escape": 2,
            }),
        )
        self.assertEqual(
            Counter(trial["recipient_id"] for trial in manifest["trials"]),
            Counter({"neutral": 2, "westboro": 6, "patriot_front": 6}),
        )

    def test_arm_filter_supports_neutral_directional_supplement(self):
        config = {
            "protocol_version": "neutral-supplement",
            "model": "test-model",
            "max_output_tokens": 100,
            "trials_per_arm": 4,
            "seed": 13,
            "neutral_recipient": "Neutral Org",
            "disliked_recipient": "Disliked Org",
            "include_neutral_directional": True,
            "arm_ids": ["neutral_lower_escape", "neutral_higher_escape"],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = create_manifest(config_path, root / "manifest.json")
        self.assertEqual(
            Counter(trial["arm_id"] for trial in manifest["trials"]),
            Counter({"neutral_lower_escape": 4, "neutral_higher_escape": 4}),
        )
        self.assertTrue(all(trial["recipient_id"] == "neutral" for trial in manifest["trials"]))


if __name__ == "__main__":
    unittest.main()
