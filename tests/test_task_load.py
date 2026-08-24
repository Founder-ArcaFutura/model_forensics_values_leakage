import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from value_leakage_experiment.task_load import (
    _mover_difference_interval,
    _task_compliance,
    create_task_load_manifest,
    task_load_prompt,
)


class TaskLoadTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "protocol_version": "test-load",
            "model": "claude-test",
            "trials_per_cell": 2,
            "seed": 11,
            "analysis_seed": 12,
            "neutral_recipient": "Neutral Group",
            "disliked_recipient": "Disliked Group",
        }

    def test_manifest_is_balanced_3_by_2_by_2(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(self.config), encoding="utf-8")
            manifest = create_task_load_manifest(config_path, root / "manifest.json")
        self.assertEqual(len(manifest["trials"]), 24)
        self.assertEqual(set(manifest["cell_counts"].values()), {2})
        self.assertEqual(len(manifest["cell_counts"]), 12)

    def test_value_and_control_prompts_differ_only_in_value_prefix_within_cell(self):
        for load in (0, 1, 2):
            for direction in ("lower", "higher"):
                value = task_load_prompt("veiled_value", direction, load, self.config)
                neutral = task_load_prompt("neutral_control", direction, load, self.config)
                self.assertIn(f"revise your estimate {direction}", value)
                self.assertIn(f"revise your estimate {direction}", neutral)
                self.assertEqual(value.split(" If you revise", 1)[1], neutral.split(" If you revise", 1)[1])

    def test_tasks_are_cumulative(self):
        prompts = [task_load_prompt("veiled_value", "higher", load, self.config) for load in (0, 1, 2)]
        self.assertNotIn("Memory token:", prompts[0])
        self.assertIn("Memory token:", prompts[1])
        self.assertNotIn("Arithmetic check:", prompts[1])
        self.assertIn("Memory token:", prompts[2])
        self.assertIn("Arithmetic check:", prompts[2])

    def test_task_compliance_accepts_expression_format(self):
        result = _task_compliance("Memory token: LANTERN-47\nArithmetic check: 37 × 24 = 888", 2)
        self.assertEqual(result, {"memory": True, "arithmetic": True})

    def test_mover_interaction_preserves_zero_cell_uncertainty(self):
        interval = _mover_difference_interval(0.0, [-0.0876, 0.0876], 0.025, [-0.065, 0.1288])
        self.assertLess(interval[0], -0.1)
        self.assertGreater(interval[1], 0.09)


if __name__ == "__main__":
    unittest.main()
