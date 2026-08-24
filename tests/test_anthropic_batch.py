import unittest

from value_leakage_experiment.anthropic_api import AnthropicSettings
from value_leakage_experiment.anthropic_batch import (
    _assemble_records,
    _batch_request,
    _parse_batch_results,
)


class AnthropicBatchTests(unittest.TestCase):
    def setUp(self):
        self.settings = AnthropicSettings(
            model="claude-test",
            max_tokens=1000,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "high"},
            request_timeout_seconds=30,
            base_url="https://example.invalid",
            api_key="not-a-real-key",
            anthropic_version="2023-06-01",
            http_transport="urllib",
        )
        self.trial = {
            "trial_id": "t-1",
            "order": 1,
            "arm_id": "westboro_lower_escape",
            "recipient_id": "westboro",
            "recipient_name": "Westboro Baptist Church",
            "recipient_valence": "disliked",
            "direction": "lower",
            "replicate": 1,
            "turn1_prompt": "Estimate.",
            "turn2_prompt": "Revise?",
            "turn1_prompt_sha256": "a",
            "turn2_prompt_sha256": "b",
        }

    def test_phase_two_round_trips_thinking_blocks_unchanged(self):
        content = [
            {"type": "thinking", "thinking": "summary", "signature": "signed"},
            {"type": "text", "text": "Final estimate: 10"},
        ]
        request = _batch_request(self.settings, self.trial, 2, {"content": content})
        self.assertIs(request["params"]["messages"][1]["content"], content)
        self.assertNotIn("temperature", request["params"])
        self.assertEqual(request["params"]["thinking"]["type"], "adaptive")

    def test_parse_rejects_duplicate_custom_ids(self):
        text = '{"custom_id":"a","result":{"type":"errored"}}\n' * 2
        with self.assertRaises(ValueError):
            _parse_batch_results(text)

    def test_assemble_preserves_batch_failure_denominator(self):
        manifest = {"manifest_core_sha256": "hash", "trials": [self.trial]}
        rows = _assemble_records(
            manifest,
            {"t-1": {"custom_id": "t-1", "result": {"type": "errored", "error": {"type": "invalid_request"}}}},
            {},
        )
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["incomplete_turn"], 1)


if __name__ == "__main__":
    unittest.main()
