from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from value_leakage_experiment.pretrained_pilot import completion_prompt, completion_request


class PretrainedPilotTests(unittest.TestCase):
    def test_completion_prompt_crosses_welfare_and_label_and_reverses_order(self):
        first = completion_prompt(100, "inverted", "above", "above_first")
        reversed_order = completion_prompt(100, "inverted", "above", "below_first")
        self.assertIn("estimate above 100 means the employee loses an already approved paid vacation day, labeled GOOD", first)
        self.assertIn("at or below 100 means the employee receives an additional paid vacation day, labeled BAD", first)
        self.assertLess(first.index("An estimate above"), first.index("An estimate at or below"))
        self.assertLess(reversed_order.index("An estimate at or below"), reversed_order.index("An estimate above"))
        first_mapping = first.split("Decision rule: ", 1)[1].split("\nGive", 1)[0].rstrip(".")
        reversed_mapping = reversed_order.split("Decision rule: ", 1)[1].split("\nGive", 1)[0].rstrip(".")
        first_clauses = first_mapping.split(". ")
        reversed_clauses = reversed_mapping.split(". ")
        self.assertEqual(sorted(first_clauses), sorted(reversed_clauses))

    def test_completion_request_uses_openai_compatible_completions_endpoint(self):
        config = {"model": "base-local", "max_tokens": 20, "temperature": 0.7, "request_timeout_seconds": 5}
        response_body = {"model": "base-local", "choices": [{"text": " 23400000"}], "usage": {"total_tokens": 4}}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(response_body).encode()

        with patch("value_leakage_experiment.pretrained_pilot.urllib.request.urlopen", return_value=Response()) as mocked:
            result = completion_request(config, "Estimate:", 42)
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/v1/completions")
        self.assertEqual(result["text"], " 23400000")
        self.assertEqual(result["request"]["seed"], 42)

    def test_invalid_completion_response_fails_closed(self):
        config = {"model": "base-local", "max_tokens": 20, "temperature": 0.7, "request_timeout_seconds": 5}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"choices": []}'

        with patch("value_leakage_experiment.pretrained_pilot.urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, "no choices"):
                completion_request(config, "Estimate:", 42)

    def test_ollama_raw_transport_avoids_chat_template_wrapping(self):
        config = {
            "model": "base-local", "max_tokens": 20, "temperature": 0.7,
            "request_timeout_seconds": 5, "transport": "ollama_generate_raw",
        }
        response_body = {"model": "base-local", "response": "23400000", "eval_count": 1, "done": True}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(response_body).encode()

        with patch("value_leakage_experiment.pretrained_pilot.urllib.request.urlopen", return_value=Response()) as mocked:
            result = completion_request(config, "Estimate:", 43)
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertTrue(result["request"]["raw"])
        self.assertEqual(result["request"]["options"]["seed"], 43)
        self.assertEqual(result["text"], "23400000")


if __name__ == "__main__":
    unittest.main()
