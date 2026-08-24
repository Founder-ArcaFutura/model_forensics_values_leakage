import json
import unittest

from value_leakage_experiment.openai_batch import output_text, parse_output


class OpenAIBatchTests(unittest.TestCase):
    def test_parse_output_and_text(self):
        body = {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Final estimate: 24,000,000"}]}],
        }
        line = json.dumps({"custom_id": "a", "response": {"status_code": 200, "body": body}, "error": None})
        rows = parse_output(line)
        self.assertEqual(output_text(rows["a"]["response"]["body"]), "Final estimate: 24,000,000")

    def test_duplicate_custom_id_fails_closed(self):
        line = json.dumps({"custom_id": "a", "response": {}, "error": None})
        with self.assertRaises(ValueError):
            parse_output(line + "\n" + line)


if __name__ == "__main__":
    unittest.main()
