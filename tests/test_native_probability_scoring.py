from __future__ import annotations

import unittest
from unittest.mock import patch

from value_leakage_experiment.native_probability_scoring import (
    FULL_N_PROBS,
    INITIAL_N_PROBS,
    _candidate_spec,
    _score_sequence,
)


class NativeProbabilityScoringTests(unittest.TestCase):
    def test_multi_token_candidate_scores_sum_conditional_token_logprobs(self):
        tokens = [{"id": 101, "piece": " G"}, {"id": 202, "piece": "AIN"}]
        prefixes = []

        def fake_distribution(prefix, seed, n_probs):
            prefixes.append(prefix)
            target_id = 101 if prefix == "Question:" else 202
            score = -0.4 if target_id == 101 else -0.7
            return {"probabilities": {"top_logprobs": [{"id": target_id, "logprob": score,
                                                          "bytes": list(tokens[0 if target_id == 101 else 1]["piece"].encode())}]},
                    "n_probs_requested": n_probs}

        with patch("value_leakage_experiment.native_probability_scoring.distribution", fake_distribution):
            row = _score_sequence("Question:", tokens, 9, {})

        self.assertAlmostEqual(row["logprob"], -1.1)
        self.assertEqual(prefixes, ["Question:", "Question: G"])
        self.assertEqual(row["token_count"], 2)

    def test_missing_candidate_in_initial_top_set_uses_full_vocabulary(self):
        token = {"id": 42, "piece": " BAD"}
        calls = []

        def fake_distribution(prefix, seed, n_probs):
            calls.append(n_probs)
            items = [] if n_probs == INITIAL_N_PROBS else [{"id": 42, "logprob": -3.25, "bytes": list(b" BAD")}]
            return {"probabilities": {"top_logprobs": items}, "n_probs_requested": n_probs}

        with patch("value_leakage_experiment.native_probability_scoring.FULL_N_PROBS", 1), \
             patch("value_leakage_experiment.native_probability_scoring.distribution", fake_distribution):
            row = _score_sequence("Question:", [token], 9, {})

        self.assertEqual(calls, [INITIAL_N_PROBS, 1])
        self.assertEqual(row["tokens"][0]["n_probs_used"], 1)
        self.assertAlmostEqual(row["logprob"], -3.25)

    def test_direct_candidate_sets_are_distinct_by_elicitation(self):
        self.assertEqual(_candidate_spec({"answer_format": "letter"}), ("A", " A", "B", " B"))
        self.assertEqual(_candidate_spec({"answer_format": "category"}), ("GOOD", " GOOD", "BAD", " BAD"))
        self.assertEqual(_candidate_spec({"answer_format": "welfare_category"}),
                         ("GAIN", " GAIN", "LOSS", " LOSS"))


if __name__ == "__main__":
    unittest.main()
