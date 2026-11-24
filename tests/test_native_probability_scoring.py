from __future__ import annotations

import unittest
from unittest.mock import patch

from value_leakage_experiment.native_probability_scoring import (
    FULL_N_PROBS,
    INITIAL_N_PROBS,
    _candidate_spec,
    _score_sequence,
)
from value_leakage_experiment.pretrained_probability_probe import (
    _make_trials,
    _probability_record,
    _summarize,
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

    def test_neutral_welfare_category_scores_gain_loss_candidates(self):
        trial = next(row for row in _make_trials(9) if row["answer_format"] == "welfare_category")
        response = {"response": " GAIN", "logprobs": [{"bytes": list(b" GAIN"), "top_logprobs": [
            {"token": " GAIN", "bytes": list(b" GAIN"), "logprob": -0.25},
            {"token": " LOSS", "bytes": list(b" LOSS"), "logprob": -1.25},
        ]}]}

        row = _probability_record(trial, response)

        self.assertEqual(set(row["candidate_scores"]), {"GAIN", "LOSS"})
        self.assertAlmostEqual(row["logprob_first_minus_second"], 1.0)
        self.assertAlmostEqual(row["welfare_margin_gain_minus_loss"], 1.0)
        self.assertIsNone(row["logprob_A_minus_B"])

    def test_summary_counts_scored_category_pairs_without_letter_margin(self):
        trial = next(row for row in _make_trials(12) if row["answer_format"] == "category")
        response = {"response": " GOOD", "logprobs": [{"bytes": list(b" GOOD"), "top_logprobs": [
            {"token": " GOOD", "bytes": list(b" GOOD"), "logprob": -0.2},
            {"token": " BAD", "bytes": list(b" BAD"), "logprob": -1.0},
        ]}]}

        row = _probability_record(trial, response)
        summary = _summarize([row])

        self.assertIsNone(row["logprob_A_minus_B"])
        self.assertEqual(summary["candidate_pairs_scored"], 1)
        self.assertEqual(summary["candidate_pair_missing_n"], 0)
        self.assertEqual(summary["label_margin"]["label_token_probability"][trial["alignment"]]
                         ["n_distinct_prompts"], 1)


if __name__ == "__main__":
    unittest.main()
