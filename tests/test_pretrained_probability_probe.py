from __future__ import annotations

import unittest
import json

from value_leakage_experiment.pretrained_probability_probe import (
    _candidate_score,
    _labels_for_trial,
    _make_trials,
    _probability_record,
)


class PretrainedProbabilityProbeTests(unittest.TestCase):
    def test_candidate_score_requires_exact_whitespace_prefixed_token(self):
        top = [
            {"token": "A", "bytes": [65], "logprob": -1.0},
            {"token": " B", "bytes": [32, 66], "logprob": -2.0},
        ]
        self.assertFalse(_candidate_score(top, " A")["available_in_top_set"])
        self.assertEqual(_candidate_score(top, " B")["logprob"], -2.0)

    def test_corrected_inverted_mapping_is_good_label_on_welfare_loss(self):
        for good_side in ("above", "below"):
            mapping = _labels_for_trial({"alignment": "inverted", "good_side": good_side})
            good_outcome = mapping[good_side + "_outcome"]
            bad_side = "below" if good_side == "above" else "above"
            bad_outcome = mapping[bad_side + "_outcome"]
            self.assertEqual(good_outcome, "the employee loses an already approved paid vacation day")
            self.assertEqual(bad_outcome, "the employee receives an additional paid vacation day")

    def test_margin_signs_separate_good_label_from_welfare_under_inversion(self):
        for good_side in ("above", "below"):
            for order in ("above_first", "below_first"):
                trial = {
                    "alignment": "inverted", "good_side": good_side, "option_order": order,
                    "elicitation": "choice_answer_token_probability", "seed": 1,
                    "prompt": "p", "prompt_sha256": "h", "wording": "original",
                    "protocol_version": "test",
                }
                generated = " A" if order == "above_first" else " B"
                response = {"response": generated, "model": "fixture", "created_at": "now",
                            "logprobs": [{"bytes": list(generated.encode()), "top_logprobs": [
                                {"token": " A", "bytes": [32, 65], "logprob": -0.2},
                                {"token": " B", "bytes": [32, 66], "logprob": -0.8},
                            ]}]}
                row = _probability_record(trial, response)
                expected_sign = 1 if (good_side == "above") == (order == "above_first") else -1
                self.assertGreater(row["label_margin_good_minus_bad"] * expected_sign, 0)
                self.assertLess(row["welfare_margin_gain_minus_loss"] * expected_sign, 0)

    def test_manifest_trials_cross_alignment_side_order_and_wording(self):
        trials = _make_trials(41)
        choice = [t for t in trials if t["alignment"] != "neutral"]
        self.assertEqual(len(choice), 32)
        self.assertEqual(len({t["prompt_sha256"] for t in choice}), 32)
        self.assertEqual({t["answer_format"] for t in choice}, {"letter", "category"})
        self.assertEqual({t["alignment"] for t in choice}, {"congruent", "inverted"})
        self.assertEqual({t["good_side"] for t in choice}, {"above", "below"})
        self.assertEqual({t["option_order"] for t in choice}, {"above_first", "below_first"})
        self.assertEqual({t["wording"] for t in choice}, {"original", "completion_style"})

    def test_summary_is_json_serializable_with_unparsed_generations(self):
        from value_leakage_experiment.pretrained_probability_probe import _summarize

        summary = _summarize([{"elicitation": "choice_answer_token_probability", "alignment": "neutral",
                               "option_order": "above_first", "generated_choice": None,
                               "logprob_A_minus_B": None, "label_margin_good_minus_bad": None,
                               "welfare_margin_gain_minus_loss": None}])
        json.dumps(summary, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
