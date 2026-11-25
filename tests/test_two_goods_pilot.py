import json
import math

from value_leakage_experiment import two_goods_pilot as pilot


def test_preference_manifest_crosses_symbols_order_and_wording_without_labels(tmp_path):
    pilot.prepare(tmp_path, 77, "model.gguf")
    manifest = json.loads((tmp_path / "preference_manifest.json").read_text(encoding="utf-8"))
    trials = manifest["trials"]
    preference = [r for r in trials if r["stage"] == "independent_preference"]
    assert len(preference) == 64
    assert len({r["prompt_sha256"] for r in trials}) == 72
    for pair in pilot.BENEFIT_PAIRS:
        rows = [r for r in preference if r["pair_id"] == pair["pair_id"]]
        assert len(rows) == 16
        assert {r["wording_id"] for r in rows} == set(pilot.PREFERENCE_WORDINGS)
        assert {r["symbol_scheme"] for r in rows} == set(pilot.SYMBOL_SCHEMES)
        assert {r["presentation_order"] for r in rows} == {"zero_first", "one_first"}
        for row in rows:
            assert "GOOD" not in row["prompt"] and "BAD" not in row["prompt"]
            assert set(row["outcome_symbols"].values()) == set(pilot.SYMBOL_SCHEMES[row["symbol_scheme"]])


def test_outcome_to_threshold_mapping_reverses_only_beneficiary_assignment():
    pair = pilot.BENEFIT_PAIRS[0]
    prompt_high, map_high = pilot._donation_prompt("Question?", 100, pair, "outcome_0", "favored_above", "above_first")
    prompt_low, map_low = pilot._donation_prompt("Question?", 100, pair, "outcome_0", "favored_below", "above_first")
    assert map_high == {"above": "outcome_0", "at_or_below": "outcome_1"}
    assert map_low == {"above": "outcome_1", "at_or_below": "outcome_0"}
    assert "GOOD" not in prompt_high and "BAD" not in prompt_high
    assert "unrelated to the quantity" in prompt_low


def test_numeric_completion_parser_separates_valid_malformed_and_unparseable():
    valid = pilot._number_parse(" 1.5 million")
    assert valid["validity"] == "valid_single_number"
    assert valid["estimate"] == 1_500_000
    assert pilot._number_parse("10,000 grains")["estimate"] == 10_000
    noncompliant = pilot._number_parse("The number of hairs on an adult head is approximately 100,000.")
    assert noncompliant["validity"] == "parsed_single_numeric_noncompliant"
    assert noncompliant["salvaged_estimate"] == 100_000
    malformed = pilot._number_parse("My estimate is 1.5 million, though it could be 2 million.")
    assert malformed["validity"] == "parsed_but_malformed_or_noncompliant"
    assert malformed["salvaged_estimate"] is not None
    assert pilot._number_parse("I cannot estimate this.")["validity"] == "unparseable_completion"


def test_candidate_probability_margin_is_stable_and_normalized():
    assert math.isclose(pilot._logistic(0), 0.5)
    assert pilot._logistic(2) > pilot._logistic(1) > 0.5
    assert pilot._logistic(-1) < 0.5


def test_preference_freeze_uses_predeclared_stability_rule(tmp_path):
    pilot.prepare(tmp_path, 9, "model.gguf")
    manifest = json.loads((tmp_path / "preference_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for trial in manifest["trials"]:
        row = {**trial, "candidate_logprobs": {"A": -1.0, "B": -1.2},
               "candidate_argmax": "A", "candidate_set_probability": {"A": 0.55, "B": 0.45}}
        if trial["stage"] == "independent_preference":
            margin = 1.0
            row.update({"logprob_margin_outcome_0_minus_outcome_1": margin,
                        "outcome_logprobs": {"outcome_0": -0.5, "outcome_1": -0.5 - margin}})
        else:
            row["diagnostic_correct_by_argmax"] = True
        rows.append(row)
    (tmp_path / "preference_raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    frozen = pilot.freeze_preferences(tmp_path)
    assert len(frozen["pairs"]) == 4
    assert all(row["stable_by_preregistered_rule"] for row in frozen["pairs"])
    assert all(row["frozen_ranked_outcome"] == "outcome_0" for row in frozen["pairs"])
