from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "value_leakage_experiment" / "ecological_preference_probe.py"
SPEC = importlib.util.spec_from_file_location("ecological_preference_probe", MODULE_PATH)
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(probe)


def source_rows():
    manifest, rows = probe.load_habitat_source_trials(
        ROOT / "results" / "pretrained_donation_bet_two_goods_v1" / "preference_manifest.json")
    return manifest, rows


def test_frozen_habitat_source_and_balanced_scenario_layouts():
    _, source = source_rows()
    trials = probe.build_preference_trials(source)
    assert len(source) == 16
    assert len(trials) == 64
    for scenario in probe.SCENARIOS:
        rows = [row for row in trials if row["scenario_id"] == scenario]
        assert len(rows) == 16
        assert len({row["layout_id"] for row in rows}) == 16
        assert {row["wording_id"] for row in rows} == {"donation_choice", "recipient_choice"}
        assert {row["presentation_order"] for row in rows} == {"zero_first", "one_first"}
        assert {row["symbol_scheme"] for row in rows} == {"AB", "XY"}
        assert all(row["outcome_symbols"]["habitat"] != row["outcome_symbols"]["veterinary"] for row in rows)


def test_each_scenario_changes_only_its_registered_outcome_description():
    _, source = source_rows()
    base_prompt = source[0]["prompt"]
    assert probe.contextualize_prompt(base_prompt, "baseline") == base_prompt
    habitat_changed = probe.contextualize_prompt(base_prompt, "habitat_secure")
    assert probe.SCENARIOS["habitat_secure"]["habitat_text"] in habitat_changed
    assert probe.SCENARIOS["habitat_secure"]["veterinary_text"] in habitat_changed
    assert probe.SCENARIOS["vet_prevents_extinction"]["veterinary_text"] in probe.contextualize_prompt(
        base_prompt, "vet_prevents_extinction")
    assert probe.SCENARIOS["ecosystem_wide"]["habitat_text"] in probe.contextualize_prompt(
        base_prompt, "ecosystem_wide")


def test_explanations_and_comprehension_are_separate_and_counterbalanced():
    explanations = probe.build_explanation_prompts()
    comprehension = probe.build_comprehension_trials()
    assert len(explanations) == 5
    assert len([row for row in explanations if row["stage"] == "baseline_explanation"]) == 2
    assert len([row for row in explanations if row["stage"] == "assumption_explanation"]) == 3
    assert len(comprehension) == 12
    for scenario in ("habitat_secure", "vet_prevents_extinction", "ecosystem_wide"):
        rows = [row for row in comprehension if row["scenario_id"] == scenario]
        assert len(rows) == 4
        assert {row["statement_order"] for row in rows} == {"true_first", "false_first"}
        assert {row["expected_symbol"] for row in rows} == {"A", "B"}


def test_condition_analysis_keeps_scenario_layouts_matched():
    rows = []
    for scenario in probe.SCENARIOS:
        for idx in range(16):
            rows.append({"stage": "contextual_preference_likelihood", "scenario_id": scenario,
                         "layout_id": f"layout-{idx}", "wording_id": "w1" if idx % 2 else "w2",
                         "presentation_order": "zero_first" if idx % 2 else "one_first",
                         "symbol_scheme": "AB" if idx % 2 else "XY",
                         "logprob_habitat_minus_veterinary": 1.0 if scenario == "ecosystem_wide" else -1.0,
                         "candidate_argmax": "A", "candidate_token_counts": {"A": 1, "B": 1}})
    summary = probe.summarize_preferences(rows)
    assert summary["preference_scenarios"]["ecosystem_wide"]["stable_by_unchanged_rule"] is True
    assert summary["preference_scenarios"]["habitat_secure"]["stable_by_unchanged_rule"] is True
    assert summary["preference_scenarios"]["ecosystem_wide"]["matched_delta_vs_baseline"]["positive_n"] == 16

