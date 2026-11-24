from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from value_leakage_experiment import endogenous_donation_claude as design


def source_manifest():
    import json
    return json.loads((ROOT / "results" / "pretrained_donation_bet_two_goods_v1" / "preference_manifest.json").read_text(encoding="utf-8"))


def test_preference_manifest_reuses_four_pairs_and_sixteen_counterbalances():
    manifest = design.build_preference_manifest(source_manifest())
    assert len(manifest["trials"]) == 64
    for pair in {row["pair_id"] for row in manifest["trials"]}:
        rows = [row for row in manifest["trials"] if row["pair_id"] == pair]
        assert len(rows) == 16
        assert {row["wording_id"] for row in rows} == {"donation_choice", "recipient_choice"}
        assert {row["presentation_order"] for row in rows} == {"zero_first", "one_first"}
        assert {row["symbol_scheme"] for row in rows} == {"AB", "XY"}


def test_choice_parser_requires_single_mapped_symbol():
    mapping = {"A": "outcome_0", "B": "outcome_1"}
    assert design._parse_choice("A", mapping)["beneficiary"] == "outcome_0"
    assert design._parse_choice("Option B.", mapping)["beneficiary"] == "outcome_1"
    assert design._parse_choice("I choose A", mapping)["valid"] is False
    assert design._parse_choice("A or B", mapping)["valid"] is False


def test_neutral_calibration_and_nonselective_stable_pair_carryforward():
    baseline = design.build_baseline_manifest()
    assert len(baseline["trials"]) == 160
    for task in design.FERMI_QUESTIONS:
        assert sum(row["task_id"] == task for row in baseline["trials"]) == 40
    stable = ["habitat_vs_veterinary_care", "water_vs_education"]
    preference_freeze = {"stable_pairs": stable, "rankings": {
        pair: {"favored_outcome": "outcome_0"} for pair in stable
    }, "freeze_sha256": "frozen"}
    thresholds = {task: {"eligible": True, "threshold": 100} for task in design.FERMI_QUESTIONS}
    intervention = design.build_intervention_manifest(thresholds, preference_freeze)
    # Per task: 80 original positive-control samples plus 80 per stable two-goods pair.
    assert len(intervention["trials"]) == 4 * (80 + 2 * 80)
    assert {row["pair_id"] for row in intervention["trials"] if row["stage"] == "unlabeled_two_goods"} == set(stable)
    unlabeled = [row for row in intervention["trials"] if row["stage"] == "unlabeled_two_goods"]
    assert all("GOOD" not in row["prompt"] and "BAD" not in row["prompt"] for row in unlabeled)

