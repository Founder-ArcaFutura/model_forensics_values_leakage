from __future__ import annotations

import random

from value_leakage_experiment import claude_congruence_pilot as pilot
from value_leakage_experiment.util import sha256_json, sha256_text


def _balanced_fixture_manifest():
    provenance = {"source_provenance_sha256": "fixture-provenance"}
    rows = []
    base = "Question prompt. So, your answer must include one number."
    for condition in pilot.CONDITIONS:
        for mapping in pilot.MAPPINGS:
            good = pilot.HABITAT if condition == "congruent" else pilot.VETERINARY if condition == "incongruent" else None
            sentence = pilot._label_sentence(condition)
            prompt = base if condition == "unlabeled" else base.replace("So, your answer", sentence + "So, your answer")
            above = pilot.HABITAT if mapping == "habitat_above" else pilot.VETERINARY
            at_below = pilot.VETERINARY if mapping == "habitat_above" else pilot.HABITAT
            for replicate in range(1, 13):
                rows.append({"custom_id": f"{condition}-{mapping}-{replicate}", "condition": condition,
                             "mapping": mapping, "replicate": replicate, "prompt": prompt,
                             "prompt_sha256": sha256_text(prompt), "above_beneficiary": above,
                             "at_or_below_beneficiary": at_below, "good_beneficiary": good})
    random.Random(99).shuffle(rows)
    core = {"protocol": pilot.PROTOCOL, "model": pilot.MODEL, "config": {
        "model": pilot.MODEL, "max_output_tokens": 2048, "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"}}, "threshold": pilot.THRESHOLD, "question": pilot.QUESTION,
        "source_provenance_sha256": provenance["source_provenance_sha256"],
        "source_provenance": {"forced_preference": {"batch_id": "msgbatch_fixture"}},
        "conditions": {condition: condition for condition in pilot.CONDITIONS}, "trials": rows}
    manifest = {**core, "manifest_core_sha256": sha256_json(core)}
    return manifest, provenance


def test_manifest_validation_checks_all_six_cells_and_label_only_edits():
    manifest, provenance = _balanced_fixture_manifest()
    result = pilot.validate_manifest(manifest, provenance)
    assert result["request_count"] == 72
    assert result["order_interleaving_pass"] is True
    assert set(result["cell_counts"].values()) == {12}

    changed = {**manifest, "trials": [dict(row) for row in manifest["trials"]]}
    target = next(row for row in changed["trials"] if row["condition"] == "incongruent")
    target["prompt"] += " Extra wording."
    target["prompt_sha256"] = sha256_text(target["prompt"])
    core = {key: value for key, value in changed.items() if key not in {"manifest_core_sha256", "created_at_utc"}}
    changed["manifest_core_sha256"] = sha256_json(core)
    try:
        pilot.validate_manifest(changed, provenance)
    except ValueError as exc:
        assert "differs from unlabeled" in str(exc)
    else:
        raise AssertionError("Validation must reject added text beyond the label substitution")


def test_analysis_reports_preference_and_label_alignment_ties_and_usage():
    manifest, _ = _balanced_fixture_manifest()
    records = []
    for trial in manifest["trials"]:
        condition, mapping, replicate = trial["condition"], trial["mapping"], trial["replicate"]
        if condition == "incongruent":
            estimate = pilot.THRESHOLD - 1 if mapping == "habitat_above" else pilot.THRESHOLD + 1
            if mapping == "habitat_at_or_below" and replicate == 1:
                estimate = pilot.THRESHOLD
        else:
            estimate = pilot.THRESHOLD + 1 if mapping == "habitat_above" else pilot.THRESHOLD - 1
        status = "completed"
        stop = "end_turn"
        if condition == "unlabeled" and mapping == "habitat_above" and replicate == 1:
            status = "incomplete"
            stop = "max_tokens"
        records.append({**trial, "estimate": estimate, "valid_numeric_estimate": True,
                        "valid_for_primary_analysis": status == "completed", "status": status,
                        "stop_reason": stop, "usage": {"input_tokens": 5, "output_tokens": 10,
                                                           "output_tokens_details": {"thinking_tokens": 2}}})
    analysis = pilot.analyze_records(manifest, records)
    unlabeled = analysis["condition_results"]["unlabeled"]
    congruent = analysis["condition_results"]["congruent"]
    incongruent = analysis["condition_results"]["incongruent"]
    assert unlabeled["valid_completed_numeric_n"] == 23
    assert unlabeled["parsing_valid_count"] == 24
    assert unlabeled["stop_reasons"] == {"end_turn": 23, "max_tokens": 1}
    assert unlabeled["token_usage"]["input_tokens"] == 120
    assert unlabeled["token_usage"]["thinking_tokens"] == 48
    assert unlabeled["label_aligned_good_beneficiary_funding"] is None
    assert congruent["preference_aligned_habitat_funding"]["successes"] == 24
    assert congruent["label_aligned_good_beneficiary_funding"]["successes"] == 24
    assert incongruent["beneficiary_funding_counts"]["habitat"] == 1
    assert incongruent["label_aligned_good_beneficiary_funding"]["successes"] == 23
    assert incongruent["threshold_counts"]["tie_at_threshold"] == 1
    assert analysis["preference_aligned_pairwise_contrasts"]["congruent_minus_incongruent"]["ci95"]
    assert analysis["label_aligned_congruent_vs_incongruent_contrast"]["incongruent_minus_congruent"]["difference"] < 0
