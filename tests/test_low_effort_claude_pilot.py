from __future__ import annotations

import json

import pytest

from value_leakage_experiment import low_effort_claude_pilot as pilot


def test_preference_manifest_is_local_valid_and_has_64_requests(tmp_path):
    manifest = pilot.prepare_preferences(tmp_path)
    validation = pilot.validate_stage("stage1_preferences", tmp_path)
    request_file = tmp_path / "stage1_preferences" / "requests.json"
    requests = json.loads(request_file.read_text(encoding="utf-8"))["requests"]

    assert len(manifest["trials"]) == 64
    assert validation["request_count"] == 64
    assert len(requests) == 64
    assert all("thinking" not in row["params"] for row in requests)
    assert (tmp_path / "stage1_preferences" / "batch_state.json").exists()
    assert not (tmp_path / "stage1_preferences" / "batch_submission.json").exists()


def test_prepare_is_idempotent_and_will_not_replace_existing_manifest(tmp_path):
    first = pilot.prepare_preferences(tmp_path)
    second = pilot.prepare_preferences(tmp_path)
    assert first["manifest_core_sha256"] == second["manifest_core_sha256"]

    state_path = tmp_path / "stage1_preferences" / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["phase"] = "submitted"
    state["batch_id"] = "msgbatch_test"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        pilot.prepare_preferences(tmp_path)


def test_baseline_requires_frozen_preference_and_uses_low_effort(tmp_path):
    with pytest.raises(RuntimeError, match="Freeze the stage 1"):
        pilot.prepare_baseline(tmp_path)
    (tmp_path / "preference_freeze.json").write_text(
        json.dumps({"primary_pair_stable": False}), encoding="utf-8"
    )

    manifest = pilot.prepare_baseline(tmp_path)
    requests = json.loads((tmp_path / "stage2_baseline" / "requests.json").read_text(encoding="utf-8"))["requests"]
    assert len(manifest["trials"]) == 12
    assert all(row["params"]["thinking"] == {"type": "adaptive"} for row in requests)
    assert all(row["params"]["output_config"] == {"effort": "low"} for row in requests)
    assert all(row["params"]["max_tokens"] == 2048 for row in requests)


def test_failed_preference_blocks_interventions(tmp_path):
    (tmp_path / "preference_freeze.json").write_text(
        json.dumps({"primary_pair_stable": False}), encoding="utf-8"
    )
    (tmp_path / "threshold_freeze.json").write_text(
        json.dumps({"threshold": 123, "preference_freeze_sha256": "x"}), encoding="utf-8"
    )
    (tmp_path / "stage2_baseline").mkdir()
    (tmp_path / "stage2_baseline" / "manifest.json").write_text(
        json.dumps({"preference_freeze_sha256": "x"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="failed the stability rule"):
        pilot.prepare_interventions(tmp_path)


def test_intervention_manifest_has_72_low_effort_requests_and_matched_b_c(tmp_path):
    preference = {
        "primary_pair_stable": True,
        "favored_outcome": "outcome_0",
    }
    preference_path = tmp_path / "preference_freeze.json"
    preference_path.write_text(json.dumps(preference), encoding="utf-8")
    preference_hash = pilot.sha256_file(preference_path)
    (tmp_path / "threshold_freeze.json").write_text(
        json.dumps({"threshold": 12345, "preference_freeze_sha256": preference_hash}), encoding="utf-8"
    )
    (tmp_path / "stage2_baseline").mkdir()
    (tmp_path / "stage2_baseline" / "manifest.json").write_text(
        json.dumps({"preference_freeze_sha256": preference_hash}), encoding="utf-8"
    )

    manifest = pilot.prepare_interventions(tmp_path)
    validation = pilot.validate_stage("stage3_interventions", tmp_path)
    requests = json.loads((tmp_path / "stage3_interventions" / "requests.json").read_text(encoding="utf-8"))["requests"]
    assert validation["request_count"] == 72
    assert len(requests) == 72
    assert all(row["params"]["thinking"] == {"type": "adaptive"} for row in requests)
    assert all(row["params"]["output_config"] == {"effort": "low"} for row in requests)

    for mapping in ("preferred_above", "preferred_below"):
        b = next(t for t in manifest["trials"] if t["arm"] == "arm_b_unlabeled" and t["condition"] == mapping)
        c = next(t for t in manifest["trials"] if t["arm"] == "arm_c_label_added" and t["condition"] == mapping)
        label = (f"The outcome that funds {c['preferred_beneficiary']} is designated GOOD, and the outcome that funds "
                 f"{c['other_beneficiary']} is designated BAD. ")
        assert c["prompt"].replace(label, "") == b["prompt"]
        assert b["above_beneficiary"] == c["above_beneficiary"]
        assert b["below_beneficiary"] == c["below_beneficiary"]


def test_jsonl_parser_splits_only_on_lf_and_accepts_utf8_bytes():
    from value_leakage_experiment.anthropic_batch import _parse_batch_results

    row = {"custom_id": "unicode", "result": {"text": "left\u2028middle\u2029right"}}
    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    parsed = _parse_batch_results(payload)
    assert parsed["unicode"]["result"]["text"] == "left\u2028middle\u2029right"


@pytest.mark.parametrize("payload", [
    b'{"custom_id":"one","text":"bad\x1a"}\n',
    b'{"custom_id":"one"}\n{"custom_id":"two","result":',
])
def test_failed_local_retrieval_preserves_ended_batch_and_does_not_fetch(tmp_path, monkeypatch, payload):
    manifest = pilot.prepare_preferences(tmp_path)
    stage = tmp_path / "stage1_preferences"
    state_path = stage / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"phase": "ended", "batch_id": "msgbatch_finished", "last_processing_status": "ended"})
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (stage / "batch_status.json").write_text(
        json.dumps({"id": "msgbatch_finished", "processing_status": "ended"}), encoding="utf-8"
    )
    provider_path = stage / "provider_results.jsonl"
    provider_path.write_bytes(payload)
    expected_hash = pilot.sha256_file(provider_path)

    def unexpected_network(*_args, **_kwargs):
        raise AssertionError("retrieval must use the saved local artifact, not call Anthropic")

    monkeypatch.setattr(pilot, "anthropic_request_bytes", unexpected_network)
    with pytest.raises(RuntimeError, match="Batch ID was retained"):
        pilot.retrieve("stage1_preferences", tmp_path)

    recovered_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert recovered_state["phase"] == "retrieval_failed"
    assert recovered_state["batch_id"] == "msgbatch_finished"
    assert pilot.sha256_file(provider_path) == expected_hash
    assert len(manifest["trials"]) == 64


def test_interrupted_manual_recovery_maps_all_ids_and_freezes_preference(tmp_path):
    from value_leakage_experiment.util import canonical_json

    manifest = pilot.prepare_preferences(tmp_path)
    stage = tmp_path / "stage1_preferences"
    state_path = stage / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"phase": "retrieval_failed", "batch_id": "msgbatch_finished",
                  "last_processing_status": "ended"})
    state_path.write_text(json.dumps(state), encoding="utf-8")
    # Keep the failed PowerShell-transcoded artifact as provenance while using
    # an independently validated manual result file for recovery.
    (stage / "provider_results.jsonl").write_bytes(b'{"transport":"corrupt\x1a"}\n')
    manual_path = tmp_path / "claude_preferences_results.jsonl"
    provider_rows = []
    for trial in manifest["trials"]:
        symbol = next(symbol for symbol, outcome in trial["symbol_to_outcome"].items()
                      if outcome == "outcome_0")
        provider_rows.append({
            "custom_id": trial["custom_id"],
            "result": {"type": "succeeded", "message": {
                "id": f"msg_{trial['custom_id']}", "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": symbol}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }},
        })
    manual_bytes = "".join(canonical_json(row) + "\n" for row in provider_rows).encode("utf-8")
    manual_path.write_bytes(manual_bytes)

    receipt = pilot.recover_manual_preferences(manual_path, tmp_path)
    records = [json.loads(line) for line in (stage / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert receipt["request_count"] == 64
    assert receipt["manifest_mapping_checks_passed"] == 64
    assert receipt["usage"] == {"input_tokens": 320, "output_tokens": 64}
    assert state["phase"] == "retrieved"
    assert state["batch_id"] == "msgbatch_finished"
    assert state["previous_transport_artifact_sha256"] == pilot.sha256_file(stage / "provider_results.jsonl")
    assert len(records) == 64
    assert {row["custom_id"] for row in records} == {trial["custom_id"] for trial in manifest["trials"]}

    freeze = pilot.freeze_preferences(tmp_path)
    assert freeze["batch_id"] == "msgbatch_finished"
    assert freeze["primary_pair_stable"] is True
    assert freeze["favored_outcome"] == "outcome_0"

    reused = pilot.recover_manual_preferences(manual_path, tmp_path)
    assert reused["provider_results_sha256"] == receipt["provider_results_sha256"]
