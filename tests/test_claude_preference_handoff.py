from __future__ import annotations

import json

import pytest

from value_leakage_experiment import claude_preference_format_probe as probe
from value_leakage_experiment import claude_preference_handoff as handoff
from value_leakage_experiment import low_effort_claude_pilot as pilot
from value_leakage_experiment.util import canonical_json, sha256_json, write_json


def _make_frozen_source(tmp_path, monkeypatch):
    source = tmp_path / "structured-source"
    format_core = {"protocol": probe.PREFERENCE_PROTOCOL, "format_passed": True,
                   "measurement_mode": "forced conditional choice; not spontaneous preference"}
    source.mkdir()
    write_json(source / "format_freeze.json", {**format_core, "freeze_sha256": sha256_json(format_core)})
    manifest = probe.prepare_preferences(source)
    probe.freeze_manifest(source)
    stage = source / "preferences"
    state_path = stage / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"phase": "ended", "batch_id": "msgbatch_fixture"})
    write_json(state_path, state)
    write_json(stage / "batch_status.json", {"id": "msgbatch_fixture", "processing_status": "ended"})

    failures = 0
    results = []
    for trial in manifest["trials"]:
        outcome = "outcome_0"
        if trial["pair_id"] == handoff.PRIMARY_PAIR and failures < 3:
            outcome = "outcome_1"
            failures += 1
        symbol = trial["outcome_symbols"][outcome]
        message = {
            "id": f"msg_{trial['custom_id']}", "model": probe.MODEL,
            "content": [{"type": "text", "text": canonical_json({"choice": symbol})}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 8, "output_tokens": 10},
        }
        results.append({"custom_id": trial["custom_id"], "result": {"type": "succeeded", "message": message}})
    provider_bytes = "".join(canonical_json(row) + "\n" for row in results).encode("utf-8")
    monkeypatch.setattr(probe, "anthropic_settings_from_config", lambda *_: object())
    monkeypatch.setattr(probe, "anthropic_request_bytes", lambda *_: provider_bytes)
    probe.retrieve("preferences", source)
    probe.freeze_preferences(source)
    return source


def test_structured_freeze_handoff_validates_and_prepares_low_effort_baseline(tmp_path, monkeypatch):
    source = _make_frozen_source(tmp_path, monkeypatch)
    continuation = tmp_path / "continuation"
    result = handoff.create_handoff(source, continuation)
    assert result["verification"]["primary_pair_count"] == 13
    assert result["verification"]["primary_pair_stable"] is True
    assert handoff.validate_handoff(continuation)["valid"] is True
    assert ((continuation / "preference_freeze_structured_source.json").read_bytes()
            == (source / "preference_freeze_structured.json").read_bytes())

    manifest = pilot.prepare_baseline(continuation)
    validation = pilot.validate_stage("stage2_baseline", continuation)
    requests = json.loads((continuation / "stage2_baseline" / "requests.json").read_text(encoding="utf-8"))["requests"]
    state = json.loads((continuation / "stage2_baseline" / "batch_state.json").read_text(encoding="utf-8"))
    assert len(manifest["trials"]) == validation["request_count"] == len(requests) == 12
    assert all(item["params"]["thinking"] == {"type": "adaptive"} for item in requests)
    assert all(item["params"]["output_config"] == {"effort": "low"} for item in requests)
    assert all(item["params"]["max_tokens"] == 2048 for item in requests)
    assert manifest["preference_measurement_mode"].startswith("forced comparative choice")
    assert state["phase"] == "prepared" and state.get("batch_id") is None
    assert not (continuation / "stage2_baseline" / "batch_submission.json").exists()
    assert not (continuation / "stage3_interventions").exists()


def test_handoff_fails_closed_if_provider_artifact_changes(tmp_path, monkeypatch):
    source = _make_frozen_source(tmp_path, monkeypatch)
    continuation = tmp_path / "continuation"
    handoff.create_handoff(source, continuation)
    provider_path = source / "preferences" / "provider_results.jsonl"
    provider_path.write_bytes(provider_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="provider_results_sha256"):
        handoff.validate_handoff(continuation)
