from __future__ import annotations

import json
from collections import Counter

import pytest

from value_leakage_experiment import claude_preference_format_probe as probe
from value_leakage_experiment.util import canonical_json, write_json


def test_probe_manifest_is_local_balanced_and_schema_constrained(tmp_path):
    manifest = probe.prepare_probe(tmp_path)
    checked = probe.validate("format_probe", tmp_path)
    requests = json.loads((tmp_path / "format_probe" / "requests.json").read_text(encoding="utf-8"))["requests"]

    assert checked["valid"] is True
    assert len(manifest["trials"]) == len(requests) == 4
    assert Counter(row["wording_id"] for row in manifest["trials"]) == {"donation_choice": 2, "recipient_choice": 2}
    assert Counter(row["symbol_scheme"] for row in manifest["trials"]) == {"AB": 2, "XY": 2}
    assert Counter(row["presentation_order"] for row in manifest["trials"]) == {"zero_first": 2, "one_first": 2}
    for trial, request in zip(manifest["trials"], requests):
        assert request["custom_id"] == trial["custom_id"]
        assert request["params"]["output_config"]["format"]["type"] == "json_schema"
        assert request["params"]["output_config"]["format"]["schema"] == trial["json_schema"]
        assert request["params"]["max_tokens"] == 128
        assert "thinking" not in request["params"]
    state = json.loads((tmp_path / "format_probe" / "batch_state.json").read_text(encoding="utf-8"))
    assert state["phase"] == "prepared"
    assert not (tmp_path / "format_probe" / "batch_submission.json").exists()
    assert "does not measure spontaneous preference" in manifest["measurement_statement"]


def test_preferences_require_successfully_frozen_probe(tmp_path):
    probe.prepare_probe(tmp_path)
    with pytest.raises(RuntimeError, match="Freeze a successful"):
        probe.prepare_preferences(tmp_path)
    write_json(tmp_path / "format_freeze.json", {"format_passed": False})
    with pytest.raises(RuntimeError, match="did not pass"):
        probe.prepare_preferences(tmp_path)


def test_format_freeze_requires_all_four_valid_schema_responses(tmp_path):
    manifest = probe.prepare_probe(tmp_path)
    folder = tmp_path / "format_probe"
    state_path = folder / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"phase": "retrieved", "batch_id": "msgbatch_probe"})
    write_json(state_path, state)
    records = []
    for trial in manifest["trials"]:
        symbol = trial["allowed_symbols"][0]
        records.append({**trial, "valid": True, "strict_response_format": True, "strict_symbol": symbol})
    (folder / "raw.jsonl").write_text("".join(canonical_json(row) + "\n" for row in records), encoding="utf-8")
    write_json(folder / "receipt.json", {"status_counts": {"completed": 4}})
    freeze = probe.freeze_format(tmp_path)
    assert freeze["format_passed"] is True
    assert freeze["response_count"] == 4
    assert "not spontaneous preference" in freeze["measurement_mode"]

    records[-1]["valid"] = False
    (folder / "raw.jsonl").write_text("".join(canonical_json(row) + "\n" for row in records), encoding="utf-8")
    freeze = probe.freeze_format(tmp_path)
    assert freeze["format_passed"] is False


def test_retrieval_parse_failure_keeps_batch_id_and_can_poll_or_retry(tmp_path, monkeypatch):
    probe.prepare_probe(tmp_path)
    folder = tmp_path / "format_probe"
    state_path = folder / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"phase": "ended", "batch_id": "msgbatch_probe", "last_processing_status": "ended"})
    write_json(state_path, state)
    write_json(folder / "batch_status.json", {"id": "msgbatch_probe", "processing_status": "ended"})
    monkeypatch.setattr(probe, "anthropic_settings_from_config", lambda *_: object())
    monkeypatch.setattr(probe, "anthropic_request_bytes", lambda *_: b'{"custom_id":"bad","text":"x\x1a"}\n')

    with pytest.raises(json.JSONDecodeError):
        probe.retrieve("format_probe", tmp_path)
    recovered = json.loads(state_path.read_text(encoding="utf-8"))
    assert recovered["phase"] == "retrieval_failed"
    assert recovered["batch_id"] == "msgbatch_probe"
    assert probe.poll("format_probe", tmp_path)["processing_status"] == "ended"
    assert list(folder.glob("provider_results_attempt_*.jsonl"))


def test_definitive_dns_failure_is_logged_as_preconnection_and_not_unknown(tmp_path, monkeypatch):
    probe.prepare_probe(tmp_path)
    monkeypatch.setattr(probe, "anthropic_settings_from_config", lambda *_: object())

    def dns_failure(*_args, **_kwargs):
        raise RuntimeError("Anthropic HTTP request failed: The remote name could not be resolved: 'api.anthropic.com'")

    monkeypatch.setattr(probe, "anthropic_request_json", dns_failure)
    with pytest.raises(RuntimeError, match="not accepted by Anthropic"):
        probe.submit("format_probe", tmp_path)
    state = json.loads((tmp_path / "format_probe" / "batch_state.json").read_text(encoding="utf-8"))
    assert state["phase"] == "prepared"
    assert state["preconnection_failures"][-1]["request_could_have_reached_provider"] is False
    assert state["batch_id"] is None


def test_authentication_rejection_is_logged_as_no_batch_created(tmp_path, monkeypatch):
    probe.prepare_probe(tmp_path)
    monkeypatch.setattr(probe, "anthropic_settings_from_config", lambda *_: object())
    monkeypatch.setattr(probe, "anthropic_request_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError('{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}')
    ))
    with pytest.raises(RuntimeError, match="not accepted by Anthropic"):
        probe.submit("format_probe", tmp_path)
    state = json.loads((tmp_path / "format_probe" / "batch_state.json").read_text(encoding="utf-8"))
    assert state["phase"] == "prepared"
    assert state["preconnection_failures"][-1]["provider_rejected_before_batch_creation"] is True
    assert state["batch_id"] is None


def test_v2_preference_revision_preserves_64_original_layouts_and_freezes(tmp_path):
    from value_leakage_experiment.util import sha256_json

    core = {
        "protocol": probe.PREFERENCE_PROTOCOL,
        "format_passed": True,
        "measurement_mode": "forced conditional choice; not spontaneous preference",
        "successful_probe": {"output_tokens": 41, "valid_choices": 4},
    }
    write_json(tmp_path / "format_freeze.json", {**core, "freeze_sha256": sha256_json(core)})
    manifest = probe.prepare_preferences(tmp_path)
    validation = probe.validate("preferences", tmp_path)
    freeze = probe.freeze_manifest(tmp_path)
    revalidated = probe.validate("preferences", tmp_path)

    assert len(manifest["trials"]) == 64
    assert validation["schema_neutrality"]["pass"] is True
    assert revalidated["schema_neutrality"]["mapping_is_fully_counterbalanced"] is True
    assert freeze["request_count"] == 64
    assert freeze["predeclared_primary_pair_id"] == "habitat_vs_veterinary_care"
    by_pair = Counter(row["pair_id"] for row in manifest["trials"])
    assert by_pair == {pair["pair_id"]: 16 for pair in probe.design.BENEFIT_PAIRS}
    state = json.loads((tmp_path / "preferences" / "batch_state.json").read_text(encoding="utf-8"))
    assert state["phase"] == "prepared"
    assert state["batch_id"] is None
    assert not (tmp_path / "preferences" / "batch_submission.json").exists()


def test_v2_analysis_reports_forced_choice_position_and_mapping_effects(tmp_path):
    from value_leakage_experiment.util import sha256_json

    core = {"protocol": probe.PREFERENCE_PROTOCOL, "format_passed": True,
            "measurement_mode": "forced conditional choice; not spontaneous preference"}
    write_json(tmp_path / "format_freeze.json", {**core, "freeze_sha256": sha256_json(core)})
    manifest = probe.prepare_preferences(tmp_path)
    records = []
    for trial in manifest["trials"]:
        symbol = trial["outcome_symbols"]["outcome_0"]
        records.append({**trial, "valid": True, "beneficiary": "outcome_0", "strict_symbol": symbol,
                        "status": "completed"})
    analysis = probe.analyze_preference_records(records)
    primary = analysis["choice_frequencies"]["habitat_vs_veterinary_care"]
    assert analysis["measurement_mode"].startswith("forced comparative choice")
    assert primary["choice_frequency_by_beneficiary"] == {"outcome_0": 16, "outcome_1": 0}
    assert primary["first_position_choice"]["n_valid"] == 16
    assert primary["first_position_choice"]["chosen_first_rate"] == 0.5
    assert set(primary["outcome_choice_rates_by_symbol_mapping"]) == {
        f"{outcome}_as_{symbol}" for outcome in ("outcome_0", "outcome_1") for symbol in ("A", "B", "X", "Y")
    }
    assert analysis["primary_pair_stability"]["stable"] is True

