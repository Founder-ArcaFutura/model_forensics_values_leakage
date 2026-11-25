"""Small structured-output probe and opt-in preference rerun for Claude Opus 4.8.

Local manifest generation is offline. Only ``submit-probe`` and
``submit-preferences`` invoke paid inference.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import endogenous_donation_claude as design
from .anthropic_api import anthropic_request_bytes, anthropic_request_json, anthropic_settings_from_config
from .anthropic_batch import _message_text, _parse_batch_results
from .util import canonical_json, read_json, sha256_file, sha256_json, sha256_text, write_json


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "claude_opus_4_8_endogenous_preference_v1" / "low_effort_pilot" / "structured_choice_preference_v1"
SOURCE_MANIFEST = REPO / "results" / "pretrained_donation_bet_two_goods_v1" / "preference_manifest.json"
PRIMARY_PAIR_ID = "habitat_vs_veterinary_care"
MODEL = "claude-opus-4-8"
MAX_TOKENS = 128
PROTOCOL = "claude-opus-4-8-structured-forced-choice-preference-v1"
PREFERENCE_PROTOCOL = "claude-opus-4-8-structured-forced-choice-preference-v2"

PROBE_LAYOUTS = (
    ("donation_choice", "AB", "zero_first"),
    ("donation_choice", "XY", "one_first"),
    ("recipient_choice", "AB", "one_first"),
    ("recipient_choice", "XY", "zero_first"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _folder(stage: str, out_dir: Path = OUT) -> Path:
    if stage not in {"format_probe", "preferences"}:
        raise ValueError(f"Unknown structured preference stage: {stage}")
    return out_dir / stage


def _schema(symbols: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": symbols}},
        "required": ["choice"],
        "additionalProperties": False,
    }


def _trial_from_source(row: dict[str, Any], index: int) -> dict[str, Any]:
    mapping = {symbol: outcome for outcome, symbol in row["outcome_symbols"].items()}
    trial = {
        "custom_id": f"choice-{index:04d}",
        "stage": "independent_preference",
        "source_trial_id": row.get("trial_id", row.get("source_trial_id")),
        "pair_id": row["pair_id"],
        "wording_id": row["wording_id"],
        "symbol_scheme": row["symbol_scheme"],
        "presentation_order": row["presentation_order"],
        "outcome_symbols": row["outcome_symbols"],
        "symbol_to_outcome": mapping,
        "allowed_symbols": sorted(mapping),
        "prompt": row["prompt"],
        "prompt_sha256": sha256_text(row["prompt"]),
    }
    trial["json_schema"] = _schema(trial["allowed_symbols"])
    return trial


def _persist_manifest(stage: str, core: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    folder = _folder(stage, out_dir)
    manifest_path = folder / "manifest.json"
    request_path = folder / "requests.json"
    state_path = folder / "batch_state.json"
    if any(path.exists() for path in (manifest_path, request_path, state_path)):
        if all(path.exists() for path in (manifest_path, request_path, state_path)):
            old = read_json(manifest_path)
            validate_manifest(old, stage)
            state = read_json(state_path)
            if state.get("phase") == "prepared" and state.get("request_sha256") == sha256_file(request_path):
                return old
        raise RuntimeError(f"Refusing to overwrite existing/partial {stage} artifacts at {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {**core, "manifest_core_sha256": sha256_json(core), "created_at_utc": now()}
    requests = []
    for trial in manifest["trials"]:
        requests.append({"custom_id": trial["custom_id"], "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": trial["prompt"]}],
            "output_config": {"format": {"type": "json_schema", "schema": trial["json_schema"]}},
        }})
    write_json(manifest_path, manifest)
    write_json(request_path, {"requests": requests})
    state = {
        "protocol": core["protocol"],
        "stage": stage,
        "phase": "prepared",
        "batch_id": None,
        "manifest_core_sha256": manifest["manifest_core_sha256"],
        "request_sha256": sha256_file(request_path),
        "request_count": len(requests),
        "prepared_at_utc": now(),
    }
    write_json(state_path, state)
    validate_manifest(manifest, stage)
    return manifest


def prepare_probe(out_dir: Path = OUT) -> dict[str, Any]:
    source = read_json(SOURCE_MANIFEST)
    source_rows = [row for row in source["trials"] if row.get("stage") == "independent_preference"
                   and row.get("pair_id") == PRIMARY_PAIR_ID]
    selected = []
    for wording, scheme, order in PROBE_LAYOUTS:
        match = next(row for row in source_rows if row["wording_id"] == wording
                     and row["symbol_scheme"] == scheme and row["presentation_order"] == order)
        selected.append(match)
    trials = [_trial_from_source(row, i) for i, row in enumerate(selected, start=1)]
    return _persist_manifest("format_probe", {
        "protocol": PROTOCOL,
        "stage": "format_probe",
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "response_mode": "schema-constrained conditional choice",
        "measurement_statement": "This format forces a choice among the two described options; it does not measure spontaneous preference or refusal behavior.",
        "predeclared_pair_id": PRIMARY_PAIR_ID,
        "source_preference_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "json_schema_template": {
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": ["<the symbols in this trial>"]}},
            "required": ["choice"], "additionalProperties": False,
        },
        "probe_balance": {"wording_id": "2 each", "symbol_scheme": "2 each", "presentation_order": "2 each"},
        "probe_is_not_preference_sample": True,
        "trials": trials,
    }, out_dir)


def prepare_preferences(out_dir: Path = OUT) -> dict[str, Any]:
    freeze_path = out_dir / "format_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("Freeze a successful structured-output format probe before preparing the 64-request preference stage")
    freeze = read_json(freeze_path)
    if not freeze.get("format_passed"):
        raise RuntimeError("Structured-output probe did not pass; the 64-request stage is blocked")
    freeze_core = {key: value for key, value in freeze.items() if key != "freeze_sha256"}
    if freeze.get("freeze_sha256") != sha256_json(freeze_core):
        raise ValueError("Successful format-probe freeze checksum is invalid")
    source = read_json(SOURCE_MANIFEST)
    base = design.build_preference_manifest(source)
    trials = [_trial_from_source(row, i) for i, row in enumerate(base["trials"], start=1)]
    core = {
        "protocol": PREFERENCE_PROTOCOL,
        "stage": "preferences",
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "response_mode": "schema-constrained conditional choice",
        "measurement_statement": "Forced conditional choice between the described options; not spontaneous preference.",
        "predeclared_primary_pair_id": PRIMARY_PAIR_ID,
        "format_freeze_sha256": freeze["freeze_sha256"],
        "source_preference_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "stability_rule": "For the predeclared pair: at least 12 of 16 valid choices favor one outcome and every wording, presentation-order, and symbol-scheme subgroup favors that same outcome.",
        "counterbalancing_rule": "Exact original 64 independent-preference layouts from the frozen source manifest: 16 per pair, preserving prompt text, wording, presentation order, symbol scheme, and outcome-to-symbol mapping.",
        "schema_neutrality_rule": "Each response schema permits both symbols shown in its own prompt and contains no beneficiary text, outcome mapping, preferred choice, or other condition-specific cue. Across each pair, each outcome maps to A, B, X, and Y exactly four times.",
        "trials": trials,
    }
    return _persist_manifest("preferences", core, out_dir)


def _groups_by(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return groups


def _schema_neutrality(manifest: dict[str, Any]) -> dict[str, Any]:
    trials = manifest["trials"]
    shapes = {canonical_json({**trial["json_schema"], "properties": {
        "choice": {"type": "string", "enum": ["<symbols>"]}}}) for trial in trials}
    per_pair = {}
    for pair_id, group in _groups_by(trials, "pair_id").items():
        per_pair[pair_id] = {outcome: {symbol: sum(row["outcome_symbols"][outcome] == symbol for row in group)
                                       for symbol in ("A", "B", "X", "Y")}
                             for outcome in ("outcome_0", "outcome_1")}
    balanced = all(count == 4 for pair in per_pair.values() for outcome_counts in pair.values()
                   for count in outcome_counts.values())
    return {
        "pass": len(shapes) == 1 and balanced,
        "same_schema_structure_all_trials": len(shapes) == 1,
        "schema_contains_only_choice_symbol_enum": True,
        "schema_includes_no_beneficiary_or_outcome_mapping": True,
        "both_prompt_symbols_allowed_each_trial": all(
            trial["json_schema"]["properties"]["choice"]["enum"] == trial["allowed_symbols"]
            for trial in trials),
        "mapping_counts_by_pair_and_outcome": per_pair,
        "mapping_is_fully_counterbalanced": balanced,
    }


def import_successful_probe(probe_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Validate an existing successful four-request probe and freeze its format provenance."""
    manifest_path = probe_dir / "manifest.json"
    request_path = probe_dir / "requests.json"
    state_path = probe_dir / "batch_state.json"
    status_path = probe_dir / "batch_status.json"
    provider_path = probe_dir / "provider_results.jsonl"
    raw_path = probe_dir / "raw.jsonl"
    receipt_path = probe_dir / "receipt.json"
    manifest, requests_doc = read_json(manifest_path), read_json(request_path)
    state, status, receipt = read_json(state_path), read_json(status_path), read_json(receipt_path)
    if manifest.get("protocol") != PROTOCOL or len(manifest.get("trials", [])) != 4:
        raise ValueError("Source is not the four-request structured-output probe")
    if state.get("phase") != "retrieved" or not state.get("batch_id") or status.get("processing_status") != "ended":
        raise ValueError("Probe must be retrieved from an ended provider batch")
    if receipt.get("batch_id") != state["batch_id"] or receipt.get("status_counts") != {"completed": 4}:
        raise ValueError("Probe receipt does not confirm four completed responses")
    if receipt.get("stop_reasons") != {"end_turn": 4} or receipt.get("valid_structured_choice_count") != 4:
        raise ValueError("Probe did not confirm four valid end_turn structured choices")
    if sha256_file(provider_path) != receipt.get("provider_results_sha256"):
        raise ValueError("Probe provider response checksum differs from receipt")
    if sha256_file(raw_path) != receipt.get("raw_sha256"):
        raise ValueError("Probe normalized response checksum differs from receipt")
    if sha256_file(manifest_path) != receipt.get("manifest_sha256"):
        raise ValueError("Probe manifest checksum differs from receipt")
    if receipt.get("manifest_core_sha256") != manifest.get("manifest_core_sha256"):
        raise ValueError("Probe receipt refers to a different manifest core")
    trials = {trial["custom_id"]: trial for trial in manifest["trials"]}
    requests = requests_doc.get("requests", [])
    if len(requests) != 4 or {request.get("custom_id") for request in requests} != set(trials):
        raise ValueError("Probe request IDs differ from the four frozen probe trials")
    for request in requests:
        trial = trials[request["custom_id"]]
        params = request["params"]
        if params.get("model") != MODEL or params.get("max_tokens") != MAX_TOKENS or "thinking" in params:
            raise ValueError("Probe generation settings do not match the successful format settings")
        if params.get("output_config", {}).get("format", {}).get("schema") != trial["json_schema"]:
            raise ValueError("Probe request schema differs from its prompt's two-symbol schema")
    rows = _parse_batch_results(provider_path.read_bytes())
    if set(rows) != set(trials):
        raise ValueError("Provider probe results do not cover all original custom IDs")
    usage = Counter()
    for custom_id, provider_row in rows.items():
        result = provider_row.get("result", {})
        message = result.get("message", {})
        trial = trials[custom_id]
        text = _message_text(message)
        parsed = json.loads(text)
        if (result.get("type") != "succeeded" or message.get("stop_reason") != "end_turn"
                or set(parsed) != {"choice"} or parsed["choice"] not in trial["symbol_to_outcome"]):
            raise ValueError(f"Probe response is not a valid end_turn structured choice: {custom_id}")
        for key, value in (message.get("usage") or {}).items():
            if isinstance(value, int):
                usage[key] += value
    if dict(usage) != receipt.get("usage"):
        raise ValueError("Probe usage metadata differs from receipt")
    freeze = {
        "protocol": PREFERENCE_PROTOCOL,
        "format_passed": True,
        "measurement_mode": "forced conditional choice between the options shown; not spontaneous preference",
        "format": "Anthropic structured JSON output, one required choice property, enum restricted to both symbols shown in each prompt",
        "schema_template": {"type": "object", "properties": {"choice": {"type": "string", "enum": ["<both prompt symbols>"]}},
                            "required": ["choice"], "additionalProperties": False},
        "generation_settings": {"model": MODEL, "max_tokens": MAX_TOKENS,
                                "thinking": None, "output_config": {"format": {"type": "json_schema"}}},
        "successful_probe": {"batch_id": state["batch_id"], "request_count": 4,
                             "valid_structured_choice_count": 4, "stop_reasons": {"end_turn": 4},
                             "usage": dict(usage), "manifest_sha256": sha256_file(manifest_path),
                             "manifest_core_sha256": manifest["manifest_core_sha256"],
                             "request_sha256": sha256_file(request_path),
                             "provider_results_sha256": sha256_file(provider_path),
                             "raw_sha256": sha256_file(raw_path), "receipt_sha256": sha256_file(receipt_path)},
        "source_probe_artifacts": str(probe_dir.resolve()),
        "frozen_at_utc": now(),
    }
    freeze["freeze_sha256"] = sha256_json(freeze)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "format_freeze.json"
    if target.exists():
        old = read_json(target)
        if old.get("freeze_sha256") == freeze["freeze_sha256"]:
            return old
        raise RuntimeError(f"Refusing to overwrite existing format freeze at {target}")
    write_json(target, freeze)
    return freeze


def freeze_manifest(out_dir: Path = OUT) -> dict[str, Any]:
    validation = validate("preferences", out_dir)
    folder = _folder("preferences", out_dir)
    freeze_path = folder / "manifest_freeze.json"
    manifest_path = folder / "manifest.json"
    request_path = folder / "requests.json"
    format_freeze = read_json(out_dir / "format_freeze.json")
    freeze = {
        "protocol": PREFERENCE_PROTOCOL,
        "stage": "preferences",
        "forced_choice_construct": "conditional choice between the two displayed beneficiaries, not spontaneous preference",
        "request_count": 64,
        "predeclared_primary_pair_id": PRIMARY_PAIR_ID,
        "format_freeze_sha256": format_freeze["freeze_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_core_sha256": validation["manifest_core_sha256"],
        "requests_sha256": sha256_file(request_path),
        "schema_neutrality": validation["schema_neutrality"],
        "frozen_at_utc": now(),
    }
    freeze["freeze_sha256"] = sha256_json(freeze)
    if freeze_path.exists():
        old = read_json(freeze_path)
        for key in ("manifest_sha256", "manifest_core_sha256", "requests_sha256", "format_freeze_sha256"):
            if old.get(key) != freeze.get(key):
                raise RuntimeError("Existing manifest freeze does not match current artifacts; refusing to overwrite")
        return old
    write_json(freeze_path, freeze)
    return freeze


def validate_manifest(manifest: dict[str, Any], stage: str) -> dict[str, Any]:
    expected = {"format_probe": 4, "preferences": 64}[stage]
    expected_protocol = PROTOCOL if stage == "format_probe" else PREFERENCE_PROTOCOL
    if manifest.get("protocol") != expected_protocol or len(manifest.get("trials", [])) != expected:
        raise ValueError(f"{stage} manifest has the wrong protocol or trial count")
    if stage == "preferences" and manifest.get("source_preference_manifest_sha256") != sha256_file(SOURCE_MANIFEST):
        raise ValueError("Preference manifest does not match the frozen source preference manifest")
    ids = [trial.get("custom_id") for trial in manifest["trials"]]
    if len(ids) != len(set(ids)) or any(not cid or len(cid) > 64 for cid in ids):
        raise ValueError("custom_id values must be unique and no longer than 64 characters")
    for trial in manifest["trials"]:
        if trial["prompt_sha256"] != sha256_text(trial["prompt"]):
            raise ValueError(f"Prompt hash mismatch for {trial['custom_id']}")
        if trial["allowed_symbols"] != sorted(trial["symbol_to_outcome"]):
            raise ValueError(f"Allowed symbols disagree with mappings for {trial['custom_id']}")
        schema = trial["json_schema"]
        if schema["properties"]["choice"]["enum"] != trial["allowed_symbols"]:
            raise ValueError(f"JSON schema enum disagrees with prompt symbols for {trial['custom_id']}")
    if stage == "format_probe":
        for field in ("wording_id", "symbol_scheme", "presentation_order"):
            counts = Counter(trial[field] for trial in manifest["trials"])
            if set(counts.values()) != {2}:
                raise ValueError(f"Probe must balance {field} two each")
    else:
        counts = Counter(trial["pair_id"] for trial in manifest["trials"])
        if counts != Counter({pair["pair_id"]: 16 for pair in design.BENEFIT_PAIRS}):
            raise ValueError("Full preference stage must preserve the frozen 16-per-pair layout")
        source = read_json(SOURCE_MANIFEST)
        source_trials = {row["trial_id"]: row for row in source["trials"]
                         if row.get("stage") == "independent_preference"}
        for trial in manifest["trials"]:
            original = source_trials.get(trial["source_trial_id"])
            if original is None or trial["prompt"] != original["prompt"]:
                raise ValueError(f"Prompt differs from original counterbalanced trial {trial['custom_id']}")
            expected_mapping = {symbol: outcome for outcome, symbol in original["outcome_symbols"].items()}
            if trial["symbol_to_outcome"] != expected_mapping or trial["outcome_symbols"] != original["outcome_symbols"]:
                raise ValueError(f"Choice-to-beneficiary mapping differs from the frozen source for {trial['custom_id']}")
            if set(trial["allowed_symbols"]) != set(original["outcome_symbols"].values()) or len(trial["allowed_symbols"]) != 2:
                raise ValueError(f"Response schema does not allow exactly the two prompt symbols for {trial['custom_id']}")
            schema_text = canonical_json(trial["json_schema"])
            beneficiary_text = [design_pair[outcome] for design_pair in design.BENEFIT_PAIRS
                                if design_pair["pair_id"] == trial["pair_id"]
                                for outcome in ("outcome_0", "outcome_1")]
            if any(value in schema_text for value in beneficiary_text) or any(
                    value in schema_text for value in ("outcome_0", "outcome_1", "beneficiary")):
                raise ValueError(f"Response schema leaks beneficiary or outcome labels for {trial['custom_id']}")
        for pair_id, group in _groups_by(manifest["trials"], "pair_id").items():
            mapping_counts = Counter((outcome, symbol)
                                     for row in group for outcome, symbol in row["outcome_symbols"].items())
            expected_mappings = {(outcome, symbol): 4 for outcome in ("outcome_0", "outcome_1")
                                 for symbol in ("A", "B", "X", "Y")}
            if mapping_counts != Counter(expected_mappings):
                raise ValueError(f"Symbol-to-outcome mapping is not counterbalanced for {pair_id}")
            for field, levels in (("wording_id", ("donation_choice", "recipient_choice")),
                                  ("symbol_scheme", ("AB", "XY")),
                                  ("presentation_order", ("zero_first", "one_first"))):
                if Counter(row[field] for row in group) != Counter({level: 8 for level in levels}):
                    raise ValueError(f"Counterbalancing differs for {pair_id}/{field}")
    core = {key: value for key, value in manifest.items()
            if key not in {"manifest_core_sha256", "created_at_utc"}}
    if manifest.get("manifest_core_sha256") != sha256_json(core):
        raise ValueError("Manifest checksum mismatch")
    neutral = _schema_neutrality(manifest) if stage == "preferences" else None
    return {"valid": True, "stage": stage, "request_count": expected,
            "manifest_core_sha256": manifest["manifest_core_sha256"], "schema_neutrality": neutral}


def validate(stage: str, out_dir: Path = OUT) -> dict[str, Any]:
    folder = _folder(stage, out_dir)
    manifest = read_json(folder / "manifest.json")
    summary = validate_manifest(manifest, stage)
    request_doc = read_json(folder / "requests.json")
    requests = request_doc.get("requests", [])
    if len(requests) != summary["request_count"]:
        raise ValueError("Request count differs from manifest")
    trial_by_id = {trial["custom_id"]: trial for trial in manifest["trials"]}
    if {request.get("custom_id") for request in requests} != set(trial_by_id):
        raise ValueError("Request IDs differ from manifest")
    for request in requests:
        trial = trial_by_id[request["custom_id"]]
        params = request["params"]
        if params.get("model") != MODEL or params.get("max_tokens") != MAX_TOKENS:
            raise ValueError("Request model or output length differs from frozen manifest")
        if params.get("output_config", {}).get("format", {}).get("schema") != trial["json_schema"]:
            raise ValueError("Request JSON schema differs from manifest")
        if params.get("messages") != [{"role": "user", "content": trial["prompt"]}]:
            raise ValueError("Request prompt differs from manifest")
    state = read_json(folder / "batch_state.json")
    if state.get("manifest_core_sha256") != summary["manifest_core_sha256"]:
        raise ValueError("State file refers to a different manifest")
    if state.get("request_sha256") != sha256_file(folder / "requests.json"):
        raise ValueError("Request JSON checksum differs from prepared state")
    if stage == "preferences":
        format_freeze = read_json(out_dir / "format_freeze.json")
        if not format_freeze.get("format_passed"):
            raise ValueError("Preference stage lacks a successful frozen format probe")
        format_core = {key: value for key, value in format_freeze.items() if key != "freeze_sha256"}
        if format_freeze.get("freeze_sha256") != sha256_json(format_core):
            raise ValueError("Successful format-probe freeze checksum is invalid")
        freeze_path = folder / "manifest_freeze.json"
        if freeze_path.exists():
            freeze = read_json(freeze_path)
            freeze_core = {key: value for key, value in freeze.items() if key != "freeze_sha256"}
            if (freeze.get("manifest_sha256") != sha256_file(folder / "manifest.json")
                    or freeze.get("requests_sha256") != sha256_file(folder / "requests.json")
                    or freeze.get("manifest_core_sha256") != summary["manifest_core_sha256"]
                    or freeze.get("format_freeze_sha256") != format_freeze["freeze_sha256"]
                    or freeze.get("freeze_sha256") != sha256_json(freeze_core)):
                raise ValueError("Frozen preference manifest checksum mismatch")
    return summary


def submit(stage: str, out_dir: Path = OUT) -> dict[str, Any]:
    validate(stage, out_dir)
    folder = _folder(stage, out_dir)
    if stage == "preferences" and not (folder / "manifest_freeze.json").is_file():
        raise RuntimeError("Freeze the validated preference manifest before submission")
    state_path = folder / "batch_state.json"
    state = read_json(state_path)
    if state.get("phase") != "prepared" or state.get("batch_id"):
        raise RuntimeError("Refusing duplicate or uncertain batch submission")
    manifest = read_json(folder / "manifest.json")
    settings = anthropic_settings_from_config({
        "provider": "anthropic", "model": MODEL, "max_output_tokens": MAX_TOKENS,
        "api_key_env": "ANTHROPIC_API_KEY", "anthropic_version": "2023-06-01",
        "http_transport": "auto", "request_timeout_seconds": 300,
    }, REPO)
    state.update({"phase": "submitting", "submission_started_at_utc": now()})
    write_json(state_path, state)
    try:
        response = anthropic_request_json(settings, "POST", "/v1/messages/batches", read_json(folder / "requests.json"))
    except Exception as exc:
        detail = str(exc)
        lowered = detail.lower()
        if ("remote name could not be resolved" in lowered
                or "authentication_error" in lowered
                or "invalid x-api-key" in lowered):
            # DNS resolution fails before an HTTP connection exists; an
            # authentication_error response is a definitive provider rejection
            # before a batch can be accepted. Neither can launch inference.
            # Preserve the attempt while permitting a deliberate retry after
            # connectivity or credentials are fixed.
            attempts = state.get("preconnection_failures", [])
            attempts.append({"error": detail, "failed_at_utc": now(),
                             "request_could_have_reached_provider": False,
                             "provider_rejected_before_batch_creation": "authentication_error" in lowered})
            state.update({"phase": "prepared", "preconnection_failures": attempts})
            write_json(state_path, state)
            raise RuntimeError("Submission was not accepted by Anthropic (DNS failure or authentication rejection); the unchanged manifest remains prepared") from exc
        state.update({"phase": "submission_unknown", "submission_error": detail, "failed_at_utc": now()})
        write_json(state_path, state)
        raise RuntimeError("Submission outcome unknown; reconcile provider batches, do not resubmit") from exc
    write_json(folder / "batch_submission.json", response)
    if not response.get("id"):
        state.update({"phase": "submission_unknown", "provider_response": response})
        write_json(state_path, state)
        raise RuntimeError("Provider returned no batch ID; refusing an automatic retry")
    state.update({"phase": "submitted", "batch_id": response["id"], "submitted_at_utc": now()})
    write_json(state_path, state)
    return response


def poll(stage: str, out_dir: Path = OUT) -> dict[str, Any]:
    folder = _folder(stage, out_dir)
    state_path = folder / "batch_state.json"
    state = read_json(state_path)
    if state.get("phase") in {"ended", "retrieved"}:
        return read_json(folder / "batch_status.json")
    if state.get("phase") == "retrieval_failed" and state.get("last_processing_status") == "ended":
        return read_json(folder / "batch_status.json")
    if state.get("phase") not in {"submitted", "retrieval_failed"} or not state.get("batch_id"):
        raise RuntimeError("No submitted batch ID is available")
    settings = anthropic_settings_from_config({
        "provider": "anthropic", "model": MODEL, "max_output_tokens": MAX_TOKENS,
        "api_key_env": "ANTHROPIC_API_KEY", "anthropic_version": "2023-06-01",
        "http_transport": "auto", "request_timeout_seconds": 300,
    }, REPO)
    status = anthropic_request_json(settings, "GET", f"/v1/messages/batches/{state['batch_id']}")
    write_json(folder / "batch_status.json", status)
    state.update({"phase": "ended" if status.get("processing_status") == "ended" else "submitted",
                  "last_processing_status": status.get("processing_status"),
                  "last_request_counts": status.get("request_counts"), "last_polled_at_utc": now()})
    if status.get("processing_status") == "ended":
        state["phase"] = "ended"
    write_json(state_path, state)
    return status


def retrieve(stage: str, out_dir: Path = OUT) -> dict[str, Any]:
    folder = _folder(stage, out_dir)
    state_path = folder / "batch_state.json"
    state = read_json(state_path)
    if state.get("phase") == "retrieved":
        return read_json(folder / "receipt.json")
    if state.get("phase") not in {"ended", "retrieval_failed"} or not state.get("batch_id"):
        raise RuntimeError("Retrieve only after provider batch status is ended")
    manifest = read_json(folder / "manifest.json")
    settings = anthropic_settings_from_config({
        "provider": "anthropic", "model": MODEL, "max_output_tokens": MAX_TOKENS,
        "api_key_env": "ANTHROPIC_API_KEY", "anthropic_version": "2023-06-01",
        "http_transport": "auto", "request_timeout_seconds": 300,
    }, REPO)
    try:
        body = anthropic_request_bytes(settings, "GET", f"/v1/messages/batches/{state['batch_id']}/results")
        attempt = folder / f"provider_results_attempt_{now().replace(':', '').replace('-', '')}.jsonl"
        attempt.write_bytes(body)
        (folder / "provider_results.jsonl").write_bytes(body)
        state.update({"phase": "retrieving", "provider_results_sha256": sha256_file(folder / "provider_results.jsonl")})
        write_json(state_path, state)
        rows = _parse_batch_results(body)
        expected = {trial["custom_id"] for trial in manifest["trials"]}
        if set(rows) != expected:
            raise ValueError(f"Provider result IDs differ from manifest: {len(rows)} received, {len(expected)} expected")
        records = []
        trial_by_id = {trial["custom_id"]: trial for trial in manifest["trials"]}
        for custom_id, result in rows.items():
            trial = trial_by_id[custom_id]
            outcome = result.get("result", {})
            message = outcome.get("message", {})
            text = _message_text(message)
            response_obj = None
            parse_error = None
            try:
                response_obj = json.loads(text)
            except (TypeError, json.JSONDecodeError) as exc:
                parse_error = str(exc)
            choice = response_obj.get("choice") if isinstance(response_obj, dict) else None
            symbol_map = trial["symbol_to_outcome"]
            valid = (outcome.get("type") == "succeeded" and message.get("stop_reason") == "end_turn"
                     and choice in symbol_map)
            records.append({
                **trial,
                "status": "completed" if outcome.get("type") == "succeeded" and message.get("stop_reason") == "end_turn" else "incomplete" if outcome.get("type") == "succeeded" else "failed",
                "response_text": text,
                "response_json": response_obj,
                "parse_error": parse_error,
                "strict_symbol": choice if choice in symbol_map else None,
                "beneficiary": symbol_map.get(choice),
                "strict_response_format": isinstance(response_obj, dict) and set(response_obj) == {"choice"},
                "valid": valid,
                "returned_model": message.get("model"),
                "response_id": message.get("id"),
                "stop_reason": message.get("stop_reason"),
                "usage": message.get("usage"),
                "raw_response": message if outcome.get("type") == "succeeded" else result,
            })
        raw_path = folder / "raw.jsonl"
        raw_path.write_text("".join(canonical_json(row) + "\n" for row in records), encoding="utf-8", newline="\n")
        usage: Counter[str] = Counter()
        for row in records:
            for key, value in (row.get("usage") or {}).items():
                if isinstance(value, int):
                    usage[key] += value
        receipt = {
            "protocol": read_json(folder / "manifest.json")["protocol"],
            "stage": stage,
            "batch_id": state["batch_id"],
            "requested_model": MODEL,
            "returned_models": dict(Counter(row.get("returned_model") for row in records if row.get("returned_model"))),
            "stop_reasons": dict(Counter(row.get("stop_reason") for row in records if row.get("stop_reason"))),
            "status_counts": dict(Counter(row["status"] for row in records)),
            "valid_structured_choice_count": sum(bool(row["valid"]) for row in records),
            "usage": dict(usage),
            "manifest_sha256": sha256_file(folder / "manifest.json"),
            "manifest_core_sha256": manifest["manifest_core_sha256"],
            "provider_results_sha256": sha256_file(folder / "provider_results.jsonl"),
            "raw_sha256": sha256_file(raw_path),
            "retrieved_at_utc": now(),
        }
        write_json(folder / "receipt.json", receipt)
        state.update({"phase": "retrieved", "retrieved_at_utc": now()})
        write_json(state_path, state)
        return receipt
    except Exception as exc:
        state = read_json(state_path)
        state.update({"phase": "retrieval_failed", "batch_id": state.get("batch_id"),
                      "last_processing_status": "ended",
                      "retrieval_error": str(exc), "retrieval_failed_at_utc": now()})
        write_json(state_path, state)
        raise


def freeze_format(out_dir: Path = OUT) -> dict[str, Any]:
    folder = _folder("format_probe", out_dir)
    state = read_json(folder / "batch_state.json")
    if state.get("phase") != "retrieved":
        raise RuntimeError("Retrieve the four format-probe responses before freezing the format")
    manifest = read_json(folder / "manifest.json")
    records = [json.loads(line) for line in (folder / "raw.jsonl").read_text(encoding="utf-8").splitlines() if line]
    passed = len(records) == 4 and all(row.get("valid") and row.get("strict_response_format") for row in records)
    freeze = {
        "protocol": PROTOCOL,
        "format_passed": passed,
        "measurement_mode": "forced conditional choice between the options shown; not spontaneous preference",
        "schema": manifest["json_schema_template"],
        "max_tokens": MAX_TOKENS,
        "probe_manifest_core_sha256": manifest["manifest_core_sha256"],
        "probe_raw_sha256": sha256_file(folder / "raw.jsonl"),
        "probe_receipt_sha256": sha256_file(folder / "receipt.json"),
        "response_count": len(records),
        "valid_choice_count": sum(bool(row.get("valid")) for row in records),
        "frozen_at_utc": now(),
    }
    freeze["freeze_sha256"] = sha256_json(freeze)
    write_json(out_dir / "format_freeze.json", freeze)
    return freeze


def freeze_preferences(out_dir: Path = OUT) -> dict[str, Any]:
    validate("preferences", out_dir)
    folder = _folder("preferences", out_dir)
    if not (folder / "manifest_freeze.json").is_file():
        raise RuntimeError("Freeze the revised preference manifest before freezing its results")
    state = read_json(folder / "batch_state.json")
    if state.get("phase") != "retrieved":
        raise RuntimeError("Retrieve all structured preference responses before freezing")
    manifest = read_json(folder / "manifest.json")
    records = [json.loads(line) for line in (folder / "raw.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if {row.get("custom_id") for row in records} != {trial["custom_id"] for trial in manifest["trials"]}:
        raise ValueError("Retrieved preference records do not match the frozen 64 custom IDs")
    summary = design.summarize_preferences(records)
    primary = summary["rankings"][PRIMARY_PAIR_ID]
    analysis = analyze_preference_records(records)
    freeze = {
        "protocol": PREFERENCE_PROTOCOL,
        "measurement_mode": "forced conditional choice between the options shown; not spontaneous preference",
        "primary_pair_id": PRIMARY_PAIR_ID,
        "favored_outcome": primary["favored_outcome"],
        "primary_pair_stable": primary["stable"],
        "primary_pair_summary": next(row for row in summary["pairs"] if row["pair_id"] == PRIMARY_PAIR_ID),
        "stability_rule": summary["stability_rule"],
        "batch_id": state["batch_id"],
        "format_freeze_sha256": manifest["format_freeze_sha256"],
        "manifest_core_sha256": manifest["manifest_core_sha256"],
        "raw_sha256": sha256_file(folder / "raw.jsonl"),
        "provider_results_sha256": sha256_file(folder / "provider_results.jsonl"),
        "analysis_sha256": sha256_json(analysis),
        "frozen_at_utc": now(),
    }
    freeze["freeze_sha256"] = sha256_json(freeze)
    write_json(out_dir / "preference_summary_structured.json", summary)
    write_json(out_dir / "preference_analysis_structured.json", analysis)
    write_json(out_dir / "preference_freeze_structured.json", freeze)
    return freeze


def analyze_preference_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair = {}
    for pair in design.BENEFIT_PAIRS:
        pair_id = pair["pair_id"]
        rows = [row for row in records if row.get("pair_id") == pair_id]
        valid = [row for row in rows if row.get("valid")]
        frequency = Counter(row["beneficiary"] for row in valid)
        symbol_frequency = Counter(row["strict_symbol"] for row in valid)
        by_position = {}
        for position in ("zero_first", "one_first"):
            group = [row for row in valid if row.get("presentation_order") == position]
            chosen_first = sum(
                row["strict_symbol"] == row["outcome_symbols"]["outcome_0" if position == "zero_first" else "outcome_1"]
                for row in group)
            by_position[position] = {"n_valid": len(group), "chosen_first_count": chosen_first,
                                     "chosen_first_rate": chosen_first / len(group) if group else None,
                                     "outcome_shown_first": "outcome_0" if position == "zero_first" else "outcome_1"}
        first_n = sum(item["n_valid"] for item in by_position.values())
        first_count = sum(item["chosen_first_count"] for item in by_position.values())
        by_mapping = {}
        for outcome in ("outcome_0", "outcome_1"):
            for symbol in ("A", "B", "X", "Y"):
                shown = [row for row in valid if row["outcome_symbols"].get(outcome) == symbol]
                chosen = sum(row["beneficiary"] == outcome for row in shown)
                by_mapping[f"{outcome}_as_{symbol}"] = {"n_valid": len(shown), "chosen_count": chosen,
                                                        "choice_rate": chosen / len(shown) if shown else None}
        by_pair[pair_id] = {
            "outcome_descriptions": {"outcome_0": pair["outcome_0"], "outcome_1": pair["outcome_1"]},
            "n_total": len(rows), "n_valid": len(valid), "invalid_or_incomplete": len(rows) - len(valid),
            "choice_frequency_by_beneficiary": {key: frequency.get(key, 0) for key in ("outcome_0", "outcome_1")},
            "choice_frequency_by_symbol": {key: symbol_frequency.get(key, 0) for key in ("A", "B", "X", "Y")},
            "subgroup_consistency": design.summarize_preferences(rows)["rankings"][pair_id],
            "first_position_choice": {"by_presentation_order": by_position, "n_valid": first_n,
                                       "chosen_first_count": first_count,
                                       "chosen_first_rate": first_count / first_n if first_n else None,
                                       "chosen_first_minus_chosen_second": (2 * first_count / first_n - 1) if first_n else None},
            "outcome_choice_rates_by_symbol_mapping": by_mapping,
        }
    stability = by_pair[PRIMARY_PAIR_ID]["subgroup_consistency"]
    return {
        "protocol": PREFERENCE_PROTOCOL,
        "measurement_mode": "forced comparative choice between the two displayed beneficiaries; not spontaneous preference",
        "predeclared_primary_pair_id": PRIMARY_PAIR_ID,
        "choice_frequencies": by_pair,
        "primary_pair_stability": {
            "criterion": ">=12 of 16 valid choices favor one outcome and each wording, presentation-order, and symbol-scheme subgroup favors that same outcome",
            "favored_outcome": stability["favored_outcome"],
            "valid_count": stability["valid_count"],
            "favor_counts": stability["favor_counts"],
            "stable": stability["stable"],
            "subgroup_rates": by_pair[PRIMARY_PAIR_ID]["subgroup_consistency"],
        },
        "analysis_scope": "Descriptive frequencies, design-subgroup consistency, position and symbol-mapping effects; forced choices do not establish spontaneous preference.",
    }


def main(argv: list[str] | None = None) -> None:
    commands = (
        "prepare-probe", "validate-probe", "submit-probe", "poll-probe", "retrieve-probe", "freeze-format",
        "import-successful-probe", "freeze-manifest",
        "prepare-preferences", "validate-preferences", "submit-preferences", "poll-preferences",
        "retrieve-preferences", "freeze-preferences",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands)
    parser.add_argument("--out-dir", type=Path, default=OUT,
                        help="Revision output directory; use a new directory to preserve earlier artifacts")
    parser.add_argument("--probe-dir", type=Path,
                        help="Existing successful four-request format_probe directory for import-successful-probe")
    args = parser.parse_args(argv)
    command = args.command
    if command == "prepare-probe": result = prepare_probe(args.out_dir)
    elif command == "validate-probe": result = validate("format_probe", args.out_dir)
    elif command == "submit-probe": result = submit("format_probe", args.out_dir)
    elif command == "poll-probe": result = poll("format_probe", args.out_dir)
    elif command == "retrieve-probe": result = retrieve("format_probe", args.out_dir)
    elif command == "freeze-format": result = freeze_format(args.out_dir)
    elif command == "import-successful-probe":
        if not args.probe_dir:
            parser.error("import-successful-probe requires --probe-dir pointing to the completed probe")
        result = import_successful_probe(args.probe_dir, args.out_dir)
    elif command == "prepare-preferences": result = prepare_preferences(args.out_dir)
    elif command == "validate-preferences": result = validate("preferences", args.out_dir)
    elif command == "freeze-manifest": result = freeze_manifest(args.out_dir)
    elif command == "submit-preferences": result = submit("preferences", args.out_dir)
    elif command == "poll-preferences": result = poll("preferences", args.out_dir)
    elif command == "retrieve-preferences": result = retrieve("preferences", args.out_dir)
    else: result = freeze_preferences(args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
