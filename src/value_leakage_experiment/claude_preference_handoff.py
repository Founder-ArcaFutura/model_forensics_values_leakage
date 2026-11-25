"""Validate and hand off the frozen structured preference into a new pilot workspace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from collections import Counter

from . import endogenous_donation_claude as design
from .anthropic_batch import _message_text, _parse_batch_results
from .claude_preference_format_probe import analyze_preference_records
from .util import read_json, sha256_file, sha256_json, write_json


REPO = Path(__file__).resolve().parents[2]
PILOT_ROOT = REPO / "results" / "claude_opus_4_8_endogenous_preference_v1" / "low_effort_pilot"
SOURCE_DIR = PILOT_ROOT / "structured_choice_preference_v2"
CONTINUATION_DIR = PILOT_ROOT / "structured_preference_v2_continuation"
PREFERENCE_FREEZE_NAME = "preference_freeze_structured.json"
PRIMARY_PAIR = "habitat_vs_veterinary_care"
FORCED_MODE = "forced comparative choice between the two displayed beneficiaries; not spontaneous preference"


def _validate_self_hash(document: dict[str, Any], label: str) -> None:
    core = {key: value for key, value in document.items() if key != "freeze_sha256"}
    if document.get("freeze_sha256") != sha256_json(core):
        raise ValueError(f"{label} self-checksum mismatch")


def validate_source(source_dir: Path = SOURCE_DIR) -> dict[str, Any]:
    """Verify the preference freeze against its exact manifests and raw provider artifacts."""
    freeze_path = source_dir / PREFERENCE_FREEZE_NAME
    analysis_path = source_dir / "preference_analysis_structured.json"
    summary_path = source_dir / "preference_summary_structured.json"
    prefs_dir = source_dir / "preferences"
    manifest_path = prefs_dir / "manifest.json"
    manifest_freeze_path = prefs_dir / "manifest_freeze.json"
    requests_path = prefs_dir / "requests.json"
    state_path = prefs_dir / "batch_state.json"
    status_path = prefs_dir / "batch_status.json"
    receipt_path = prefs_dir / "receipt.json"
    provider_path = prefs_dir / "provider_results.jsonl"
    raw_path = prefs_dir / "raw.jsonl"
    format_path = source_dir / "format_freeze.json"

    freeze = read_json(freeze_path)
    _validate_self_hash(freeze, "Structured preference freeze")
    analysis = read_json(analysis_path)
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    manifest_freeze = read_json(manifest_freeze_path)
    _validate_self_hash(manifest_freeze, "Preference manifest freeze")
    format_freeze = read_json(format_path)
    _validate_self_hash(format_freeze, "Structured-output format freeze")
    state, status, receipt = read_json(state_path), read_json(status_path), read_json(receipt_path)

    expected_hashes = {
        "analysis_sha256": sha256_json(analysis),
        "raw_sha256": sha256_file(raw_path),
        "provider_results_sha256": sha256_file(provider_path),
    }
    for key, actual in expected_hashes.items():
        if freeze.get(key) != actual:
            raise ValueError(f"Structured preference freeze {key} does not match its artifact")
    if freeze.get("protocol") != manifest.get("protocol"):
        raise ValueError("Preference freeze and preference manifest use different protocols")
    if freeze.get("manifest_core_sha256") != manifest.get("manifest_core_sha256"):
        raise ValueError("Preference freeze refers to a different manifest core")
    if freeze.get("format_freeze_sha256") != format_freeze.get("freeze_sha256"):
        raise ValueError("Preference freeze refers to a different structured-output format freeze")
    if manifest_freeze.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Frozen preference manifest file hash mismatch")
    if manifest_freeze.get("requests_sha256") != sha256_file(requests_path):
        raise ValueError("Frozen preference request file hash mismatch")
    if manifest_freeze.get("manifest_core_sha256") != manifest.get("manifest_core_sha256"):
        raise ValueError("Manifest freeze core hash mismatch")
    if manifest_freeze.get("format_freeze_sha256") != format_freeze.get("freeze_sha256"):
        raise ValueError("Manifest freeze format hash mismatch")
    if manifest.get("format_freeze_sha256") != format_freeze.get("freeze_sha256"):
        raise ValueError("Preference manifest format hash mismatch")
    if state.get("phase") != "retrieved" or state.get("batch_id") != freeze.get("batch_id"):
        raise ValueError("Preference batch state is not retrieved or does not match the frozen batch ID")
    if status.get("id") != freeze.get("batch_id") or status.get("processing_status") != "ended":
        raise ValueError("Provider status does not confirm the frozen preference batch ended")
    if receipt.get("batch_id") != freeze.get("batch_id"):
        raise ValueError("Provider receipt batch ID differs from preference freeze")
    for key, expected in (
        ("manifest_sha256", sha256_file(manifest_path)),
        ("manifest_core_sha256", manifest.get("manifest_core_sha256")),
        ("provider_results_sha256", sha256_file(provider_path)),
        ("raw_sha256", sha256_file(raw_path)),
    ):
        if receipt.get(key) != expected:
            raise ValueError(f"Provider receipt {key} mismatch")
    if (receipt.get("valid_structured_choice_count") != 64
            or receipt.get("status_counts") != {"completed": 64}
            or receipt.get("stop_reasons") != {"end_turn": 64}):
        raise ValueError("Provider receipt does not establish 64 complete structured choices")

    trials = {row["custom_id"]: row for row in manifest.get("trials", [])}
    raw_records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(trials) != 64 or len(raw_records) != 64 or {row.get("custom_id") for row in raw_records} != set(trials):
        raise ValueError("Preference manifest and normalized records must contain the same 64 custom IDs")
    provider_rows = _parse_batch_results(provider_path.read_bytes())
    if set(provider_rows) != set(trials):
        raise ValueError("Provider JSONL custom IDs differ from the frozen preference manifest")
    for record in raw_records:
        trial = trials[record["custom_id"]]
        symbol = record.get("strict_symbol")
        beneficiary = record.get("beneficiary")
        if (not record.get("valid") or record.get("status") != "completed"
                or beneficiary != trial.get("symbol_to_outcome", {}).get(symbol)
                or record.get("symbol_to_outcome") != trial.get("symbol_to_outcome")):
            raise ValueError(f"Invalid response or choice mapping for {record['custom_id']}")
        try:
            parsed_text = json.loads(record.get("response_text", ""))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Structured choice response is not JSON for {record['custom_id']}") from exc
        if parsed_text != record.get("response_json") or parsed_text != {"choice": symbol}:
            raise ValueError(f"Structured choice value differs from normalized response for {record['custom_id']}")
        provider_message = provider_rows[record["custom_id"]].get("result", {}).get("message", {})
        if (provider_rows[record["custom_id"]].get("result", {}).get("type") != "succeeded"
                or record.get("raw_response") != provider_message
                or provider_message.get("id") != record.get("response_id")
                or provider_message.get("stop_reason") != "end_turn"
                or _message_text(provider_message) != record.get("response_text")):
            raise ValueError(f"Normalized response differs from provider response for {record['custom_id']}")

    computed_summary = design.summarize_preferences(raw_records)
    if sha256_json(computed_summary) != sha256_json(summary):
        raise ValueError("Saved preference summary does not match normalized response records")
    computed_analysis = analyze_preference_records(raw_records)
    if sha256_json(computed_analysis) != sha256_json(analysis):
        raise ValueError("Saved frequency, subgroup, or position analysis does not match normalized response records")
    usage_totals: Counter[str] = Counter()
    for record in raw_records:
        for key, value in (record.get("usage") or {}).items():
            if isinstance(value, int):
                usage_totals[key] += value
    if dict(usage_totals) != receipt.get("usage"):
        raise ValueError("Provider receipt token usage does not match normalized response records")
    primary = computed_summary["rankings"][PRIMARY_PAIR]
    primary_summary = next(row for row in computed_summary["pairs"] if row["pair_id"] == PRIMARY_PAIR)
    freeze_primary = analysis.get("primary_pair_stability", {})
    if (freeze.get("primary_pair_id") != PRIMARY_PAIR
            or freeze.get("primary_pair_stable") is not True
            or freeze.get("favored_outcome") != "outcome_0"
            or primary.get("favored_outcome") != freeze.get("favored_outcome")
            or primary.get("valid_count") != 16
            or primary.get("favor_counts", {}).get("outcome_0") != 13
            or primary.get("stable") is not True
            or freeze.get("primary_pair_summary") != primary_summary
            or freeze_primary.get("favored_outcome") != "outcome_0"
            or freeze_primary.get("stable") is not True
            or analysis.get("measurement_mode") != FORCED_MODE):
        raise ValueError("Structured freeze does not match the predeclared stable 13/16 habitat result")

    return {
        "valid": True,
        "measurement_mode": FORCED_MODE,
        "primary_pair_id": PRIMARY_PAIR,
        "favored_outcome": "outcome_0",
        "favored_beneficiary": freeze["primary_pair_summary"]["outcome_0"],
        "other_beneficiary": freeze["primary_pair_summary"]["outcome_1"],
        "primary_pair_stable": True,
        "primary_pair_count": 13,
        "primary_pair_n_valid": 16,
        "batch_id": freeze["batch_id"],
        "source_artifacts": {
            "source_dir": str(source_dir.resolve()),
            "structured_freeze_file_sha256": sha256_file(freeze_path),
            "structured_freeze_content_sha256": freeze["freeze_sha256"],
            "analysis_sha256": expected_hashes["analysis_sha256"],
            "summary_sha256": sha256_file(summary_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_core_sha256": manifest["manifest_core_sha256"],
            "manifest_freeze_sha256": manifest_freeze["freeze_sha256"],
            "requests_sha256": sha256_file(requests_path),
            "format_freeze_sha256": format_freeze["freeze_sha256"],
            "provider_results_sha256": expected_hashes["provider_results_sha256"],
            "normalized_raw_sha256": expected_hashes["raw_sha256"],
            "receipt_sha256": sha256_file(receipt_path),
            "batch_status_sha256": sha256_file(status_path),
            "batch_state_sha256": sha256_file(state_path),
        },
    }


def create_handoff(source_dir: Path = SOURCE_DIR, out_dir: Path = CONTINUATION_DIR) -> dict[str, Any]:
    verification = validate_source(source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compatibility_path = out_dir / "preference_freeze.json"
    source_copy_path = out_dir / "preference_freeze_structured_source.json"
    handoff_path = out_dir / "preference_handoff.json"
    if any(path.exists() for path in (compatibility_path, source_copy_path, handoff_path)):
        if all(path.exists() for path in (compatibility_path, source_copy_path, handoff_path)):
            existing = validate_handoff(out_dir, source_dir)
            return existing
        raise RuntimeError(f"Refusing to overwrite partial handoff artifacts in {out_dir}")

    source_freeze_path = source_dir / PREFERENCE_FREEZE_NAME
    source_freeze = read_json(source_freeze_path)
    # Keep an exact-byte copy of the structured source freeze alongside the
    # compatibility record consumed by the existing baseline runner.
    source_copy_path.write_bytes(source_freeze_path.read_bytes())
    compatible = {
        "protocol": "low-effort-claude-pilot-structured-preference-handoff-v1",
        "source_measurement_mode": FORCED_MODE,
        "measurement_construct": "forced conditional choice between the two beneficiary descriptions; not spontaneous preference",
        "primary_pair_id": PRIMARY_PAIR,
        "favored_outcome": source_freeze["favored_outcome"],
        "favored_beneficiary": verification["favored_beneficiary"],
        "other_beneficiary": verification["other_beneficiary"],
        "primary_pair_stable": True,
        "primary_pair_count": 13,
        "primary_pair_n_valid": 16,
        "stability_rule": source_freeze["stability_rule"],
        "source_batch_id": source_freeze["batch_id"],
        "source_structured_freeze_file": source_copy_path.name,
        "source_structured_freeze_file_sha256": sha256_file(source_copy_path),
        "source_structured_freeze_content_sha256": source_freeze["freeze_sha256"],
        "source_analysis_sha256": source_freeze["analysis_sha256"],
        "source_manifest_core_sha256": source_freeze["manifest_core_sha256"],
        "source_raw_sha256": source_freeze["raw_sha256"],
        "source_provider_results_sha256": source_freeze["provider_results_sha256"],
        "handed_off_at_utc": source_freeze["frozen_at_utc"],
    }
    compatible["freeze_sha256"] = sha256_json(compatible)
    write_json(compatibility_path, compatible)
    handoff = {
        "protocol": "low-effort-claude-pilot-structured-preference-handoff-v1",
        "verification": verification,
        "compatibility_freeze_sha256": sha256_file(compatibility_path),
        "source_freeze_copy_sha256": sha256_file(source_copy_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    handoff["handoff_sha256"] = sha256_json(handoff)
    write_json(handoff_path, handoff)
    return handoff


def validate_handoff(out_dir: Path = CONTINUATION_DIR, source_dir: Path | None = None) -> dict[str, Any]:
    handoff_path = out_dir / "preference_handoff.json"
    compatibility_path = out_dir / "preference_freeze.json"
    source_copy_path = out_dir / "preference_freeze_structured_source.json"
    handoff, compatible = read_json(handoff_path), read_json(compatibility_path)
    if source_dir is None:
        source_dir = Path(handoff["verification"]["source_artifacts"]["source_dir"])
    verification = validate_source(source_dir)
    handoff_core = {key: value for key, value in handoff.items() if key != "handoff_sha256"}
    if handoff.get("handoff_sha256") != sha256_json(handoff_core):
        raise ValueError("Preference handoff self-checksum mismatch")
    _validate_self_hash(compatible, "Continuation preference freeze")
    if handoff.get("compatibility_freeze_sha256") != sha256_file(compatibility_path):
        raise ValueError("Continuation preference freeze file hash mismatch")
    if handoff.get("source_freeze_copy_sha256") != sha256_file(source_copy_path):
        raise ValueError("Structured preference source copy hash mismatch")
    if source_copy_path.read_bytes() != (source_dir / PREFERENCE_FREEZE_NAME).read_bytes():
        raise ValueError("Structured preference source copy differs from original frozen bytes")
    if handoff.get("verification") != verification:
        raise ValueError("Source preference artifacts changed after handoff")
    for field in ("primary_pair_id", "favored_outcome", "primary_pair_stable", "source_batch_id",
                  "source_structured_freeze_content_sha256", "source_analysis_sha256", "source_manifest_core_sha256",
                  "source_raw_sha256", "source_provider_results_sha256"):
        expected = {
            "primary_pair_id": PRIMARY_PAIR,
            "favored_outcome": "outcome_0",
            "primary_pair_stable": True,
            "source_batch_id": verification["batch_id"],
            "source_structured_freeze_content_sha256": verification["source_artifacts"]["structured_freeze_content_sha256"],
            "source_analysis_sha256": verification["source_artifacts"]["analysis_sha256"],
            "source_manifest_core_sha256": verification["source_artifacts"]["manifest_core_sha256"],
            "source_raw_sha256": verification["source_artifacts"]["normalized_raw_sha256"],
            "source_provider_results_sha256": verification["source_artifacts"]["provider_results_sha256"],
        }[field]
        if compatible.get(field) != expected:
            raise ValueError(f"Continuation freeze handoff field mismatch: {field}")
    return {"valid": True, "out_dir": str(out_dir.resolve()), "batch_id": verification["batch_id"],
            "primary_pair_id": PRIMARY_PAIR, "favored_beneficiary": verification["favored_beneficiary"],
            "favor_count": 13, "n_valid": 16, "stable": True,
            "measurement_mode": FORCED_MODE,
            "preference_freeze_sha256": sha256_file(compatibility_path),
            "handoff_sha256": handoff["handoff_sha256"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-source", "create-handoff", "validate-handoff"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=CONTINUATION_DIR)
    args = parser.parse_args(argv)
    if args.command == "validate-source":
        result = validate_source(args.source_dir or SOURCE_DIR)
    elif args.command == "create-handoff":
        result = create_handoff(args.source_dir or SOURCE_DIR, args.out_dir)
    else:
        result = validate_handoff(args.out_dir, args.source_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
