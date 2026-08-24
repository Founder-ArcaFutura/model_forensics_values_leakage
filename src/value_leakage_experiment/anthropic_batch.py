from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .anthropic_api import (
    AnthropicSettings,
    anthropic_request_json,
    anthropic_request_text,
    anthropic_settings_from_config,
    message_params,
)
from .util import canonical_json, read_json, read_jsonl, sha256_file, write_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _message_text(message: dict[str, Any]) -> str:
    return "\n".join(
        block["text"]
        for block in message.get("content", [])
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def _thinking_summary(message: dict[str, Any]) -> str | None:
    parts = [
        block["thinking"]
        for block in message.get("content", [])
        if block.get("type") == "thinking" and isinstance(block.get("thinking"), str)
    ]
    return "\n".join(parts) if parts else None


def _turn_receipt(message: dict[str, Any]) -> dict[str, Any]:
    incomplete = message.get("stop_reason") == "max_tokens"
    return {
        "text": _message_text(message),
        "thinking_summary": _thinking_summary(message),
        "content": message.get("content"),
        "response_id": message.get("id"),
        "returned_model": message.get("model"),
        "status": "incomplete" if incomplete else "completed",
        "stop_reason": message.get("stop_reason"),
        "stop_sequence": message.get("stop_sequence"),
        "usage": message.get("usage"),
        "raw_response": message,
    }


def _batch_request(settings: AnthropicSettings, trial: dict[str, Any], phase: int, turn1_message: dict[str, Any] | None = None) -> dict[str, Any]:
    if phase == 1:
        messages = [{"role": "user", "content": trial["turn1_prompt"]}]
    elif phase == 2 and turn1_message is not None:
        messages = [
            {"role": "user", "content": trial["turn1_prompt"]},
            # Round-trip all signed thinking/text blocks unchanged, as Anthropic requires.
            {"role": "assistant", "content": turn1_message["content"]},
            {"role": "user", "content": trial["turn2_prompt"]},
        ]
    else:
        raise ValueError("phase 2 requires the complete turn-1 message")
    return {"custom_id": trial["trial_id"], "params": message_params(settings, messages)}


def _parse_batch_results(text: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = row.get("custom_id")
        if not custom_id or custom_id in rows:
            raise ValueError(f"Missing or duplicate custom_id in batch results line {line_number}")
        rows[custom_id] = row
    return rows


def _succeeded_message(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row or row.get("result", {}).get("type") != "succeeded":
        return None
    return row["result"].get("message")


def _base_record(trial: dict[str, Any], manifest_hash: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "trial_id": trial["trial_id"],
        "order": trial["order"],
        "arm_id": trial["arm_id"],
        "recipient_id": trial.get("recipient_id"),
        "recipient_name": trial.get("recipient_name"),
        "recipient_valence": trial["recipient_valence"],
        "direction": trial["direction"],
        "replicate": trial["replicate"],
        "turn1_prompt": trial["turn1_prompt"],
        "turn2_prompt": trial["turn2_prompt"],
        "turn1_prompt_sha256": trial["turn1_prompt_sha256"],
        "turn2_prompt_sha256": trial["turn2_prompt_sha256"],
        "manifest_core_sha256": manifest_hash,
    }


def _assemble_records(
    manifest: dict[str, Any],
    turn1_rows: dict[str, dict[str, Any]],
    turn2_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trial in manifest["trials"]:
        trial_id = trial["trial_id"]
        base = _base_record(trial, manifest["manifest_core_sha256"])
        turn1_row = turn1_rows.get(trial_id)
        turn1_message = _succeeded_message(turn1_row)
        if turn1_message is None:
            records.append({
                **base,
                "status": "failed",
                "incomplete_turn": 1,
                "batch_result": turn1_row,
                "completed_at_utc": _now(),
            })
            continue
        turn1 = _turn_receipt(turn1_message)
        if turn1["status"] == "incomplete":
            records.append({
                **base,
                "status": "incomplete",
                "incomplete_turn": 1,
                "turn1": turn1,
                "completed_at_utc": _now(),
            })
            continue
        turn2_row = turn2_rows.get(trial_id)
        turn2_message = _succeeded_message(turn2_row)
        if turn2_message is None:
            records.append({
                **base,
                "status": "failed",
                "incomplete_turn": 2,
                "turn1": turn1,
                "batch_result": turn2_row,
                "completed_at_utc": _now(),
            })
            continue
        turn2 = _turn_receipt(turn2_message)
        records.append({
            **base,
            "status": "incomplete" if turn2["status"] == "incomplete" else "completed",
            **({"incomplete_turn": 2} if turn2["status"] == "incomplete" else {}),
            "turn1": turn1,
            "turn2": turn2,
            "completed_at_utc": _now(),
        })
    return records


def _write_raw(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def _usage_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for record in records:
        for turn_name in ("turn1", "turn2"):
            for key, value in (record.get(turn_name, {}).get("usage") or {}).items():
                if isinstance(value, int):
                    totals[key] += value
    return dict(sorted(totals.items()))


def _finalize(manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    turn1_text = (output_dir / "turn1_results.jsonl").read_text(encoding="utf-8")
    turn2_path = output_dir / "turn2_results.jsonl"
    turn2_text = turn2_path.read_text(encoding="utf-8") if turn2_path.exists() else ""
    records = _assemble_records(manifest, _parse_batch_results(turn1_text), _parse_batch_results(turn2_text))
    raw_path = output_dir / "raw.jsonl"
    _write_raw(raw_path, records)
    usage = _usage_totals(records)
    config = manifest["config"]
    estimated_cost = None
    if "batch_input_usd_per_million" in config and "batch_output_usd_per_million" in config:
        estimated_cost = (
            usage.get("input_tokens", 0) * float(config["batch_input_usd_per_million"])
            + usage.get("output_tokens", 0) * float(config["batch_output_usd_per_million"])
        ) / 1_000_000
    receipt = {
        "schema_version": 1,
        "provider": "anthropic",
        "requested_model": config["model"],
        "manifest_core_sha256": manifest["manifest_core_sha256"],
        "turn1_batch": read_json(output_dir / "turn1_batch.json"),
        "turn2_batch": read_json(output_dir / "turn2_batch.json") if (output_dir / "turn2_batch.json").exists() else None,
        "records_total": len(records),
        "status_counts": dict(sorted(Counter(record["status"] for record in records).items())),
        "usage": usage,
        "estimated_batch_cost_usd": estimated_cost,
        "raw_jsonl_sha256": sha256_file(raw_path),
        "completed_at_utc": _now(),
    }
    write_json(output_dir / "run_receipt.json", receipt)
    state = read_json(output_dir / "batch_state.json")
    write_json(output_dir / "batch_state.json", {**state, "phase": "completed", "completed_at_utc": _now()})
    return {"action": "completed", **receipt}


def run_anthropic_smoke(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    settings = anthropic_settings_from_config(manifest["config"], Path.cwd())
    trial = manifest["trials"][0]
    output_dir.mkdir(parents=True, exist_ok=True)
    turn1_message = anthropic_request_json(
        settings,
        "POST",
        "/v1/messages",
        message_params(settings, [{"role": "user", "content": trial["turn1_prompt"]}]),
    )
    turn1 = _turn_receipt(turn1_message)
    if turn1["status"] != "completed":
        record = {**_base_record(trial, manifest["manifest_core_sha256"]), "status": "incomplete", "incomplete_turn": 1, "turn1": turn1}
    else:
        turn2_message = anthropic_request_json(
            settings,
            "POST",
            "/v1/messages",
            message_params(
                settings,
                [
                    {"role": "user", "content": trial["turn1_prompt"]},
                    {"role": "assistant", "content": turn1_message["content"]},
                    {"role": "user", "content": trial["turn2_prompt"]},
                ],
            ),
        )
        turn2 = _turn_receipt(turn2_message)
        record = {
            **_base_record(trial, manifest["manifest_core_sha256"]),
            "status": "completed" if turn2["status"] == "completed" else "incomplete",
            **({"incomplete_turn": 2} if turn2["status"] != "completed" else {}),
            "turn1": turn1,
            "turn2": turn2,
            "completed_at_utc": _now(),
        }
    _write_raw(output_dir / "raw.jsonl", [record])
    write_json(output_dir / "smoke_receipt.json", {
        "provider": "anthropic",
        "requested_model": settings.model,
        "trial_id": trial["trial_id"],
        "status": record["status"],
        "thinking_blocks_round_tripped": any(block.get("type") == "thinking" for block in turn1_message.get("content", [])),
        "completed_at_utc": _now(),
    })
    return record


def step_anthropic_batch(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    settings = anthropic_settings_from_config(manifest["config"], Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "batch_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "schema_version": 1,
        "phase": "new",
        "manifest_core_sha256": manifest["manifest_core_sha256"],
        "requested_model": settings.model,
        "created_at_utc": _now(),
    }
    if state["manifest_core_sha256"] != manifest["manifest_core_sha256"]:
        raise ValueError("Existing batch state belongs to a different frozen manifest")

    if state["phase"] == "new":
        requests = [_batch_request(settings, trial, 1) for trial in manifest["trials"]]
        response = anthropic_request_json(settings, "POST", "/v1/messages/batches", {"requests": requests})
        write_json(output_dir / "turn1_batch.json", response)
        write_json(state_path, {**state, "phase": "turn1_submitted", "turn1_batch_id": response["id"], "turn1_submitted_at_utc": _now()})
        return {"action": "turn1_submitted", "batch_id": response["id"], "request_count": len(requests), "processing_status": response.get("processing_status")}

    if state["phase"] == "turn1_submitted":
        response = anthropic_request_json(settings, "GET", f"/v1/messages/batches/{state['turn1_batch_id']}")
        write_json(output_dir / "turn1_batch.json", response)
        if response.get("processing_status") != "ended":
            return {"action": "turn1_polled", "batch_id": response["id"], "processing_status": response.get("processing_status"), "request_counts": response.get("request_counts")}
        results_text = anthropic_request_text(settings, "GET", f"/v1/messages/batches/{state['turn1_batch_id']}/results")
        (output_dir / "turn1_results.jsonl").write_text(results_text, encoding="utf-8", newline="\n")
        turn1_rows = _parse_batch_results(results_text)
        requests = []
        for trial in manifest["trials"]:
            message = _succeeded_message(turn1_rows.get(trial["trial_id"]))
            if message is not None and message.get("stop_reason") != "max_tokens" and _message_text(message):
                requests.append(_batch_request(settings, trial, 2, message))
        write_json(output_dir / "turn2_request_receipt.json", {
            "request_count": len(requests),
            "custom_ids": [request["custom_id"] for request in requests],
            "prepared_at_utc": _now(),
        })
        # The payload is prompt-only plus API responses already preserved in turn1_results.jsonl.
        write_json(output_dir / "turn2_requests.json", {"requests": requests})
        write_json(state_path, {**state, "phase": "turn2_ready", "turn1_ended_at_utc": _now(), "turn2_request_count": len(requests)})
        return {"action": "turn1_retrieved_turn2_prepared", "turn2_request_count": len(requests)}

    if state["phase"] == "turn2_ready":
        payload = read_json(output_dir / "turn2_requests.json")
        if not payload["requests"]:
            return _finalize(manifest, output_dir)
        response = anthropic_request_json(settings, "POST", "/v1/messages/batches", payload)
        write_json(output_dir / "turn2_batch.json", response)
        write_json(state_path, {**state, "phase": "turn2_submitted", "turn2_batch_id": response["id"], "turn2_submitted_at_utc": _now()})
        return {"action": "turn2_submitted", "batch_id": response["id"], "request_count": len(payload["requests"]), "processing_status": response.get("processing_status")}

    if state["phase"] == "turn2_submitted":
        response = anthropic_request_json(settings, "GET", f"/v1/messages/batches/{state['turn2_batch_id']}")
        write_json(output_dir / "turn2_batch.json", response)
        if response.get("processing_status") != "ended":
            return {"action": "turn2_polled", "batch_id": response["id"], "processing_status": response.get("processing_status"), "request_counts": response.get("request_counts")}
        results_text = anthropic_request_text(settings, "GET", f"/v1/messages/batches/{state['turn2_batch_id']}/results")
        (output_dir / "turn2_results.jsonl").write_text(results_text, encoding="utf-8", newline="\n")
        return _finalize(manifest, output_dir)

    if state["phase"] == "completed":
        receipt = read_json(output_dir / "run_receipt.json")
        return {"action": "already_completed", **receipt}
    raise ValueError(f"Unknown batch state phase: {state['phase']}")
