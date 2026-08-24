from __future__ import annotations

import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .api import APISettings, create_response, settings_from_config
from .util import append_jsonl, read_json, read_jsonl, sha256_file, write_json


def _turn_receipt(text: str, raw: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    return {
        "text": text,
        "response_id": raw.get("id"),
        "returned_model": raw.get("model"),
        "status": raw.get("status"),
        "service_tier": raw.get("service_tier"),
        "usage": raw.get("usage"),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "raw_response": raw,
    }


def _run_trial(trial: dict[str, Any], settings: APISettings, manifest_hash: str) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    base = {
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
        "started_at_utc": started.isoformat(),
    }
    try:
        t0 = time.monotonic()
        turn1_text, turn1_raw = create_response(settings, trial["turn1_prompt"])
        turn1_receipt = _turn_receipt(turn1_text, turn1_raw, time.monotonic() - t0)
        if turn1_raw.get("status") != "completed":
            return {
                **base,
                "status": "incomplete",
                "incomplete_turn": 1,
                "turn1": turn1_receipt,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }

        second_input = [
            {"role": "user", "content": trial["turn1_prompt"]},
            {"role": "assistant", "content": turn1_text},
            {"role": "user", "content": trial["turn2_prompt"]},
        ]
        t1 = time.monotonic()
        turn2_text, turn2_raw = create_response(settings, second_input)
        turn2_receipt = _turn_receipt(turn2_text, turn2_raw, time.monotonic() - t1)
        if turn2_raw.get("status") != "completed":
            return {
                **base,
                "status": "incomplete",
                "incomplete_turn": 2,
                "turn1": turn1_receipt,
                "turn2": turn2_receipt,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        return {
            **base,
            "status": "completed",
            "turn1": turn1_receipt,
            "turn2": turn2_receipt,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:  # result records must preserve failed trials for denominators
        return {
            **base,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }


def run_manifest(
    manifest_path: Path,
    output_dir: Path,
    concurrency_override: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    config = manifest["config"]
    settings = settings_from_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.jsonl"
    completed_ids = {row["trial_id"] for row in read_jsonl(raw_path)}
    pending = [trial for trial in manifest["trials"] if trial["trial_id"] not in completed_ids]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        pending = pending[:limit]
    concurrency = concurrency_override or int(config.get("concurrency", 1))

    run_receipt = {
        "schema_version": 1,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_core_sha256": manifest["manifest_core_sha256"],
        "config_sha256": manifest["config_sha256"],
        "requested_model": settings.model,
        "reasoning_effort": settings.reasoning_effort,
        "temperature": settings.temperature,
        "max_output_tokens": settings.max_output_tokens,
        "base_url_origin": settings.base_url,
        "http_transport": settings.http_transport,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "concurrency": concurrency,
        "already_present": len(completed_ids),
        "pending_at_start": len(pending),
        "limit": limit,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # The receipt itself must never include the credential.
    write_json(output_dir / "run_receipt.json", run_receipt)

    lock = threading.Lock()
    newly_written = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_run_trial, trial, settings, manifest["manifest_core_sha256"]): trial
            for trial in pending
        }
        for future in as_completed(futures):
            record = future.result()
            with lock:
                append_jsonl(raw_path, [record])
                newly_written += 1
            print(f"[{newly_written}/{len(pending)}] {record['trial_id']} {record['arm_id']} {record['status']}", flush=True)

    all_rows = read_jsonl(raw_path)
    final_receipt = {
        **run_receipt,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "newly_written": newly_written,
        "records_total": len(all_rows),
        "status_counts": {
            status: sum(1 for row in all_rows if row.get("status") == status)
            for status in sorted({row.get("status", "unknown") for row in all_rows})
        },
        "raw_jsonl_sha256": sha256_file(raw_path),
    }
    write_json(output_dir / "run_receipt.json", final_receipt)
    return final_receipt
