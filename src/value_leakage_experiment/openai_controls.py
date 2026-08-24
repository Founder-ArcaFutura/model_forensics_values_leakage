from __future__ import annotations

import random
import statistics
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analysis import _fisher_two_sided, _newcombe_difference_interval, _wilson_interval
from .api import settings_from_config
from .donation_positive_control import donation_prompt
from .openai_batch import output_text, parse_output, request_json, request_text, submit_batch, write_batch_input
from .parsing import extract_final_estimate
from .task_load import _mover_difference_interval
from .util import canonical_json, read_json, sha256_file, sha256_json, sha256_text, write_json
from .valence_factorial import CONDITIONS, _condition_effect, outcome_prompt


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def _record(trial: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    response_wrapper = (row or {}).get("response") or {}
    body = response_wrapper.get("body") or {}
    text = output_text(body)
    estimate, source = extract_final_estimate(text)
    succeeded = response_wrapper.get("status_code") == 200 and body.get("status") == "completed"
    return {
        **trial,
        "status": "completed" if succeeded else "failed",
        "estimate": estimate,
        "estimate_source": source,
        "parse_review_required": estimate is None,
        "text": text,
        "returned_model": body.get("model"),
        "response_id": body.get("id"),
        "response_status": body.get("status"),
        "usage": body.get("usage"),
        "batch_request_id": response_wrapper.get("request_id"),
        "batch_error": (row or {}).get("error"),
        "raw_response": body,
    }


def _usage(records: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for record in records:
        for key, value in (record.get("usage") or {}).items():
            if isinstance(value, int):
                totals[key] += value
    return dict(sorted(totals.items()))


def _cost(config: dict[str, Any], usage: dict[str, int]) -> float:
    return (
        usage.get("input_tokens", 0) * float(config["batch_input_usd_per_million"])
        + usage.get("output_tokens", 0) * float(config["batch_output_usd_per_million"])
    ) / 1_000_000


def _completed_output(settings: Any, batch: dict[str, Any]) -> str:
    if batch.get("status") != "completed" or not batch.get("output_file_id"):
        raise ValueError(f"Batch is not retrievable: {batch.get('status')}")
    return request_text(settings, f"/v1/files/{batch['output_file_id']}/content")


def _prepare_positive_intervention(manifest: dict[str, Any], output_dir: Path, output_text_value: str) -> dict[str, Any]:
    rows = parse_output(output_text_value)
    baseline = [_record(trial, rows.get(trial["custom_id"])) for trial in manifest["baseline_trials"]]
    _write_jsonl(output_dir / "baseline_raw.jsonl", baseline)
    valid = [r["estimate"] for r in baseline if r["status"] == "completed" and r["estimate"] is not None]
    minimum = int(manifest["config"].get("minimum_valid_baseline", 36))
    if len(valid) < minimum:
        raise ValueError(f"Only {len(valid)} valid baseline estimates; require {minimum}")
    threshold = statistics.median(valid)
    if isinstance(threshold, float) and threshold.is_integer():
        threshold = int(threshold)
    n = int(manifest["config"]["trials_per_intervention"])
    trials: list[dict[str, Any]] = []
    for condition, side in (("above_good", "above"), ("below_good", "below")):
        prompt = donation_prompt(threshold, side)
        for replicate in range(1, n + 1):
            trials.append({
                "custom_id": f"{manifest['config']['protocol_version']}-{condition}-{replicate:03d}",
                "condition": condition,
                "good_side": side,
                "replicate": replicate,
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
            })
    random.Random(int(manifest["config"]["intervention_seed"])).shuffle(trials)
    core = {
        "schema_version": 1,
        "parent_manifest_core_sha256": manifest["manifest_core_sha256"],
        "threshold": threshold,
        "threshold_method": "median of valid completed same-model baseline estimates",
        "n_valid_baseline": len(valid),
        "baseline_estimate_counts": dict(sorted(Counter(str(v) for v in valid).items())),
        "trials": trials,
    }
    frozen = {**core, "intervention_core_sha256": sha256_json(core), "frozen_at_utc": _now()}
    write_json(output_dir / "intervention_manifest.json", frozen)
    return frozen


def _analyze_positive(manifest: dict[str, Any], output_dir: Path, output_text_value: str) -> dict[str, Any]:
    frozen = read_json(output_dir / "intervention_manifest.json")
    rows = parse_output(output_text_value)
    records = [_record(trial, rows.get(trial["custom_id"])) for trial in frozen["trials"]]
    _write_jsonl(output_dir / "intervention_raw.jsonl", records)
    valid = [r for r in records if r["status"] == "completed" and r["estimate"] is not None]
    above = [r for r in valid if r["good_side"] == "above"]
    below = [r for r in valid if r["good_side"] == "below"]
    threshold = frozen["threshold"]
    a = sum(r["estimate"] > threshold for r in above)
    b = sum(r["estimate"] > threshold for r in below)
    favored = a + len(below) - b
    total = len(above) + len(below)
    low, high = _wilson_interval(favored, total)
    baseline_records = [json.loads(line) for line in (output_dir / "baseline_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    usage = _usage(baseline_records + records)
    summary = {
        "schema_version": 1,
        "model": manifest["config"]["model"],
        "reasoning_effort": manifest["config"].get("reasoning_effort"),
        "threshold": threshold,
        "baseline": {"n_valid": frozen["n_valid_baseline"], "estimate_counts": frozen["baseline_estimate_counts"]},
        "above_good": {"n_valid": len(above), "favored": a, "favored_rate": a / len(above)},
        "below_good": {"n_valid": len(below), "favored": len(below) - b, "favored_rate": (len(below) - b) / len(below)},
        "donation_bet": {
            "n_valid": total,
            "favored": favored,
            "favored_rate": favored / total,
            "bias": 2 * (favored / total - 0.5),
            "bias_wilson_95_ci": [2 * (low - 0.5), 2 * (high - 0.5)],
            "p_above_given_above_good": a / len(above),
            "p_above_given_below_good": b / len(below),
            "above_probability_difference": a / len(above) - b / len(below),
            "above_probability_difference_newcombe_95_ci": list(_newcombe_difference_interval(a, len(above), b, len(below))),
            "fisher_exact_two_sided_p": _fisher_two_sided(a, len(above) - a, b, len(below) - b),
        },
        "status_counts": dict(sorted(Counter(r["status"] for r in records).items())),
        "usage": usage,
        "estimated_batch_cost_usd": _cost(manifest["config"], usage),
        "baseline_raw_sha256": sha256_file(output_dir / "baseline_raw.jsonl"),
        "intervention_raw_sha256": sha256_file(output_dir / "intervention_raw.jsonl"),
        "completed_at_utc": _now(),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def step_openai_positive(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    settings = settings_from_config(manifest["config"])
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "batch_state.json"
    state = read_json(state_path) if state_path.exists() else {"phase": "new", "manifest_core_sha256": manifest["manifest_core_sha256"], "created_at_utc": _now()}
    if state["manifest_core_sha256"] != manifest["manifest_core_sha256"]:
        raise ValueError("Existing state belongs to a different manifest")
    if state["phase"] == "new":
        path = output_dir / "baseline_batch_input.jsonl"
        write_batch_input(path, [(t["custom_id"], t["prompt"]) for t in manifest["baseline_trials"]], settings)
        receipt = submit_batch(settings, path, f"{manifest['config']['protocol_version']} baseline")
        write_json(output_dir / "baseline_batch.json", receipt)
        write_json(state_path, {**state, "phase": "baseline_submitted", "baseline_batch_id": receipt["batch"]["id"], "submitted_at_utc": _now()})
        return {"action": "baseline_submitted", "request_count": len(manifest["baseline_trials"]), "status": receipt["batch"]["status"]}
    if state["phase"] == "baseline_submitted":
        batch = request_json(settings, "GET", f"/v1/batches/{state['baseline_batch_id']}")
        write_json(output_dir / "baseline_batch_status.json", batch)
        if batch["status"] != "completed":
            if batch["status"] in {"failed", "expired", "cancelled"}:
                raise ValueError(f"Baseline batch terminal status: {batch['status']}")
            return {"action": "baseline_polled", "status": batch["status"], "request_counts": batch.get("request_counts")}
        text = _completed_output(settings, batch)
        (output_dir / "baseline_batch_output.jsonl").write_text(text, encoding="utf-8", newline="\n")
        frozen = _prepare_positive_intervention(manifest, output_dir, text)
        path = output_dir / "intervention_batch_input.jsonl"
        write_batch_input(path, [(t["custom_id"], t["prompt"]) for t in frozen["trials"]], settings)
        receipt = submit_batch(settings, path, f"{manifest['config']['protocol_version']} intervention")
        write_json(output_dir / "intervention_batch.json", receipt)
        write_json(state_path, {**state, "phase": "intervention_submitted", "threshold": frozen["threshold"], "intervention_core_sha256": frozen["intervention_core_sha256"], "intervention_batch_id": receipt["batch"]["id"], "intervention_submitted_at_utc": _now()})
        return {"action": "baseline_complete_intervention_submitted", "threshold": frozen["threshold"], "n_valid_baseline": frozen["n_valid_baseline"], "status": receipt["batch"]["status"]}
    if state["phase"] == "intervention_submitted":
        batch = request_json(settings, "GET", f"/v1/batches/{state['intervention_batch_id']}")
        write_json(output_dir / "intervention_batch_status.json", batch)
        if batch["status"] != "completed":
            if batch["status"] in {"failed", "expired", "cancelled"}:
                raise ValueError(f"Intervention batch terminal status: {batch['status']}")
            return {"action": "intervention_polled", "status": batch["status"], "request_counts": batch.get("request_counts")}
        text = _completed_output(settings, batch)
        (output_dir / "intervention_batch_output.jsonl").write_text(text, encoding="utf-8", newline="\n")
        summary = _analyze_positive(manifest, output_dir, text)
        write_json(state_path, {**state, "phase": "completed", "completed_at_utc": _now()})
        return {"action": "completed", **summary}
    if state["phase"] == "completed":
        return {"action": "already_completed", **read_json(output_dir / "summary.json")}
    raise ValueError(f"Unknown phase: {state['phase']}")


def _analyze_valence(manifest: dict[str, Any], output_dir: Path, output_text_value: str) -> dict[str, Any]:
    rows = parse_output(output_text_value)
    records = [_record(trial, rows.get(trial["custom_id"])) for trial in manifest["trials"]]
    _write_jsonl(output_dir / "raw.jsonl", records)
    valid = [r for r in records if r["status"] == "completed" and r["estimate"] is not None]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        grouped[record["condition"]].append(record)
    effects = {condition: _condition_effect(grouped[condition], manifest["config"]["threshold"]) for condition in CONDITIONS}

    def contrast(first: str, second: str) -> dict[str, Any]:
        e1, e0 = effects[first], effects[second]
        return {
            "first_condition": first,
            "second_condition": second,
            "difference": e1["effect"] - e0["effect"],
            "mover_95_ci": _mover_difference_interval(
                e1["effect"], e1["effect_newcombe_95_ci"],
                e0["effect"], e0["effect_newcombe_95_ci"],
            ),
        }

    usage = _usage(records)
    summary = {
        "schema_version": 1,
        "model": manifest["config"]["model"],
        "reasoning_effort": manifest["config"].get("reasoning_effort"),
        "threshold": manifest["config"]["threshold"],
        "records_total": len(records),
        "valid_estimates": len(valid),
        "status_counts": dict(sorted(Counter(r["status"] for r in records).items())),
        "conditions": effects,
        "registered_contrasts": {
            "label_increment_with_semantics": contrast("semantic_labeled", "semantic_unlabeled"),
            "semantic_inference_without_labels": contrast("semantic_unlabeled", "abstract_unlabeled"),
            "supplied_label_without_semantics": contrast("abstract_labeled", "abstract_unlabeled"),
            "semantic_increment_with_labels": contrast("semantic_labeled", "abstract_labeled"),
        },
        "usage": usage,
        "estimated_batch_cost_usd": _cost(manifest["config"], usage),
        "manifest_core_sha256": manifest["manifest_core_sha256"],
        "raw_sha256": sha256_file(output_dir / "raw.jsonl"),
        "completed_at_utc": _now(),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def step_openai_valence(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    settings = settings_from_config(manifest["config"])
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "batch_state.json"
    state = read_json(state_path) if state_path.exists() else {"phase": "new", "manifest_core_sha256": manifest["manifest_core_sha256"], "created_at_utc": _now()}
    if state["manifest_core_sha256"] != manifest["manifest_core_sha256"]:
        raise ValueError("Existing state belongs to a different manifest")
    if state["phase"] == "new":
        path = output_dir / "batch_input.jsonl"
        write_batch_input(path, [(t["custom_id"], t["prompt"]) for t in manifest["trials"]], settings)
        receipt = submit_batch(settings, path, f"{manifest['config']['protocol_version']} valence factorial")
        write_json(output_dir / "batch.json", receipt)
        write_json(state_path, {**state, "phase": "submitted", "batch_id": receipt["batch"]["id"], "submitted_at_utc": _now()})
        return {"action": "submitted", "request_count": len(manifest["trials"]), "status": receipt["batch"]["status"]}
    if state["phase"] == "submitted":
        batch = request_json(settings, "GET", f"/v1/batches/{state['batch_id']}")
        write_json(output_dir / "batch_status.json", batch)
        if batch["status"] != "completed":
            if batch["status"] in {"failed", "expired", "cancelled"}:
                raise ValueError(f"Valence batch terminal status: {batch['status']}")
            return {"action": "polled", "status": batch["status"], "request_counts": batch.get("request_counts")}
        text = _completed_output(settings, batch)
        (output_dir / "batch_output.jsonl").write_text(text, encoding="utf-8", newline="\n")
        summary = _analyze_valence(manifest, output_dir, text)
        write_json(state_path, {**state, "phase": "completed", "completed_at_utc": _now()})
        return {"action": "completed", **summary}
    if state["phase"] == "completed":
        return {"action": "already_completed", **read_json(output_dir / "summary.json")}
    raise ValueError(f"Unknown phase: {state['phase']}")
