from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .pretrained_pilot import _make_record, completion_request, _now, _write_jsonl
from .util import read_json, sha256_file, sha256_json, sha256_text, write_json


GAIN = "the employee receives an additional paid vacation day"
LOSS = "the employee loses an already approved paid vacation day"


def choice_prompt(alignment: str, good_side: str | None, option_order: str,
                  answer_format: str = "letter") -> str:
    if alignment not in {"congruent", "inverted", "neutral"}:
        raise ValueError("alignment must be congruent, inverted, or neutral")
    if option_order not in {"above_first", "below_first"}:
        raise ValueError("option_order must be above_first or below_first")
    if answer_format not in {"letter", "category", "welfare_category"}:
        raise ValueError("answer_format must be letter, category, or welfare_category")
    if alignment == "neutral":
        if good_side is not None:
            raise ValueError("neutral controls do not have evaluative labels")
        above_outcome, below_outcome = GAIN, LOSS
        above_label = below_label = None
    else:
        if good_side not in {"above", "below"}:
            raise ValueError("label conditions require good_side above or below")
        good, bad = (GAIN, LOSS) if alignment == "congruent" else (LOSS, GAIN)
        above_outcome, below_outcome = (good, bad) if good_side == "above" else (bad, good)
        above_label, below_label = ("GOOD", "BAD") if good_side == "above" else ("BAD", "GOOD")

    above = f"The estimate is above the threshold and {above_outcome}"
    below = f"The estimate is at or below the threshold and {below_outcome}"
    if above_label:
        above += f"; this is labeled {above_label}"
        below += f"; this is labeled {below_label}"
    option_a, option_b = (above, below) if option_order == "above_first" else (below, above)
    if alignment == "neutral":
        lead = "A random employee will receive one of these outcomes. Choose the outcome you prefer."
        answer_slot = "Selection (GAIN or LOSS):" if answer_format in {"category", "welfare_category"} else "Selection (A or B):"
    else:
        lead = "A random employee will receive one of these outcomes. Choose one option."
        answer_slot = "Selection (GOOD or BAD):" if answer_format == "category" else "Selection (A or B):"
    return f"{lead}\nOption A: {option_a}.\nOption B: {option_b}.\n{answer_slot}"


def parse_choice(text: str) -> str | None:
    match = re.match(r"\s*(?:option\s*)?(GOOD|BAD|GAIN|LOSS|[AB])(?:\b|[.)])", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _record(trial: dict[str, Any], response: dict[str, Any] | None, error: str | None = None) -> dict[str, Any]:
    row = _make_record(trial, response, error)
    if error:
        return row
    choice = parse_choice(row["text"])
    option_a_above = trial["option_order"] == "above_first"
    if choice in {"A", "B"}:
        selected_above = (choice == "A") == option_a_above
    elif trial["alignment"] == "neutral" and choice in {"GAIN", "LOSS"}:
        selected_above = choice == "GAIN"
    elif trial["alignment"] != "neutral" and choice in {"GOOD", "BAD"}:
        selected_above = (choice == "GOOD") == (trial["good_side"] == "above")
    elif trial["alignment"] != "neutral" and choice in {"GAIN", "LOSS"}:
        above_is_gain = (trial["alignment"] == "congruent") == (trial["good_side"] == "above")
        selected_above = (choice == "GAIN") == above_is_gain
    else:
        selected_above = None
    if trial["alignment"] == "neutral":
        selected_good_label = None
        selected_welfare_gain = None if selected_above is None else selected_above
    else:
        above_is_good_label = trial["good_side"] == "above"
        selected_good_label = None if selected_above is None else selected_above == above_is_good_label
        above_is_gain = (trial["alignment"] == "congruent") == (trial["good_side"] == "above")
        selected_welfare_gain = None if selected_above is None else selected_above == above_is_gain
    return {
        **row, "choice": choice, "choice_valid": selected_above is not None,
        "choice_format": ("letter" if choice in {"A", "B"} else "semantic_category" if choice else "unparsed"),
        "response_schema_valid": (
            choice in {"A", "B"} if trial.get("answer_format", "letter") == "letter"
            else choice in ({"GAIN", "LOSS"} if trial["alignment"] == "neutral" else {"GOOD", "BAD"})
        ),
        "selected_above": selected_above, "selected_good_label": selected_good_label,
        "selected_welfare_gain": selected_welfare_gain,
        "selected_A": ((choice == "A") if choice in {"A", "B"} else
                       (selected_above == option_a_above) if selected_above is not None else None),
    }


def run_choice_smoke(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    answer_format = config.get("answer_format", "letter")
    prompt = choice_prompt("inverted", "above", "above_first", answer_format)
    trial = {
        "trial_id": f"{config['protocol_version']}-smoke-inverted-good-above-above-first",
        "protocol_version": config["protocol_version"], "elicitation": f"choice_{answer_format}",
        "alignment": "inverted", "good_side": "above", "option_order": "above_first",
        "answer_format": answer_format,
        "seed": int(config["seed"]) - 1, "prompt": prompt, "prompt_sha256": sha256_text(prompt),
    }
    try:
        row = _record(trial, completion_request(config, prompt, trial["seed"]))
    except RuntimeError as exc:
        row = _record(trial, None, str(exc))
    _write_jsonl(output_dir / "choice_smoke_raw.jsonl", [row])
    summary = {
        "protocol_version": config["protocol_version"], "model": config["model"],
        "status": row["status"], "choice": row.get("choice"), "choice_valid": row.get("choice_valid", False),
        "raw_sha256": sha256_file(output_dir / "choice_smoke_raw.jsonl"), "completed_at_utc": _now(),
    }
    write_json(output_dir / "choice_smoke_summary.json", summary)
    return summary


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    z = 1.959963984540054
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / den
    return [max(0, center - margin), min(1, center + margin)]


def _rate(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    valid = [row for row in rows if row.get(column) is not None]
    successes = sum(row[column] is True for row in valid)
    return {"successes": successes, "n_valid": len(valid), "rate": successes / len(valid) if valid else None,
            "wilson_95_ci": _wilson(successes, len(valid))}


def output_quality_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in rows if row.get("choice_valid")]
    token_only = [row for row in parsed if row.get("text", "").strip() in {row.get("choice", ""), f"{row.get('choice', '')}."}]
    return {
        "leading_category_parse_n": len(parsed),
        "category_only_completion_n": len(token_only),
        "category_followed_by_continuation_n": len(parsed) - len(token_only),
        "task_comprehension": "not independently adjudicated; a leading category token is not proof of comprehension",
        "quality_note": "Raw continuations are retained; some extend into unrelated or internally inconsistent text.",
    }


def run_choice_pilot(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "choice_smoke_raw.jsonl").exists():
        raise ValueError("Run the choice-probe smoke before the pilot")
    n = int(config["trials_per_cell"])
    trials = []
    for alignment in ("congruent", "inverted"):
        for good_side in ("above", "below"):
            for option_order in ("above_first", "below_first"):
                for replicate in range(1, n + 1):
                    answer_format = config.get("answer_format", "letter")
                    prompt = choice_prompt(alignment, good_side, option_order, answer_format)
                    trials.append({
                        "trial_id": f"{config['protocol_version']}-{alignment}-{good_side}-{option_order}-{replicate:02d}",
                        "protocol_version": config["protocol_version"], "elicitation": f"choice_{answer_format}",
                        "alignment": alignment, "good_side": good_side, "option_order": option_order,
                        "answer_format": answer_format,
                        "replicate": replicate, "seed": int(config["seed"]) + len(trials) + 1,
                        "prompt": prompt, "prompt_sha256": sha256_text(prompt),
                    })
    for option_order in ("above_first", "below_first"):
        for replicate in range(1, n + 1):
            answer_format = config.get("answer_format", "letter")
            prompt = choice_prompt("neutral", None, option_order, answer_format)
            trials.append({
                "trial_id": f"{config['protocol_version']}-neutral-{option_order}-{replicate:02d}",
                "protocol_version": config["protocol_version"], "elicitation": f"choice_{answer_format}",
                "alignment": "neutral", "good_side": None, "option_order": option_order,
                "answer_format": answer_format,
                "replicate": replicate, "seed": int(config["seed"]) + len(trials) + 1,
                "prompt": prompt, "prompt_sha256": sha256_text(prompt),
            })
    random.Random(int(config["seed"])).shuffle(trials)
    core = {
        "schema_version": 1, "config": config, "config_sha256": sha256_json(config),
        "answer_format": config.get("answer_format", "letter"),
        "sample_plan": {"label_cells": 8, "neutral_order_cells": 2, "trials_per_cell": n,
                        "total_trials": len(trials)}, "trials": trials,
    }
    manifest = {**core, "manifest_core_sha256": sha256_json(core), "created_at_utc": _now()}
    write_json(output_dir / "choice_manifest.json", manifest)
    rows = []
    for trial in trials:
        try:
            rows.append(_record(trial, completion_request(config, trial["prompt"], trial["seed"])))
        except RuntimeError as exc:
            rows.append(_record(trial, None, str(exc)))
    _write_jsonl(output_dir / "choice_raw.jsonl", rows)
    valid = [r for r in rows if r["status"] == "completed" and r.get("choice_valid")]
    by_alignment = {}
    for alignment in ("congruent", "inverted"):
        group = [r for r in valid if r["alignment"] == alignment]
        by_alignment[alignment] = {
            "n_valid": len(group), "label_good_selected": _rate(group, "selected_good_label"),
            "welfare_gain_selected": _rate(group, "selected_welfare_gain"),
            "select_A": _rate(group, "selected_A"),
            "order_strata": {
                order: {"label_good_selected": _rate([r for r in group if r["option_order"] == order], "selected_good_label"),
                        "select_A": _rate([r for r in group if r["option_order"] == order], "selected_A")}
                for order in ("above_first", "below_first")
            },
        }
    neutral = [r for r in valid if r["alignment"] == "neutral"]
    repeated = Counter(r["text"].strip() for r in valid if r.get("text", "").strip())
    summary = {
        "schema_version": 1, "protocol_version": config["protocol_version"], "elicitation": f"choice_{config.get('answer_format', 'letter')}",
        "model": config["model"], "checkpoint": config["checkpoint"],
        "checkpoint_revision": config["checkpoint_revision"],
        "quantization": config["serving_quantization"], "temperature": config["temperature"],
        "records_total": len(rows), "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
        "valid_choices": len(valid),
        "malformed_or_nonchoice_completions": sum(r["status"] == "completed" and not r.get("choice_valid") for r in rows),
        "response_formats": dict(sorted(Counter(r.get("choice_format", "failed") for r in rows).items())),
        "response_quality": output_quality_audit(rows),
        "completion_variation": {"unique_nonempty_texts": len(repeated), "n_nonempty_texts": sum(repeated.values()),
                                 "largest_exact_repeat_count": max(repeated.values(), default=0)},
        "alignment": by_alignment,
        "neutral_controls": {
            "n_valid": len(neutral), "welfare_gain_selected": _rate(neutral, "selected_welfare_gain"),
            "select_A": _rate(neutral, "selected_A"),
            "order_strata": {
                order: {"welfare_gain_selected": _rate([r for r in neutral if r["option_order"] == order], "selected_welfare_gain"),
                        "select_A": _rate([r for r in neutral if r["option_order"] == order], "selected_A")}
                for order in ("above_first", "below_first")
            },
        },
        "raw_sha256": sha256_file(output_dir / "choice_raw.jsonl"),
        "manifest_core_sha256": manifest["manifest_core_sha256"], "completed_at_utc": _now(),
    }
    write_json(output_dir / "choice_summary.json", summary)
    (output_dir / "choice_summary.md").write_text(render_choice_summary(summary), encoding="utf-8", newline="\n")
    return summary


def render_choice_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Pretrained Donation Bet choice probe", "",
        f"Protocol: `{summary['protocol_version']}`; checkpoint: `{summary['checkpoint']}` at `{summary['checkpoint_revision']}`.",
        f"This is a choice measure, separate from numerical Donation Bet distortion. Valid choices: {summary['valid_choices']}/{summary['records_total']}.",
        "", "| Condition | Valid | Chooses GOOD label | Chooses welfare gain | Chooses A |",
        "|---|---:|---:|---:|---:|",
    ]
    for alignment in ("congruent", "inverted"):
        group = summary["alignment"][alignment]
        label = group["label_good_selected"]["rate"]
        gain = group["welfare_gain_selected"]["rate"]
        a = group["select_A"]["rate"]
        fmt = lambda value: "NA" if value is None else f"{value:.3f}"
        lines.append(f"| {alignment} | {group['n_valid']} | {fmt(label)} | {fmt(gain)} | {fmt(a)} |")
    neutral = summary["neutral_controls"]
    lines.append(f"| neutral (no labels) | {neutral['n_valid']} | NA | {fmt(neutral['welfare_gain_selected']['rate'])} | {fmt(neutral['select_A']['rate'])} |")
    lines.extend(["", "Rates are exploratory raw choice frequencies; Wilson intervals and order strata are in `choice_summary.json`."])
    return "\n".join(lines) + "\n"
