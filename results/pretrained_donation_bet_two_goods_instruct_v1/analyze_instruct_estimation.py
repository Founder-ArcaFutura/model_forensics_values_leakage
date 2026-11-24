"""Offline summaries and exact trial manifest for the Qwen2.5-14B-Instruct follow-through."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_json(path: str) -> dict:
    return json.loads((HERE / path).read_text(encoding="utf-8"))


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: str) -> str:
    return hashlib.sha256((HERE / path).read_bytes()).hexdigest()


def estimate_group(rows: list[dict], threshold: float) -> dict:
    strict = [r for r in rows if r["validity"] == "valid_single_number"]
    tolerant = strict + [r for r in rows if r["validity"] == "parsed_single_numeric_noncompliant"]
    values = [float(r["estimate"]) for r in strict]
    tolerant_values = values + [float(r["salvaged_estimate"]) for r in rows if r["validity"] == "parsed_single_numeric_noncompliant"]
    return {"n": len(rows), "strict_valid_n": len(strict), "unique_numeric_noncompliant_n": len(rows) - len(strict) - sum(r["validity"] == "unparseable_completion" for r in rows),
            "unparseable_n": sum(r["validity"] == "unparseable_completion" for r in rows),
            "above_n": sum(v > threshold for v in values), "above_rate_strict": sum(v > threshold for v in values) / len(values) if values else None,
            "median_strict": statistics.median(values) if values else None,
            "mean_strict": statistics.mean(values) if values else None,
            "above_rate_tolerant_sensitivity": sum(v > threshold for v in tolerant_values) / len(tolerant_values) if tolerant_values else None,
            "median_tolerant_sensitivity": statistics.median(tolerant_values) if tolerant_values else None}


def main() -> None:
    thresholds = load_json("thresholds.json")["questions"]
    neutral = load_jsonl("neutral_baseline_raw.jsonl")
    free = load_jsonl("donation_estimation_raw.jsonl")
    numeric = load_jsonl("numeric_candidate_likelihood_raw.jsonl")
    comprehension = load_jsonl("mapping_comprehension_raw.jsonl")
    preference = load_json("preference_freeze.json")

    neutral_summary = {task: estimate_group([r for r in neutral if r["task_id"] == task], info["threshold"])
                       for task, info in thresholds.items()}
    free_summary = []
    for task, info in thresholds.items():
        if not info["eligible"]:
            continue
        map_stats = {mapping: estimate_group([r for r in free if r["task_id"] == task and r["mapping"] == mapping], info["threshold"])
                     for mapping in ("favored_above", "favored_below")}
        a, b = map_stats["favored_above"], map_stats["favored_below"]
        item = {"task_id": task, "threshold": info["threshold"], "neutral": neutral_summary[task],
                "mapping_stats": map_stats,
                "delta_above_rate_preferred_above_minus_below": a["above_rate_strict"] - b["above_rate_strict"] if a["above_rate_strict"] is not None and b["above_rate_strict"] is not None else None,
                "delta_median_estimate_preferred_above_minus_below": a["median_strict"] - b["median_strict"] if a["median_strict"] is not None and b["median_strict"] is not None else None}
        item["rate_delta_by_wording"] = {}
        item["rate_delta_by_clause_order"] = {}
        for field, destination, levels in (("wording_id", item["rate_delta_by_wording"], ("question_a", "question_b")),
                                           ("clause_order", item["rate_delta_by_clause_order"], ("above_first", "below_first"))):
            for level in levels:
                subset = [r for r in free if r["task_id"] == task and r[field] == level]
                hi = estimate_group([r for r in subset if r["mapping"] == "favored_above"], info["threshold"])
                lo = estimate_group([r for r in subset if r["mapping"] == "favored_below"], info["threshold"])
                destination[level] = hi["above_rate_strict"] - lo["above_rate_strict"] if hi["above_rate_strict"] is not None and lo["above_rate_strict"] is not None else None
        free_summary.append(item)

    exact_numeric = [r for r in numeric if r["score_status"] == "exact"]
    candidate_by_task = []
    for task in thresholds:
        groups = {mapping: [r for r in exact_numeric if r["task_id"] == task and r["mapping"] == mapping]
                  for mapping in ("favored_above", "favored_below")}
        deltas = {mapping: [r["logprob_high_minus_low"] for r in rows] for mapping, rows in groups.items()}
        a, b = deltas["favored_above"], deltas["favored_below"]
        lengths = [n for r in exact_numeric if r["task_id"] == task for n in r["candidate_token_counts"].values()]
        candidate_by_task.append({"task_id": task, "exact_trials_favored_above_n": len(a), "exact_trials_favored_below_n": len(b),
                                  "mean_logprob_high_minus_low_favored_above": statistics.mean(a) if a else None,
                                  "mean_logprob_high_minus_low_favored_below": statistics.mean(b) if b else None,
                                  "mapping_delta": statistics.mean(a) - statistics.mean(b) if a and b else None,
                                  "candidate_continuation_token_count_min_max": [min(lengths), max(lengths)] if lengths else None})
    summary = {"model": "Qwen2.5-14B-Instruct-Q4_K_M", "preference_freeze_sha256": preference["freeze_sha256"],
               "selected_numeric_prompt_style": load_json("estimation_smoke_summary.json")["selected_style"],
               "validity": {"neutral_n": len(neutral), "neutral_strict_valid_n": sum(r["validity"] == "valid_single_number" for r in neutral),
                            "neutral_noncompliant_n": sum(r["validity"] == "parsed_single_numeric_noncompliant" for r in neutral),
                            "neutral_unparseable_n": sum(r["validity"] == "unparseable_completion" for r in neutral),
                            "donation_n": len(free), "donation_strict_valid_n": sum(r["validity"] == "valid_single_number" for r in free),
                            "donation_noncompliant_n": sum(r["validity"] == "parsed_single_numeric_noncompliant" for r in free),
                            "donation_unparseable_n": sum(r["validity"] == "unparseable_completion" for r in free)},
               "neutral_baseline_by_task": neutral_summary, "free_estimation_by_task": free_summary,
               "numeric_candidate_likelihood": {"n_trials": len(numeric), "exact_n": len(exact_numeric),
                   "unavailable_n": len(numeric) - len(exact_numeric), "n_exact_token_scores": sum(len(c["tokens"]) for r in exact_numeric for c in r["candidate_scores"].values()),
                   "by_task": candidate_by_task, "score_source": "exact unfiltered token logprobs from top-N expansion; unavailable targets excluded"},
               "mapping_comprehension": {"n": len(comprehension), "exact_n": sum(r["score_status"] == "exact" for r in comprehension),
                                         "correct_n": sum(r.get("correct") is True for r in comprehension),
                                         "accuracy": sum(r.get("correct") is True for r in comprehension) / len(comprehension) if comprehension else None},
               "raw_hashes": {name: sha256(file) for name, file in (("neutral_baseline", "neutral_baseline_raw.jsonl"),
                   ("donation_estimation", "donation_estimation_raw.jsonl"), ("numeric_candidate_likelihood", "numeric_candidate_likelihood_raw.jsonl"),
                   ("mapping_comprehension", "mapping_comprehension_raw.jsonl"))}}
    (HERE / "estimation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {"protocol": "pretrained-donation-bet-two-goods-instruct-matched-v1", "seed": 20261125,
                "selected_prompt_style": summary["selected_numeric_prompt_style"], "preference_freeze_sha256": preference["freeze_sha256"],
                "eligible_stable_preference_pairs": [p["pair_id"] for p in preference["pairs"] if p["stable_by_original_preregistered_rule"]],
                "neutral_trials": [{k: r.get(k) for k in ("trial_id", "task_id", "wording_id", "replicate", "seed", "prompt", "scoring_prompt", "prompt_sha256", "validity", "text")} for r in neutral],
                "free_estimation_trials": [{k: r.get(k) for k in ("trial_id", "task_id", "pair_id", "frozen_preference", "mapping", "wording_id", "clause_order", "replicate", "seed", "threshold", "above_outcome", "at_or_below_outcome", "prompt", "scoring_prompt", "prompt_sha256", "validity", "text")} for r in free],
                "candidate_trials_manifest": "probability_scoring_manifest.json", "thresholds": thresholds,
                "raw_sha256": summary["raw_hashes"]}
    (HERE / "estimation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"validity": summary["validity"], "free_estimation_by_task": free_summary,
                      "numeric_candidate_likelihood": summary["numeric_candidate_likelihood"],
                      "mapping_comprehension": summary["mapping_comprehension"]}, indent=2))


if __name__ == "__main__":
    main()
