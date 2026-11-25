from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .util import read_json, sha256_file, write_json


CHOICE_TYPES = {"choice_answer_token_probability", "label_token_probability", "welfare_token_probability"}


def _group(rows: list[dict[str, Any]], elicitation: str, alignment: str, field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if row["elicitation"] == elicitation and
              row["alignment"] == alignment and row.get(field) is not None]
    return {"n_distinct_prompts": len(values), "mean_logprob_margin": sum(values) / len(values) if values else None,
            "positive_n": sum(value > 0 for value in values), "negative_n": sum(value < 0 for value in values),
            "min_prompt_margin": min(values) if values else None, "max_prompt_margin": max(values) if values else None}


def _analyze_model(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = read_json(path / "summary.json")
    rows = [json.loads(line) for line in (path / "raw_probability_results.jsonl").read_text(encoding="utf-8").splitlines()]
    choices = [row for row in rows if row.get("elicitation") in CHOICE_TYPES]
    diagnostics = [row for row in rows if row.get("elicitation") == "task_comprehension_diagnostic"]
    fallback_counts = Counter(
        str(token["n_probs_used"]) for row in choices for score in row["candidate_scores"].values()
        for token in score["tokens"])
    fallback_by_elicitation = {}
    for elicitation in CHOICE_TYPES:
        fallback_by_elicitation[elicitation] = sum(
            token["n_probs_used"] == 151936 for row in choices if row["elicitation"] == elicitation
            for score in row["candidate_scores"].values() for token in score["tokens"])
    candidate_argmax = {}
    for elicitation in sorted(CHOICE_TYPES):
        candidate_argmax[elicitation] = dict(Counter(row["candidate_argmax"] for row in choices
                                                     if row["elicitation"] == elicitation))
    summary = {
        "model_name": meta["model_name"], "checkpoint": meta["checkpoint"],
        "quantization": meta["quantization"], "protocol_version": meta["protocol_version"],
        "records_total": len(rows), "choice_records": len(choices), "diagnostic_records": len(diagnostics),
        "all_candidate_scores_exact": all(row.get("all_candidate_tokens_exactly_scored") for row in choices),
        "full_vocabulary_fallback_token_scores": fallback_counts.get("151936", 0),
        "fallback_token_scores_by_elicitation": fallback_by_elicitation,
        "candidate_argmax_counts": candidate_argmax,
        "unconstrained_greedy_first_token_counts": {
            elicitation: dict(Counter(row.get("response", "") for row in choices
                                      if row["elicitation"] == elicitation))
            for elicitation in sorted(CHOICE_TYPES)},
        "label_good_minus_bad": {
            alignment: _group(rows, "label_token_probability", alignment, "label_margin_good_minus_bad")
            for alignment in ("congruent", "inverted")},
        "letter_choice_welfare_gain_minus_loss": {
            alignment: _group(rows, "choice_answer_token_probability", alignment, "welfare_margin_gain_minus_loss")
            for alignment in ("congruent", "inverted", "neutral")},
        "letter_choice_label_good_minus_bad": {
            alignment: _group(rows, "choice_answer_token_probability", alignment, "label_margin_good_minus_bad")
            for alignment in ("congruent", "inverted")},
        "neutral_gain_minus_loss_category": _group(rows, "welfare_token_probability", "neutral",
                                                    "welfare_margin_gain_minus_loss"),
        "label_margin_by_wording_and_order": {
            f"{wording}/{order}": {
                alignment: _group([row for row in rows if row["wording"] == wording and row["option_order"] == order],
                                  "label_token_probability", alignment, "label_margin_good_minus_bad")
                for alignment in ("congruent", "inverted")}
            for wording in ("original", "completion_style") for order in ("above_first", "below_first")},
        "letter_choice_label_margin_by_wording_and_order": {
            f"{wording}/{order}": {
                alignment: _group([row for row in rows if row["wording"] == wording and row["option_order"] == order],
                                  "choice_answer_token_probability", alignment, "label_margin_good_minus_bad")
                for alignment in ("congruent", "inverted")}
            for wording in ("original", "completion_style") for order in ("above_first", "below_first")},
        "welfare_margin_by_wording_and_order": {
            f"{wording}/{order}": {
                alignment: _group([row for row in rows if row["wording"] == wording and row["option_order"] == order],
                                  "choice_answer_token_probability", alignment, "welfare_margin_gain_minus_loss")
                for alignment in ("congruent", "inverted", "neutral")}
            for wording in ("original", "completion_style") for order in ("above_first", "below_first")},
        "answer_position_logprob_A_minus_B": {
            order: {"n": sum(row["option_order"] == order for row in rows
                              if row["elicitation"] == "choice_answer_token_probability"),
                    "mean_logprob_A_minus_B": (sum(row["logprob_A_minus_B"] for row in rows
                                                   if row["option_order"] == order and
                                                   row["elicitation"] == "choice_answer_token_probability") /
                                                sum(row["option_order"] == order for row in rows
                                                    if row["option_order"] == order and
                                                    row["elicitation"] == "choice_answer_token_probability"))
                    if any(row["option_order"] == order and row["elicitation"] == "choice_answer_token_probability"
                           for row in rows) else None}
            for order in ("above_first", "below_first")},
        "comprehension": {"n": len(diagnostics),
                          "argmax_correct_n": sum(row["diagnostic_correct_by_argmax"] is True for row in diagnostics),
                          "items": [{"task": row["alignment"], "ordering": row["wording"],
                                     "expected": row["expected_answer"], "argmax": row["candidate_argmax"],
                                     "correct": row["diagnostic_correct_by_argmax"]}
                                    for row in diagnostics]},
        "candidate_score_ranges": {
            elicitation: {
                alignment: {"logprob_difference_range": [
                    min((row["candidate_logprob_difference_first_minus_second"] for row in rows
                         if row["elicitation"] == elicitation and row["alignment"] == alignment), default=None),
                    max((row["candidate_logprob_difference_first_minus_second"] for row in rows
                         if row["elicitation"] == elicitation and row["alignment"] == alignment), default=None)]}
                for alignment in sorted({row["alignment"] for row in rows if row["elicitation"] == elicitation})}
            for elicitation in sorted(CHOICE_TYPES)},
    }
    table_rows = []
    for row in choices:
        first, second = row["candidate_names"]
        table_rows.append({
            "model": meta["model_name"], "elicitation": row["elicitation"], "answer_format": row["answer_format"],
            "alignment": row["alignment"], "good_side": row.get("good_side"),
            "option_order": row["option_order"], "wording": row["wording"],
            "candidate_1": first, "candidate_1_text": row["candidate_strings"][0],
            "candidate_1_logprob": row["candidate_scores"][first]["logprob"],
            "candidate_1_token_count": row["candidate_scores"][first]["token_count"],
            "candidate_2": second, "candidate_2_text": row["candidate_strings"][1],
            "candidate_2_logprob": row["candidate_scores"][second]["logprob"],
            "candidate_2_token_count": row["candidate_scores"][second]["token_count"],
            "candidate_logprob_difference_1_minus_2": row["candidate_logprob_difference_first_minus_second"],
            "defined_candidate_set_probability_mass": row["defined_candidate_set_probability_mass"],
            "candidate_argmax": row["candidate_argmax"], "greedy_first_token": row["response"],
            "full_vocab_fallback_token_count": sum(token["n_probs_used"] == 151936
                for score in row["candidate_scores"].values() for token in score["tokens"]),
            "prompt_sha256": row["prompt_sha256"],
        })
    return summary, table_rows


def analyze(root: Path) -> dict[str, Any]:
    models = {}
    all_table_rows = []
    for folder in ("qwen25_3b", "qwen25_14b"):
        summary, table_rows = _analyze_model(root / folder)
        models[folder] = summary
        all_table_rows.extend(table_rows)
        write_json(root / folder / "analysis_summary.json", summary)
        diagnostics = [row for row in [json.loads(line) for line in
                        (root / folder / "raw_probability_results.jsonl").read_text(encoding="utf-8").splitlines()]
                      if row.get("elicitation") == "task_comprehension_diagnostic"]
        with (root / folder / "comprehension.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["task", "ordering", "expected", "argmax", "correct", "prompt_sha256"])
            writer.writeheader()
            for row in diagnostics:
                writer.writerow({"task": row["alignment"], "ordering": row["wording"],
                                 "expected": row["expected_answer"], "argmax": row["candidate_argmax"],
                                 "correct": row["diagnostic_correct_by_argmax"], "prompt_sha256": row["prompt_sha256"]})
    table_path = root / "condition_analysis.csv"
    with table_path.open("w", encoding="utf-8", newline="") as file:
        fields = list(all_table_rows[0])
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_table_rows)
    result = {"protocol_version": "pretrained-donation-bet-full-likelihood-v1",
              "models": models,
              "comparison_note": "These are deterministic candidate probabilities for distinct prompts, not independent resamples of identical prompts.",
              "condition_analysis_csv_sha256": sha256_file(table_path), "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(root / "side_by_side_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze matched native full-likelihood Donation Bet results")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps({"models": {name: {"choice_records": model["choice_records"],
                                         "all_candidate_scores_exact": model["all_candidate_scores_exact"],
                                         "comprehension": model["comprehension"]}
                              for name, model in result["models"].items()},
                      "condition_analysis_csv_sha256": result["condition_analysis_csv_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
