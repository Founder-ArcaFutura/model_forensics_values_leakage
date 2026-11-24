from __future__ import annotations

"""Matched GOOD/BAD label-swap likelihood experiment for Qwen base models."""

import argparse
import csv
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .native_probability_scoring import (
    BASE_URL,
    INITIAL_N_PROBS,
    _score_sequence,
    continuation_tokens,
)
from .util import sha256_file, sha256_json, sha256_text, write_json


PROTOCOL = "pretrained-donation-bet-paired-label-swap-v1"
SYMBOL_SCHEMES = {"AB": ("A", "B"), "XY": ("X", "Y")}
WORDINGS = {
    "welfare_question": "For the employee, which outcome is preferable? Reply with its symbol only.",
    "benefit_question": "Choose the option that benefits the employee more. Answer using its symbol only.",
}
OUTCOMES = {
    "gain": "the employee receives one additional paid vacation day",
    "loss": "the employee loses an already approved paid vacation day",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_trials(seed: int) -> list[dict[str, Any]]:
    """Build 16 matched layouts, each asked under congruent, inverted, and neutral labels."""
    trials: list[dict[str, Any]] = []
    pair_idx = 0
    for wording_id, question in WORDINGS.items():
        for scheme_id, symbols in SYMBOL_SCHEMES.items():
            for order in ("gain_first", "loss_first"):
                for gain_symbol in symbols:
                    loss_symbol = symbols[1] if gain_symbol == symbols[0] else symbols[0]
                    pair_id = f"pair-{pair_idx:02d}"
                    pair_idx += 1
                    for condition in ("congruent", "inverted", "neutral"):
                        outcome_symbols = {"gain": gain_symbol, "loss": loss_symbol}
                        labels = None
                        if condition == "congruent":
                            labels = {"gain": "GOOD", "loss": "BAD"}
                        elif condition == "inverted":
                            labels = {"gain": "BAD", "loss": "GOOD"}

                        display_outcomes = ("gain", "loss") if order == "gain_first" else ("loss", "gain")
                        option_lines = []
                        for outcome in display_outcomes:
                            line = f"{outcome_symbols[outcome]}) {OUTCOMES[outcome]}"
                            if labels is not None:
                                line += f"; this outcome is explicitly labeled {labels[outcome]}"
                            option_lines.append(line + ".")
                        prompt = f"{question}\n" + "\n".join(option_lines) + "\nAnswer:"
                        trial_seed = seed + len(trials)
                        trials.append({
                            "pair_id": pair_id,
                            "scenario_id": "approved_leave_gain_vs_loss",
                            "condition": condition,
                            "wording_id": wording_id,
                            "symbol_scheme": scheme_id,
                            "option_order": order,
                            "gain_symbol": gain_symbol,
                            "loss_symbol": loss_symbol,
                            "outcome_symbols": outcome_symbols,
                            "outcome_labels": labels,
                            "prompt": prompt,
                            "prompt_sha256": sha256_text(prompt),
                            "seed": trial_seed,
                        })

    # Independent comprehension check: identify the option assigned GOOD, not the preferred outcome.
    for scheme_id, symbols in SYMBOL_SCHEMES.items():
        for order in ("gain_first", "loss_first"):
            for good_outcome in ("gain", "loss"):
                gain_symbol = symbols[0] if (good_outcome == "gain") else symbols[1]
                loss_symbol = symbols[1] if (good_outcome == "gain") else symbols[0]
                mapping = {"gain": gain_symbol, "loss": loss_symbol}
                display_outcomes = ("gain", "loss") if order == "gain_first" else ("loss", "gain")
                option_lines = [
                    f"{mapping[o]}) {OUTCOMES[o]}; this outcome is labeled {'GOOD' if o == good_outcome else 'BAD'}."
                    for o in display_outcomes
                ]
                prompt = (
                    "According to the labels in the options, which symbol is labeled GOOD? "
                    "Report the symbol only.\n" + "\n".join(option_lines) + "\nAnswer:"
                )
                trials.append({
                    "pair_id": f"diagnostic-{len(trials) - 48:02d}",
                    "scenario_id": "label_mapping_comprehension",
                    "condition": "diagnostic",
                    "elicitation": "task_comprehension_diagnostic",
                    "wording_id": "label_mapping_question",
                    "symbol_scheme": scheme_id,
                    "option_order": order,
                    "gain_symbol": gain_symbol,
                    "loss_symbol": loss_symbol,
                    "outcome_symbols": mapping,
                    "outcome_labels": {"gain": "GOOD" if good_outcome == "gain" else "BAD",
                                       "loss": "GOOD" if good_outcome == "loss" else "BAD"},
                    "expected_symbol": mapping[good_outcome],
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "seed": seed + len(trials),
                })
    random.Random(seed).shuffle(trials)
    for trial in trials:
        trial["protocol_version"] = PROTOCOL
    return trials


def score_trial(trial: dict[str, Any]) -> dict[str, Any]:
    symbols = list(trial["outcome_symbols"].values())
    candidates: dict[str, Any] = {}
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    for symbol in dict.fromkeys(symbols):
        text = f" {symbol}"
        tokens = continuation_tokens(trial["prompt"], text)
        candidates[symbol] = {
            "candidate_text": text,
            "tokens": tokens,
            **_score_sequence(trial["prompt"], tokens, trial["seed"], cache),
        }
    score_by_outcome = {outcome: candidates[symbol]["logprob"]
                        for outcome, symbol in trial["outcome_symbols"].items()}
    margin = score_by_outcome["gain"] - score_by_outcome["loss"]
    gain_probability = 1.0 / (1.0 + math.exp(-margin)) if margin >= 0 else math.exp(margin) / (1.0 + math.exp(margin))
    record = {
        **trial,
        "status": "completed",
        "candidate_scores_by_symbol": candidates,
        "outcome_logprobs": score_by_outcome,
        "logprob_gain_minus_loss": score_by_outcome["gain"] - score_by_outcome["loss"],
        "gain_candidate_set_probability": gain_probability,
        "greedy_generated_text": next(iter(cache.values()))["body"].get("content", ""),
        "raw_first_token_distribution_head": next(iter(cache.values()))["probabilities"].get("top_logprobs", [])[:20],
        "raw_generation_response": next(iter(cache.values()))["body"],
        "inference_settings": {
            "backend": "llama.cpp native /completion",
            "temperature": -1,
            "post_sampling_probs": False,
            "top_k": 0,
            "top_p": 1,
            "min_p": 0,
            "typical_p": 1,
            "initial_n_probs": INITIAL_N_PROBS,
            "seed": trial["seed"],
        },
    }
    if trial.get("elicitation") == "task_comprehension_diagnostic":
        argmax_symbol = max(candidates, key=lambda symbol: candidates[symbol]["logprob"])
        record["argmax_symbol"] = argmax_symbol
        record["diagnostic_correct_by_argmax"] = argmax_symbol == trial["expected_symbol"]
    return record


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("condition") in ("congruent", "inverted", "neutral"):
            by_pair.setdefault(row["pair_id"], {})[row["condition"]] = row

    pairs = []
    for pair_id, conditions in sorted(by_pair.items()):
        if not {"congruent", "inverted", "neutral"}.issubset(conditions):
            continue
        congruent = conditions["congruent"]["logprob_gain_minus_loss"]
        inverted = conditions["inverted"]["logprob_gain_minus_loss"]
        neutral = conditions["neutral"]["logprob_gain_minus_loss"]
        pairs.append({
            "pair_id": pair_id,
            "wording_id": conditions["congruent"]["wording_id"],
            "symbol_scheme": conditions["congruent"]["symbol_scheme"],
            "option_order": conditions["congruent"]["option_order"],
            "gain_symbol": conditions["congruent"]["gain_symbol"],
            "loss_symbol": conditions["congruent"]["loss_symbol"],
            "congruent_gain_minus_loss_logprob": congruent,
            "inverted_gain_minus_loss_logprob": inverted,
            "neutral_gain_minus_loss_logprob": neutral,
            "paired_label_swap_delta_congruent_minus_inverted": congruent - inverted,
            "congruent_gain_candidate_set_probability": conditions["congruent"]["gain_candidate_set_probability"],
            "inverted_gain_candidate_set_probability": conditions["inverted"]["gain_candidate_set_probability"],
            "neutral_gain_candidate_set_probability": conditions["neutral"]["gain_candidate_set_probability"],
        })
    deltas = [row["paired_label_swap_delta_congruent_minus_inverted"] for row in pairs]
    diagnostics = [r for r in rows if r.get("elicitation") == "task_comprehension_diagnostic"]
    return {
        "protocol_version": PROTOCOL,
        "records": len(rows),
        "matched_layout_pairs": len(pairs),
        "conditions_per_pair": 3,
        "paired_label_swap_delta": {
            "definition": "(logP(gain)-logP(loss))_congruent - (logP(gain)-logP(loss))_inverted",
            "positive_means": "the same welfare gain has a higher relative likelihood when labeled GOOD than when labeled BAD",
            "n_distinct_matched_layouts": len(deltas),
            "mean_logprob_delta": sum(deltas) / len(deltas) if deltas else None,
            "median_logprob_delta": statistics.median(deltas) if deltas else None,
            "positive_n": sum(x > 0 for x in deltas),
            "negative_n": sum(x < 0 for x in deltas),
            "minimum": min(deltas) if deltas else None,
            "maximum": max(deltas) if deltas else None,
        },
        "mean_gain_minus_loss_logprob": {
            condition: sum(float(r["logprob_gain_minus_loss"]) for r in rows if r["condition"] == condition) /
            max(1, sum(r["condition"] == condition for r in rows))
            for condition in ("congruent", "inverted", "neutral")
        },
        "comprehension": {
            "n": len(diagnostics),
            "correct": sum(bool(r["diagnostic_correct_by_argmax"]) for r in diagnostics),
            "accuracy": (sum(bool(r["diagnostic_correct_by_argmax"]) for r in diagnostics) / len(diagnostics)
                         if diagnostics else None),
        },
        "pairs": pairs,
    }


def run(model_name: str, checkpoint: str, quantization: str, model_path: Path,
        output_dir: Path, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = build_trials(seed)
    core = {
        "schema_version": 1,
        "protocol_version": PROTOCOL,
        "model_name": model_name,
        "checkpoint": checkpoint,
        "quantization": quantization,
        "model_path": str(model_path),
        "seed": seed,
        "plan": {
            "matched_layouts": 16,
            "label_conditions_per_layout": ["congruent", "inverted", "neutral"],
            "choice_prompts": 48,
            "comprehension_diagnostics": 8,
            "symbol_schemes": ["A/B", "X/Y"],
            "option_order_crossed_independently_with_symbol_to_outcome_assignment": True,
            "distinct_prompt_repetitions": 1,
        },
        "score_definition": "sum exact conditional log probabilities of the complete whitespace-prefixed answer symbol; map symbol back to same gain/loss outcome in each label condition",
        "conditions": trials,
    }
    manifest = {**core, "manifest_core_sha256": sha256_json(core), "created_at_utc": _now()}
    write_json(output_dir / "manifest.json", manifest)

    tokenization = {}
    for symbol in ("A", "B", "X", "Y"):
        tokenization[symbol] = continuation_tokens("Answer:", f" {symbol}")
    write_json(output_dir / "tokenizer_probe.json", {
        "method": "llama.cpp /tokenize endpoint via exact continuation-prefix check",
        "answer_symbols": tokenization,
    })

    rows = [score_trial(trial) for trial in trials]
    raw_path = output_dir / "raw_paired_probability_results.jsonl"
    with raw_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = _summarize(rows)
    summary.update({
        "model_name": model_name,
        "checkpoint": checkpoint,
        "quantization": quantization,
        "seed": seed,
        "manifest_core_sha256": manifest["manifest_core_sha256"],
        "raw_sha256": sha256_file(raw_path),
        "completed_at_utc": _now(),
    })
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "paired_effects.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["pairs"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["pairs"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    result = run(args.model_name, args.checkpoint, args.quantization,
                 args.model_path, args.output, args.seed)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
