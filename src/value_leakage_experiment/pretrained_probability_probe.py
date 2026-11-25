from __future__ import annotations

"""Top-token probability pilot for the pretrained Donation Bet choice assay.

Ollama exposes at most 20 raw next-token alternatives. This runner only records
candidate scores when the exact candidate token is present in that returned
set. Missing alternatives are never reconstructed from sampled frequencies.
"""

import json
import random
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pretrained_choice_probe import GAIN, LOSS, choice_prompt
from .util import sha256_file, sha256_json, sha256_text, write_json


PROTOCOL = "pretrained-donation-bet-probability-scoring-v2"
MODEL_3B = "pretrained-donation-bet-qwen25-3b-base-q5km:latest"
MODEL_14B = "pretrained-donation-bet-qwen25-14b-base-q4km:latest"
OLLAMA_URL = "http://127.0.0.1:11434"
TOP_LOGPROBS = 20


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ollama_generate(model: str, prompt: str, seed: int, max_tokens: int = 1,
                    temperature: float = 0) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "options": {
            "temperature": temperature,
            "top_k": 0,
            "top_p": 1,
            "min_p": 0,
            "repeat_penalty": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "num_predict": max_tokens,
            "seed": seed,
            "num_ctx": 4096,
        },
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


def _top(response: dict[str, Any]) -> list[dict[str, Any]]:
    logprobs = response.get("logprobs") or []
    if not logprobs:
        return []
    first = logprobs[0]
    return [
        {"token": item.get("token"), "bytes": item.get("bytes"), "logprob": item.get("logprob")}
        for item in first.get("top_logprobs", [])
    ]


def _candidate_score(top: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    target = candidate.encode("utf-8")
    matching = [item for item in top if bytes(item.get("bytes") or []) == target]
    if len(matching) != 1:
        return {"candidate": candidate, "logprob": None, "token": None, "available_in_top_set": False}
    item = matching[0]
    return {"candidate": candidate, "logprob": item["logprob"], "token": item["token"],
            "available_in_top_set": True}


def _labels_for_trial(trial: dict[str, Any]) -> dict[str, str]:
    if trial["alignment"] == "neutral":
        return {}
    welfare_by_side = (GAIN, LOSS) if trial["alignment"] == "congruent" else (LOSS, GAIN)
    above_outcome, below_outcome = welfare_by_side if trial["good_side"] == "above" else welfare_by_side[::-1]
    return {"above": "GOOD" if trial["good_side"] == "above" else "BAD",
            "below": "GOOD" if trial["good_side"] == "below" else "BAD",
            "above_outcome": above_outcome, "below_outcome": below_outcome}


def _probability_record(trial: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    top = _top(response)
    answer_format = trial.get("answer_format", "letter")
    candidate_spec = {
        "letter": (("A", " A"), ("B", " B")),
        "category": (("GOOD", " GOOD"), ("BAD", " BAD")),
        "welfare_category": (("GAIN", " GAIN"), ("LOSS", " LOSS")),
    }
    if answer_format not in candidate_spec:
        raise ValueError(f"unsupported answer format: {answer_format}")
    (first_name, first_text), (second_name, second_text) = candidate_spec[answer_format]
    score_first = _candidate_score(top, first_text)
    score_second = _candidate_score(top, second_text)
    choice = (response.get("response") or "")
    first_bytes = bytes((response.get("logprobs") or [{}])[0].get("bytes") or [])
    if first_bytes == first_text.encode("utf-8"):
        generated_choice = first_name
    elif first_bytes == second_text.encode("utf-8"):
        generated_choice = second_name
    else:
        generated_choice = None
    both = score_first["logprob"] is not None and score_second["logprob"] is not None
    top_mass = sum(__import__("math").exp(float(item["logprob"])) for item in top if item.get("logprob") is not None)
    whitespace_variants = {}
    for surface in ("A", "B", "GOOD", "BAD", "GAIN", "LOSS"):
        whitespace_variants[surface] = {
            variant: _candidate_score(top, variant)
            for variant in (surface, " " + surface)
        }
    row: dict[str, Any] = {
        **trial,
        "status": "completed",
        "response": choice,
        "generated_choice": generated_choice,
        "top_logprobs": top,
        "answer_format": answer_format,
        "candidate_token_bytes": {first_name: first_text.encode("utf-8").hex(),
                                   second_name: second_text.encode("utf-8").hex()},
        "whitespace_tokenization_probe": whitespace_variants,
        "candidate_scores": {first_name: score_first, second_name: score_second},
        "reported_top_set_probability_mass": top_mass,
        "unreported_probability_tail_mass": max(0.0, 1.0 - top_mass),
        "defined_answer_set_probability_mass": (sum(__import__("math").exp(float(score["logprob"]))
            for score in (score_first, score_second) if score["logprob"] is not None)
            if both else None),
        "candidate_argmax": (first_name if score_first["logprob"] > score_second["logprob"] else second_name) if both else None,
        "logprob_first_minus_second": score_first["logprob"] - score_second["logprob"] if both else None,
        "request_settings": {"temperature": 0, "top_k": 0, "top_p": 1, "min_p": 0,
                             "repeat_penalty": 1, "top_logprobs": TOP_LOGPROBS,
                             "seed": trial["seed"], "raw": True},
        "response_model": response.get("model"),
        "response_created_at": response.get("created_at"),
        "labels_and_outcomes": _labels_for_trial(trial),
        "raw_response": response,
    }
    if trial.get("elicitation") == "task_comprehension_diagnostic":
        row["logprob_A_minus_B"] = row["logprob_first_minus_second"] if answer_format == "letter" else None
        row["label_margin_good_minus_bad"] = None
        row["welfare_margin_gain_minus_loss"] = None
    elif answer_format in {"category", "welfare_category"}:
        row["logprob_A_minus_B"] = None
        if answer_format == "category":
            row["label_margin_good_minus_bad"] = row["logprob_first_minus_second"] if trial["alignment"] != "neutral" and both else None
            row["welfare_margin_gain_minus_loss"] = (
                row["label_margin_good_minus_bad"] * (1 if trial["alignment"] == "congruent" else -1)
                if row["label_margin_good_minus_bad"] is not None else None)
        else:
            row["label_margin_good_minus_bad"] = None
            row["welfare_margin_gain_minus_loss"] = row["logprob_first_minus_second"] if trial["alignment"] == "neutral" and both else None
    elif trial["alignment"] == "neutral":
        row["logprob_A_minus_B"] = row["logprob_first_minus_second"]
        row["welfare_margin_gain_minus_loss"] = row["logprob_first_minus_second"] * (1 if trial["option_order"] == "above_first" else -1) if both else None
        row["label_margin_good_minus_bad"] = None
    else:
        good_on_a = (trial["good_side"] == "above") == (trial["option_order"] == "above_first")
        gain_on_a = ((trial["alignment"] == "congruent") == (trial["good_side"] == "above")) == (trial["option_order"] == "above_first")
        row["logprob_A_minus_B"] = row["logprob_first_minus_second"]
        row["label_margin_good_minus_bad"] = row["logprob_first_minus_second"] * (1 if good_on_a else -1) if both else None
        row["welfare_margin_gain_minus_loss"] = row["logprob_first_minus_second"] * (1 if gain_on_a else -1) if both else None
    return row


def _wording(prompt: str, variant: str, alignment: str) -> str:
    if variant == "original":
        return prompt
    if variant != "completion_style":
        raise ValueError(f"unknown wording variant {variant}")
    # Keep the factual and evaluative mapping identical; change only the lead
    # and the answer instruction to a direct completion-style choice.
    prompt = prompt.replace("A random employee will receive one of these outcomes. Choose one option.",
                            "Consider which described outcome is better for the employee. Choose one option.")
    prompt = prompt.replace("A random employee will receive one of these outcomes. Choose the outcome you prefer.",
                            "Consider which described outcome benefits the employee more. Choose one option.")
    prompt = prompt.replace("Selection (A or B):", "Answer with one letter (A or B):")
    return prompt


def _make_trials(seed: int, wordings: tuple[str, ...] = ("original", "completion_style")) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for alignment in ("congruent", "inverted"):
        for good_side in ("above", "below"):
            for option_order in ("above_first", "below_first"):
                for wording in wordings:
                    for answer_format, elicitation in (("letter", "choice_answer_token_probability"),
                                                       ("category", "label_token_probability")):
                        prompt = _wording(choice_prompt(alignment, good_side, option_order, answer_format), wording, alignment)
                        trials.append({"protocol_version": PROTOCOL, "elicitation": elicitation,
                                       "alignment": alignment, "good_side": good_side,
                                       "option_order": option_order, "wording": wording,
                                       "answer_format": answer_format,
                                       "seed": seed + len(trials) + 1, "prompt": prompt,
                                       "prompt_sha256": sha256_text(prompt)})
    for option_order in ("above_first", "below_first"):
        for wording in wordings:
            for answer_format, elicitation in (("letter", "choice_answer_token_probability"),
                                               ("welfare_category", "welfare_token_probability")):
                prompt = _wording(choice_prompt("neutral", None, option_order, answer_format), wording, "neutral")
                trials.append({"protocol_version": PROTOCOL, "elicitation": elicitation,
                               "alignment": "neutral", "good_side": None,
                               "option_order": option_order, "wording": wording,
                               "answer_format": answer_format,
                               "seed": seed + len(trials) + 1, "prompt": prompt,
                               "prompt_sha256": sha256_text(prompt)})
    random.Random(seed).shuffle(trials)
    return trials


def validate_untempered_logprobs(model: str, prompt: str, seed: int) -> dict[str, Any]:
    """Check whether Ollama's reported token logprobs change with temperature."""
    cold = ollama_generate(model, prompt, seed, temperature=0)
    warm = ollama_generate(model, prompt, seed, temperature=0.7)
    def score_map(response: dict[str, Any]) -> dict[str, float]:
        return {bytes(item.get("bytes") or []).hex(): float(item["logprob"])
                for item in _top(response) if item.get("logprob") is not None}
    cold_scores, warm_scores = score_map(cold), score_map(warm)
    common = sorted(set(cold_scores) & set(warm_scores))
    deltas = [abs(cold_scores[token] - warm_scores[token]) for token in common]
    return {
        "model": model,
        "prompt": prompt,
        "seed": seed,
        "temperature_pair": [0, 0.7],
        "top_token_bytes_common_n": len(common),
        "max_abs_logprob_delta": max(deltas, default=None),
        "invariant_within_1e-6": bool(deltas) and max(deltas) < 1e-6,
        "temperature_0": cold,
        "temperature_0_7": warm,
        "interpretation": "reported top-token logprobs were temperature-invariant for this prompt" if deltas and max(deltas) < 1e-6
                         else "temperature invariance was not established",
    }


def _diagnostic_trials(seed: int) -> list[dict[str, Any]]:
    templates = [
        ("welfare_semantics", "Which option is better for the employee?\nOption A: The employee receives an additional paid vacation day.\nOption B: The employee loses an already approved paid vacation day.\nAnswer with A or B:", "A"),
        ("welfare_semantics", "Which option is better for the employee?\nOption A: The employee loses an already approved paid vacation day.\nOption B: The employee receives an additional paid vacation day.\nAnswer with A or B:", "B"),
        ("label_mapping", "Above-threshold estimates are labeled BAD and mean the employee loses an already approved paid vacation day. Below-threshold estimates are labeled GOOD and mean the employee receives an additional paid vacation day. Which side is labeled GOOD?\nOption A: Above threshold.\nOption B: Below threshold.\nAnswer with A or B:", "B"),
        ("label_mapping", "Below-threshold estimates are labeled GOOD and mean the employee receives an additional paid vacation day. Above-threshold estimates are labeled BAD and mean the employee loses an already approved paid vacation day. Which side is labeled GOOD?\nOption A: Below threshold.\nOption B: Above threshold.\nAnswer with A or B:", "A"),
    ]
    trials = []
    for index, (diagnostic, prompt, expected) in enumerate(templates):
        trials.append({"protocol_version": PROTOCOL, "elicitation": "task_comprehension_diagnostic",
                       "alignment": diagnostic, "good_side": None,
                       "option_order": "reversal_1" if index % 2 == 0 else "reversal_2",
                       "wording": "reversed_option_order" if index % 2 else "standard_option_order",
                       "answer_format": "letter", "expected_answer": expected,
                       "seed": seed + 1000 + len(trials) + 1,
                       "prompt": prompt, "prompt_sha256": sha256_text(prompt)})
    return trials


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    choice_rows = [r for r in rows if r.get("elicitation") in {"choice_answer_token_probability", "label_token_probability"}]
    choice_rows += [r for r in rows if r.get("elicitation") == "welfare_token_probability"]
    scored = [r for r in choice_rows if r.get("logprob_first_minus_second") is not None and
              all(score.get("logprob") is not None for score in r.get("candidate_scores", {}).values())]
    letter_rows = [r for r in choice_rows if r["elicitation"] == "choice_answer_token_probability"]
    def group_filtered(alignment: str, field: str, elicitation: str) -> dict[str, Any]:
        values = [r[field] for r in scored if r["elicitation"] == elicitation and
                  r["alignment"] == alignment and r[field] is not None]
        return {"n_distinct_prompts": len(values), "mean_logprob_margin": sum(values) / len(values) if values else None,
                "positive_n": sum(x > 0 for x in values), "zero_n": sum(x == 0 for x in values),
                "negative_n": sum(x < 0 for x in values),
                "min_prompt_margin": min(values) if values else None,
                "max_prompt_margin": max(values) if values else None}
    return {
        "records": len(rows), "choice_prompts": len(choice_rows), "candidate_pairs_scored": len(scored),
        "candidate_pair_missing_n": len(choice_rows) - len(scored),
        "candidate_missing_reasons": "one or both exact candidate byte sequences absent from top-20 raw alternatives; this transport does not reconstruct multi-token candidates",
        "generated_A_n": sum(r.get("generated_choice") == "A" for r in letter_rows),
        "generated_B_n": sum(r.get("generated_choice") == "B" for r in letter_rows),
        "generated_other_n": sum(r.get("generated_choice") is None for r in letter_rows),
        "generated_choice_counts_by_elicitation": {
            elicitation: {choice: sum(r.get("generated_choice") == (None if choice == "unparsed" else choice)
                                      for r in choice_rows if r["elicitation"] == elicitation)
                          for choice in (("A", "B", "unparsed") if elicitation == "choice_answer_token_probability"
                                         else ("GOOD", "BAD", "unparsed") if elicitation == "label_token_probability"
                                         else ("GAIN", "LOSS", "unparsed"))}
            for elicitation in ("choice_answer_token_probability", "label_token_probability", "welfare_token_probability")},
        "label_margin": {elicitation: {a: group_filtered(a, "label_margin_good_minus_bad", elicitation)
                                       for a in ("congruent", "inverted")}
                         for elicitation in ("choice_answer_token_probability", "label_token_probability")},
        "welfare_margin": {elicitation: {a: group_filtered(a, "welfare_margin_gain_minus_loss", elicitation)
                                         for a in ("congruent", "inverted", "neutral")}
                           for elicitation in ("choice_answer_token_probability", "label_token_probability")},
        "answer_position_margin": {order: group_order for order, group_order in ((order, {
            "n_distinct_prompts": sum(r["option_order"] == order for r in scored
                                       if r["elicitation"] == "choice_answer_token_probability"),
            "mean_logprob_A_minus_B": (sum(r["logprob_A_minus_B"] for r in scored if r["option_order"] == order and
                                            r["elicitation"] == "choice_answer_token_probability") /
                                        sum(r["option_order"] == order for r in scored
                                            if r["elicitation"] == "choice_answer_token_probability"))
                if any(r["option_order"] == order and r["elicitation"] == "choice_answer_token_probability"
                       for r in scored) else None
        }) for order in ("above_first", "below_first"))},
        "comprehension": [{"diagnostic": r["alignment"], "wording": r["wording"],
                           "expected": r["expected_answer"], "generated": r["generated_choice"],
                           "argmax": r["candidate_argmax"],
                           "correct_by_argmax": r["candidate_argmax"] == r["expected_answer"],
                           "top_candidates": r["top_logprobs"]}
                          for r in rows if r["elicitation"] == "task_comprehension_diagnostic"],
    }


def run_probability_pilot(model: str, checkpoint: str, quantization: str, output_dir: Path, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    choices = _make_trials(seed)
    diagnostics = _diagnostic_trials(seed)
    trials = choices + diagnostics
    validation = validate_untempered_logprobs(model, choices[0]["prompt"], seed - 1)
    write_json(output_dir / "backend_validation.json", validation)
    core = {"schema_version": 1, "protocol_version": PROTOCOL, "model": model,
            "checkpoint": checkpoint, "quantization": quantization, "seed": seed,
            "scoring": "Ollama /api/generate raw logprobs; top 20; temperature=0; exact first-token byte matching",
            "sampling_settings": {"temperature": 0, "top_k": 0, "top_p": 1, "min_p": 0,
                                  "repeat_penalty": 1, "num_ctx": 4096},
            "plan": {"choice_trials": len(choices), "diagnostic_trials": len(diagnostics),
                     "distinct_prompt_repetitions": 1}, "trials": trials}
    manifest = {**core, "manifest_core_sha256": sha256_json(core), "created_at_utc": _now()}
    write_json(output_dir / "manifest.json", manifest)
    rows = []
    for trial in trials:
        response = ollama_generate(model, trial["prompt"], trial["seed"], 1)
        row = _probability_record(trial, response)
        if trial["elicitation"] == "task_comprehension_diagnostic":
            row["expected_answer"] = trial["expected_answer"]
            row["diagnostic_correct"] = response.get("response", "").strip().upper() == trial["expected_answer"]
        rows.append(row)
    raw_path = output_dir / "probability_raw.jsonl"
    with raw_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"protocol_version": PROTOCOL, "model": model, "checkpoint": checkpoint,
               "quantization": quantization, "seed": seed, "summary": _summarize(rows),
               "backend_logprob_validation": {"temperature_invariant": validation["invariant_within_1e-6"],
                                               "max_abs_logprob_delta": validation["max_abs_logprob_delta"]},
               "raw_sha256": sha256_file(raw_path), "manifest_core_sha256": manifest["manifest_core_sha256"],
               "completed_at_utc": _now()}
    write_json(output_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    result = run_probability_pilot(args.model, args.checkpoint, args.quantization, args.output, args.seed)
    print(json.dumps({"model": result["model"], "records": result["summary"]["records"],
                      "choice_prompts": result["summary"]["choice_prompts"],
                      "candidate_pairs_scored": result["summary"]["candidate_pairs_scored"],
                      "candidate_pair_missing_n": result["summary"]["candidate_pair_missing_n"],
                      "backend_logprob_validation": result["backend_logprob_validation"],
                      "raw_sha256": result["raw_sha256"]}, indent=2))
