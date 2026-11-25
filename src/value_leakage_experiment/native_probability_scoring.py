from __future__ import annotations

"""Exact candidate-continuation likelihoods from llama.cpp native completion."""

import argparse
import json
import math
import random
import subprocess
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .pretrained_probability_probe import _diagnostic_trials, _labels_for_trial, _make_trials
from .util import sha256_file, sha256_json, sha256_text, write_json


PROTOCOL = "pretrained-donation-bet-full-likelihood-v1"
BASE_URL = "http://127.0.0.1:8081"
INITIAL_N_PROBS = 5000
FULL_N_PROBS = 151936  # Qwen2.5 tokenizer vocabulary size; checked against returned count.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenize(text: str) -> list[dict[str, Any]]:
    payload = {"content": text, "add_special": False, "with_pieces": True}
    request = urllib.request.Request(
        f"{BASE_URL}/tokenize", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["tokens"]


def continuation_tokens(prompt: str, candidate: str) -> list[dict[str, Any]]:
    prompt_tokens = tokenize(prompt)
    joint_tokens = tokenize(prompt + candidate)
    prefix = [token["id"] for token in prompt_tokens]
    joint = [token["id"] for token in joint_tokens]
    if joint[:len(prefix)] != prefix:
        raise ValueError("Prompt tokenization is not a prefix of prompt plus candidate")
    suffix = joint_tokens[len(prefix):]
    if not suffix:
        raise ValueError("Candidate produces no continuation tokens")
    if "".join(token["piece"] for token in suffix) != candidate:
        raise ValueError("Candidate token pieces do not reconstruct the exact continuation")
    return suffix


def _curl_json(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="donation_bet_logits_") as tmp:
        request_path = Path(tmp) / "request.json"
        response_path = Path(tmp) / "response.json"
        request_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        proc = subprocess.run(
            ["curl.exe", "-sS", "--fail", "--max-time", "600", "-H", "Content-Type: application/json",
             "--data-binary", f"@{request_path}", "-o", str(response_path), "-w", "%{http_code}", endpoint],
            capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"curl completion failed ({proc.returncode}): {proc.stderr.strip()} {proc.stdout.strip()}")
        status = proc.stdout.strip()
        if status != "200":
            raise RuntimeError(f"llama.cpp returned HTTP {status}: {proc.stderr.strip()}")
        return json.loads(response_path.read_text(encoding="utf-8"))


def distribution(prompt: str, seed: int, n_probs: int) -> dict[str, Any]:
    payload = {
        "prompt": prompt, "seed": seed, "n_predict": 1, "n_probs": n_probs,
        "temperature": -1, "top_k": 0, "top_p": 1, "min_p": 0,
        "typical_p": 1, "repeat_penalty": 1, "frequency_penalty": 0,
        "presence_penalty": 0, "post_sampling_probs": False,
        "return_tokens": True, "stream": False,
    }
    body = _curl_json(f"{BASE_URL}/completion", payload)
    probabilities = body.get("completion_probabilities") or []
    if not probabilities:
        raise RuntimeError("llama.cpp response had no completion_probabilities")
    return {"body": body, "probabilities": probabilities[0], "n_probs_requested": n_probs}


def _score_sequence(prompt: str, tokens: list[dict[str, Any]], seed: int,
                    cache: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    prefix = prompt
    token_scores = []
    for token in tokens:
        key = (prefix, INITIAL_N_PROBS)
        scored = cache.get(key)
        if scored is None:
            scored = distribution(prefix, seed, INITIAL_N_PROBS)
            cache[key] = scored
        top = scored["probabilities"].get("top_logprobs") or []
        target = next((item for item in top if int(item.get("id", -1)) == int(token["id"])), None)
        if target is None:
            full = distribution(prefix, seed, FULL_N_PROBS)
            full_top = full["probabilities"].get("top_logprobs") or []
            if len(full_top) < FULL_N_PROBS:
                raise RuntimeError(f"Full-vocabulary request returned only {len(full_top)} tokens")
            target = next((item for item in full_top if int(item.get("id", -1)) == int(token["id"])), None)
            if target is None:
                raise RuntimeError(f"Target token id {token['id']} is absent from the full vocabulary")
            scored = full
            top = full_top
        token_scores.append({"token_id": int(token["id"]), "token_piece": token["piece"],
                             "logprob": float(target["logprob"]),
                             "n_probs_used": scored["n_probs_requested"],
                             "returned_alternatives": len(top),
                             "distribution_prompt_sha256": sha256_text(prefix)})
        prefix += token["piece"]
    return {"logprob": sum(item["logprob"] for item in token_scores),
            "token_count": len(token_scores), "tokens": token_scores}


def _candidate_spec(trial: dict[str, Any]) -> tuple[str, str, str, str]:
    answer_format = trial["answer_format"]
    if answer_format == "category":
        return "GOOD", " GOOD", "BAD", " BAD"
    if answer_format == "welfare_category":
        return "GAIN", " GAIN", "LOSS", " LOSS"
    return "A", " A", "B", " B"


def score_trial(trial: dict[str, Any]) -> dict[str, Any]:
    first_name, first_text, second_name, second_text = _candidate_spec(trial)
    first_tokens = continuation_tokens(trial["prompt"], first_text)
    second_tokens = continuation_tokens(trial["prompt"], second_text)
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    first = _score_sequence(trial["prompt"], first_tokens, trial["seed"], cache)
    second = _score_sequence(trial["prompt"], second_tokens, trial["seed"], cache)
    margin = first["logprob"] - second["logprob"]
    top = next(iter(cache.values()))["probabilities"].get("top_logprobs") or []
    top20_mass = sum(math.exp(float(item["logprob"])) for item in top[:20])
    response = next(iter(cache.values()))["body"]
    generated = response.get("content", "")
    labels = _labels_for_trial(trial)
    record = {
        **trial, "status": "completed", "response": generated,
        "candidate_names": [first_name, second_name],
        "candidate_strings": [first_text, second_text],
        "candidate_tokenizations": {first_name: first_tokens, second_name: second_tokens},
        "candidate_scores": {first_name: first, second_name: second},
        "candidate_logprob_difference_first_minus_second": margin,
        "candidate_argmax": first_name if margin > 0 else second_name if margin < 0 else "tie",
        "defined_candidate_set_probability_mass": math.exp(first["logprob"]) + math.exp(second["logprob"]),
        "first_token_reported_top20_probability_mass": top20_mass,
        "first_token_unreported_tail_after_top20": max(0.0, 1.0 - top20_mass),
        "all_candidate_tokens_exactly_scored": True,
        "labels_and_outcomes": labels,
        "raw_first_distribution_head": top[:20],
        "raw_generation_response": response,
        "inference_settings": {"backend": "llama.cpp native /completion", "temperature": -1,
                               "probabilities": "softmax of raw logits; temp<0 greedy sampling, post_sampling_probs=false",
                               "top_k": 0, "top_p": 1, "min_p": 0, "typical_p": 1,
                               "repeat_penalty": 1, "frequency_penalty": 0, "presence_penalty": 0,
                               "initial_n_probs": INITIAL_N_PROBS, "full_fallback_n_probs": FULL_N_PROBS,
                               "seed": trial["seed"]},
    }
    if trial.get("elicitation") == "task_comprehension_diagnostic":
        record["diagnostic_correct_by_argmax"] = record["candidate_argmax"] == trial["expected_answer"]
        record["logprob_A_minus_B"] = margin
        record["label_margin_good_minus_bad"] = None
        record["welfare_margin_gain_minus_loss"] = None
    elif trial["alignment"] == "neutral":
        record["diagnostic_correct_by_argmax"] = None
        record["logprob_A_minus_B"] = margin if trial["answer_format"] == "letter" else None
        record["label_margin_good_minus_bad"] = None
        if trial["answer_format"] == "letter":
            gain_on_first = trial["option_order"] == "above_first"
            record["welfare_margin_gain_minus_loss"] = margin if gain_on_first else -margin
        else:
            record["welfare_margin_gain_minus_loss"] = margin
    elif trial["answer_format"] == "category":
        record["diagnostic_correct_by_argmax"] = None
        record["logprob_A_minus_B"] = None
        record["label_margin_good_minus_bad"] = margin
        record["welfare_margin_gain_minus_loss"] = margin if trial["alignment"] == "congruent" else -margin
    else:
        record["diagnostic_correct_by_argmax"] = None
        record["logprob_A_minus_B"] = margin
        good_on_a = (trial["good_side"] == "above") == (trial["option_order"] == "above_first")
        gain_on_a = (((trial["alignment"] == "congruent") == (trial["good_side"] == "above")) ==
                     (trial["option_order"] == "above_first"))
        record["label_margin_good_minus_bad"] = margin if good_on_a else -margin
        record["welfare_margin_gain_minus_loss"] = margin if gain_on_a else -margin
    return record


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group(elicitation: str, alignment: str, field: str) -> dict[str, Any]:
        values = [float(row[field]) for row in rows if row["elicitation"] == elicitation and
                  row["alignment"] == alignment and row.get(field) is not None]
        return {"n_distinct_prompts": len(values), "mean_logprob_margin": sum(values) / len(values) if values else None,
                "positive_n": sum(value > 0 for value in values), "negative_n": sum(value < 0 for value in values),
                "min_prompt_margin": min(values) if values else None, "max_prompt_margin": max(values) if values else None}
    return {
        "records": len(rows),
        "candidate_pairs_scored": sum(row.get("all_candidate_tokens_exactly_scored", False) for row in rows
                                       if row.get("elicitation") != "task_comprehension_diagnostic"),
        "logprob_margin": {
            "label_token_probability": {alignment: group("label_token_probability", alignment, "label_margin_good_minus_bad")
                                         for alignment in ("congruent", "inverted")},
            "letter_outcome_choice": {alignment: group("choice_answer_token_probability", alignment,
                                                        "welfare_margin_gain_minus_loss")
                                      for alignment in ("congruent", "inverted", "neutral")},
            "neutral_welfare_category": {"neutral": group("welfare_token_probability", "neutral",
                                                           "welfare_margin_gain_minus_loss")},
        },
        "generated_first_token_counts": {
            "letter_choice": {letter: sum(row.get("response", "").strip() == letter for row in rows
                                           if row["elicitation"] == "choice_answer_token_probability")
                              for letter in ("A", "B")},
            "label_category": {category: sum(row.get("response", "").strip() == category for row in rows
                                             if row["elicitation"] == "label_token_probability")
                              for category in ("GOOD", "BAD")},
            "neutral_category_first_token": {"G_or_fragment": sum(row.get("response", "").strip().startswith("G")
                                                                   for row in rows if row["elicitation"] == "welfare_token_probability"),
                                             "LOSS": sum(row.get("response", "").strip() == "LOSS"
                                                         for row in rows if row["elicitation"] == "welfare_token_probability")},
        },
        "comprehension": [{"task": row["alignment"], "ordering": row["wording"],
                           "expected": row["expected_answer"], "argmax": row["candidate_argmax"],
                           "correct": row["diagnostic_correct_by_argmax"]}
                          for row in rows if row["elicitation"] == "task_comprehension_diagnostic"],
        "answer_position_logprob_A_minus_B": {
            order: {"n": sum(row["option_order"] == order for row in rows
                              if row["elicitation"] == "choice_answer_token_probability"),
                    "mean": (sum(row["logprob_A_minus_B"] for row in rows if row["option_order"] == order and
                                 row["elicitation"] == "choice_answer_token_probability") /
                             sum(row["option_order"] == order for row in rows
                                 if row["elicitation"] == "choice_answer_token_probability"))
                    if any(row["option_order"] == order and row["elicitation"] == "choice_answer_token_probability"
                           for row in rows) else None}
            for order in ("above_first", "below_first")},
    }


def run(model_path: Path, model_name: str, checkpoint: str, quantization: str,
        output_dir: Path, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = _make_trials(seed) + _diagnostic_trials(seed)
    for trial in trials:
        trial["protocol_version"] = PROTOCOL
    random.Random(seed).shuffle(trials)
    core = {"schema_version": 1, "protocol_version": PROTOCOL, "model_name": model_name,
            "checkpoint": checkpoint, "quantization": quantization, "model_path": str(model_path),
            "seed": seed, "plan": {"letter_outcome_trials": 20, "direct_label_probability_trials": 16,
                                     "neutral_welfare_category_trials": 4, "comprehension_controls": 4,
                                     "distinct_prompt_repetitions": 1},
            "token_scoring": {"method": "sum exact conditional raw token logprobs for entire candidate continuation",
                              "tokenizer_endpoint": f"{BASE_URL}/tokenize", "initial_n_probs": INITIAL_N_PROBS,
                              "full_vocab_fallback_n_probs": FULL_N_PROBS,
                              "temperature": -1, "post_sampling_probs": False}, "trials": trials}
    manifest = {**core, "manifest_core_sha256": sha256_json(core), "created_at_utc": _now()}
    write_json(output_dir / "manifest.json", manifest)
    token_probe = {}
    for candidate in ("A", " A", "B", " B", "GOOD", " GOOD", "BAD", " BAD", "GAIN", " GAIN", "LOSS", " LOSS"):
        token_probe[candidate] = tokenize(candidate)
    write_json(output_dir / "tokenizer_probe.json", {"model_path": str(model_path),
                "vocabulary_size": FULL_N_PROBS, "candidates": token_probe,
                "method": "llama.cpp /tokenize; exact token ids and pieces retained"})
    rows = []
    for trial in trials:
        row = score_trial(trial)
        if trial.get("elicitation") == "task_comprehension_diagnostic":
            row["expected_answer"] = trial["expected_answer"]
        rows.append(row)
    raw_path = output_dir / "raw_probability_results.jsonl"
    with raw_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"protocol_version": PROTOCOL, "model_name": model_name, "checkpoint": checkpoint,
               "quantization": quantization, "seed": seed, "summary": _summary(rows),
               "raw_sha256": sha256_file(raw_path), "manifest_core_sha256": manifest["manifest_core_sha256"],
               "completed_at_utc": _now()}
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    result = run(args.model_path, args.model_name, args.checkpoint, args.quantization, args.output, args.seed)
    print(json.dumps({"model": result["model_name"], "records": result["summary"]["records"],
                      "candidate_pairs_scored": result["summary"]["candidate_pairs_scored"],
                      "raw_sha256": result["raw_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
