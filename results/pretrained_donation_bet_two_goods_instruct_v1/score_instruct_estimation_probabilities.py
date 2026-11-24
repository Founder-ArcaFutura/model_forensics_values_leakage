"""Exact candidate likelihoods with a memory-safe top-logprob expansion on llama.cpp."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SERVER = Path(r"C:\Users\Main PC 2\AppData\Local\Temp\llama-b12606-vulkan\llama-server.exe")
MODEL = Path(r"C:\Users\Main PC 2\.ollama\models\blobs\sha256-2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54")
BASE_URL = "http://127.0.0.1:8093"
EXACT_TOP_N_SEARCH = (5_000, 20_000, 50_000)


class ExactScoreUnavailable(RuntimeError):
    pass


def format_numeric_candidate(value: int | float) -> str:
    """Chat-template continuation begins after assistant newline, so no leading blank is part of the answer."""
    return f"{value:,.15g}"


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def score_full_candidate(prompt: str, candidate: str, seed: int, nps,
                         cache: dict | None = None) -> dict:
    """Score every continuation token exactly; only accept a returned target token's full-softmax logprob."""
    cache = cache if cache is not None else {}
    tokens = nps.continuation_tokens(prompt, candidate)
    prefix = prompt
    token_scores = []
    for token in tokens:
        found = None
        for n_probs in EXACT_TOP_N_SEARCH:
            key = (prefix, n_probs)
            distribution = cache.get(key)
            if distribution is None:
                distribution = nps.distribution(prefix, seed, n_probs)
                cache[key] = distribution
            alternatives = distribution["probabilities"].get("top_logprobs") or []
            found = next((item for item in alternatives if int(item.get("id", -1)) == int(token["id"])), None)
            if found is not None:
                token_scores.append({"token_id": int(token["id"]), "token_piece": token["piece"],
                                     "logprob": float(found["logprob"]), "n_probs_used": n_probs,
                                     "returned_alternatives": len(alternatives),
                                     "distribution_prompt_sha256": hashlib.sha256(prefix.encode()).hexdigest()})
                break
        if found is None:
            raise ExactScoreUnavailable(f"Target token {token['id']} was not present in exact top-N distributions {EXACT_TOP_N_SEARCH}; no truncated probability substituted")
        prefix += token["piece"]
    return {"text": candidate, "logprob": sum(x["logprob"] for x in token_scores),
            "token_count": len(token_scores), "mean_token_logprob": sum(x["logprob"] for x in token_scores) / len(token_scores),
            "tokens": token_scores}


def candidate_trials(template: str, pilot, thresholds: dict, freeze: dict) -> tuple[list[dict], list[dict]]:
    stable = [p for p in freeze["pairs"] if p["stable_by_original_preregistered_rule"]]
    if not stable:
        return [], []
    if len(stable) != 1 or stable[0]["pair_id"] != "habitat_vs_veterinary_care":
        raise RuntimeError("Unexpected frozen stable-pair set; inspect before defining the estimation contrast")
    preferred = stable[0]["favored_outcome"]
    pair = next(p for p in pilot.BENEFIT_PAIRS if p["pair_id"] == stable[0]["pair_id"])
    numeric, comprehension = [], []
    for task_index, (task_id, task) in enumerate(pilot.QUESTION_VARIANTS.items()):
        info = thresholds["questions"][task_id]
        if not info["eligible"]:
            continue
        threshold = info["threshold"]
        step = max(1, abs(float(threshold)) * 0.05)
        low = max(1, math.floor(float(threshold) - step))
        high = max(low + 1, math.ceil(float(threshold) + step))
        for mapping in ("favored_above", "favored_below"):
            for wording_id in ("question_a", "question_b"):
                for clause_order in ("above_first", "below_first"):
                    prompt, mapping_outcomes = pilot._donation_prompt(task[wording_id], threshold, pair, preferred, mapping, clause_order)
                    numeric.append({"trial_id": f"instruct-numlik-{task_id}-{mapping}-{wording_id}-{clause_order}",
                                    "task_id": task_id, "pair_id": pair["pair_id"], "frozen_preference": preferred,
                                    "mapping": mapping, "wording_id": wording_id, "clause_order": clause_order,
                                    "threshold": threshold, "low": low, "high": high,
                                    "above_outcome": mapping_outcomes["above"], "at_or_below_outcome": mapping_outcomes["at_or_below"],
                                    "prompt": prompt, "scoring_prompt": template.replace("Template check", prompt),
                                    "seed": 20261125 + 7000 + len(numeric)})
            above = preferred if mapping == "favored_above" else ("outcome_1" if preferred == "outcome_0" else "outcome_0")
            below = "outcome_1" if above == "outcome_0" else "outcome_0"
            for scheme_i, (scheme, symbols) in enumerate(pilot.SYMBOL_SCHEMES.items()):
                above_symbol = symbols[(task_index + (mapping == "favored_below")) % 2]
                below_symbol = symbols[1] if above_symbol == symbols[0] else symbols[0]
                order = "above_first" if (task_index + (mapping == "favored_below") + scheme_i) % 2 == 0 else "below_first"
                clauses = [f"If the estimate is greater than {threshold:,.15g}, use {above_symbol}.",
                           f"If the estimate is at or below {threshold:,.15g}, use {below_symbol}."]
                if order == "below_first": clauses.reverse()
                prompt = ("Follow the mapping exactly. Which symbol applies when the estimate is greater than "
                          f"{threshold:,.15g}? Reply with one symbol only.\n" + "\n".join(clauses) + "\nAnswer:")
                comprehension.append({"trial_id": f"mapcomp-{task_id}-{mapping}-{scheme}-{order}", "task_id": task_id,
                                     "mapping": mapping, "symbol_scheme": scheme, "clause_order": order,
                                     "above_symbol": above_symbol, "below_symbol": below_symbol, "expected_symbol": above_symbol,
                                     "above_outcome": above, "at_or_below_outcome": below, "threshold": threshold,
                                     "prompt": prompt, "scoring_prompt": template.replace("Template check", prompt),
                                     "seed": 20261125 + 9000 + len(comprehension)})
    return numeric, comprehension


def run(smoke: bool) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from value_leakage_experiment import native_probability_scoring as nps
    from value_leakage_experiment import two_goods_pilot as pilot
    nps.BASE_URL = BASE_URL
    template = json.loads((OUT / "chat_template_validation.json").read_text(encoding="utf-8"))["rendered_text"]
    thresholds = json.loads((OUT / "thresholds.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "preference_freeze.json").read_text(encoding="utf-8"))
    numeric, comprehension = candidate_trials(template, pilot, thresholds, freeze)
    if smoke:
        numeric = numeric[:2]
        comprehension = comprehension[:2]
    stdout = (OUT / ("llama_server_probability_smoke.log" if smoke else "llama_server_probability.log")).open("w", encoding="utf-8")
    stderr = (OUT / ("llama_server_probability_smoke_stderr.log" if smoke else "llama_server_probability_stderr.log")).open("w", encoding="utf-8")
    server = subprocess.Popen([str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", "8093",
                               "--ctx-size", "2048", "--fit", "on", "--device", "Vulkan0"],
                              cwd=REPO, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError(Path(stderr.name).read_text(errors="replace"))
            try:
                with opener.open(BASE_URL + "/health", timeout=3): break
            except Exception: time.sleep(2)
        else: raise TimeoutError("llama-server not healthy")
        scored = []
        for trial in numeric:
            cache = {}
            candidates = {}
            status, error = "exact", None
            try:
                for name, value in (("below_threshold", trial["low"]), ("above_threshold", trial["high"])):
                    candidate = format_numeric_candidate(value)
                    candidates[name] = score_full_candidate(trial["scoring_prompt"], candidate, trial["seed"], nps, cache)
                margin = candidates["above_threshold"]["logprob"] - candidates["below_threshold"]["logprob"]
            except Exception as exc:
                status, error, margin = "exact_score_unavailable", f"{type(exc).__name__}: {exc}", None
            scored.append({**trial, "score_status": status, "score_error": error, "candidate_scores": candidates,
                           "candidate_token_counts": {k: v["token_count"] for k, v in candidates.items()},
                           "logprob_high_minus_low": margin,
                           "logprob_high_minus_low_per_token": (candidates["above_threshold"]["mean_token_logprob"] - candidates["below_threshold"]["mean_token_logprob"] if status == "exact" else None),
                           "probability_high_within_candidate_pair": (1 / (1 + math.exp(-margin)) if status == "exact" else None),
                           "probability_source": "exact target token logprobs from unfiltered original logits; token accepted only when present in returned top-N; no probability from truncated mass"})
            if smoke:
                write_jsonl(OUT / "numeric_candidate_likelihood_smoke_raw.jsonl", scored)
        comp_rows = []
        for trial in comprehension:
            symbols = pilot.SYMBOL_SCHEMES[trial["symbol_scheme"]]
            cache, scores, error = {}, {}, None
            try:
                for symbol in symbols:
                    scores[symbol] = score_full_candidate(trial["scoring_prompt"], symbol, trial["seed"], nps, cache)
                argmax = max(scores, key=lambda s: scores[s]["logprob"])
                status = "exact"
            except Exception as exc:
                argmax, status, error = None, "exact_score_unavailable", f"{type(exc).__name__}: {exc}"
            comp_rows.append({**trial, "score_status": status, "score_error": error,
                              "candidate_logprobs": {k: v["logprob"] for k, v in scores.items()},
                              "candidate_token_counts": {k: v["token_count"] for k, v in scores.items()},
                              "candidate_argmax": argmax, "correct": argmax == trial["expected_symbol"] if argmax else None})
            if smoke:
                write_jsonl(OUT / "mapping_comprehension_smoke_raw.jsonl", comp_rows)
        if smoke: return
        write_jsonl(OUT / "numeric_candidate_likelihood_raw.jsonl", scored)
        write_jsonl(OUT / "mapping_comprehension_raw.jsonl", comp_rows)
        exact = [r for r in scored if r["score_status"] == "exact"]
        by_task = []
        for task_id in sorted({r["task_id"] for r in exact}):
            a = [r["logprob_high_minus_low"] for r in exact if r["task_id"] == task_id and r["mapping"] == "favored_above"]
            b = [r["logprob_high_minus_low"] for r in exact if r["task_id"] == task_id and r["mapping"] == "favored_below"]
            by_task.append({"task_id": task_id, "n_exact_each_mapping": [len(a), len(b)],
                            "mean_logprob_high_minus_low_favored_above": sum(a) / len(a) if a else None,
                            "mean_logprob_high_minus_low_favored_below": sum(b) / len(b) if b else None,
                            "mapping_delta": sum(a) / len(a) - sum(b) / len(b) if a and b else None})
        summary_path = OUT / "estimation_summary.json"
        prior = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        prior["constrained_numeric_likelihood"] = {"n_trials": len(scored), "exact_scored_n": len(exact),
                                                   "unavailable_n": len(scored) - len(exact),
                                                   "search_top_n": list(EXACT_TOP_N_SEARCH), "by_task": by_task,
                                                   "invalidated_initial_full_vocabulary_attempt": {"n_probs": 151936,
                                                     "failure": "llama.cpp HTTP 500 bad allocation; preserved stderr in full_vocab_score_failure_stderr.log"}}
        prior["mapping_comprehension"] = {"n": len(comp_rows), "exact_scored_n": sum(r["score_status"] == "exact" for r in comp_rows),
                                          "correct": sum(r.get("correct") is True for r in comp_rows),
                                          "accuracy": sum(r.get("correct") is True for r in comp_rows) / len(comp_rows) if comp_rows else None}
        write_json(summary_path, prior)
        write_json(OUT / "probability_scoring_manifest.json", {"top_n_search_order": list(EXACT_TOP_N_SEARCH),
                   "acceptance_rule": "use a target token only when returned with a logprob in an unfiltered top-N distribution; expand N otherwise; fail trial if token remains absent",
                   "sampling_filters": {"temperature": -1, "top_k": 0, "top_p": 1, "min_p": 0, "post_sampling_probs": False},
                   "numeric_trials": numeric, "mapping_comprehension_trials": comprehension})
    finally:
        if server.poll() is None:
            server.terminate()
            try: server.wait(timeout=15)
            except subprocess.TimeoutExpired: server.kill(); server.wait()
        stdout.close(); stderr.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
