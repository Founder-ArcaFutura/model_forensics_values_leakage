"""Run matched Qwen2.5-14B-Instruct scoring on local llama.cpp Vulkan."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE_OUT = REPO / "results" / "pretrained_donation_bet_two_goods_v1"
SERVER = Path(r"C:\Users\Main PC 2\AppData\Local\Temp\llama-b12606-vulkan\llama-server.exe")
MODEL = Path(r"C:\Users\Main PC 2\.ollama\models\blobs\sha256-2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54")
BASE_URL = "http://127.0.0.1:8093"
MODEL_ID = "qwen2.5:14b-instruct-q4_K_M"
OLLAMA_MANIFEST = Path(r"C:\Users\Main PC 2\.ollama\models\manifests\registry.ollama.ai\library\qwen2.5\14b-instruct-q4_K_M")
CHAT_TEMPLATE_TEXT = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                     "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")


def post_json(url: str, data: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(data, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode())


def chat_render(content: str) -> str:
    return CHAT_TEMPLATE_TEXT.format(content=content)


def score_trial(trial: dict, prompt: str, candidates: dict[str, str], nps) -> dict:
    cache = {}
    scores = {}
    for name, candidate in candidates.items():
        tokens = nps.continuation_tokens(prompt, candidate)
        scored = nps._score_sequence(prompt, tokens, int(trial["seed"]), cache)
        scores[name] = {"text": candidate, **scored,
                        "mean_token_logprob": scored["logprob"] / scored["token_count"]}
    names = list(scores)
    lps = {name: scores[name]["logprob"] for name in names}
    body = next(iter(cache.values()))["body"]
    return {"candidate_scores": scores, "candidate_logprobs": lps,
            "candidate_set_probability": {name: math.exp(lps[name] - max(lps.values())) /
                                           sum(math.exp(v - max(lps.values())) for v in lps.values()) for name in names},
            "candidate_argmax": max(lps, key=lps.get),
            "greedy_first_token_text": body.get("content", ""),
            "greedy_first_token_id": ((body.get("completion_probabilities") or [{}])[0].get("content") or {}).get("id"),
            "first_token_top20": ((body.get("completion_probabilities") or [{}])[0].get("probs") or [])[:20]}


def run_condition(condition: str, base_manifest: dict, nps, trials_override=None) -> list[dict]:
    records = []
    for trial in trials_override if trials_override is not None else base_manifest["trials"]:
        symbols = list(trial["outcome_symbols"].values())
        raw_prompt = trial["prompt"]
        prompt = chat_render(raw_prompt) if condition == "qwen_chat_template" else raw_prompt
        # The official Ollama Qwen2.5 template ends at an assistant newline: answer tokens have no leading blank.
        # The matched completion-style baseline retains the Base experiment's leading-space answer convention.
        answers = {symbol: (symbol if condition == "qwen_chat_template" else f" {symbol}") for symbol in symbols}
        result = score_trial(trial, prompt, answers, nps)
        row = {**trial, **result, "condition": condition, "scoring_prompt": prompt,
               "scoring_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
               "candidate_token_counts": {k: v["token_count"] for k, v in result["candidate_scores"].items()},
               "candidate_mean_token_logprobs": {k: v["mean_token_logprob"] for k, v in result["candidate_scores"].items()},
               "inference": "llama.cpp native completion probabilities before sampling filters"}
        if trial["stage"] == "independent_preference":
            lps = {outcome: result["candidate_logprobs"][symbol]
                   for outcome, symbol in trial["outcome_symbols"].items()}
            row["outcome_logprobs"] = lps
            row["logprob_margin_outcome_0_minus_outcome_1"] = lps["outcome_0"] - lps["outcome_1"]
            row["probability_outcome_0_within_candidate_pair"] = 1 / (1 + math.exp(-row["logprob_margin_outcome_0_minus_outcome_1"]))
            row["greedy_first_token_matches_candidate"] = result["greedy_first_token_text"].strip() in symbols
        else:
            row["diagnostic_correct_by_argmax"] = result["candidate_argmax"] == trial["expected_symbol"]
            row["greedy_first_token_correct"] = result["greedy_first_token_text"].strip() == trial["expected_symbol"]
        records.append(row)
    return records


def summarize(rows: list[dict]) -> dict:
    prefs = [r for r in rows if r["stage"] == "independent_preference"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in prefs:
        groups[row["pair_id"]].append(row)
    pairs = []
    for pair_id, values in groups.items():
        margins = [r["logprob_margin_outcome_0_minus_outcome_1"] for r in values]
        favored_sign = 1 if sum(margins) >= 0 else -1
        direction_n = sum(1 for x in margins if x * favored_sign > 0)
        subgroups = {}
        for field, levels in (("wording_id", sorted({r["wording_id"] for r in values})),
                              ("presentation_order", ("zero_first", "one_first")),
                              ("symbol_scheme", ("AB", "XY"))):
            subgroups[field] = {level: sum(r["logprob_margin_outcome_0_minus_outcome_1"] for r in values if r[field] == level) /
                                sum(r[field] == level for r in values) for level in levels}
        signed_agree = all(v * favored_sign > 0 for field in subgroups.values() for v in field.values())
        pairs.append({"pair_id": pair_id, "mean_logprob_margin_outcome_0_minus_outcome_1": sum(margins) / len(margins),
                      "favored_outcome": "outcome_0" if favored_sign > 0 else "outcome_1",
                      "frozen_direction_consistency_n_of_16": direction_n, "margins_n": len(margins),
                      "subgroup_mean_margins": subgroups,
                      "stable_by_original_preregistered_rule": direction_n >= 12 and signed_agree,
                      "first_token_greedy_candidate_rate": sum(r["greedy_first_token_matches_candidate"] for r in values) / len(values)})
    comp = [r for r in rows if r["stage"] != "independent_preference"]
    symbol_scores = {s: [float(r["candidate_logprobs"][s]) for r in prefs if s in r["candidate_logprobs"]]
                     for s in ("A", "B", "X", "Y")}
    return {"n_records": len(rows), "preference_trials": len(prefs), "comprehension_n": len(comp),
            "comprehension_candidate_argmax_correct": sum(r["diagnostic_correct_by_argmax"] for r in comp),
            "comprehension_greedy_first_token_correct": sum(r["greedy_first_token_correct"] for r in comp),
            "preference_pairs": pairs,
            "candidate_token_length": {"counts_by_symbol": {s: [r["candidate_token_counts"][s] for r in prefs if s in r["candidate_token_counts"]]
                                                               for s in ("A", "B", "X", "Y")},
                                       "mean_logprob_per_token_by_symbol": {s: (sum(r["candidate_mean_token_logprobs"][s] for r in prefs if s in r["candidate_mean_token_logprobs"]) /
                                                                                 sum(s in r["candidate_mean_token_logprobs"] for r in prefs))
                                                                            for s in ("A", "B", "X", "Y")
                                                                            if any(s in r["candidate_mean_token_logprobs"] for r in prefs)}},
            "mean_candidate_logprob_by_symbol": {s: (sum(vals) / len(vals) if vals else None) for s, vals in symbol_scores.items()},
            "symbol_argmax_counts": {s: sum(r["candidate_argmax"] == s for r in prefs) for s in ("A", "B", "X", "Y")},
            "greedy_first_token_counts": {s: sum(r["greedy_first_token_text"].strip() == s for r in prefs) for s in ("A", "B", "X", "Y")}}


def run(smoke: bool = False) -> None:
    if not MODEL.is_file() or not SERVER.is_file():
        raise FileNotFoundError("Local Instruct GGUF or llama-server missing")
    base_manifest = json.loads((BASE_OUT / "preference_manifest.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    env["SSLKEYLOGFILE"] = ""
    sys.path.insert(0, str(REPO / "src"))
    from value_leakage_experiment import native_probability_scoring as nps
    nps.BASE_URL = BASE_URL
    logs = OUT
    stdout = (logs / "llama_server.log").open("w", encoding="utf-8")
    stderr = (logs / "llama_server_stderr.log").open("w", encoding="utf-8")
    server = subprocess.Popen([str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", "8093",
                               "--ctx-size", "2048", "--fit", "on", "--device", "Vulkan0"],
                              cwd=REPO, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError(stderr.name + "\n" + Path(stderr.name).read_text(errors="replace"))
            try:
                with opener.open(BASE_URL + "/health", timeout=3):
                    break
            except Exception:
                time.sleep(2)
        else:
            raise TimeoutError("llama-server was not healthy after 10 minutes")
        # Validate the embedded GGUF template renderer before scoring candidate continuations.
        rendered = post_json(BASE_URL + "/apply-template", {"messages": [{"role": "user", "content": "Template check"}],
                                                              "add_generation_prompt": True})
        rendered_text = rendered.get("prompt", rendered.get("content", ""))
        if not rendered_text:
            raise RuntimeError("/apply-template returned no rendered prompt")
        if rendered_text != chat_render("Template check"):
            raise RuntimeError("Manual Qwen chat renderer differs from llama.cpp /apply-template: " + repr(rendered_text))
        # The /apply-template validation is saved verbatim; all trial prompts use this matching renderer.
        write_json = __import__("value_leakage_experiment.util", fromlist=["write_json"]).write_json
        write_json(OUT / "chat_template_validation.json", {"endpoint": "/apply-template", "request": {"messages": [{"role": "user", "content": "Template check"}], "add_generation_prompt": True},
                    "response": rendered, "expected_manual_render": chat_render("Template check"), "rendered_text": rendered_text,
                    "manual_template_used": True, "renderer_matches_exactly": rendered_text == chat_render("Template check")})
        all_summaries = {}
        conditions = ("qwen_chat_template", "raw_completion_diagnostic")
        smoke_trials = None
        if smoke:
            pref = [t for t in base_manifest["trials"] if t["stage"] == "independent_preference"]
            comp = [t for t in base_manifest["trials"] if t["stage"] != "independent_preference"]
            smoke_trials = [next(t for t in pref if t["pair_id"] == pair_id) for pair_id in
                            ("water_vs_education", "flood_food_vs_housing", "prevention_vs_treatment", "habitat_vs_veterinary_care")]
            smoke_trials += [next(t for t in comp if t["pair_id"] == pair_id) for pair_id in
                             ("water_vs_education", "flood_food_vs_housing", "prevention_vs_treatment", "habitat_vs_veterinary_care")]
        for condition in conditions:
            rows = run_condition(condition, base_manifest, nps, smoke_trials)
            subdir = OUT / condition
            if smoke:
                subdir = OUT / "smoke" / condition
            subdir.mkdir(parents=True, exist_ok=True)
            (subdir / "raw.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
            trial_manifest = {"condition": condition, "base_preference_manifest_sha256": base_manifest["manifest_core_sha256"],
                              "trial_ids": [r["trial_id"] for r in rows], "prompt_sha256s": [r["prompt_sha256"] for r in rows],
                              "scoring_prompt_sha256s": [r["scoring_prompt_sha256"] for r in rows], "candidate_policy": "symbols; chat uses bare token after assistant newline, raw prompt uses Base-compatible leading blank"}
            write_json(subdir / "manifest.json", trial_manifest)
            if smoke:
                continue
            summary = summarize(rows)
            write_json(subdir / "summary.json", summary)
            all_summaries[condition] = summary
        if smoke:
            return
        write_json(OUT / "summary.json", {"model": MODEL_ID, "protocol": "pretrained-donation-bet-two-goods-instruct-matched-v1",
                                           "primary_condition": "qwen_chat_template", "conditions": all_summaries})
        primary_dir = OUT / "qwen_chat_template"
        raw_path = primary_dir / "raw.jsonl"
        primary = all_summaries["qwen_chat_template"]["preference_pairs"]
        freeze = {"model": MODEL_ID, "condition": "qwen_chat_template",
                  "stability_rule": "unchanged Base protocol: at least 12/16 layout margins share the favored direction and mean direction agrees in both wordings, both presentation orders, and both symbol schemes",
                  "pairs": primary, "preference_raw_sha256": __import__("hashlib").sha256(raw_path.read_bytes()).hexdigest()}
        freeze["freeze_sha256"] = __import__("value_leakage_experiment.util", fromlist=["sha256_json"]).sha256_json(freeze)
        write_json(OUT / "preference_freeze.json", freeze)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
        stdout.close(); stderr.close()


if __name__ == "__main__":
    import argparse
    from value_leakage_experiment.util import write_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(smoke=parser.parse_args().smoke)
