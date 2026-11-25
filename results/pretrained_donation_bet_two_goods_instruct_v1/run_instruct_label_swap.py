"""Optional matched explicit GOOD/BAD control, templated for Qwen2.5-14B-Instruct."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE_LABEL_RESULTS = REPO / "results" / "pretrained_donation_bet_paired_label_swap_v1"
SERVER = Path(r"C:\Users\Main PC 2\AppData\Local\Temp\llama-b12606-vulkan\llama-server.exe")
MODEL = Path(r"C:\Users\Main PC 2\.ollama\models\blobs\sha256-2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54")
OUT = HERE / "explicit_label_swap_chat_template"
BASE_URL = "http://127.0.0.1:8093"


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(REPO / "src"))
    from value_leakage_experiment import native_probability_scoring as nps
    from value_leakage_experiment import paired_label_swap as label_swap
    scorer_path = HERE / "score_instruct_estimation_probabilities.py"
    spec = importlib.util.spec_from_file_location("instruct_exact_candidate_scorer", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(scorer)
    nps.BASE_URL = BASE_URL
    base_manifest_path = BASE_LABEL_RESULTS / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    template_record = json.loads((HERE / "chat_template_validation.json").read_text(encoding="utf-8"))
    template = template_record["rendered_text"]
    trials = []
    for original in base_manifest["conditions"]:
        trial = dict(original)
        trial["source_prompt_sha256"] = original.get("prompt_sha256")
        trial["prompt"] = template.replace("Template check", original["prompt"])
        trial["prompt_sha256"] = hashlib.sha256(trial["prompt"].encode()).hexdigest()
        trial["condition_prompt_sha256"] = trial["prompt_sha256"]
        trial["prompt_format"] = "llama.cpp-applied Qwen2.5-14B-Instruct chat template"
        trials.append(trial)
    OUT.mkdir(parents=True, exist_ok=True)
    core = {"protocol_version": "pretrained-donation-bet-paired-label-swap-v1-instruct-chat-template",
            "checkpoint": "Qwen/Qwen2.5-14B-Instruct via Ollama qwen2.5:14b-instruct-q4_K_M",
            "quantization": "Q4_K_M", "base_manifest_sha256": hashlib.sha256(base_manifest_path.read_bytes()).hexdigest(),
            "chat_template": template, "score_definition": "exact raw-logit logP(gain)-logP(loss) from single-symbol completions; chat answer tokens have no leading blank",
            "trials": trials}
    manifest = {**core, "manifest_core_sha256": hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}
    write_json(OUT / "manifest.json", manifest)

    stdout = (OUT / "llama_server.log").open("w", encoding="utf-8")
    stderr = (OUT / "llama_server_stderr.log").open("w", encoding="utf-8")
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
        rows = []
        for trial in trials:
            symbols = list(dict.fromkeys(trial["outcome_symbols"].values()))
            cache, scores = {}, {}
            for symbol in symbols:
                scored = scorer.score_full_candidate(trial["prompt"], symbol, int(trial["seed"]), nps, cache)
                scores[symbol] = {"candidate_text": symbol, **scored}
            outcomes = {name: scores[symbol]["logprob"] for name, symbol in trial["outcome_symbols"].items()}
            margin = outcomes["gain"] - outcomes["loss"]
            body = cache[(trial["prompt"], 5000)]["body"]
            row = {**trial, "status": "completed", "candidate_scores_by_symbol": scores,
                   "outcome_logprobs": outcomes, "logprob_gain_minus_loss": margin,
                   "gain_candidate_set_probability": 1 / (1 + math.exp(-margin)),
                   "greedy_first_token_text": body.get("content", ""),
                   "candidate_token_counts": {k: v["token_count"] for k, v in scores.items()},
                   "inference_settings": {"backend": "llama.cpp native /completion", "temperature": -1,
                                          "post_sampling_probs": False, "top_k": 0, "top_p": 1, "min_p": 0}}
            if trial.get("elicitation") == "task_comprehension_diagnostic":
                argmax = max(scores, key=lambda s: scores[s]["logprob"])
                row["argmax_symbol"] = argmax
                row["diagnostic_correct_by_argmax"] = argmax == trial["expected_symbol"]
            rows.append(row)
            (OUT / "raw_paired_probability_results.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        summary = label_swap._summarize(rows)
        summary.update({"model_name": "Qwen2.5-14B-Instruct-Q4_K_M", "checkpoint": core["checkpoint"],
                        "quantization": "Q4_K_M", "seed": base_manifest["seed"],
                        "manifest_core_sha256": manifest["manifest_core_sha256"],
                        "raw_sha256": hashlib.sha256((OUT / "raw_paired_probability_results.jsonl").read_bytes()).hexdigest(),
                        "answer_token_format": "bare symbol immediately after assistant newline; see Base raw-completion token format difference"})
        write_json(OUT / "summary.json", summary)
    finally:
        if server.poll() is None:
            server.terminate()
            try: server.wait(timeout=15)
            except subprocess.TimeoutExpired: server.kill(); server.wait()
        stdout.close(); stderr.close()


if __name__ == "__main__":
    main()
