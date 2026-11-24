"""Run the scenario-sensitive habitat/veterinary probe on local Qwen Instruct."""
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
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SERVER = Path(r"C:\Users\Main PC 2\AppData\Local\Temp\llama-b12606-vulkan\llama-server.exe")
MODEL = Path(r"C:\Users\Main PC 2\.ollama\models\blobs\sha256-2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54")
BASE_URL = "http://127.0.0.1:8093"
CHAT_TEMPLATE_TEXT = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                      "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def chat_render(prompt: str) -> str:
    return CHAT_TEMPLATE_TEXT.format(content=prompt)


def prepare() -> dict[str, Any]:
    sys.path.insert(0, str(REPO / "src"))
    from value_leakage_experiment import ecological_preference_probe as probe
    base_path = REPO / "results" / "pretrained_donation_bet_two_goods_v1" / "preference_manifest.json"
    freeze_path = REPO / "results" / "pretrained_donation_bet_two_goods_instruct_v1" / "preference_freeze.json"
    prereg_path = OUT / "hypothesis_design.md"
    base_manifest, source_rows = probe.load_habitat_source_trials(base_path)
    prereg_hash = hashlib.sha256(prereg_path.read_bytes()).hexdigest()
    manifest = probe.make_protocol_manifest(base_manifest, source_rows, prereg_hash)
    manifest["model"] = "Qwen/Qwen2.5-14B-Instruct via qwen2.5:14b-instruct-q4_K_M"
    manifest["model_artifact_sha256"] = "2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54"
    manifest["chat_template"] = CHAT_TEMPLATE_TEXT
    for collection in ("trials", "comprehension_trials", "explanation_prompts"):
        for row in manifest[collection]:
            row["scoring_prompt"] = chat_render(row["prompt"])
            row["scoring_prompt_sha256"] = hashlib.sha256(row["scoring_prompt"].encode()).hexdigest()
    freeze = {
        "protocol": probe.PROTOCOL,
        "manifest_sha256": manifest["manifest_sha256"],
        "preregistration_sha256": prereg_hash,
        "source_instruct_preference_freeze_sha256": json.loads(freeze_path.read_text(encoding="utf-8"))["freeze_sha256"],
        "frozen_preference_stage_before_inference": True,
        "layout_counts": {"preference": len(manifest["trials"]), "comprehension": len(manifest["comprehension_trials"]),
                          "explanations": len(manifest["explanation_prompts"])},
    }
    freeze["freeze_sha256"] = probe.sha256_json(freeze)
    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUT / "manifest.json", manifest)
    _write_json(OUT / "preference_freeze.json", freeze)
    _write_json(OUT / "model_provenance.json", {
        "model": manifest["model"], "checkpoint_type": "public instruction-tuned checkpoint",
        "ollama_tag": "qwen2.5:14b-instruct-q4_K_M", "quantization": "Q4_K_M",
        "gguf_sha256": manifest["model_artifact_sha256"],
        "backend": "llama.cpp llama-server b12606, Vulkan", "device_expected_and_to_be_verified": "AMD Radeon RX 9060 XT",
        "server_args": ["--ctx-size", "2048", "--fit", "on", "--device", "Vulkan0"],
        "candidate_probability_source": "llama.cpp native /completion raw next-token logits; temp=-1, post_sampling_probs=false",
        "chat_template": CHAT_TEMPLATE_TEXT,
        "source_preferences_sha256": freeze["source_instruct_preference_freeze_sha256"],
    })
    return manifest


def _get_json(url: str) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def _score_row(row: dict[str, Any], nps, base_template: str) -> dict[str, Any]:
    prompt = row["scoring_prompt"]
    symbols = ("A", "B", "X", "Y") if row["stage"] == "contextual_preference_likelihood" else ("A", "B")
    if row["stage"] == "contextual_preference_likelihood":
        candidates = {symbol: symbol for symbol in (row["habitat_symbol"], row["veterinary_symbol"])}
    else:
        candidates = {"A": "A", "B": "B"}
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    scores = {}
    tokenizations = {}
    for name, text in candidates.items():
        tokens = nps.continuation_tokens(prompt, text)
        tokenizations[name] = tokens
        scored = nps._score_sequence(prompt, tokens, int(row["seed"]), cache)
        scores[name] = {"text": text, **scored,
                        "mean_token_logprob": scored["logprob"] / scored["token_count"]}
    lps = {name: data["logprob"] for name, data in scores.items()}
    names = list(scores)
    max_lp = max(lps.values())
    probabilities = {name: math.exp(lp - max_lp) / sum(math.exp(v - max_lp) for v in lps.values())
                     for name, lp in lps.items()}
    base = next(iter(cache.values()))["body"]
    response = base.get("content", "")
    record = {**row, "candidate_scores": scores, "candidate_tokenizations": tokenizations,
              "candidate_logprobs": lps, "candidate_set_probability": probabilities,
              "candidate_argmax": max(lps, key=lps.get), "greedy_response": response,
              "raw_first_distribution": (base.get("completion_probabilities") or [{}])[0],
              "scoring_source": "exact unfiltered token logprobs before sampling filters"}
    if row["stage"] == "contextual_preference_likelihood":
        h = lps[row["habitat_symbol"]]
        v = lps[row["veterinary_symbol"]]
        record["logprob_habitat_minus_veterinary"] = h - v
        record["conditional_probability_habitat_over_two_candidates"] = probabilities[row["habitat_symbol"]]
        record["candidate_token_counts"] = {k: value["token_count"] for k, value in scores.items()}
    else:
        record["candidate_token_counts"] = {k: value["token_count"] for k, value in scores.items()}
        record["correct_by_argmax"] = record["candidate_argmax"] == row["expected_symbol"]
    return record


def _generate_explanation(row: dict[str, Any], nps) -> dict[str, Any]:
    prompt = row["scoring_prompt"]
    generation = row["generation"]
    payload = {"prompt": prompt, "seed": int(row["seed"]), "n_predict": generation["max_tokens"],
               "temperature": generation["temperature"], "top_k": generation["top_k"],
               "top_p": generation["top_p"], "repeat_penalty": 1.0,
               "stop": ["<|im_end|>"], "stream": False, "n_probs": 0}
    body = nps._curl_json(BASE_URL + "/completion", payload)
    return {**row, "response_text": body.get("content", ""), "raw_response": body,
            "generation_parameters": {k: payload[k] for k in ("seed", "n_predict", "temperature", "top_k", "top_p", "repeat_penalty", "stop")}}


def extend_explanations() -> None:
    """Repeat only the qualitative rationale prompts with more completion room."""
    sys.path.insert(0, str(REPO / "src"))
    from value_leakage_experiment import ecological_preference_probe as probe
    from value_leakage_experiment import native_probability_scoring as nps
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    nps.BASE_URL = BASE_URL
    stdout = (OUT / "llama_server_explanations_extension.log").open("w", encoding="utf-8")
    stderr = (OUT / "llama_server_explanations_extension_stderr.log").open("w", encoding="utf-8")
    server = subprocess.Popen([str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", "8093",
                               "--ctx-size", "2048", "--fit", "on", "--device", "Vulkan0"],
                              cwd=REPO, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError((OUT / "llama_server_explanations_extension_stderr.log").read_text(errors="replace")[-8000:])
            try:
                _get_json(BASE_URL + "/health")
                break
            except Exception:
                time.sleep(2)
        else:
            raise TimeoutError("llama-server did not become healthy for explanation extension")
        rows = []
        extension_manifest = {"protocol": "pretrained-donation-bet-ecological-explanations-extension-v1",
                              "parent_manifest_sha256": manifest["manifest_sha256"],
                              "elicitation": "same five rationale prompts, separate completion-cap condition; not pooled with first pass",
                              "generation": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "max_tokens": 420},
                              "trials": []}
        for index, source in enumerate(manifest["explanation_prompts"]):
            row = {**source, "trial_id": source["trial_id"] + "::completion_extension",
                   "seed": int(source["seed"]) + 30000,
                   "generation": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "max_tokens": 420}}
            extension_manifest["trials"].append(row)
            rows.append(_generate_explanation(row, nps))
            print(f"explanation extension {index + 1}/5", flush=True)
        _write_json(OUT / "explanations_extension_manifest.json", extension_manifest)
        _write_jsonl(OUT / "explanations_extension_raw.jsonl", rows)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
        stdout.close(); stderr.close()


def run_brief_explanation_adaptation() -> None:
    """Collect concise complete rationales after the first instruction ran long."""
    sys.path.insert(0, str(REPO / "src"))
    from value_leakage_experiment import native_probability_scoring as nps
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    nps.BASE_URL = BASE_URL
    stdout = (OUT / "llama_server_brief_explanations.log").open("w", encoding="utf-8")
    stderr = (OUT / "llama_server_brief_explanations_stderr.log").open("w", encoding="utf-8")
    server = subprocess.Popen([str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", "8093",
                               "--ctx-size", "2048", "--fit", "on", "--device", "Vulkan0"],
                              cwd=REPO, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError((OUT / "llama_server_brief_explanations_stderr.log").read_text(errors="replace")[-8000:])
            try:
                _get_json(BASE_URL + "/health")
                break
            except Exception:
                time.sleep(2)
        else:
            raise TimeoutError("llama-server did not become healthy for brief explanations")
        rows = []
        brief_manifest = {"protocol": "pretrained-donation-bet-ecological-explanations-brief-v1",
                          "parent_manifest_sha256": manifest["manifest_sha256"],
                          "elicitation": "adapted short-response prompt after the original rationale prompts hit the 180 and 420 token caps",
                          "instruction": "Name one consideration favoring each program, then recommend one. Limit the whole response to 80 words.",
                          "generation": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "max_tokens": 180}, "trials": []}
        for index, source in enumerate(manifest["explanation_prompts"]):
            base = source["prompt"].split("What considerations", 1)[0] if source["scenario_id"] == "baseline" else source["prompt"].split("What considerations", 1)[0]
            prompt = base + "Name one consideration favoring each program, then recommend one. Limit the whole response to 80 words."
            row = {**source, "trial_id": source["trial_id"] + "::brief_adaptation",
                   "prompt": prompt, "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                   "scoring_prompt": chat_render(prompt), "scoring_prompt_sha256": __import__("hashlib").sha256(chat_render(prompt).encode()).hexdigest(),
                   "seed": int(source["seed"]) + 60000,
                   "generation": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "max_tokens": 180}}
            brief_manifest["trials"].append(row)
            rows.append(_generate_explanation(row, nps))
            print(f"brief explanation adaptation {index + 1}/5", flush=True)
        _write_json(OUT / "explanations_brief_adaptation_manifest.json", brief_manifest)
        _write_jsonl(OUT / "explanations_brief_adaptation_raw.jsonl", rows)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
        stdout.close(); stderr.close()


def execute() -> None:
    sys.path.insert(0, str(REPO / "src"))
    from value_leakage_experiment import ecological_preference_probe as probe
    from value_leakage_experiment import native_probability_scoring as nps
    manifest_path = OUT / "manifest.json"
    freeze_path = OUT / "preference_freeze.json"
    if not manifest_path.exists() or not freeze_path.exists():
        raise RuntimeError("Run --prepare-only first and review the frozen manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("Frozen manifest hash mismatch")
    prereg_path = OUT / "hypothesis_design.md"
    if hashlib.sha256(prereg_path.read_bytes()).hexdigest() != freeze["preregistration_sha256"]:
        raise RuntimeError("Preregisration changed after freezing")
    if not MODEL.is_file() or not SERVER.is_file():
        raise FileNotFoundError("Local Qwen Instruct GGUF or llama-server is missing")
    nps.BASE_URL = BASE_URL
    stdout = (OUT / "llama_server.log").open("w", encoding="utf-8")
    stderr = (OUT / "llama_server_stderr.log").open("w", encoding="utf-8")
    server = subprocess.Popen([str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", "8093",
                               "--ctx-size", "2048", "--fit", "on", "--device", "Vulkan0"],
                              cwd=REPO, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError((OUT / "llama_server_stderr.log").read_text(errors="replace")[-8000:])
            try:
                _get_json(BASE_URL + "/health")
                break
            except Exception:
                time.sleep(2)
        else:
            raise TimeoutError("llama-server did not become healthy")
        rendered = _post_json(BASE_URL + "/apply-template", {"messages": [{"role": "user", "content": "Template check"}],
                                                                "add_generation_prompt": True})
        expected = chat_render("Template check")
        actual = rendered.get("prompt", rendered.get("content", ""))
        _write_json(OUT / "chat_template_validation.json", {"endpoint": "/apply-template", "response": rendered,
                                                            "expected_manual_render": expected, "rendered_text": actual,
                                                            "matches_exactly": actual == expected})
        if actual != expected:
            raise RuntimeError("Local Qwen renderer does not match the tested template")
        # Stage order follows the preregistration: rationales first, then scoring.
        explanations = [_generate_explanation(row, nps) for row in manifest["explanation_prompts"]]
        _write_jsonl(OUT / "explanations_raw.jsonl", explanations)
        preference_rows = []
        for i, row in enumerate(manifest["trials"], start=1):
            preference_rows.append(_score_row(row, nps, CHAT_TEMPLATE_TEXT))
            if i % 8 == 0:
                print(f"preference {i}/{len(manifest['trials'])}", flush=True)
        comprehension_rows = []
        for i, row in enumerate(manifest["comprehension_trials"], start=1):
            comprehension_rows.append(_score_row(row, nps, CHAT_TEMPLATE_TEXT))
            if i % 4 == 0:
                print(f"comprehension {i}/{len(manifest['comprehension_trials'])}", flush=True)
        _write_jsonl(OUT / "preference_raw.jsonl", preference_rows)
        _write_jsonl(OUT / "comprehension_raw.jsonl", comprehension_rows)
        summary = probe.summarize_preferences(preference_rows + comprehension_rows)
        summary["manifest_sha256"] = manifest["manifest_sha256"]
        summary["preference_freeze_sha256"] = freeze["freeze_sha256"]
        summary["explanation_valid_count"] = sum(bool(r["response_text"].strip()) for r in explanations)
        summary["explanation_prompts"] = len(explanations)
        summary["raw_sha256"] = {
            name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            for name in ("explanations_raw.jsonl", "preference_raw.jsonl", "comprehension_raw.jsonl")}
        _write_json(OUT / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
        stdout.close(); stderr.close()


def analyze_existing() -> None:
    sys.path.insert(0, str(REPO / "src"))
    from value_leakage_experiment import ecological_preference_probe as probe
    preference = [json.loads(line) for line in (OUT / "preference_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    comprehension = [json.loads(line) for line in (OUT / "comprehension_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    explanations = [json.loads(line) for line in (OUT / "explanations_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = probe.summarize_preferences(preference + comprehension)
    summary["manifest_sha256"] = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))["manifest_sha256"]
    summary["preference_freeze_sha256"] = json.loads((OUT / "preference_freeze.json").read_text(encoding="utf-8"))["freeze_sha256"]
    summary["explanation_valid_count"] = sum(bool(row.get("response_text", "").strip()) for row in explanations)
    summary["explanation_prompts"] = len(explanations)
    summary["raw_sha256"] = {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                             for name in ("explanations_raw.jsonl", "preference_raw.jsonl", "comprehension_raw.jsonl")}
    _write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--extend-explanations", action="store_true")
    parser.add_argument("--brief-explanations", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        analyze_existing()
    elif args.extend_explanations:
        extend_explanations()
    elif args.brief_explanations:
        run_brief_explanation_adaptation()
    elif args.execute:
        execute()
    else:
        payload = prepare()
        print(json.dumps({"manifest_sha256": payload["manifest_sha256"],
                          "preference_trials": len(payload["trials"]),
                          "comprehension_trials": len(payload["comprehension_trials"]),
                          "explanations": len(payload["explanation_prompts"])}, indent=2))
