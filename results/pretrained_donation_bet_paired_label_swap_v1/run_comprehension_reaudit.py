"""Run only the corrected label-mapping comprehension diagnostic on local Qwen."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
SERVER = Path(r"C:\Users\Main PC 2\AppData\Local\Temp\llama-b12606-vulkan\llama-server.exe")
MODEL = Path(r"C:\Users\Main PC 2\AppData\Local\Temp\Qwen2.5-14B.Q4_K_M.gguf")
sys.path.insert(0, str(ROOT / "src"))
from value_leakage_experiment.paired_label_swap import PROTOCOL, build_trials, score_trial  # noqa: E402
from value_leakage_experiment.util import sha256_file, sha256_json, write_json  # noqa: E402

SEED = 20261126


def main() -> int:
    for path in (SERVER, MODEL):
        if not path.is_file():
            raise FileNotFoundError(f"Required local model file is unavailable: {path}")
    diagnostics = [row for row in build_trials(SEED)
                   if row.get("elicitation") == "task_comprehension_diagnostic"]
    core = {
        "protocol_version": PROTOCOL,
        "purpose": "corrected comprehension-only re-audit; the original 8-item diagnostic was symbol-confounded",
        "model": "Qwen2.5-14B-Base-Q4_K_M",
        "checkpoint": "Qwen/Qwen2.5-14B@97e1e76335b7017d8f67c08a19d103c0504298c9",
        "quantization": "Q4_K_M",
        "backend": "llama.cpp b12606, Vulkan0, same local Radeon RX 9060 XT setup as primary run",
        "seed": SEED,
        "scoring": "full conditional candidate symbol likelihoods; greedy argmax over A/B or X/Y",
        "planned_trials": len(diagnostics),
        "trials": diagnostics,
    }
    manifest = {**core, "manifest_core_sha256": sha256_json(core)}
    write_json(OUTPUT / "comprehension_reaudit_manifest.json", manifest)
    server_out = (OUTPUT / "llama_server_comprehension_reaudit.log").open("w", encoding="utf-8")
    server_err = (OUTPUT / "llama_server_comprehension_reaudit_stderr.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        [str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", "8081",
         "--ctx-size", "4096", "--n-gpu-layers", "99", "--device", "Vulkan0"],
        cwd=ROOT, stdout=server_out, stderr=server_err,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        while server.poll() is None:
            try:
                with opener.open("http://127.0.0.1:8081/health", timeout=3):
                    break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError((OUTPUT / "llama_server_comprehension_reaudit_stderr.log").read_text(
                encoding="utf-8", errors="replace"))
        os.environ["PYTHONPATH"] = str(ROOT / "src")
        rows = [score_trial(trial) for trial in diagnostics]
        raw_path = OUTPUT / "comprehension_reaudit_raw.jsonl"
        with raw_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        expected_counts: dict[str, int] = {}
        correct = 0
        for row in rows:
            expected_counts[row["expected_symbol"]] = expected_counts.get(row["expected_symbol"], 0) + 1
            correct += row["diagnostic_correct_by_argmax"]
        summary = {
            "protocol_version": PROTOCOL,
            "manifest_core_sha256": manifest["manifest_core_sha256"],
            "raw_sha256": sha256_file(raw_path),
            "n": len(rows),
            "correct_by_candidate_argmax": correct,
            "accuracy": correct / len(rows) if rows else None,
            "expected_symbol_counts": expected_counts,
            "diagnostic_correct_by_argmax_counts": {
                str(symbol): sum(row["diagnostic_correct_by_argmax"] and row["argmax_symbol"] == symbol for row in rows)
                for symbol in sorted(expected_counts)
            },
            "independent_unit_note": "16 counterbalanced prompts test this mapping comprehension only; they are not independent model samples",
        }
        write_json(OUTPUT / "comprehension_reaudit_summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        server_out.close()
        server_err.close()


if __name__ == "__main__":
    raise SystemExit(main())
