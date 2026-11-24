"""Run one two-goods scoring phase against the local Radeon-backed server."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
SERVER = Path(r"C:\Users\Main PC 2\AppData\Local\Temp\llama-b12606-vulkan\llama-server.exe")
MODEL = Path(r"C:\Users\Main PC 2\AppData\Local\Temp\Qwen2.5-14B.Q4_K_M.gguf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preferences", "smoke", "estimate", "comprehension"), required=True)
    args = parser.parse_args()
    for path in (SERVER, MODEL):
        if not path.is_file():
            raise FileNotFoundError(f"Required local runtime file is missing: {path}")
    stdout = (OUTPUT / f"llama_server_{args.phase}.log").open("w", encoding="utf-8")
    stderr = (OUTPUT / f"llama_server_{args.phase}_stderr.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        [str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", "8081",
         "--ctx-size", "4096", "--n-gpu-layers", "99", "--device", "Vulkan0"],
        cwd=REPO, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    print(f"llama.cpp PID: {server.pid}", flush=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        while server.poll() is None:
            try:
                with opener.open("http://127.0.0.1:8081/health", timeout=3) as response:
                    print("GPU server health: " + response.read().decode("utf-8"), flush=True)
                    break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError((OUTPUT / f"llama_server_{args.phase}_stderr.log").read_text(encoding="utf-8", errors="replace"))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO / "src")
        env["SSLKEYLOGFILE"] = ""
        command = [sys.executable, "-m", "value_leakage_experiment.two_goods_pilot", args.phase,
                   "--output", str(OUTPUT), "--seed", "20261125"]
        with (OUTPUT / f"{args.phase}_phase.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        print((OUTPUT / f"{args.phase}_phase.log").read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
        return result.returncode
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        stdout.close()
        stderr.close()
        print(f"llama.cpp stopped (return code {server.returncode})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
