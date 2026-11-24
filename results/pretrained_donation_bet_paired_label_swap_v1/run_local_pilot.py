"""Launch the local 14B Vulkan server, score the paired pilot, and stop the server."""

from __future__ import annotations

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
CHECKPOINT = "Qwen/Qwen2.5-14B@97e1e76335b7017d8f67c08a19d103c0504298c9"


def main() -> int:
    for path in (SERVER, MODEL):
        if not path.is_file():
            raise FileNotFoundError(f"Required local model file is unavailable: {path}")
    server_log = (OUTPUT / "llama_server.log").open("w", encoding="utf-8")
    server_error = (OUTPUT / "llama_server_stderr.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        [str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", "8081",
         "--ctx-size", "4096", "--n-gpu-layers", "99", "--device", "Vulkan0"],
        cwd=REPO,
        stdout=server_log,
        stderr=server_error,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    print(f"llama.cpp PID: {server.pid}", flush=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        while server.poll() is None:
            try:
                with opener.open("http://127.0.0.1:8081/health", timeout=3) as response:
                    print(f"GPU server health: {response.read().decode('utf-8')}", flush=True)
                    break
            except Exception:
                time.sleep(2)
        else:
            print((OUTPUT / "llama_server_stderr.log").read_text(encoding="utf-8", errors="replace"))
            raise RuntimeError("llama.cpp exited before becoming healthy")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO / "src")
        env["SSLKEYLOGFILE"] = ""
        command = [
            sys.executable, "-m", "value_leakage_experiment.paired_label_swap",
            "--model-path", str(MODEL),
            "--model-name", "Qwen2.5-14B-Base-Q4_K_M",
            "--checkpoint", CHECKPOINT,
            "--quantization", "Q4_K_M",
            "--output", str(OUTPUT),
            "--seed", "20261125",
        ]
        with (OUTPUT / "run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        print((OUTPUT / "run.log").read_text(encoding="utf-8", errors="replace"), flush=True)
        return result.returncode
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        server_log.close()
        server_error.close()
        print(f"llama.cpp exit code: {server.returncode}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
