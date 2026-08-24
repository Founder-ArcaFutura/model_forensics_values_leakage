from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from pathlib import Path


class APIError(RuntimeError):
    pass


@dataclass(frozen=True)
class APISettings:
    model: str
    reasoning_effort: str | None
    temperature: float | None
    max_output_tokens: int
    request_timeout_seconds: int
    max_retries: int
    base_url: str
    api_key: str
    http_transport: str


def settings_from_config(config: dict[str, Any]) -> APISettings:
    key_env = config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(key_env, "")
    if not api_key:
        for env_path in (Path.cwd() / ".env.local", Path.cwd() / ".env"):
            if not env_path.exists():
                continue
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith(f"{key_env}="):
                    api_key = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                    break
            if api_key:
                break
    if not api_key:
        raise APIError(f"Required API key environment variable is not set: {key_env}")
    base_url_env = config.get("base_url_env", "OPENAI_BASE_URL")
    base_url = os.environ.get(base_url_env, "https://api.openai.com").rstrip("/")
    return APISettings(
        model=config["model"],
        reasoning_effort=config.get("reasoning_effort"),
        temperature=config.get("temperature"),
        max_output_tokens=int(config["max_output_tokens"]),
        request_timeout_seconds=int(config.get("request_timeout_seconds", 180)),
        max_retries=int(config.get("max_retries", 4)),
        base_url=base_url,
        api_key=api_key,
        http_transport=config.get("http_transport", "auto"),
    )


def _output_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                texts.append(content["text"])
    if not texts:
        raise APIError("API response contained no output_text message")
    return "\n".join(texts)


def _post_urllib(settings: APISettings, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url=f"{settings.base_url}/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=settings.request_timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_powershell(settings: APISettings, payload: dict[str, Any]) -> dict[str, Any]:
    if platform.system() != "Windows":
        raise APIError("PowerShell transport is only supported on Windows")
    script = r"""
$PayloadPath = $env:VL_EXPERIMENT_PAYLOAD_PATH
$Url = $env:VL_EXPERIMENT_URL
$TimeoutSeconds = [int]$env:VL_EXPERIMENT_TIMEOUT_SECONDS
$headers = @{ Authorization = "Bearer $env:VL_EXPERIMENT_API_KEY" }
try {
    $response = Invoke-WebRequest -Uri $Url -Method Post -Headers $headers -ContentType 'application/json' -InFile $PayloadPath -UseBasicParsing -TimeoutSec $TimeoutSeconds
    [Console]::Out.Write($response.Content)
} catch {
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
        [Console]::Error.Write($_.ErrorDetails.Message)
    } else {
        [Console]::Error.Write($_.Exception.Message)
    }
    exit 1
}
"""
    payload_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=False)
            payload_path = handle.name
        child_env = os.environ.copy()
        child_env["VL_EXPERIMENT_API_KEY"] = settings.api_key
        child_env["VL_EXPERIMENT_PAYLOAD_PATH"] = payload_path
        child_env["VL_EXPERIMENT_URL"] = f"{settings.base_url}/v1/responses"
        child_env["VL_EXPERIMENT_TIMEOUT_SECONDS"] = str(settings.request_timeout_seconds)
        process = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=settings.request_timeout_seconds + 30,
            env=child_env,
            check=False,
        )
        if process.returncode != 0:
            raise APIError(f"PowerShell HTTP transport failed: {process.stderr[:1000]}")
        return json.loads(process.stdout)
    finally:
        if payload_path:
            try:
                os.unlink(payload_path)
            except FileNotFoundError:
                pass


def create_response(settings: APISettings, input_value: Any) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": settings.model,
        "input": input_value,
        "max_output_tokens": settings.max_output_tokens,
        "store": False,
    }
    if settings.reasoning_effort:
        payload["reasoning"] = {"effort": settings.reasoning_effort}
    if settings.temperature is not None:
        payload["temperature"] = settings.temperature

    last_error: Exception | None = None
    retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
    for attempt in range(settings.max_retries + 1):
        try:
            transport = settings.http_transport
            if transport == "auto":
                transport = "powershell" if platform.system() == "Windows" else "urllib"
            if transport == "powershell":
                body = _post_powershell(settings, payload)
            elif transport == "urllib":
                body = _post_urllib(settings, payload)
            else:
                raise APIError(f"Unknown HTTP transport: {transport}")
            return _output_text(body), body
        except urllib.error.HTTPError as exc:
            safe_body = exc.read().decode("utf-8", errors="replace")[:1000]
            last_error = APIError(f"HTTP {exc.code} from Responses API: {safe_body}")
            if exc.code not in retryable_statuses:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, APIError) as exc:
            last_error = exc
        if attempt < settings.max_retries:
            delay = min(30.0, (2**attempt) + random.random())
            time.sleep(delay)
    raise APIError(str(last_error) if last_error else "Unknown Responses API error")
