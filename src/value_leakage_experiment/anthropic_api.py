from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .api import APIError


@dataclass(frozen=True)
class AnthropicSettings:
    model: str
    max_tokens: int
    thinking: dict[str, Any] | None
    output_config: dict[str, Any] | None
    request_timeout_seconds: int
    base_url: str
    api_key: str
    anthropic_version: str
    http_transport: str


def _load_secret(name: str, search_dir: Path) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    for path in (search_dir / ".env.local", search_dir / ".env"):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, candidate = stripped.split("=", 1)
                if key.strip() == name:
                    return candidate.strip().strip('"').strip("'")
    raise APIError(f"Required API key is not available in the process environment or .env.local: {name}")


def anthropic_settings_from_config(config: dict[str, Any], search_dir: Path) -> AnthropicSettings:
    key_env = config.get("api_key_env", "ANTHROPIC_API_KEY")
    return AnthropicSettings(
        model=config["model"],
        max_tokens=int(config["max_output_tokens"]),
        thinking=config.get("thinking"),
        output_config=config.get("output_config"),
        request_timeout_seconds=int(config.get("request_timeout_seconds", 180)),
        base_url=os.environ.get(config.get("base_url_env", "ANTHROPIC_BASE_URL"), "https://api.anthropic.com").rstrip("/"),
        api_key=_load_secret(key_env, search_dir),
        anthropic_version=config.get("anthropic_version", "2023-06-01"),
        http_transport=config.get("http_transport", "auto"),
    )


def _urllib_request(
    settings: AnthropicSettings,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
) -> str:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url=f"{settings.base_url}{path}",
        data=data,
        headers={
            "x-api-key": settings.api_key,
            "anthropic-version": settings.anthropic_version,
            "content-type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(request, timeout=settings.request_timeout_seconds) as response:
        return response.read().decode("utf-8")


def _powershell_request(
    settings: AnthropicSettings,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
) -> str:
    if platform.system() != "Windows":
        raise APIError("PowerShell transport is only supported on Windows")
    script = r"""
$headers = @{
    'x-api-key' = $env:VL_ANTHROPIC_API_KEY
    'anthropic-version' = $env:VL_ANTHROPIC_VERSION
}
$params = @{
    Uri = $env:VL_ANTHROPIC_URL
    Method = $env:VL_ANTHROPIC_METHOD
    Headers = $headers
    UseBasicParsing = $true
    TimeoutSec = [int]$env:VL_ANTHROPIC_TIMEOUT_SECONDS
}
if ($env:VL_ANTHROPIC_PAYLOAD_PATH) {
    $params.ContentType = 'application/json'
    $params.InFile = $env:VL_ANTHROPIC_PAYLOAD_PATH
}
try {
    $response = Invoke-WebRequest @params
    if ($response.Content -is [byte[]]) {
        [Console]::Out.Write([System.Text.Encoding]::UTF8.GetString($response.Content))
    } else {
        [Console]::Out.Write($response.Content)
    }
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
        if payload is not None:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as handle:
                json.dump(payload, handle, ensure_ascii=False)
                payload_path = handle.name
        child_env = os.environ.copy()
        child_env["VL_ANTHROPIC_API_KEY"] = settings.api_key
        child_env["VL_ANTHROPIC_VERSION"] = settings.anthropic_version
        child_env["VL_ANTHROPIC_URL"] = f"{settings.base_url}{path}"
        child_env["VL_ANTHROPIC_METHOD"] = method
        child_env["VL_ANTHROPIC_TIMEOUT_SECONDS"] = str(settings.request_timeout_seconds)
        child_env["VL_ANTHROPIC_PAYLOAD_PATH"] = payload_path or ""
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
            raise APIError(f"Anthropic HTTP request failed: {process.stderr[:1000]}")
        return process.stdout
    finally:
        if payload_path:
            try:
                os.unlink(payload_path)
            except FileNotFoundError:
                pass


def anthropic_request_text(
    settings: AnthropicSettings,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> str:
    transport = settings.http_transport
    if transport == "auto":
        transport = "powershell" if platform.system() == "Windows" else "urllib"
    if transport == "powershell":
        return _powershell_request(settings, method, path, payload)
    if transport == "urllib":
        return _urllib_request(settings, method, path, payload)
    raise APIError(f"Unknown HTTP transport: {transport}")


def anthropic_request_json(
    settings: AnthropicSettings,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return json.loads(anthropic_request_text(settings, method, path, payload))


def message_params(
    settings: AnthropicSettings,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "messages": messages,
    }
    if settings.thinking:
        params["thinking"] = settings.thinking
    if settings.output_config:
        params["output_config"] = settings.output_config
    return params
