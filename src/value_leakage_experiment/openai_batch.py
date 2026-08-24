from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .api import APIError, APISettings
from .util import canonical_json


_POWERSHELL_HTTP = r"""
Add-Type -AssemblyName System.Net.Http
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds([int]$env:VL_OPENAI_TIMEOUT)
$client.DefaultRequestHeaders.Authorization = [System.Net.Http.Headers.AuthenticationHeaderValue]::new("Bearer", $env:VL_OPENAI_KEY)
$url = $env:VL_OPENAI_URL
$method = $env:VL_OPENAI_METHOD
$bodyPath = $env:VL_OPENAI_BODY_PATH
$uploadPath = $env:VL_OPENAI_UPLOAD_PATH
try {
    if ($uploadPath) {
        $multipart = [System.Net.Http.MultipartFormDataContent]::new()
        $stream = [IO.File]::OpenRead($uploadPath)
        $fileContent = [System.Net.Http.StreamContent]::new($stream)
        $fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::new("application/jsonl")
        $multipart.Add($fileContent, "file", [IO.Path]::GetFileName($uploadPath))
        $multipart.Add([System.Net.Http.StringContent]::new("batch"), "purpose")
        $response = $client.PostAsync($url, $multipart).Result
    } elseif ($method -eq "POST") {
        $json = [IO.File]::ReadAllText($bodyPath, [Text.Encoding]::UTF8)
        $content = [System.Net.Http.StringContent]::new($json, [Text.Encoding]::UTF8, "application/json")
        $response = $client.PostAsync($url, $content).Result
    } else {
        $response = $client.GetAsync($url).Result
    }
    $text = $response.Content.ReadAsStringAsync().Result
    if (-not $response.IsSuccessStatusCode) {
        [Console]::Error.Write($text)
        exit 1
    }
    [Console]::Out.Write($text)
} finally {
    if ($stream) { $stream.Dispose() }
    if ($multipart) { $multipart.Dispose() }
    $client.Dispose()
}
"""


def _request(settings: APISettings, method: str, path: str, body: dict[str, Any] | None = None, upload: Path | None = None) -> str:
    if platform.system() != "Windows":
        raise APIError("OpenAI batch transport currently requires Windows PowerShell")
    body_path: str | None = None
    try:
        if body is not None:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as handle:
                json.dump(body, handle, ensure_ascii=False)
                body_path = handle.name
        child_env = os.environ.copy()
        child_env.update({
            "VL_OPENAI_KEY": settings.api_key,
            "VL_OPENAI_URL": f"{settings.base_url}{path}",
            "VL_OPENAI_METHOD": method,
            "VL_OPENAI_TIMEOUT": str(settings.request_timeout_seconds),
            "VL_OPENAI_BODY_PATH": body_path or "",
            "VL_OPENAI_UPLOAD_PATH": str(upload.resolve()) if upload else "",
        })
        process = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", _POWERSHELL_HTTP],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=settings.request_timeout_seconds + 30,
            env=child_env,
            check=False,
        )
        if process.returncode != 0:
            raise APIError(f"OpenAI batch transport failed: {process.stderr[:1500]}")
        return process.stdout
    finally:
        if body_path:
            try:
                os.unlink(body_path)
            except FileNotFoundError:
                pass


def request_json(settings: APISettings, method: str, path: str, body: dict[str, Any] | None = None, upload: Path | None = None) -> dict[str, Any]:
    return json.loads(_request(settings, method, path, body, upload))


def request_text(settings: APISettings, path: str) -> str:
    return _request(settings, "GET", path)


def response_body(settings: APISettings, prompt: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": settings.model,
        "input": prompt,
        "max_output_tokens": settings.max_output_tokens,
        "store": False,
    }
    if settings.reasoning_effort:
        body["reasoning"] = {"effort": settings.reasoning_effort}
    if settings.temperature is not None:
        body["temperature"] = settings.temperature
    return body


def write_batch_input(path: Path, requests: list[tuple[str, str]], settings: APISettings) -> None:
    path.write_text(
        "".join(
            canonical_json({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": response_body(settings, prompt),
            }) + "\n"
            for custom_id, prompt in requests
        ),
        encoding="utf-8",
        newline="\n",
    )


def submit_batch(settings: APISettings, input_path: Path, description: str) -> dict[str, Any]:
    uploaded = request_json(settings, "POST", "/v1/files", upload=input_path)
    batch = request_json(settings, "POST", "/v1/batches", {
        "input_file_id": uploaded["id"],
        "endpoint": "/v1/responses",
        "completion_window": "24h",
        "metadata": {"description": description[:512]},
    })
    return {"input_file": uploaded, "batch": batch}


def parse_output(text: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = row.get("custom_id")
        if not custom_id or custom_id in rows:
            raise ValueError(f"Missing or duplicate custom_id at output line {line_number}")
        rows[custom_id] = row
    return rows


def output_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts)
