"""
Local LLM runtime helpers.

Currently supports Ollama for low-latency sidecar intelligence. The startup
path preloads the configured model and pins it in memory with keep_alive=-1 so
later classifier/shaping calls do not pay model load latency.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import time
from typing import Any

import requests

import config

_log = logging.getLogger(__name__)
_serve_process: subprocess.Popen | None = None


def _base_url() -> str:
    return str(getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")


def _model() -> str:
    return str(getattr(config, "OLLAMA_MODEL", "qwen2.5:1.5b")).strip()


def enabled() -> bool:
    return (
        bool(getattr(config, "LOCAL_LLM_ENABLED", False))
        and str(getattr(config, "LOCAL_LLM_PROVIDER", "ollama")).lower() == "ollama"
        and bool(_model())
    )


def _server_ready(timeout_secs: float = 0.5) -> bool:
    try:
        resp = requests.get(_base_url(), timeout=timeout_secs)
        return resp.status_code < 500
    except requests.RequestException:
        return False


def _try_start_server() -> None:
    """Best-effort local Ollama start.

    On macOS the normal install is the Ollama.app background service. On other
    platforms, or if only the CLI exists, fall back to `ollama serve`.
    """
    global _serve_process
    if _server_ready():
        return

    if platform.system() == "Darwin":
        try:
            proc = subprocess.run(
                ["open", "-ga", "Ollama", "--args", "hidden"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if proc.returncode == 0:
                return
        except Exception as exc:
            _log.debug("[local_llm] open -ga Ollama failed: %s", exc)

    if shutil.which("ollama") and _serve_process is None:
        try:
            _serve_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            _log.debug("[local_llm] ollama serve failed: %s", exc)


def wait_for_server(timeout_secs: float | None = None) -> bool:
    if not enabled():
        return False
    timeout = (
        float(getattr(config, "OLLAMA_STARTUP_TIMEOUT_SECS", 30.0))
        if timeout_secs is None
        else float(timeout_secs)
    )
    _try_start_server()
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if _server_ready():
            return True
        time.sleep(0.25)
    return _server_ready()


def preload() -> bool:
    """Load the configured Ollama model and keep it resident."""
    if not enabled() or not bool(getattr(config, "OLLAMA_PRELOAD_ON_STARTUP", True)):
        return True

    model = _model()
    if not wait_for_server():
        _log.error("[local_llm] Ollama server not reachable at %s", _base_url())
        return False

    keep_alive: Any = getattr(config, "OLLAMA_KEEP_ALIVE", -1)
    started = time.monotonic()
    try:
        resp = requests.post(
            f"{_base_url()}/api/generate",
            json={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": keep_alive,
            },
            timeout=max(5.0, float(getattr(config, "OLLAMA_STARTUP_TIMEOUT_SECS", 30.0))),
        )
        resp.raise_for_status()
        payload = resp.json() if resp.content else {}
    except Exception as exc:
        _log.error("[local_llm] preload failed for %s: %s", model, exc)
        return False

    total = (time.monotonic() - started)
    load_ns = int(payload.get("load_duration") or 0) if isinstance(payload, dict) else 0
    _log.info(
        "[local_llm] preloaded %s in %.3fs (ollama_load=%.3fs keep_alive=%r)",
        model,
        total,
        load_ns / 1_000_000_000 if load_ns else 0.0,
        keep_alive,
    )
    return True


def generate(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 32,
    timeout_secs: float = 1.0,
) -> str:
    """Run a short non-streaming local generation request.

    Designed for sidecar classifiers, not long conversational replies.
    Raises on transport/server errors so callers can fall back cheaply.
    """
    if not enabled():
        raise RuntimeError("local LLM is disabled")
    if not wait_for_server(timeout_secs=min(0.5, max(0.0, timeout_secs))):
        raise RuntimeError(f"Ollama server not reachable at {_base_url()}")

    payload: dict[str, Any] = {
        "model": _model(),
        "prompt": prompt,
        "stream": False,
        "keep_alive": getattr(config, "OLLAMA_KEEP_ALIVE", -1),
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    if system:
        payload["system"] = system

    resp = requests.post(
        f"{_base_url()}/api/generate",
        json=payload,
        timeout=max(0.1, float(timeout_secs)),
    )
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    return str(data.get("response") or "").strip()


def unload() -> None:
    """Release the model at process shutdown."""
    if not enabled():
        return
    model = _model()
    try:
        requests.post(
            f"{_base_url()}/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=3.0,
        )
        _log.info("[local_llm] unloaded %s", model)
    except Exception as exc:
        _log.debug("[local_llm] unload failed for %s: %s", model, exc)
