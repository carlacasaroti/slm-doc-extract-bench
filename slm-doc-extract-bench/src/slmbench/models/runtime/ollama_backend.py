"""Thin client for a local Ollama server.

Requires `ollama serve` running (default http://localhost:11434) and the
target model already pulled, e.g.:

    ollama pull qwen2.5vl:3b
    ollama pull qwen2.5:3b

We call the raw HTTP API directly (instead of the `ollama` pip package)
to keep the dependency footprint small and make timeouts explicit.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_TIMEOUT = int(os.environ.get("SLMBENCH_OLLAMA_TIMEOUT", "300"))


def generate_vision(model_tag: str, prompt: str, image_path: Path, json_schema: dict | None = None) -> str:
    """Call an Ollama vision model with one image + prompt."""
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

    payload = {
        "model": model_tag,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    if json_schema:
        payload["format"] = json_schema

    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["response"]


def generate_text(model_tag: str, prompt: str, json_schema: dict | None = None) -> str:
    """Call an Ollama text-only model with a prompt (used for text_ocr SLMs)."""
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
    }
    if json_schema:
        payload["format"] = json_schema

    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["response"]
