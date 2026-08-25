"""Thin client for a local Ollama server.

Requires `ollama serve` running (default http://localhost:11434) and the
target model already pulled, e.g.:

    ollama pull qwen2.5vl:3b
    ollama pull qwen2.5:3b

We call the raw HTTP API directly (instead of the `ollama` pip package)
to keep the dependency footprint small and make timeouts explicit.

Two knobs matter for full-page documents on CPU:
  * num_ctx: vision models encode the image as tokens, so a full-page
    invoice can overflow Ollama's default 4096 context and 400 with
    "exceeds the available context size". We raise it (SLMBENCH_OLLAMA_NUM_CTX).
  * image size: a 200 dpi page is far more detail than a VLM needs and makes
    prefill painfully slow on CPU. We downscale the longest side to
    SLMBENCH_VLM_MAX_DIM before sending — this only affects the image handed
    to the VLM, not the on-disk PNG the OCR engines read.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import requests
from PIL import Image

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_TIMEOUT = int(os.environ.get("SLMBENCH_OLLAMA_TIMEOUT", "300"))
NUM_CTX = int(os.environ.get("SLMBENCH_OLLAMA_NUM_CTX", "8192"))
VLM_MAX_DIM = int(os.environ.get("SLMBENCH_VLM_MAX_DIM", "1400"))


def _encode_image(image_path: Path) -> str:
    """Downscale (longest side -> VLM_MAX_DIM) and base64-encode as PNG."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        if max(im.size) > VLM_MAX_DIM:
            im.thumbnail((VLM_MAX_DIM, VLM_MAX_DIM))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_vision(model_tag: str, prompt: str, image_path: Path, json_schema: dict | None = None) -> str:
    """Call an Ollama vision model with one image + prompt."""
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "images": [_encode_image(Path(image_path))],
        "stream": False,
        "options": {"num_ctx": NUM_CTX, "temperature": 0},
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
        "options": {"num_ctx": NUM_CTX, "temperature": 0},
    }
    if json_schema:
        payload["format"] = json_schema

    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["response"]
