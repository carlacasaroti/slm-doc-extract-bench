"""Anthropic (Claude) backend — cloud multimodal model as a ceiling reference.

Requires ANTHROPIC_API_KEY in the environment (or an `ant auth login` profile).
Only used by models.yaml entries with `backend: anthropic`.
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path

MAX_TOKENS = int(os.environ.get("SLMBENCH_ANTHROPIC_MAX_TOKENS", "8192"))
_MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}


@lru_cache(maxsize=1)
def _client():
    import anthropic

    return anthropic.Anthropic()


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text")


def generate_vision(model_id: str, prompt: str, image_path: Path, json_schema: dict | None = None) -> str:
    image_path = Path(image_path)
    media_type = _MEDIA_TYPES.get(image_path.suffix.lower(), "image/png")
    image_b64 = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    resp = _client().messages.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return _text(resp)


def generate_text(model_id: str, prompt: str, json_schema: dict | None = None) -> str:
    resp = _client().messages.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return _text(resp)
