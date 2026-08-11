"""Abstract extractor interface.

Both families (multimodal VLM extractors and text+OCR extractors)
implement this same interface, which is what lets the pipeline and
evaluator treat them identically — the benchmark doesn't care *how* a
model reads a document, only that it returns a JSON-shaped guess.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from slmbench.datasets.base import ExtractionResult


class BaseExtractor(ABC):
    """Common contract for every extractor.

    Subclasses implement `_generate()`, which does the actual model call
    and returns the raw text response. `extract()` wraps it with timing,
    JSON parsing and error handling so subclasses stay focused on the
    model-calling logic.
    """

    family: str  # "multimodal" or "text_ocr" — set by subclass

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def _generate(self, image_path: Path, schema: dict[str, Any], prompt: str) -> str:
        """Call the underlying model and return its raw text output."""
        raise NotImplementedError

    def extract(
        self,
        image_path: Path,
        schema: dict[str, Any],
        prompt: str,
        sample_id: str,
    ) -> ExtractionResult:
        start = time.perf_counter()
        error: str | None = None
        raw_output = ""
        parsed: dict[str, Any] | None = None

        try:
            raw_output = self._generate(image_path=image_path, schema=schema, prompt=prompt)
            parsed = _extract_json(raw_output)
        except Exception as exc:  # noqa: BLE001 — we want every failure recorded, not raised
            error = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - start

        return ExtractionResult(
            sample_id=sample_id,
            model_id=self.model_id,
            family=self.family,
            raw_output=raw_output,
            parsed_output=parsed,
            latency_seconds=elapsed,
            json_valid=parsed is not None,
            error=error,
        )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from a model's raw text output.

    Models frequently wrap JSON in markdown fences or add a sentence
    before/after it. We try progressively looser strategies rather than
    failing the whole sample over formatting noise the model itself
    introduced.
    """
    text = text.strip()

    # 1. Straight parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences.
    if "```" in text:
        candidate = text.split("```")[1]
        candidate = candidate.removeprefix("json").strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3. Grab the outermost {...} block.
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(text[start_idx : end_idx + 1])
        except json.JSONDecodeError:
            pass

    return None
