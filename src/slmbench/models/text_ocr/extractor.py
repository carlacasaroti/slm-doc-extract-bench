"""Text+OCR extractor: OCR reads the document first, then a text-only SLM
extracts structured fields from the resulting text.

This is the classic pre-VLM pipeline and remains a fair, often-cheaper
baseline — the whole point of this benchmark is to measure, per document
domain, whether the extra cost of a multimodal model is actually justified
over this two-stage approach.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slmbench.models.base import BaseExtractor
from slmbench.models.ocr.engines import run_ocr
from slmbench.models.runtime import ollama_backend, transformers_backend


class TextOCRExtractor(BaseExtractor):
    family = "text_ocr"

    def __init__(self, model_id: str, backend: str, ocr_engine: str, **backend_kwargs: Any):
        super().__init__(model_id=f"{ocr_engine}+{model_id}")
        self.backend = backend
        self.ocr_engine = ocr_engine
        self.backend_kwargs = backend_kwargs

    def _generate(self, image_path: Path, schema: dict[str, Any], prompt: str) -> str:
        ocr_text = run_ocr(self.ocr_engine, image_path)
        full_prompt = f"{prompt}\n\n--- OCR TEXT ---\n{ocr_text}\n--- END OCR TEXT ---"

        if self.backend == "ollama":
            return ollama_backend.generate_text(
                model_tag=self.backend_kwargs["ollama_tag"],
                prompt=full_prompt,
                json_schema=schema,
            )
        if self.backend == "transformers":
            return transformers_backend.generate_text(
                hf_repo=self.backend_kwargs["hf_repo"],
                prompt=full_prompt,
            )
        raise ValueError(
            f"Unsupported backend '{self.backend}'. Supported: ollama, transformers."
        )

    @classmethod
    def from_config(cls, slm_entry: dict[str, Any], ocr_engine: str) -> TextOCRExtractor:
        """Build an extractor from one `slms` entry + one OCR engine id."""
        kwargs = {
            k: v for k, v in slm_entry.items() if k not in {"id", "backend", "params", "notes"}
        }
        return cls(model_id=slm_entry["id"], backend=slm_entry["backend"], ocr_engine=ocr_engine, **kwargs)
