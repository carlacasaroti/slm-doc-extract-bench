"""Multimodal extractor: the model reads the document image directly.

One extractor instance wraps one entry from configs/models.yaml's
`multimodal` list. The backend (ollama / transformers / anthropic) is
resolved from that config entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slmbench.models.base import BaseExtractor
from slmbench.models.runtime import anthropic_backend, ollama_backend, transformers_backend


class MultimodalExtractor(BaseExtractor):
    family = "multimodal"

    def __init__(self, model_id: str, backend: str, **backend_kwargs: Any):
        super().__init__(model_id)
        self.backend = backend
        self.backend_kwargs = backend_kwargs

    def _generate(self, image_path: Path, schema: dict[str, Any], prompt: str) -> str:
        if self.backend == "ollama":
            return ollama_backend.generate_vision(
                model_tag=self.backend_kwargs["ollama_tag"],
                prompt=prompt,
                image_path=image_path,
                json_schema=schema,
            )
        if self.backend == "transformers":
            return transformers_backend.generate_vision(
                hf_repo=self.backend_kwargs["hf_repo"],
                prompt=prompt,
                image_path=image_path,
            )
        if self.backend == "anthropic":
            return anthropic_backend.generate_vision(
                model_id=self.backend_kwargs["anthropic_model"],
                prompt=prompt,
                image_path=image_path,
                json_schema=schema,
            )
        raise ValueError(
            f"Unsupported backend '{self.backend}' for model '{self.model_id}'. "
            f"Supported: ollama, transformers, anthropic."
        )

    @classmethod
    def from_config(cls, entry: dict[str, Any]) -> MultimodalExtractor:
        """Build an extractor from one entry of configs/models.yaml['multimodal']."""
        kwargs = {k: v for k, v in entry.items() if k not in {"id", "backend", "params", "notes"}}
        return cls(model_id=entry["id"], backend=entry["backend"], **kwargs)
