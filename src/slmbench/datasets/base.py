"""Common data structures shared by every dataset loader, extractor and
evaluator in the benchmark.

Keeping ONE normalized representation (`DocumentSample`) is what lets a
SROIE receipt, a FUNSD form and a DocILE invoice all flow through the exact
same extractor and evaluator code. Every dataset loader's only job is to
map its raw format into this shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentSample:
    """One document instance, normalized across datasets.

    Attributes:
        sample_id: Stable unique id, e.g. "sroie2019/x51005361883".
        dataset_id: Matches an entry in configs/datasets.yaml.
        task_id: Matches an entry in configs/tasks.yaml — determines the
            target schema.
        image_path: Path to the document image (page 1, if multi-page).
            Required for multimodal extractors.
        ocr_text: Pre-computed OCR text, if the dataset ships it (several
            do). If None, the text_ocr pipeline will compute it on the fly
            with the configured OCR engine.
        ground_truth: The reference structured extraction, shaped to match
            the JSON schema of `task_id` in configs/tasks.yaml.
        metadata: Anything extra worth keeping (source language, number of
            pages, original raw annotation) — not used for scoring.
    """

    sample_id: str
    dataset_id: str
    task_id: str
    image_path: Path
    ground_truth: dict[str, Any]
    ocr_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Output of running one extractor on one DocumentSample."""

    sample_id: str
    model_id: str
    family: str  # "multimodal" or "text_ocr"
    raw_output: str  # exact text returned by the model, pre-parsing
    parsed_output: dict[str, Any] | None  # None if JSON parsing failed
    latency_seconds: float
    json_valid: bool
    error: str | None = None
