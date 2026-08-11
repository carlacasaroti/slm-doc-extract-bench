"""Orchestrates: load model config -> build extractor(s) -> run over a
dataset's samples -> yield ExtractionResult per (sample, model) pair.

This is the module the CLI (`slmbench run`) calls into. Kept separate from
cli.py so it's directly reusable from a notebook or a custom script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import yaml

from slmbench.datasets.base import DocumentSample, ExtractionResult
from slmbench.extraction.prompts.builder import build_prompt, build_prompt_with_ocr_hint
from slmbench.extraction.schema import get_schema
from slmbench.models.base import BaseExtractor
from slmbench.models.multimodal.extractor import MultimodalExtractor
from slmbench.models.text_ocr.extractor import TextOCRExtractor

MODELS_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


def _load_models_config() -> dict[str, Any]:
    with open(MODELS_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_extractors(
    multimodal_ids: list[str] | None = None,
    text_slm_ids: list[str] | None = None,
    ocr_engine_ids: list[str] | None = None,
) -> list[BaseExtractor]:
    """Build the set of extractors to benchmark.

    With no filters, builds every multimodal model plus every
    (ocr_engine x text_slm) combination declared in configs/models.yaml —
    which is what makes this a full side-by-side comparison by default.
    """
    cfg = _load_models_config()
    extractors: list[BaseExtractor] = []

    for entry in cfg["multimodal"]:
        if multimodal_ids and entry["id"] not in multimodal_ids:
            continue
        extractors.append(MultimodalExtractor.from_config(entry))

    text_ocr_cfg = cfg["text_ocr"]
    ocr_engines = text_ocr_cfg["ocr_engines"]
    slms = text_ocr_cfg["slms"]
    for ocr_entry in ocr_engines:
        if ocr_engine_ids and ocr_entry["id"] not in ocr_engine_ids:
            continue
        for slm_entry in slms:
            if text_slm_ids and slm_entry["id"] not in text_slm_ids:
                continue
            extractors.append(TextOCRExtractor.from_config(slm_entry, ocr_engine=ocr_entry["id"]))

    return extractors


def run_extractor_on_samples(
    extractor: BaseExtractor,
    samples: list[DocumentSample],
) -> Iterator[ExtractionResult]:
    for sample in samples:
        schema = get_schema(sample.task_id)
        prompt = (
            build_prompt(sample.task_id)
            if extractor.family == "multimodal"
            else build_prompt_with_ocr_hint(sample.task_id)
        )
        yield extractor.extract(
            image_path=sample.image_path,
            schema=schema,
            prompt=prompt,
            sample_id=sample.sample_id,
        )
