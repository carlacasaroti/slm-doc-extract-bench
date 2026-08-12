"""Builds the extraction prompt for a given task.

Keeping ONE prompt builder (rather than per-model prompts) is intentional:
it keeps the comparison fair. If a specific model needs different phrasing
to perform well, that's a real, reportable finding — not something to
paper over by hand-tuning its prompt.
"""

from __future__ import annotations

import json

from slmbench.extraction.schema import get_task

_TEMPLATE = """You are a precise document information extraction system.

Task: {description}

Return ONLY a single JSON object matching this schema — no explanation,
no markdown fences, no extra text before or after the JSON:

{schema}

If a field is not present in the document, use null rather than guessing.
Numbers must be plain numbers (no currency symbols, no thousands separators).
"""


def build_prompt(task_id: str) -> str:
    task = get_task(task_id)
    return _TEMPLATE.format(
        description=task["description"],
        schema=json.dumps(task["schema"], indent=2, ensure_ascii=False),
    )


def build_prompt_with_ocr_hint(task_id: str) -> str:
    """Same prompt, phrased for the text_ocr family which reads noisy OCR
    text rather than the image itself — worth a small phrasing nudge since
    OCR text has no visual layout cues."""
    base = build_prompt(task_id)
    return base + (
        "\nThe text below was produced by OCR and may contain recognition "
        "errors, missing line breaks, or garbled characters — use context "
        "to infer the most likely correct value.\n"
    )
