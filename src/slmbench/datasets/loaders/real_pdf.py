"""Loader for real PDF documents.

Rasterizes page 1 of each PDF to a cached PNG and reads a sibling
ground-truth JSON. The ground truth is filtered down to the fields the
task's schema asks for, so the SAME PDFs + JSON can back several datasets
requesting different field subsets (e.g. a field-count curve). Point each
such dataset's data dir at the real one with a symlink.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from slmbench.datasets.base import DocumentSample
from slmbench.extraction.schema import get_schema


def load(
    raw_dir: Path,
    split: str,
    limit: int | None,
    dataset_id: str,
    task_id: str,
) -> list[DocumentSample]:
    schema_fields = set(get_schema(task_id)["properties"].keys())
    samples: list[DocumentSample] = []
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        stem = pdf_path.stem
        gt_path = raw_dir / f"{stem}.json"
        if not gt_path.exists():
            continue

        image_path = raw_dir / f"{stem}.png"
        if not image_path.exists():
            _render_first_page(pdf_path, image_path)

        full_gt = json.loads(gt_path.read_text(encoding="utf-8"))
        ground_truth = {k: v for k, v in full_gt.items() if k in schema_fields}

        samples.append(
            DocumentSample(
                sample_id=f"{dataset_id}/{stem}",
                dataset_id=dataset_id,
                task_id=task_id,
                image_path=image_path,
                ground_truth=ground_truth,
                metadata={"source_pdf": str(pdf_path)},
            )
        )
        if limit and len(samples) >= limit:
            break

    return samples


def _render_first_page(pdf_path: Path, out_path: Path, dpi: int = 200) -> None:
    doc = pymupdf.open(pdf_path)
    doc[0].get_pixmap(dpi=dpi).save(out_path)
    doc.close()
