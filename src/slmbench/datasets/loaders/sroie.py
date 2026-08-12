"""Loader for SROIE 2019 (Scanned Receipts OCR and Information Extraction).

Expected raw layout after scripts/download_datasets.py (matches the
official RRC release):

    data/raw/sroie2019/
        img/            *.jpg
        entities/       *.txt   (JSON: {"company", "date", "address", "total"})

Register at https://rrc.cvc.uab.es/?ch=13 to obtain the data — the download
script cannot fetch this automatically, see docs/ADDING_A_DATASET.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from slmbench.datasets.base import DocumentSample


def load(
    raw_dir: Path,
    split: str,
    limit: int | None,
    dataset_id: str,
    task_id: str,
) -> list[DocumentSample]:
    img_dir = raw_dir / "img"
    entities_dir = raw_dir / "entities"

    if not img_dir.exists() or not entities_dir.exists():
        raise FileNotFoundError(
            f"Expected {img_dir} and {entities_dir}. See the docstring of "
            f"this file / docs/ADDING_A_DATASET.md for the expected layout."
        )

    samples: list[DocumentSample] = []
    for entity_file in sorted(entities_dir.glob("*.txt")):
        stem = entity_file.stem
        image_path = img_dir / f"{stem}.jpg"
        if not image_path.exists():
            continue

        with open(entity_file, encoding="utf-8") as f:
            raw_gt = json.load(f)

        ground_truth = {
            "company_name": raw_gt.get("company", ""),
            "address": raw_gt.get("address", ""),
            "date": raw_gt.get("date", ""),
            "total": _to_number(raw_gt.get("total", "")),
        }

        samples.append(
            DocumentSample(
                sample_id=f"{dataset_id}/{stem}",
                dataset_id=dataset_id,
                task_id=task_id,
                image_path=image_path,
                ground_truth=ground_truth,
            )
        )
        if limit and len(samples) >= limit:
            break

    return samples


def _to_number(value: str) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None
