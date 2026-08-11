"""Loader for CORD (Consolidated Receipt Dataset).

Expected raw layout (matches the official clovaai/cord release):

    data/raw/cord/
        image/          *.png
        json/           *.json   (CORD's own annotation schema)

CORD's ground-truth schema nests items under gt_parse -> menu -> {nm, cnt,
price, ...} plus a "sub_total" / "total" block. We map that into our
`receipt_line_items` schema.
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
) -> list[DocumentSample]:
    json_dir = raw_dir / "json"
    image_dir = raw_dir / "image"
    if not json_dir.exists() or not image_dir.exists():
        raise FileNotFoundError(
            f"Expected {json_dir} and {image_dir}. See docs/ADDING_A_DATASET.md."
        )

    samples: list[DocumentSample] = []
    for ann_path in sorted(json_dir.glob("*.json")):
        stem = ann_path.stem
        image_path = image_dir / f"{stem}.png"
        if not image_path.exists():
            continue

        with open(ann_path, encoding="utf-8") as f:
            raw = json.load(f)

        gt_parse = raw.get("gt_parse", raw)  # some dumps skip the wrapper
        menu = gt_parse.get("menu", [])
        if isinstance(menu, dict):
            menu = [menu]

        items = []
        for entry in menu:
            items.append(
                {
                    "name": _first(entry.get("nm")),
                    "quantity": _to_number(_first(entry.get("cnt"))),
                    "unit_price": _to_number(_first(entry.get("unitprice"))),
                    "total_price": _to_number(_first(entry.get("price"))),
                }
            )

        total_block = gt_parse.get("total", {})
        ground_truth = {
            "store_name": _first(gt_parse.get("store_name")) or "",
            "date": _first(gt_parse.get("date")) or "",
            "items": items,
            "subtotal": _to_number(_first(gt_parse.get("subtotal", {}).get("subtotal_price"))),
            "tax": _to_number(_first(total_block.get("tax_price"))),
            "total": _to_number(_first(total_block.get("total_price"))),
        }

        samples.append(
            DocumentSample(
                sample_id=f"{dataset_id}/{stem}",
                dataset_id=dataset_id,
                task_id="receipt_line_items",
                image_path=image_path,
                ground_truth=ground_truth,
            )
        )
        if limit and len(samples) >= limit:
            break

    return samples


def _first(value):
    """CORD sometimes stores a field as a list of one dict; unwrap it."""
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        return value.get("text")
    return value


def _to_number(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None
