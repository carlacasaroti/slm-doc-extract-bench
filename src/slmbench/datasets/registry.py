"""Loads configs/datasets.yaml and dispatches to the right loader module.

Usage:
    from slmbench.datasets.registry import load_dataset
    samples = load_dataset("cord", split="test", limit=50)
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from slmbench.datasets.base import DocumentSample

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "datasets.yaml"


@lru_cache(maxsize=1)
def _registry() -> dict[str, dict[str, Any]]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {entry["id"]: entry for entry in raw["datasets"]}


def list_datasets() -> list[str]:
    return list(_registry().keys())


def get_dataset_config(dataset_id: str) -> dict[str, Any]:
    try:
        return _registry()[dataset_id]
    except KeyError as e:
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}'. Known datasets: {list_datasets()}"
        ) from e


def load_dataset(
    dataset_id: str,
    split: str = "test",
    limit: int | None = None,
    raw_dir: Path | None = None,
) -> list[DocumentSample]:
    """Load a dataset by id, returning normalized DocumentSample objects.

    Each loader module (src/slmbench/datasets/loaders/<loader>.py) must
    expose a `load(raw_dir, split, limit) -> list[DocumentSample]` function.
    See docs/ADDING_A_DATASET.md for the contract.
    """
    cfg = get_dataset_config(dataset_id)
    loader_name = cfg["loader"]
    module = importlib.import_module(f"slmbench.datasets.loaders.{loader_name}")

    raw_dir = raw_dir or Path("data/raw") / dataset_id
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data for '{dataset_id}' not found at {raw_dir}.\n"
            f"Run: python scripts/download_datasets.py --dataset {dataset_id}\n"
            f"(some datasets require manual registration first — see "
            f"docs/ADDING_A_DATASET.md and the 'source'/'license' fields "
            f"in configs/datasets.yaml)"
        )

    return module.load(raw_dir=raw_dir, split=split, limit=limit, dataset_id=dataset_id)
