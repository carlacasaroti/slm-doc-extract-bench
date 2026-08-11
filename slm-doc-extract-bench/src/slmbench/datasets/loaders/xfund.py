"""Loader stub for xfund.

Not implemented yet — this dataset's raw format needs a real parser
contributed. See docs/ADDING_A_DATASET.md for the contract and
sroie.py / cord.py for two complete reference implementations.

Good first contribution: open a PR implementing load() for xfund.
"""

from __future__ import annotations

from pathlib import Path

from slmbench.datasets.base import DocumentSample


def load(
    raw_dir: Path,
    split: str,
    limit: int | None,
    dataset_id: str,
) -> list[DocumentSample]:
    raise NotImplementedError(
        "Loader for 'xfund' is not implemented yet. "
        "See docs/ADDING_A_DATASET.md — contributions welcome."
    )
