"""Contract every loader module in this package must implement.

Copy this file to `<your_dataset>.py`, register it in configs/datasets.yaml
(`loader: your_dataset`), and implement `load()`. See
docs/ADDING_A_DATASET.md for a full walkthrough with a real example.
"""

from __future__ import annotations

from pathlib import Path

from slmbench.datasets.base import DocumentSample


def load(
    raw_dir: Path,
    split: str,
    limit: int | None,
    dataset_id: str,
    task_id: str,
) -> list[DocumentSample]:
    """Parse this dataset's raw files into normalized DocumentSample objects.

    Args:
        raw_dir: Where scripts/download_datasets.py placed the raw files
            for this dataset (e.g. data/raw/<dataset_id>/).
        split: "train" | "val" | "test" — loaders should degrade gracefully
            (e.g. fall back to "test" if a dataset has no val split).
        limit: If set, return at most this many samples (for fast iteration).
        dataset_id: Passed through so DocumentSample.dataset_id is set
            correctly without hardcoding it in every loader.
        task_id: The task registered for this dataset in
            configs/datasets.yaml (`task:` field) — set DocumentSample.task_id
            to this rather than hardcoding a task name, so the same loader
            can be reused with a different (e.g. reduced) schema by
            registering a second dataset entry pointing at the same loader
            with a different `task:`. See synthetic.py for an example.

    Returns:
        A list of DocumentSample, each with `ground_truth` shaped to match
        the JSON schema declared for `task_id` in configs/tasks.yaml.
    """
    raise NotImplementedError("Implement load() for this dataset.")
