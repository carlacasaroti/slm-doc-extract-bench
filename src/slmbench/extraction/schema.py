"""Loads configs/tasks.yaml — the JSON Schema per extraction task."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "tasks.yaml"


@lru_cache(maxsize=1)
def _registry() -> dict[str, dict[str, Any]]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw["tasks"]


def get_task(task_id: str) -> dict[str, Any]:
    try:
        return _registry()[task_id]
    except KeyError as e:
        raise ValueError(f"Unknown task_id '{task_id}'. Known: {list(_registry())}") from e


def get_schema(task_id: str) -> dict[str, Any]:
    return get_task(task_id)["schema"]
