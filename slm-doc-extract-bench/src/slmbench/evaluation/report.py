"""Generates results/leaderboard.md and results/<run_id>.json from
aggregated scores.

Kept deliberately simple (no plotting deps) so `slmbench report` works
in any environment, including CI, with zero extra setup.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


def write_report(
    summary: dict[str, dict[str, Any]],
    dataset_id: str,
    run_id: str | None = None,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    raw_path = RESULTS_DIR / f"{run_id}_{dataset_id}.json"
    raw_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    _append_to_leaderboard(summary, dataset_id, run_id)
    return raw_path


def _append_to_leaderboard(summary: dict[str, dict[str, Any]], dataset_id: str, run_id: str) -> None:
    leaderboard_path = RESULTS_DIR / "leaderboard.md"

    rows = sorted(summary.items(), key=lambda kv: kv[1]["mean_field_f1"], reverse=True)

    lines = [f"\n## {dataset_id} — run `{run_id}`\n"]
    lines.append("| Model | Field F1 | Exact Match | JSON Valid % | Mean Latency (s) | p95 Latency (s) |")
    lines.append("|---|---|---|---|---|---|")
    for model_id, stats in rows:
        lines.append(
            f"| {model_id} | {stats['mean_field_f1']:.3f} | {stats['exact_match_rate']:.3f} | "
            f"{stats['json_valid_rate'] * 100:.1f}% | {stats['mean_latency_seconds']:.2f} | "
            f"{stats['p95_latency_seconds']:.2f} |"
        )

    with open(leaderboard_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
