#!/usr/bin/env python3
"""Plots the field-count degradation curve: F1 vs. number of fields
requested in one schema, one line per model.

Auto-discovers whichever of the field-count result files you've already
run — you don't need all 5 points to get a chart; it plots whatever it
finds and tells you which ones are still missing.

Usage:
    pip install matplotlib
    python scripts/plot_field_count_curve.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "field_count_curve.png"

# Nested-order sweep: these datasets share the same underlying documents
# and a fixed, cumulative field order (see configs/tasks.yaml).
CURVE_DATASETS = {
    "synthetic_invoices_ptbr_2f": 2,
    "synthetic_invoices_ptbr_4f": 4,
    "synthetic_invoices_ptbr_6f": 6,
    "synthetic_invoices_ptbr_8f": 8,
    "synthetic_invoices_ptbr": 10,
}

# Different field SELECTION (the 5 hardest fields specifically), not part
# of the nested order above — plotted separately so it isn't mistaken for
# a 6th point on the same curve.
SPECIAL_DATASETS = {
    "synthetic_invoices_ptbr_reduced": ("5 hardest fields\n(different selection)", 5),
}


def latest_result_for(dataset_id: str) -> dict | None:
    matches = sorted(RESULTS_DIR.glob(f"*_{dataset_id}.json"))
    if not matches:
        return None
    with open(matches[-1], encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    import matplotlib.pyplot as plt

    # size -> {model_id: f1}
    curve_data: dict[int, dict[str, float]] = {}
    missing = []
    for dataset_id, size in CURVE_DATASETS.items():
        result = latest_result_for(dataset_id)
        if result is None:
            missing.append(dataset_id)
            continue
        curve_data[size] = {m: v["mean_field_f1"] for m, v in result.items()}

    if not curve_data:
        sys.exit(
            "No field-count result files found yet. Run at least one of:\n"
            + "\n".join(f"  slmbench run --dataset {d} ..." for d in CURVE_DATASETS)
        )

    if missing:
        print(f"Note: no results yet for {missing} — plotting the {len(curve_data)} "
              f"size(s) that are available. Run the missing ones and re-plot for the full curve.")

    all_models = sorted({m for sizes in curve_data.values() for m in sizes})
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, model_id in enumerate(all_models):
        sizes = sorted(s for s in curve_data if model_id in curve_data[s])
        f1s = [curve_data[s][model_id] for s in sizes]
        ax.plot(sizes, f1s, marker="o", markersize=8, linewidth=2,
                 color=colors[i % len(colors)], label=model_id)

    # Special (non-nested) points, plotted as standalone markers with an
    # annotation so they're clearly not part of the main curve.
    for dataset_id, (label, size) in SPECIAL_DATASETS.items():
        result = latest_result_for(dataset_id)
        if result is None:
            continue
        for i, model_id in enumerate(all_models):
            if model_id not in result:
                continue
            ax.scatter(
                [size], [result[model_id]["mean_field_f1"]],
                marker="*", s=250, color=colors[i % len(colors)],
                edgecolors="black", linewidths=0.8, zorder=5,
            )
        ax.annotate(
            label, xy=(size, max(r["mean_field_f1"] for r in result.values())),
            xytext=(8, 10), textcoords="offset points", fontsize=9, style="italic",
        )

    ax.set_xlabel("Number of fields requested in one schema")
    ax.set_ylabel("Mean field F1")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(sorted(set(list(curve_data.keys()) + [s for _, s in SPECIAL_DATASETS.values()])))
    ax.set_title(
        "Small local models: extraction quality drops as schema size grows\n"
        "(same underlying documents at every size — only field count changes)",
        fontsize=12,
    )
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
