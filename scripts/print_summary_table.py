#!/usr/bin/env python3
"""Prints (and saves) one consolidated table across every field-count run
you've done so far — the numbers behind the field_count_curve.png chart,
including latency, which the chart alone doesn't show precisely.

Usage:
    python scripts/print_summary_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUT_MD_PATH = RESULTS_DIR / "summary_table.md"
OUT_PNG_PATH = RESULTS_DIR / "summary_table.png"

DATASETS_IN_ORDER = [
    ("synthetic_invoices_ptbr_2f", 2, "nested"),
    ("synthetic_invoices_ptbr_4f", 4, "nested"),
    ("synthetic_invoices_ptbr_6f", 6, "nested"),
    ("synthetic_invoices_ptbr_8f", 8, "nested"),
    ("synthetic_invoices_ptbr", 10, "nested"),
    ("synthetic_invoices_ptbr_reduced", 5, "5 hardest fields (different selection)"),
]


def latest_result_for(dataset_id: str) -> dict | None:
    matches = sorted(RESULTS_DIR.glob(f"*_{dataset_id}.json"))
    if not matches:
        return None
    with open(matches[-1], encoding="utf-8") as f:
        return json.load(f)


def save_png(rows: list[list[str]], headers: list[str]) -> None:
    import matplotlib.pyplot as plt

    # Drop the "Dataset" column for the image — Size + Group already
    # convey the same thing more compactly, and the raw dataset_id was
    # getting clipped in the cell.
    dataset_idx = headers.index("Dataset")
    png_headers = [h for i, h in enumerate(headers) if i != dataset_idx]
    png_rows = [[v for i, v in enumerate(row) if i != dataset_idx] for row in rows]

    fig_height = 0.5 + 0.35 * len(png_rows)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=png_rows, colLabels=png_headers, loc="center", cellLoc="center",
        colWidths=[0.10, 0.22, 0.20, 0.07, 0.10, 0.11, 0.12, 0.13, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for col_idx in range(len(png_headers)):
        cell = table[0, col_idx]
        cell.set_facecolor("#333333")
        cell.set_text_props(color="white", weight="bold")

    for row_idx in range(1, len(png_rows) + 1):
        color = "#f2f2f2" if row_idx % 2 == 0 else "white"
        for col_idx in range(len(png_headers)):
            table[row_idx, col_idx].set_facecolor(color)

    fig.tight_layout()
    fig.savefig(OUT_PNG_PATH, dpi=150, bbox_inches="tight")
    print(f"Also saved to {OUT_PNG_PATH}")


def main() -> None:
    headers = ["Dataset", "Size", "Group", "Model", "N", "Mean F1", "Exact Match",
               "JSON Valid %", "Mean Latency (s)", "p95 Latency (s)"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    png_rows = []
    missing = []

    for dataset_id, size, group in DATASETS_IN_ORDER:
        result = latest_result_for(dataset_id)
        if result is None:
            missing.append(dataset_id)
            continue
        for model_id, stats in result.items():
            row = [
                dataset_id, str(size), group, model_id, str(stats["n_samples"]),
                f"{stats['mean_field_f1']:.3f}", f"{stats['exact_match_rate']:.3f}",
                f"{stats['json_valid_rate']*100:.1f}%", f"{stats['mean_latency_seconds']:.2f}",
                f"{stats['p95_latency_seconds']:.2f}",
            ]
            lines.append("| " + " | ".join(row) + " |")
            png_rows.append(row)

    table_md = "\n".join(lines)
    print(table_md)
    if missing:
        print(f"\n(No results yet for: {missing})")

    OUT_MD_PATH.write_text(table_md + "\n", encoding="utf-8")
    print(f"\nAlso saved to {OUT_MD_PATH}")

    if png_rows:
        save_png(png_rows, headers)
    else:
        print("No data to render as PNG yet.")


if __name__ == "__main__":
    main()