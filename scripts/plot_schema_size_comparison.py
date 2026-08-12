#!/usr/bin/env python3
"""The simplest possible chart of the finding you already have.

Reads your two already-saved results files:
  - the full-schema run  (dataset: synthetic_invoices_ptbr, 10 fields)
  - the reduced-schema run (dataset: synthetic_invoices_ptbr_reduced, 5 fields)

and makes ONE bar chart: F1 by model, grouped by schema size. No new model
calls, no sweep — just plotting what you already measured.

Usage:
    pip install matplotlib
    python scripts/plot_schema_size_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "schema_size_comparison.png"


def latest_result_for(dataset_id: str) -> dict:
    """Find the most recent results/<timestamp>_<dataset_id>.json file."""
    matches = sorted(RESULTS_DIR.glob(f"*_{dataset_id}.json"))
    if not matches:
        sys.exit(
            f"No results file found for '{dataset_id}' in {RESULTS_DIR}.\n"
            f"Run `slmbench run --dataset {dataset_id} ...` first."
        )
    with open(matches[-1], encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    import matplotlib.pyplot as plt

    full = latest_result_for("synthetic_invoices_ptbr")
    reduced = latest_result_for("synthetic_invoices_ptbr_reduced")

    model_ids = sorted(set(full) | set(reduced))
    x = range(len(model_ids))
    width = 0.35

    full_f1 = [full.get(m, {}).get("mean_field_f1", 0) for m in model_ids]
    reduced_f1 = [reduced.get(m, {}).get("mean_field_f1", 0) for m in model_ids]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar([i - width / 2 for i in x], full_f1, width, label="Schema completo (10 campos)", color="#d95f02")
    bars2 = ax.bar([i + width / 2 for i in x], reduced_f1, width, label="Schema reduzido (5 campos)", color="#1b9e77")

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=11,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_ids)
    ax.set_ylabel("Field F1 (média)")
    ax.set_ylim(0, 1.18)
    ax.set_title(
        "Modelos pequenos: F1 despenca ao pedir mais campos de uma vez\n"
        "(mesmos documentos e modelos — só muda quantos campos são pedidos)",
        fontsize=12,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()