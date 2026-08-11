#!/usr/bin/env python3
"""Downloads (or explains how to obtain) raw data for a dataset in the registry.

Several benchmark datasets (SROIE, DocILE) require manual registration on
their host site before download — this script cannot bypass that, and
won't pretend to. For those, it prints the exact steps.

For datasets with a plain public URL (e.g. CORD, FUNSD) it downloads and
extracts automatically. For `synthetic_invoices_ptbr`, nothing needs
downloading — it's generated on first use by `slmbench run`.

Usage:
    python scripts/download_datasets.py --dataset cord
    python scripts/download_datasets.py --list
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from slmbench.datasets.registry import get_dataset_config, list_datasets  # noqa: E402

MANUAL_INSTRUCTIONS = {
    "sroie2019": (
        "SROIE requires registration at https://rrc.cvc.uab.es/?ch=13 .\n"
        "After downloading, extract so you get:\n"
        "  data/raw/sroie2019/img/*.jpg\n"
        "  data/raw/sroie2019/entities/*.txt\n"
    ),
    "docile": (
        "DocILE requires registration at https://github.com/rossumai/docile .\n"
        "Follow their `docile` CLI download instructions, then symlink or\n"
        "copy the result into data/raw/docile/.\n"
    ),
}

DIRECT_DOWNLOAD_URLS = {
    # NOTE: verify these still resolve before relying on them — dataset
    # hosts move files around. If a URL is dead, open an issue/PR.
    "funsd": "https://guillaumejaume.github.io/FUNSD/dataset.zip",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset id from configs/datasets.yaml")
    parser.add_argument("--list", action="store_true", help="List all registered dataset ids")
    args = parser.parse_args()

    if args.list or not args.dataset:
        print("Registered datasets:")
        for ds_id in list_datasets():
            print(f"  - {ds_id}")
        return

    cfg = get_dataset_config(args.dataset)
    raw_dir = REPO_ROOT / "data" / "raw" / args.dataset

    if args.dataset == "synthetic_invoices_ptbr":
        print("Nothing to download — this dataset is generated on first "
              "`slmbench run --dataset synthetic_invoices_ptbr`.")
        return

    if args.dataset in MANUAL_INSTRUCTIONS:
        print(MANUAL_INSTRUCTIONS[args.dataset])
        return

    if args.dataset in DIRECT_DOWNLOAD_URLS:
        raw_dir.mkdir(parents=True, exist_ok=True)
        url = DIRECT_DOWNLOAD_URLS[args.dataset]
        zip_path = raw_dir / "download.zip"
        print(f"Downloading {url} ...")
        urlretrieve(url, zip_path)
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)
        zip_path.unlink()
        print(f"Done: {raw_dir}")
        return

    print(
        f"No automated or manual download instructions registered for "
        f"'{args.dataset}' yet. Source: {cfg.get('source')}. "
        f"See docs/ADDING_A_DATASET.md."
    )


if __name__ == "__main__":
    main()
