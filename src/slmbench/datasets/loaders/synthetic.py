"""Synthetic PT-BR invoice/receipt generator.

This is the one dataset in the registry that needs no download and no
registration — it renders templated invoice images on the fly with known
ground truth. Useful for:

  * A working end-to-end smoke test right after `pip install -e .`
  * Controlled ablations: vary rotation, blur, font, layout, language and
    see exactly which perturbation breaks which model.

Generated images are cached under raw_dir so repeated runs are stable
(same seed -> same documents) unless --regenerate is passed to the CLI.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from slmbench.datasets.base import DocumentSample

try:
    from faker import Faker

    _fake = Faker("pt_BR")
except ImportError:  # pragma: no cover
    _fake = None


def load(
    raw_dir: Path,
    split: str,
    limit: int | None,
    dataset_id: str,
) -> list[DocumentSample]:
    n = limit or 50
    raw_dir.mkdir(parents=True, exist_ok=True)

    samples: list[DocumentSample] = []
    rng = random.Random(42 if split == "test" else 1)

    for i in range(n):
        sample_id = f"{dataset_id}/{split}_{i:04d}"
        image_path = raw_dir / f"{split}_{i:04d}.png"
        gt_path = raw_dir / f"{split}_{i:04d}.json"

        if image_path.exists() and gt_path.exists():
            ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        else:
            ground_truth = _generate_invoice_image(image_path, rng)
            gt_path.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2))

        samples.append(
            DocumentSample(
                sample_id=sample_id,
                dataset_id=dataset_id,
                task_id="invoice_fields",
                image_path=image_path,
                ground_truth=ground_truth,
                metadata={"synthetic": True},
            )
        )

    return samples


def _generate_invoice_image(out_path: Path, rng: random.Random) -> dict:
    vendor = _fake.company() if _fake else "Empresa Exemplo LTDA"
    customer = _fake.name() if _fake else "Cliente Exemplo"
    invoice_number = f"NF-{rng.randint(100000, 999999)}"
    issue_date = _fake.date() if _fake else "2026-01-15"
    subtotal = round(rng.uniform(50, 5000), 2)
    tax = round(subtotal * rng.choice([0.0, 0.05, 0.10, 0.18]), 2)
    total = round(subtotal + tax, 2)
    currency = "BRL"

    ground_truth = {
        "invoice_number": invoice_number,
        "issue_date": issue_date,
        "due_date": None,
        "vendor_name": vendor,
        "vendor_tax_id": _fake.cnpj() if _fake else "00.000.000/0001-00",
        "customer_name": customer,
        "currency": currency,
        "subtotal": subtotal,
        "tax_amount": tax,
        "total_amount": total,
    }

    img = Image.new("RGB", (800, 1000), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    lines = [
        (f"FATURA / INVOICE {invoice_number}", font_bold),
        ("", font),
        (f"Fornecedor: {vendor}", font),
        (f"CNPJ: {ground_truth['vendor_tax_id']}", font),
        ("", font),
        (f"Cliente: {customer}", font),
        (f"Data de emissao: {issue_date}", font),
        ("", font),
        (f"Subtotal: R$ {subtotal:,.2f}", font),
        (f"Impostos: R$ {tax:,.2f}", font),
        (f"TOTAL: R$ {total:,.2f}", font_bold),
    ]

    y = 40
    for text, f in lines:
        draw.text((40, y), text, fill="black", font=f)
        y += 45

    img.save(out_path)
    return ground_truth
