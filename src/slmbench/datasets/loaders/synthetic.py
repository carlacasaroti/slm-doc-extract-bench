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
from slmbench.extraction.schema import get_schema

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
    task_id: str,
) -> list[DocumentSample]:
    n = limit or 50
    raw_dir.mkdir(parents=True, exist_ok=True)
    schema_fields = set(get_schema(task_id)["properties"].keys())

    samples: list[DocumentSample] = []
    seed = 42 if split == "test" else 1
    rng = random.Random(seed)
    if _fake is not None:
        # Faker keeps its OWN internal random state, entirely separate from
        # the `rng` above — without this, two independent load() calls
        # (e.g. synthetic_invoices_ptbr then synthetic_invoices_ptbr_reduced)
        # would each continue from wherever Faker's shared RNG last left
        # off, producing DIFFERENT names/dates/CNPJs even with the same
        # `seed`. Reseeding here is what makes every dataset that reuses
        # this loader generate byte-identical underlying documents.
        _fake.seed_instance(seed)

    for i in range(n):
        sample_id = f"{dataset_id}/{split}_{i:04d}"
        image_path = raw_dir / f"{split}_{i:04d}.png"
        gt_path = raw_dir / f"{split}_{i:04d}.json"

        if image_path.exists() and gt_path.exists():
            full_ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        else:
            full_ground_truth = _generate_invoice_image(image_path, rng)
            gt_path.write_text(json.dumps(full_ground_truth, ensure_ascii=False, indent=2))

        # Filter to only the fields this task's schema actually asks for —
        # this is what lets synthetic_invoices_ptbr_reduced reuse the exact
        # same generated documents while only scoring/prompting a subset of
        # fields (see configs/tasks.yaml::invoice_fields_reduced).
        ground_truth = {k: v for k, v in full_ground_truth.items() if k in schema_fields}

        samples.append(
            DocumentSample(
                sample_id=sample_id,
                dataset_id=dataset_id,
                task_id=task_id,
                image_path=image_path,
                ground_truth=ground_truth,
                metadata={"synthetic": True},
            )
        )

    return samples


def _load_fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    """Try a handful of common TTF locations across macOS/Linux/CI before
    giving up. A missing TrueType font silently degrades to PIL's tiny
    bitmap default font, which is nearly unreadable at this resolution —
    that would tank every extractor's score for reasons that have nothing
    to do with the model being evaluated, so we warn loudly if it happens.
    """
    candidates = [
        # Linux (matches the Dockerfile, which installs fonts-dejavu-core)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Generic — works if the font happens to be on PIL's search path
        "DejaVuSans.ttf",
        "Arial.ttf",
    ]

    def _first_working(names: list[str], size: int):
        for name in names:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return None

    font = _first_working(
        [c for c in candidates if "Bold" not in c and "bold" not in c], 20
    )
    font_bold = _first_working(
        [c for c in candidates if "Bold" in c or "bold" in c], 24
    ) or font

    if font is None:
        import warnings

        warnings.warn(
            "No TrueType font found for the synthetic document generator — "
            "falling back to PIL's tiny bitmap default font. Generated "
            "documents will likely be unreadable to any extractor, which "
            "will look like a model failure but is actually a rendering "
            "problem. Install a TTF (e.g. `apt install fonts-dejavu-core` "
            "on Linux, or point _load_fonts() at a font you have on macOS) "
            "and regenerate the dataset.",
            stacklevel=2,
        )
        try:
            font = ImageFont.load_default(size=20)
            font_bold = ImageFont.load_default(size=24)
        except TypeError:  # older Pillow without the `size` kwarg
            font = font_bold = ImageFont.load_default()

    return font, font_bold


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
    font, font_bold = _load_fonts()

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