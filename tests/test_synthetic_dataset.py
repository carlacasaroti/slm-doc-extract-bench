from pathlib import Path

from slmbench.datasets.loaders import synthetic


def test_synthetic_generates_requested_number_of_samples(tmp_path: Path):
    samples = synthetic.load(
        raw_dir=tmp_path, split="test", limit=5,
        dataset_id="synthetic_invoices_ptbr", task_id="invoice_fields",
    )
    assert len(samples) == 5


def test_synthetic_samples_have_valid_ground_truth(tmp_path: Path):
    samples = synthetic.load(
        raw_dir=tmp_path, split="test", limit=3,
        dataset_id="synthetic_invoices_ptbr", task_id="invoice_fields",
    )
    for s in samples:
        assert s.image_path.exists()
        assert s.ground_truth["invoice_number"]
        assert s.ground_truth["total_amount"] > 0
        assert s.task_id == "invoice_fields"


def test_synthetic_generation_is_deterministic_per_split(tmp_path: Path):
    first = synthetic.load(
        raw_dir=tmp_path, split="test", limit=3,
        dataset_id="synthetic_invoices_ptbr", task_id="invoice_fields",
    )
    second = synthetic.load(
        raw_dir=tmp_path, split="test", limit=3,
        dataset_id="synthetic_invoices_ptbr", task_id="invoice_fields",
    )
    assert [s.ground_truth["invoice_number"] for s in first] == [
        s.ground_truth["invoice_number"] for s in second
    ]


def test_synthetic_reduced_schema_reuses_same_underlying_documents(tmp_path: Path):
    """The controlled-experiment dataset (synthetic_invoices_ptbr_reduced)
    must generate byte-identical underlying documents to the full dataset —
    only the returned ground_truth (and therefore the schema/prompt) differs.
    """
    full_dir = tmp_path / "full"
    reduced_dir = tmp_path / "reduced"

    full = synthetic.load(
        raw_dir=full_dir, split="test", limit=2,
        dataset_id="synthetic_invoices_ptbr", task_id="invoice_fields",
    )
    reduced = synthetic.load(
        raw_dir=reduced_dir, split="test", limit=2,
        dataset_id="synthetic_invoices_ptbr_reduced", task_id="invoice_fields_reduced",
    )

    assert reduced[0].task_id == "invoice_fields_reduced"
    assert set(reduced[0].ground_truth.keys()) == {
        "issue_date", "vendor_tax_id", "customer_name", "subtotal", "tax_amount",
    }
    # Same rng seed -> same underlying document content, just filtered fields.
    assert full[0].ground_truth["issue_date"] == reduced[0].ground_truth["issue_date"]
    assert full[0].ground_truth["subtotal"] == reduced[0].ground_truth["subtotal"]
