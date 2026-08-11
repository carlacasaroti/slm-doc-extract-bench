from pathlib import Path

from slmbench.datasets.loaders import synthetic


def test_synthetic_generates_requested_number_of_samples(tmp_path: Path):
    samples = synthetic.load(raw_dir=tmp_path, split="test", limit=5, dataset_id="synthetic_invoices_ptbr")
    assert len(samples) == 5


def test_synthetic_samples_have_valid_ground_truth(tmp_path: Path):
    samples = synthetic.load(raw_dir=tmp_path, split="test", limit=3, dataset_id="synthetic_invoices_ptbr")
    for s in samples:
        assert s.image_path.exists()
        assert s.ground_truth["invoice_number"]
        assert s.ground_truth["total_amount"] > 0
        assert s.task_id == "invoice_fields"


def test_synthetic_generation_is_deterministic_per_split(tmp_path: Path):
    first = synthetic.load(raw_dir=tmp_path, split="test", limit=3, dataset_id="synthetic_invoices_ptbr")
    second = synthetic.load(raw_dir=tmp_path, split="test", limit=3, dataset_id="synthetic_invoices_ptbr")
    assert [s.ground_truth["invoice_number"] for s in first] == [
        s.ground_truth["invoice_number"] for s in second
    ]
