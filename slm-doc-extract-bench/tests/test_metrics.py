from slmbench.evaluation.metrics import aggregate, score_sample


def test_exact_match_scores_1_0():
    score = score_sample(
        sample_id="s1",
        model_id="m1",
        predicted={"total": 10.0, "company_name": "Acme"},
        expected={"total": 10.0, "company_name": "Acme"},
        json_valid=True,
        latency_seconds=1.0,
    )
    assert score.field_f1 == 1.0


def test_partial_match():
    score = score_sample(
        sample_id="s1",
        model_id="m1",
        predicted={"total": 10.0, "company_name": "Wrong Name"},
        expected={"total": 10.0, "company_name": "Acme"},
        json_valid=True,
        latency_seconds=1.0,
    )
    assert 0.0 < score.field_f1 < 1.0


def test_missing_prediction_scores_0():
    score = score_sample(
        sample_id="s1",
        model_id="m1",
        predicted=None,
        expected={"total": 10.0},
        json_valid=False,
        latency_seconds=1.0,
    )
    assert score.field_f1 == 0.0
    assert score.json_valid is False


def test_fuzzy_string_match_tolerates_minor_diffs():
    score = score_sample(
        sample_id="s1",
        model_id="m1",
        predicted={"company_name": "ACME corp."},
        expected={"company_name": "Acme Corp"},
        json_valid=True,
        latency_seconds=1.0,
    )
    assert score.field_f1 == 1.0


def test_number_tolerance_handles_formatting():
    score = score_sample(
        sample_id="s1",
        model_id="m1",
        predicted={"total": "1234.00"},
        expected={"total": 1234},
        json_valid=True,
        latency_seconds=1.0,
    )
    assert score.field_f1 == 1.0


def test_line_items_order_independent():
    predicted = {"items": [{"name": "B", "total_price": 2}, {"name": "A", "total_price": 1}]}
    expected = {"items": [{"name": "A", "total_price": 1}, {"name": "B", "total_price": 2}]}
    score = score_sample(
        sample_id="s1", model_id="m1", predicted=predicted, expected=expected,
        json_valid=True, latency_seconds=1.0,
    )
    assert score.field_f1 == 1.0


def test_aggregate_computes_expected_keys():
    scores = [
        score_sample("s1", "m1", {"total": 1}, {"total": 1}, True, 0.5),
        score_sample("s2", "m1", {"total": 2}, {"total": 1}, True, 1.5),
    ]
    summary = aggregate(scores)
    assert "m1" in summary
    for key in ["n_samples", "json_valid_rate", "mean_field_f1", "exact_match_rate",
                "mean_latency_seconds", "p95_latency_seconds"]:
        assert key in summary["m1"]
