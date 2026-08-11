"""Evaluation metrics comparing ExtractionResult.parsed_output against
DocumentSample.ground_truth.

Design choice: metrics are computed per-field, then aggregated. This
matters for document extraction specifically — a model that gets
`invoice_number` and `total_amount` right but hallucinates `due_date`
should score very differently from one that gets nothing right, and a
single "exact JSON match" metric would treat both as equally wrong.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any


@dataclass
class FieldScore:
    field: str
    match: bool
    predicted: Any
    expected: Any


@dataclass
class SampleScore:
    sample_id: str
    model_id: str
    json_valid: bool
    field_scores: list[FieldScore]
    latency_seconds: float

    @property
    def field_f1(self) -> float:
        if not self.field_scores:
            return 0.0
        return sum(1 for f in self.field_scores if f.match) / len(self.field_scores)


def score_sample(
    sample_id: str,
    model_id: str,
    predicted: dict[str, Any] | None,
    expected: dict[str, Any],
    json_valid: bool,
    latency_seconds: float,
    fuzzy_string_threshold: float = 0.85,
) -> SampleScore:
    """Score one prediction against ground truth, field by field.

    - Numbers: exact match after normalization (handles "1234.00" vs 1234.0).
    - Strings: fuzzy match above `fuzzy_string_threshold` counts as correct —
      OCR/VLM outputs commonly differ by punctuation/whitespace/case even
      when semantically correct, and penalizing that would mostly measure
      string-formatting luck rather than extraction quality.
    - Lists (e.g. line items): scored as correct only if every item matches,
      order-independent. See docs/ARCHITECTURE.md if you want a softer
      set-based line-item metric instead.
    - Missing field in `predicted` (or predicted is None) counts as a miss.
    """
    predicted = predicted or {}
    field_scores: list[FieldScore] = []

    for field, expected_value in expected.items():
        predicted_value = predicted.get(field)
        match = _values_match(predicted_value, expected_value, fuzzy_string_threshold)
        field_scores.append(
            FieldScore(field=field, match=match, predicted=predicted_value, expected=expected_value)
        )

    return SampleScore(
        sample_id=sample_id,
        model_id=model_id,
        json_valid=json_valid,
        field_scores=field_scores,
        latency_seconds=latency_seconds,
    )


def _values_match(predicted: Any, expected: Any, fuzzy_threshold: float) -> bool:
    if expected is None:
        return predicted is None
    if predicted is None:
        return False

    if isinstance(expected, (int, float)):
        try:
            return abs(float(predicted) - float(expected)) < 0.01
        except (TypeError, ValueError):
            return False

    if isinstance(expected, list):
        if not isinstance(predicted, list) or len(predicted) != len(expected):
            return False
        # Order-independent: each expected item must have a matching predicted item.
        remaining = list(predicted)
        for exp_item in expected:
            match_idx = next(
                (i for i, p in enumerate(remaining) if _dict_or_value_match(p, exp_item, fuzzy_threshold)),
                None,
            )
            if match_idx is None:
                return False
            remaining.pop(match_idx)
        return True

    if isinstance(expected, dict):
        if not isinstance(predicted, dict):
            return False
        return all(
            _values_match(predicted.get(k), v, fuzzy_threshold) for k, v in expected.items()
        )

    # String (and everything else): fuzzy match.
    return _string_similarity(str(predicted), str(expected)) >= fuzzy_threshold


def _dict_or_value_match(predicted: Any, expected: Any, fuzzy_threshold: float) -> bool:
    return _values_match(predicted, expected, fuzzy_threshold)


def _string_similarity(a: str, b: str) -> float:
    a_norm, b_norm = a.strip().lower(), b.strip().lower()
    if a_norm == b_norm:
        return 1.0
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def aggregate(scores: list[SampleScore]) -> dict[str, Any]:
    """Aggregate per-sample scores into per-model summary stats."""
    by_model: dict[str, list[SampleScore]] = {}
    for s in scores:
        by_model.setdefault(s.model_id, []).append(s)

    summary = {}
    for model_id, model_scores in by_model.items():
        n = len(model_scores)
        summary[model_id] = {
            "n_samples": n,
            "json_valid_rate": sum(s.json_valid for s in model_scores) / n,
            "mean_field_f1": sum(s.field_f1 for s in model_scores) / n,
            "exact_match_rate": sum(s.field_f1 == 1.0 for s in model_scores) / n,
            "mean_latency_seconds": sum(s.latency_seconds for s in model_scores) / n,
            "p95_latency_seconds": _percentile([s.latency_seconds for s in model_scores], 0.95),
        }
    return summary


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round(p * (len(values) - 1)))
    return values[idx]
