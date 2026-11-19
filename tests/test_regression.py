"""Tests for the saved-results latency-regression check.

Numbers below are synthetic test fixtures used to verify comparison logic,
not benchmark measurements.
"""

from __future__ import annotations

import pytest

from llm_serve_bench.bench.regression import check_regression


def _document(p95_ms: float) -> dict[str, object]:
    return {
        "aggregates": {
            "concurrency_1": {"e2e_p95_ms": {"mean": p95_ms}},
            "concurrency_4": {"e2e_p95_ms": {"mean": p95_ms * 2}},
        }
    }


def test_identical_latency_passes() -> None:
    outcome = check_regression(_document(100.0), _document(100.0), threshold=1.2)
    assert outcome.passed is True
    assert len(outcome.comparisons) == 2
    assert outcome.comparisons[0].ratio == 1.0


def test_large_regression_fails() -> None:
    outcome = check_regression(_document(100.0), _document(150.0), threshold=1.2)
    assert outcome.passed is False
    assert outcome.comparisons[0].passed is False
    assert outcome.comparisons[1].passed is False


def test_configurable_threshold_allows_small_change() -> None:
    outcome = check_regression(_document(100.0), _document(110.0), threshold=1.25)
    assert outcome.passed is True


def test_placeholder_with_null_values_is_rejected() -> None:
    placeholder = {"aggregates": {"concurrency_1": {"e2e_p95_ms": {"mean": None}}}}
    with pytest.raises(ValueError, match="placeholder"):
        check_regression(placeholder, _document(100.0), threshold=1.2)


def test_missing_fresh_level_fails_closed() -> None:
    fresh = {"aggregates": {"concurrency_1": {"e2e_p95_ms": {"mean": 100.0}}}}
    outcome = check_regression(_document(100.0), fresh, threshold=1.2)
    assert outcome.passed is False
    assert "concurrency_4" in outcome.skipped_levels


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold"):
        check_regression(_document(100.0), _document(100.0), threshold=0)
