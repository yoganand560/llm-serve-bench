"""Tests for metric math. Fixtures below are synthetic test data, not benchmark claims."""

from __future__ import annotations

from dataclasses import replace

import pytest

from llm_serve_bench.bench.metrics import (
    RequestRecord,
    RunMetrics,
    aggregate_runs,
    percentile,
    summarize_records,
)


def _record(**overrides: object) -> RequestRecord:
    values: dict[str, object] = {
        "prompt_id": 0,
        "concurrency": 1,
        "ttft_ms": 10.0,
        "itl_ms": 5.0,
        "e2e_ms": 20.0,
        "generated_tokens": 2,
        "error": None,
    }
    values.update(overrides)
    return RequestRecord(**values)  # type: ignore[arg-type]


def _summary(**overrides: object) -> RunMetrics:
    values: dict[str, object] = {
        "backend": "mock",
        "quantization": "N/A",
        "concurrency": 1,
        "repeat": 1,
        "requests": 4,
        "errors": 0,
        "elapsed_s": 1.0,
        "ttft_p50_ms": 10.0,
        "itl_mean_ms": 5.0,
        "e2e_p50_ms": 20.0,
        "e2e_p95_ms": 30.0,
        "e2e_p99_ms": 30.0,
        "tokens_per_second": 8.0,
        "requests_per_second": 4.0,
        "peak_rss_mb": 100.0,
        "memory_source": "test fixture",
        "notes": (),
    }
    values.update(overrides)
    return RunMetrics(**values)  # type: ignore[arg-type]


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert percentile([], 0.5) is None


def test_summarize_records_computes_rates_and_percentiles() -> None:
    summary = summarize_records(
        [_record(e2e_ms=20.0), _record(e2e_ms=30.0, prompt_id=1)],
        backend="mock",
        quantization="N/A",
        concurrency=1,
        repeat=1,
        elapsed_s=1.0,
        peak_rss_mb=123.4,
        memory_source="test fixture",
    )
    assert summary.e2e_p50_ms == 20.0
    assert summary.e2e_p95_ms == 30.0
    assert summary.tokens_per_second == 4.0
    assert summary.requests_per_second == 2.0
    assert summary.errors == 0


def test_missing_token_count_leaves_rate_unset() -> None:
    summary = summarize_records(
        [_record(generated_tokens=None)],
        backend="mock",
        quantization="N/A",
        concurrency=1,
        repeat=1,
        elapsed_s=1.0,
        peak_rss_mb=None,
        memory_source="test fixture",
    )
    assert summary.tokens_per_second is None
    assert any("token counts unavailable" in note for note in summary.notes)


def test_aggregate_reports_mean_and_spread() -> None:
    runs = [_summary(e2e_p95_ms=20.0), replace(_summary(), e2e_p95_ms=30.0)]
    aggregate = aggregate_runs(runs)
    assert aggregate["e2e_p95_ms"]["mean"] == 25.0
    assert aggregate["e2e_p95_ms"]["spread"] == pytest.approx(7.0710678118654755)
    assert aggregate["e2e_p95_ms"]["samples"] == 2
