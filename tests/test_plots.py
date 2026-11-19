"""Tests for chart-series extraction. Fixtures are synthetic logic tests, not results."""

from __future__ import annotations

from llm_serve_bench.bench.plots import extract_series


def _result() -> dict[str, object]:
    return {
        "metadata": {"backend": "llama_cpp", "quantization": "Q4_K_M"},
        "aggregates": {
            "concurrency_4": {
                "tokens_per_second": {"mean": 100.0},
                "e2e_p95_ms": {"mean": 50.0},
                "peak_rss_mb": {"mean": 900.0},
            },
            "concurrency_1": {
                "tokens_per_second": {"mean": 40.0},
                "e2e_p95_ms": {"mean": 20.0},
                "peak_rss_mb": {"mean": None},
            },
        },
    }


def test_extract_series_sorts_levels_and_keeps_missing_values() -> None:
    series = extract_series(_result())
    assert series.label == "llama_cpp / Q4_K_M"
    assert series.concurrency == (1, 4)
    assert series.throughput == (40.0, 100.0)
    assert series.p95_ms == (20.0, 50.0)
    assert series.memory_mb == (None, 900.0)
