"""Metric definitions and statistics for benchmark runs.

Percentiles use the nearest-rank method on successful requests. Spread across repeats
is the sample standard deviation; with a single sample it is reported as 0.0.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

_UNITS = {
    "ttft_p50_ms": "ms",
    "itl_mean_ms": "ms",
    "e2e_p50_ms": "ms",
    "e2e_p95_ms": "ms",
    "e2e_p99_ms": "ms",
    "tokens_per_second": "tokens/s",
    "requests_per_second": "requests/s",
    "peak_rss_mb": "MiB",
}


@dataclass(frozen=True)
class RequestRecord:
    """Per-request observations. Stored raw in result JSON."""

    prompt_id: int
    concurrency: int
    ttft_ms: float | None
    itl_ms: float | None
    e2e_ms: float
    generated_tokens: int | None
    error: str | None = None


@dataclass(frozen=True)
class RunMetrics:
    """Summary of one repeat at one concurrency level."""

    backend: str
    quantization: str
    concurrency: int
    repeat: int
    requests: int
    errors: int
    elapsed_s: float
    ttft_p50_ms: float | None
    itl_mean_ms: float | None
    e2e_p50_ms: float | None
    e2e_p95_ms: float | None
    e2e_p99_ms: float | None
    tokens_per_second: float | None
    requests_per_second: float | None
    peak_rss_mb: float | None
    memory_source: str
    notes: tuple[str, ...] = ()


def percentile(values: list[float], quantile: float) -> float | None:
    """Nearest-rank percentile of ``values`` at ``quantile`` in (0, 1]."""
    if not values:
        return None
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize_records(
    records: list[RequestRecord],
    *,
    backend: str,
    quantization: str,
    concurrency: int,
    repeat: int,
    elapsed_s: float,
    peak_rss_mb: float | None,
    memory_source: str,
) -> RunMetrics:
    """Turn one repeat's raw records into a typed summary."""
    successful = [record for record in records if record.error is None]
    failures = len(records) - len(successful)
    notes: list[str] = []
    if failures:
        notes.append(f"{failures} failed requests are included in wall time but not latency percentiles")

    e2e = [record.e2e_ms for record in successful]
    ttft = [record.ttft_ms for record in successful if record.ttft_ms is not None]
    itl = [record.itl_ms for record in successful if record.itl_ms is not None]
    token_counts = [record.generated_tokens for record in successful]

    if any(count is None for count in token_counts):
        tokens_per_second = None
        notes.append("token counts unavailable for at least one request; tokens/s left unset")
    else:
        total_tokens = sum(count or 0 for count in token_counts)
        tokens_per_second = total_tokens / elapsed_s if elapsed_s > 0 else None
        if not successful:
            tokens_per_second = None

    return RunMetrics(
        backend=backend,
        quantization=quantization,
        concurrency=concurrency,
        repeat=repeat,
        requests=len(records),
        errors=failures,
        elapsed_s=elapsed_s,
        ttft_p50_ms=percentile(ttft, 0.5),
        itl_mean_ms=statistics.fmean(itl) if itl else None,
        e2e_p50_ms=percentile(e2e, 0.5),
        e2e_p95_ms=percentile(e2e, 0.95),
        e2e_p99_ms=percentile(e2e, 0.99),
        tokens_per_second=tokens_per_second,
        requests_per_second=len(records) / elapsed_s if elapsed_s > 0 else None,
        peak_rss_mb=peak_rss_mb,
        memory_source=memory_source,
        notes=tuple(notes),
    )


def aggregate_runs(runs: list[RunMetrics]) -> dict[str, dict[str, object]]:
    """Compute mean and spread across repeats for each numeric metric."""
    aggregated: dict[str, dict[str, object]] = {}
    for name, unit in _UNITS.items():
        values = [float(getattr(run, name)) for run in runs if getattr(run, name) is not None]
        aggregated[name] = {
            "mean": statistics.fmean(values) if values else None,
            "spread": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            "samples": len(values),
            "unit": unit,
        }
    return aggregated
