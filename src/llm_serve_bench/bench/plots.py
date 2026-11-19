"""Pure extraction of chart series from saved result documents.

Keeping this free of matplotlib lets the data mapping be tested without plotting deps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Series:
    """One backend/quantization series ready to plot."""

    label: str
    concurrency: tuple[int, ...]
    throughput: tuple[float | None, ...]
    p95_ms: tuple[float | None, ...]
    memory_mb: tuple[float | None, ...]


def _level_number(level: str) -> int:
    suffix = level.removeprefix("concurrency_")
    return int(suffix) if suffix.isdigit() else 0


def _mean(aggregate: dict[str, object], metric: str) -> float | None:
    entry = aggregate.get(metric)
    if not isinstance(entry, dict):
        return None
    value = entry.get("mean")
    return float(value) if value is not None else None


def extract_series(result: dict[str, object]) -> Series:
    """Convert one benchmark result JSON into plottable series."""
    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("result.metadata must be a mapping")
    label = f"{metadata.get('backend', 'unknown')} / {metadata.get('quantization', 'unknown')}"

    aggregates = result.get("aggregates", {})
    if not isinstance(aggregates, dict):
        raise ValueError("result.aggregates must be a mapping")
    levels = sorted(aggregates, key=_level_number)

    concurrency: list[int] = []
    throughput: list[float | None] = []
    p95_ms: list[float | None] = []
    memory_mb: list[float | None] = []
    for level in levels:
        aggregate = aggregates[level]
        if not isinstance(aggregate, dict):
            raise ValueError(f"aggregate for {level} must be a mapping")
        concurrency.append(_level_number(level))
        throughput.append(_mean(aggregate, "tokens_per_second"))
        p95_ms.append(_mean(aggregate, "e2e_p95_ms"))
        memory_mb.append(_mean(aggregate, "peak_rss_mb"))

    return Series(
        label=label,
        concurrency=tuple(concurrency),
        throughput=tuple(throughput),
        p95_ms=tuple(p95_ms),
        memory_mb=tuple(memory_mb),
    )
