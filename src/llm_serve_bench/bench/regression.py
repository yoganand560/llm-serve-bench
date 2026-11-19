"""Latency-regression comparison between two saved benchmark result documents.

Runs on plain JSON, so it works on CI runners with no GPU, no model and no server.
A placeholder baseline with null metrics is rejected rather than silently passing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegressionComparison:
    """One concurrency level's p95 comparison."""

    level: str
    baseline_p95_ms: float
    fresh_p95_ms: float
    ratio: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class RegressionCheck:
    """Outcome across all comparable levels."""

    comparisons: tuple[RegressionComparison, ...]
    skipped_levels: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return bool(self.comparisons) and all(item.passed for item in self.comparisons) and not self.skipped_levels


def _p95_ms(document: dict[str, object], level: str) -> float | None:
    aggregates = document.get("aggregates")
    if not isinstance(aggregates, dict):
        raise ValueError("result document must contain an 'aggregates' mapping")
    level_data = aggregates.get(level)
    if not isinstance(level_data, dict):
        return None
    metric = level_data.get("e2e_p95_ms")
    if not isinstance(metric, dict):
        return None
    value = metric.get("mean")
    return float(value) if value is not None else None


def check_regression(
    baseline: dict[str, object],
    fresh: dict[str, object],
    *,
    threshold: float,
) -> RegressionCheck:
    """Compare p95 E2E latency at every level in the baseline document.

    Args:
        baseline: Stored reference result JSON.
        fresh: Newly produced result JSON.
        threshold: Maximum allowed fresh/baseline ratio (e.g. 1.2 means +20%).

    Raises:
        ValueError: if no level has two measured p95 values, or threshold is invalid.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    baseline_aggregates = baseline.get("aggregates")
    fresh_aggregates = fresh.get("aggregates")
    if not isinstance(baseline_aggregates, dict) or not isinstance(fresh_aggregates, dict):
        raise ValueError("both result documents must contain an 'aggregates' mapping")

    comparisons: list[RegressionComparison] = []
    skipped: list[str] = []
    for level in baseline_aggregates:
        base_value = _p95_ms(baseline, level)
        fresh_value = _p95_ms(fresh, level)
        if base_value is None or fresh_value is None or base_value <= 0:
            skipped.append(level)
            continue
        ratio = fresh_value / base_value
        comparisons.append(
            RegressionComparison(
                level=level,
                baseline_p95_ms=base_value,
                fresh_p95_ms=fresh_value,
                ratio=ratio,
                threshold=threshold,
                passed=ratio <= threshold,
            )
        )

    if not comparisons:
        raise ValueError(
            "no comparable measured p95 values; the baseline may still be an unmeasured placeholder"
        )
    return RegressionCheck(tuple(comparisons), tuple(skipped))
