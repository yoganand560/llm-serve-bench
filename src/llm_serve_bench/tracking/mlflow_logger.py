"""Log benchmark results to a local MLflow file store.

MLflow is imported lazily so the mapping functions can be tested without installing it.
Only values present in a real result JSON are logged; missing values are omitted.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType


def _mlflow() -> ModuleType:
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("MLflow is not installed. Install the 'tracking' extra first.") from exc
    return mlflow


def extract_params(result: dict[str, object]) -> dict[str, str]:
    """Flatten result metadata into MLflow string parameters."""
    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("result.metadata must be a mapping")
    keys = (
        "model",
        "backend",
        "quantization",
        "max_new_tokens",
        "repeats",
        "requests_per_run",
        "seed",
    )
    params = {key: str(metadata[key]) for key in keys if key in metadata}
    aggregates = result.get("aggregates", {})
    if isinstance(aggregates, dict):
        params["concurrency_levels"] = ",".join(str(key) for key in aggregates)
    return params


def flatten_metrics(result: dict[str, object]) -> dict[str, float]:
    """Flatten per-concurrency aggregates into MLflow numeric metrics."""
    metrics: dict[str, float] = {}
    aggregates = result.get("aggregates", {})
    if not isinstance(aggregates, dict):
        raise ValueError("result.aggregates must be a mapping")
    for level, aggregate in aggregates.items():
        if not isinstance(aggregate, dict):
            raise ValueError(f"aggregate for {level} must be a mapping")
        for metric_name, statistics in aggregate.items():
            if not isinstance(statistics, dict):
                raise ValueError(f"statistics for {metric_name} must be a mapping")
            for statistic in ("mean", "spread"):
                value = statistics.get(statistic)
                if value is not None:
                    metrics[f"{level}.{metric_name}.{statistic}"] = float(value)

    runs = result.get("runs", [])
    if isinstance(runs, list):
        errors = [int(run.get("errors", 0)) for run in runs if isinstance(run, dict)]
        if errors:
            metrics["errors_total"] = float(sum(errors))
    return metrics


def log_benchmark_result(
    result: dict[str, object],
    *,
    result_path: Path | None = None,
    extra_artifacts: list[Path] | None = None,
    run_name: str | None = None,
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "llm-serve-bench",
) -> str:
    """Log one benchmark document and return the created MLflow run id."""
    artifacts = list(extra_artifacts or [])
    if result_path is not None:
        artifacts.insert(0, result_path)
    for artifact in artifacts:
        if not Path(artifact).is_file():
            raise FileNotFoundError(f"artifact does not exist: {artifact}")

    mlflow = _mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(extract_params(result))
        mlflow.log_metrics(flatten_metrics(result))
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        return str(active_run.info.run_id)
