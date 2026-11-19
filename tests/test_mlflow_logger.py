"""Tests for MLflow param/metric mapping and logging calls, without MLflow installed."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_serve_bench.tracking import mlflow_logger
from llm_serve_bench.tracking.mlflow_logger import extract_params, flatten_metrics, log_benchmark_result

RESULT = {
    "metadata": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "backend": "mock",
        "quantization": "N/A",
        "max_new_tokens": 64,
        "repeats": 3,
        "requests_per_run": 32,
        "seed": 42,
    },
    "runs": [{"errors": 0}, {"errors": 1}],
    "aggregates": {
        "concurrency_1": {
            "e2e_p95_ms": {"mean": 12.5, "spread": 0.5, "samples": 3, "unit": "ms"},
            "tokens_per_second": {"mean": None, "spread": None, "samples": 0, "unit": "tokens/s"},
        }
    },
}


def test_extract_params_flattens_metadata() -> None:
    params = extract_params(RESULT)
    assert params["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert params["quantization"] == "N/A"
    assert params["concurrency_levels"] == "concurrency_1"
    assert params["seed"] == "42"


def test_flatten_metrics_skips_missing_values() -> None:
    metrics = flatten_metrics(RESULT)
    assert metrics["concurrency_1.e2e_p95_ms.mean"] == 12.5
    assert metrics["concurrency_1.e2e_p95_ms.spread"] == 0.5
    assert "concurrency_1.tokens_per_second.mean" not in metrics
    assert metrics["errors_total"] == 1.0


def test_logging_uses_fake_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    @contextmanager
    def fake_start_run(run_name: str | None = None):  # type: ignore[no-untyped-def]
        calls["run_name"] = run_name
        yield SimpleNamespace(info=SimpleNamespace(run_id="fake-id"))

    fake = SimpleNamespace(
        set_tracking_uri=lambda uri: calls.__setitem__("tracking_uri", uri),
        set_experiment=lambda name: calls.__setitem__("experiment", name),
        start_run=fake_start_run,
        log_params=lambda params: calls.__setitem__("params", params),
        log_metrics=lambda metrics: calls.__setitem__("metrics", metrics),
        log_artifact=lambda path: calls.setdefault("artifacts", []).append(path),
    )
    monkeypatch.setattr(mlflow_logger, "_mlflow", lambda: fake)

    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(RESULT), encoding="utf-8")
    run_id = log_benchmark_result(RESULT, result_path=result_path, run_name="smoke")

    assert run_id == "fake-id"
    assert calls["experiment"] == "llm-serve-bench"
    assert calls["artifacts"] == [str(result_path)]
    assert "concurrency_1.e2e_p95_ms.mean" in calls["metrics"]  # type: ignore[operator]


def test_missing_artifact_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlflow_logger, "_mlflow", lambda: SimpleNamespace())
    with pytest.raises(FileNotFoundError):
        log_benchmark_result(RESULT, result_path=Path("does_not_exist.json"))
