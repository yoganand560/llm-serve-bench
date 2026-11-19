"""YAML-backed configuration loader.

All behaviour that varies between runs (backend, model, quantization, timeouts,
concurrency, seed, regression threshold) lives here so no number is hard-coded in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

KNOWN_BACKENDS = frozenset({"mock", "llama_cpp", "vllm", "baseline_http"})


@dataclass(frozen=True)
class BackendConfig:
    """Which server adapter to use and how to reach it."""

    name: str = "mock"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_path: str | None = None
    quantization: str = "N/A"
    host: str = "127.0.0.1"
    port: int = 8080
    n_gpu_layers: int = -1
    ctx_size: int = 2048


@dataclass(frozen=True)
class GatewayConfig:
    """Gateway networking and request limits."""

    host: str = "127.0.0.1"
    port: int = 8000
    request_timeout_s: float = 60.0
    max_new_tokens_default: int = 64
    max_new_tokens_limit: int = 512


@dataclass(frozen=True)
class BenchmarkConfig:
    """Workload definition shared by every backend."""

    concurrency: tuple[int, ...] = (1, 4, 8, 16, 32)
    repeats: int = 3
    warmup_requests: int = 2
    seed: int = 42
    max_new_tokens: int = 64
    requests_per_run: int = 32
    prompt_set_path: str = "benchmarks/prompts.json"


@dataclass(frozen=True)
class RegressionConfig:
    """Inputs for the CI latency-regression check."""

    baseline_path: str = "benchmarks/regression_baseline.json"
    p95_threshold_ratio: float = 1.2


@dataclass(frozen=True)
class TrackingConfig:
    """Local MLflow tracking settings."""

    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "llm-serve-bench"


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration composed from YAML sections."""

    backend: BackendConfig
    gateway: GatewayConfig
    benchmark: BenchmarkConfig
    regression: RegressionConfig
    tracking: TrackingConfig


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section '{name}' must be a mapping")
    return value


def _integer(value: Any, name: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config file.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the YAML is malformed or a field fails validation.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    backend_raw = _section(raw, "backend")
    gateway_raw = _section(raw, "gateway")
    benchmark_raw = _section(raw, "benchmark")
    regression_raw = _section(raw, "regression")
    tracking_raw = _section(raw, "tracking")

    backend = BackendConfig(
        name=str(backend_raw.get("name", "mock")),
        model=str(backend_raw.get("model", "Qwen/Qwen2.5-1.5B-Instruct")),
        model_path=backend_raw.get("model_path"),
        quantization=str(backend_raw.get("quantization", "N/A")),
        host=str(backend_raw.get("host", "127.0.0.1")),
        port=_integer(backend_raw.get("port", 8080), "backend.port", 1, 65535),
        n_gpu_layers=_integer(backend_raw.get("n_gpu_layers", -1), "backend.n_gpu_layers"),
        ctx_size=_integer(backend_raw.get("ctx_size", 2048), "backend.ctx_size", 1),
    )
    if backend.name not in KNOWN_BACKENDS:
        raise ValueError(f"unknown backend '{backend.name}'; expected one of {sorted(KNOWN_BACKENDS)}")

    gateway = GatewayConfig(
        host=str(gateway_raw.get("host", "127.0.0.1")),
        port=_integer(gateway_raw.get("port", 8000), "gateway.port", 1, 65535),
        request_timeout_s=_positive_float(gateway_raw.get("request_timeout_s", 60.0), "gateway.request_timeout_s"),
        max_new_tokens_default=_integer(
            gateway_raw.get("max_new_tokens_default", 64), "gateway.max_new_tokens_default", 1
        ),
        max_new_tokens_limit=_integer(
            gateway_raw.get("max_new_tokens_limit", 512), "gateway.max_new_tokens_limit", 1
        ),
    )
    if gateway.max_new_tokens_default > gateway.max_new_tokens_limit:
        raise ValueError("gateway.max_new_tokens_default cannot exceed max_new_tokens_limit")

    concurrency_raw = benchmark_raw.get("concurrency", [1, 4, 8, 16, 32])
    if not isinstance(concurrency_raw, list) or not concurrency_raw:
        raise ValueError("benchmark.concurrency must be a non-empty list")
    concurrency = tuple(_integer(value, "benchmark.concurrency entry", 1) for value in concurrency_raw)

    benchmark = BenchmarkConfig(
        concurrency=concurrency,
        repeats=_integer(benchmark_raw.get("repeats", 3), "benchmark.repeats", 1),
        warmup_requests=_integer(benchmark_raw.get("warmup_requests", 2), "benchmark.warmup_requests", 0),
        seed=_integer(benchmark_raw.get("seed", 42), "benchmark.seed", 0),
        max_new_tokens=_integer(benchmark_raw.get("max_new_tokens", 64), "benchmark.max_new_tokens", 1),
        requests_per_run=_integer(benchmark_raw.get("requests_per_run", 32), "benchmark.requests_per_run", 1),
        prompt_set_path=str(benchmark_raw.get("prompt_set_path", "benchmarks/prompts.json")),
    )

    regression = RegressionConfig(
        baseline_path=str(regression_raw.get("baseline_path", "benchmarks/regression_baseline.json")),
        p95_threshold_ratio=_positive_float(
            regression_raw.get("p95_threshold_ratio", 1.2), "regression.p95_threshold_ratio"
        ),
    )

    tracking = TrackingConfig(
        tracking_uri=str(tracking_raw.get("tracking_uri", "file:./mlruns")),
        experiment_name=str(tracking_raw.get("experiment_name", "llm-serve-bench")),
    )

    return AppConfig(
        backend=backend,
        gateway=gateway,
        benchmark=benchmark,
        regression=regression,
        tracking=tracking,
    )


def load_config_from_env() -> AppConfig:
    """Load the config named by ``LLM_SERVE_CONFIG`` (default: configs/default.yaml)."""
    return load_config(os.getenv("LLM_SERVE_CONFIG", "configs/default.yaml"))
