"""Tests for the YAML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_serve_bench.config import load_config, load_config_from_env

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_default_config_loads() -> None:
    config = load_config(CONFIG_DIR / "default.yaml")
    assert config.backend.name == "mock"
    assert config.benchmark.concurrency == (1, 4, 8, 16, 32)
    assert config.benchmark.repeats == 3
    assert config.benchmark.seed == 42
    assert config.regression.p95_threshold_ratio == 1.2


def test_quant_config_uses_local_gguf_path() -> None:
    config = load_config(CONFIG_DIR / "llama_cpp_q4.yaml")
    assert config.backend.name == "llama_cpp"
    assert config.backend.quantization == "Q4_K_M"
    assert config.backend.model_path is not None
    assert config.backend.model_path.endswith(".gguf")
    assert config.backend.n_gpu_layers == -1


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(CONFIG_DIR / "does_not_exist.yaml")


def test_unknown_backend_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("backend:\n  name: bogus\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown backend"):
        load_config(path)


def test_zero_concurrency_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("benchmark:\n  concurrency: [0, 4]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="concurrency"):
        load_config(path)


def test_timeout_must_be_positive(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("gateway:\n  request_timeout_s: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="request_timeout_s"):
        load_config(path)


def test_env_override_selects_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_SERVE_CONFIG", str(CONFIG_DIR / "default.yaml"))
    assert load_config_from_env().backend.name == "mock"
