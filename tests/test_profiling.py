"""Test that the profiler harness collects statistics without a model."""

from __future__ import annotations

from pathlib import Path

from llm_serve_bench.bench.profiling import profile_requests
from llm_serve_bench.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_profile_requests_collects_calls() -> None:
    config = load_config(CONFIG_PATH)
    stats = profile_requests(config, requests=2, max_new_tokens=2, prompt="hi")
    assert stats.total_calls > 0


def test_profile_rejects_zero_requests() -> None:
    import pytest

    config = load_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="at least 1"):
        profile_requests(config, requests=0, max_new_tokens=2, prompt="hi")
