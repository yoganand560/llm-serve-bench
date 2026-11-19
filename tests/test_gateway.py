"""Tests for the gateway using the deterministic mock backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from llm_serve_bench.backends.base import MemoryStats, ServeBackend, TokenChunk
from llm_serve_bench.backends.mock import MockBackend
from llm_serve_bench.config import AppConfig, load_config
from llm_serve_bench.gateway.app import create_app

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _config() -> AppConfig:
    return load_config(CONFIG_PATH)


def _client(config: AppConfig | None = None) -> TestClient:
    resolved = config or _config()
    return TestClient(create_app(resolved, MockBackend(resolved.backend)))


def test_health_reports_mock_backend() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["backend"] == "mock"
    assert payload["ready"] is True


def test_generate_returns_aggregated_text() -> None:
    response = _client().post("/generate", json={"prompt": "hi", "max_new_tokens": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "The answer is"
    assert payload["generated_tokens"] == 1
    assert payload["backend"] == "mock"


def test_stream_endpoint_emits_sse_events() -> None:
    with _client().stream(
        "POST", "/generate/stream", json={"prompt": "hi", "max_new_tokens": 2}
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert '"is_last":true' in body
    assert '"generated_tokens":1' in body


def test_empty_prompt_is_rejected() -> None:
    assert _client().post("/generate", json={"prompt": ""}).status_code == 422


def test_token_limit_is_enforced() -> None:
    response = _client().post("/generate", json={"prompt": "hi", "max_new_tokens": 9999})
    assert response.status_code == 422


def test_metrics_endpoint_counts_requests() -> None:
    client = _client()
    client.get("/metrics")
    client.post("/generate", json={"prompt": "hi"})
    client.post("/generate/stream", json={"prompt": "hi"})
    payload = client.get("/metrics").json()
    assert payload["requests_total"] == 1
    assert payload["stream_requests_total"] == 1


class _SlowBackend(ServeBackend):
    """Backend that never finishes before a very short deadline."""

    name = "slow_mock"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def health(self) -> bool:
        return True

    def memory_stats(self) -> MemoryStats:
        return MemoryStats()

    async def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> AsyncIterator[TokenChunk]:
        import asyncio

        await asyncio.sleep(0.05)
        yield TokenChunk(text="late", is_last=True)


def test_timeout_maps_to_504() -> None:
    base = _config()
    config = replace(base, gateway=replace(base.gateway, request_timeout_s=0.001))
    client = TestClient(create_app(config, _SlowBackend(config.backend)))
    response = client.post("/generate", json={"prompt": "hi"})
    assert response.status_code == 504
