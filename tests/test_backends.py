"""Tests for backend selection, the mock stream and the vLLM stub."""

from __future__ import annotations

import asyncio

import pytest

from llm_serve_bench.backends.base import TokenChunk
from llm_serve_bench.backends.mock import MockBackend
from llm_serve_bench.backends.registry import build_backend
from llm_serve_bench.backends.vllm_stub import VLLMBackendStub
from llm_serve_bench.config import BackendConfig


def test_registry_builds_configured_backend() -> None:
    assert isinstance(build_backend(BackendConfig(name="mock")), MockBackend)
    assert isinstance(build_backend(BackendConfig(name="vllm")), VLLMBackendStub)


def test_registry_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        build_backend(BackendConfig(name="bogus"))


def _collect(backend: object) -> list[TokenChunk]:
    async def run() -> list[TokenChunk]:
        return [
            chunk
            async for chunk in backend.stream(  # type: ignore[attr-defined]
                "hi", max_new_tokens=2, temperature=0.0, seed=42
            )
        ]

    return asyncio.run(run())


def test_mock_stream_is_deterministic() -> None:
    first = _collect(MockBackend(BackendConfig()))
    second = _collect(MockBackend(BackendConfig()))
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]
    assert first[-1].is_last is True
    assert first[-1].token_count == 1


def test_vllm_stub_raises_instead_of_faking_results() -> None:
    stub = VLLMBackendStub(BackendConfig(name="vllm"))
    with pytest.raises(NotImplementedError, match="NVIDIA"):
        stub.start()
    with pytest.raises(NotImplementedError):
        _collect(stub)
