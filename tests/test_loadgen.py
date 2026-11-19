"""Load-generator test using the gateway app over an in-process HTTP transport."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from llm_serve_bench.backends.mock import MockBackend
from llm_serve_bench.bench.loadgen import run_concurrency_level, warmup
from llm_serve_bench.config import load_config
from llm_serve_bench.gateway.app import create_app

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_concurrency_level_produces_complete_records() -> None:
    config = load_config(CONFIG_PATH)
    app = create_app(config, MockBackend(config.backend))

    async def scenario() -> list[object]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await warmup(client, "/generate/stream", ["hi"], count=1, max_new_tokens=2, seed=42)
            return await run_concurrency_level(
                client,
                "/generate/stream",
                ["hi"],
                concurrency=2,
                total_requests=4,
                max_new_tokens=2,
                seed=42,
            )

    records = asyncio.run(scenario())
    assert len(records) == 4
    assert all(getattr(record, "error", None) is None for record in records)
    assert all(getattr(record, "ttft_ms", None) is not None for record in records)
    assert all(getattr(record, "generated_tokens", None) == 1 for record in records)
