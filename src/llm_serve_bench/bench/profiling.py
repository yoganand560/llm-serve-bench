"""cProfile the in-process gateway path with the configured backend.

Limitation: this profiles the Python gateway process only. If the backend is
llama.cpp or vLLM it runs in a separate native process; its internal time appears here
only as time spent waiting on the HTTP client, never as attributed native functions.
"""

from __future__ import annotations

import asyncio
import cProfile
import pstats

import httpx

from ..backends.registry import build_backend
from ..config import AppConfig
from ..gateway.app import create_app


async def _scenario(
    app: object,
    *,
    requests: int,
    max_new_tokens: int,
    prompt: str,
) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://profile") as client:
        for index in range(requests):
            payload = {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "seed": 42,
            }
            if index % 2 == 0:
                await client.post("/generate", json=payload)
            else:
                async with client.stream("POST", "/generate/stream", json=payload) as response:
                    await response.aread()


def profile_requests(
    config: AppConfig,
    *,
    requests: int,
    max_new_tokens: int,
    prompt: str,
) -> pstats.Stats:
    """Run requests under cProfile and return the collected statistics."""
    if requests < 1:
        raise ValueError("requests must be at least 1")
    backend = build_backend(config.backend)
    backend.start()
    try:
        app = create_app(config, backend)
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            asyncio.run(
                _scenario(app, requests=requests, max_new_tokens=max_new_tokens, prompt=prompt)
            )
        finally:
            profiler.disable()
        return pstats.Stats(profiler)
    finally:
        backend.stop()
