"""Adapter for the Phase 2 Hugging Face baseline server.

The baseline returns one complete response, so this backend emits a single text chunk.
The benchmark harness records that as TTFT == end-to-end latency and ITL as N/A.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from ..config import BackendConfig
from .base import MemoryStats, ServeBackend, TokenChunk


class BaselineHttpBackend(ServeBackend):
    """Talks to the baseline FastAPI server over HTTP."""

    name = "baseline_http"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._base_url = f"http://{self.config.host}:{self.config.port}"

    def start(self) -> None:
        return None  # baseline server is started separately via `make serve-baseline`

    def stop(self) -> None:
        return None

    def health(self) -> bool:
        try:
            return httpx.get(f"{self._base_url}/health", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    async def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> AsyncIterator[TokenChunk]:
        payload: dict[str, object] = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        async with httpx.AsyncClient(base_url=self._base_url, timeout=None) as client:
            response = await client.post("/generate", json=payload)
            response.raise_for_status()
            body = response.json()
        yield TokenChunk(text=body["text"], token_count=int(body["generated_tokens"]))
        yield TokenChunk(text="", is_last=True)

    def memory_stats(self) -> MemoryStats:
        return MemoryStats(source="unavailable - baseline runs in a separate process")
