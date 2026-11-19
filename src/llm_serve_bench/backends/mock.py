"""Deterministic backend used by tests and for CI regression checks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .base import MemoryStats, ServeBackend, TokenChunk

_DEMO_TOKENS = ("The", " answer", " is", " deterministic", ".")


class MockBackend(ServeBackend):
    """Emits a fixed token sequence; no model, GPU or network needed."""

    name = "mock"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def health(self) -> bool:
        return True

    def memory_stats(self) -> MemoryStats:
        return MemoryStats(source="mock backend - no real memory usage")

    async def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> AsyncIterator[TokenChunk]:
        count = min(max_new_tokens, len(_DEMO_TOKENS))
        for index in range(count):
            await asyncio.sleep(0)  # let other tasks run, as a real network stream would
            last = index == count - 1
            yield TokenChunk(text=_DEMO_TOKENS[index], is_last=last, token_count=1 if last else None)
