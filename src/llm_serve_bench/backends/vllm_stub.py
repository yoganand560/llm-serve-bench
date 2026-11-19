"""Documented vLLM adapter stub for a future NVIDIA GPU host.

This file intentionally does not run here. It implements the same ``ServeBackend``
interface so the gateway and benchmark harness need no changes on a GPU machine.
See ``docs/GPU_PENDING.md`` for the exact run instructions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from .base import MemoryStats, ServeBackend, TokenChunk

_NOT_RUNNABLE = (
    "The vLLM backend is a stub: this project was developed on an Apple Silicon Mac "
    "with no CUDA GPU. Run it on an NVIDIA machine (see docs/GPU_PENDING.md)."
)


class VLLMBackendStub(ServeBackend):
    """Interface-compatible placeholder; every operation raises ``NotImplementedError``."""

    name = "vllm"

    def start(self) -> None:
        raise NotImplementedError(_NOT_RUNNABLE)

    def stop(self) -> None:
        raise NotImplementedError(_NOT_RUNNABLE)

    def health(self) -> bool:
        return False

    async def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> AsyncIterator[TokenChunk]:
        raise NotImplementedError(_NOT_RUNNABLE)
        yield TokenChunk(text="")  # pragma: no cover - makes this an async generator

    def memory_stats(self) -> MemoryStats:
        raise NotImplementedError(_NOT_RUNNABLE)
