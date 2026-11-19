"""The interface every serving adapter implements.

Keeping one small interface lets the gateway, benchmark harness and tests treat
llama.cpp, the Hugging Face baseline, the mock and the future vLLM adapter identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..config import BackendConfig


@dataclass(frozen=True)
class TokenChunk:
    """One streamed piece of a completion.

    ``token_count`` is only set on the final chunk when the backend can report an
    exact count. Benchmark code leaves tokens/sec unavailable when it is None.
    """

    text: str
    is_last: bool = False
    token_count: int | None = None


@dataclass(frozen=True)
class MemoryStats:
    """Memory observation with an explicit source label.

    On this Mac the useful value is process RSS (a unified-memory proxy), not VRAM.
    """

    peak_rss_mb: float | None = None
    peak_vram_mb: float | None = None
    source: str = "unavailable"


class ServeBackend(ABC):
    """Abstract base class for all generation backends."""

    name: str

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @abstractmethod
    def start(self) -> None:
        """Start any local server process required by this backend."""

    @abstractmethod
    def stop(self) -> None:
        """Release resources started by :meth:`start`."""

    @abstractmethod
    def health(self) -> bool:
        """Return True when the backend can accept requests."""

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> AsyncIterator[TokenChunk]:
        """Yield text chunks for one completion, ending with an ``is_last`` chunk."""

    @abstractmethod
    def memory_stats(self) -> MemoryStats:
        """Return the most recent memory observation."""

    def reset_memory_peak(self) -> None:
        """Clear the peak-memory accumulator before a measured run (optional)."""
        return None
