"""Small in-process counter registry exposed by ``GET /metrics``.

No metrics server is needed; counters are a plain JSON snapshot. They describe gateway
activity only and are never used as benchmark latency numbers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class GatewayMetrics:
    """Thread-safe counters for gateway requests."""

    requests_total: int = 0
    stream_requests_total: int = 0
    errors_total: int = 0
    timeouts_total: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_request(self) -> None:
        with self._lock:
            self.requests_total += 1

    def record_stream_request(self) -> None:
        with self._lock:
            self.stream_requests_total += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors_total += 1

    def record_timeout(self) -> None:
        with self._lock:
            self.timeouts_total += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "stream_requests_total": self.stream_requests_total,
                "errors_total": self.errors_total,
                "timeouts_total": self.timeouts_total,
            }
