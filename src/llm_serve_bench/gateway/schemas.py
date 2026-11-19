"""Request and response models for the gateway."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Validated generation request."""

    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = Field(default=None, description="Fixed seed for reproducible runs.")


class GenerateResponse(BaseModel):
    """Aggregated completion returned by ``POST /generate``."""

    text: str
    generated_tokens: int | None = None
    backend: str
    model: str


class StreamEvent(BaseModel):
    """One Server-Sent Event payload from ``POST /generate/stream``."""

    text: str
    is_last: bool = False
    generated_tokens: int | None = None


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str
    backend: str
    model: str
    ready: bool


class MetricsResponse(BaseModel):
    """Snapshot returned by ``GET /metrics``."""

    requests_total: int
    stream_requests_total: int
    errors_total: int
    timeouts_total: int
    peak_rss_mb: float | None = None
    memory_source: str = "unavailable"
