"""FastAPI gateway in front of any ``ServeBackend``.

Responsibilities: Pydantic validation, a total request deadline, explicit HTTP error
mapping, a streamed SSE endpoint for latency measurements, and a small ``/metrics``
snapshot. Benchmarks drive the gateway so its overhead is included in measurements.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from ..backends.base import ServeBackend, TokenChunk
from ..backends.registry import build_backend
from ..config import AppConfig, load_config_from_env
from .errors import BackendError, BackendNotReadyError, BackendRequestError, BackendTimeoutError
from .metrics import GatewayMetrics
from .schemas import GenerateRequest, GenerateResponse, HealthResponse, MetricsResponse, StreamEvent


async def _with_deadline(
    chunks: AsyncIterator[TokenChunk],
    seconds: float,
) -> AsyncIterator[TokenChunk]:
    """Yield from ``chunks`` but stop if the whole stream exceeds one deadline."""
    iterator = chunks.__aiter__()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError("request deadline exceeded")
        try:
            item = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            return
        yield item


def create_app(config: AppConfig, backend: ServeBackend | None = None) -> FastAPI:
    """Build the gateway app.

    Args:
        config: Validated application config.
        backend: Injected backend (tests use the mock). Defaults to the configured one.
    """
    resolved_backend = backend or build_backend(config.backend)
    metrics = GatewayMetrics()
    app = FastAPI(title="llm-serve-bench gateway", version="0.1.0")

    @app.exception_handler(BackendNotReadyError)
    async def _handle_not_ready(_: Request, exc: BackendNotReadyError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)})

    @app.exception_handler(BackendTimeoutError)
    async def _handle_timeout(_: Request, exc: BackendTimeoutError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_504_GATEWAY_TIMEOUT, content={"detail": str(exc)})

    @app.exception_handler(BackendRequestError)
    async def _handle_backend_error(_: Request, exc: BackendRequestError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)})

    def _validate(request: GenerateRequest) -> None:
        if request.max_new_tokens > config.gateway.max_new_tokens_limit:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"max_new_tokens exceeds {config.gateway.max_new_tokens_limit}",
            )
        if not resolved_backend.health():
            raise BackendNotReadyError("Backend is not ready.")

    def _stream_args(request: GenerateRequest) -> dict[str, Any]:
        return {
            "prompt": request.prompt,
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "seed": request.seed,
        }

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        ready = resolved_backend.health()
        return HealthResponse(
            status="ok" if ready else "degraded",
            backend=resolved_backend.name,
            model=config.backend.model,
            ready=ready,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics() -> MetricsResponse:
        memory = resolved_backend.memory_stats()
        return MetricsResponse(
            **metrics.snapshot(),
            peak_rss_mb=memory.peak_rss_mb,
            memory_source=memory.source,
        )

    @app.post("/metrics/reset")
    async def reset_metrics() -> dict[str, str]:
        resolved_backend.reset_memory_peak()
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        _validate(request)
        metrics.record_request()
        parts: list[str] = []
        tokens: int | None = None
        try:
            chunks = resolved_backend.stream(**_stream_args(request))
            async for chunk in _with_deadline(chunks, config.gateway.request_timeout_s):
                parts.append(chunk.text)
                if chunk.token_count is not None:
                    tokens = chunk.token_count
        except TimeoutError as exc:
            metrics.record_timeout()
            raise BackendTimeoutError("Backend exceeded the request timeout.") from exc
        except BackendError:
            metrics.record_error()
            raise
        except Exception as exc:  # noqa: BLE001 - map unexpected backend failures to 502
            metrics.record_error()
            raise BackendRequestError(f"Backend request failed: {exc}") from exc

        return GenerateResponse(
            text="".join(parts),
            generated_tokens=tokens,
            backend=resolved_backend.name,
            model=config.backend.model,
        )

    @app.post("/generate/stream")
    async def generate_stream(request: GenerateRequest) -> StreamingResponse:
        _validate(request)
        metrics.record_stream_request()

        async def events() -> AsyncIterator[str]:
            try:
                chunks = resolved_backend.stream(**_stream_args(request))
                async for chunk in _with_deadline(chunks, config.gateway.request_timeout_s):
                    event = StreamEvent(
                        text=chunk.text,
                        is_last=chunk.is_last,
                        generated_tokens=chunk.token_count,
                    )
                    yield f"data: {event.model_dump_json()}\n\n"
            except TimeoutError:
                metrics.record_timeout()
                yield f"data: {json.dumps({'error': 'backend timeout'})}\n\n"
            except Exception as exc:  # noqa: BLE001 - keep the SSE contract on failure
                metrics.record_error()
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    """CLI entrypoint: ``python -m llm_serve_bench.gateway.app``."""
    import uvicorn

    config = load_config_from_env()
    backend = build_backend(config.backend)
    backend.start()
    try:
        app = create_app(config, backend)
        uvicorn.run(app, host=config.gateway.host, port=config.gateway.port)
    finally:
        backend.stop()


if __name__ == "__main__":
    main()
