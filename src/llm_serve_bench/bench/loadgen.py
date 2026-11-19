"""Async load generator driving the gateway's streaming endpoint."""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import time

import httpx

from .metrics import RequestRecord


async def _one_request(
    client: httpx.AsyncClient,
    endpoint: str,
    prompt: str,
    *,
    prompt_id: int,
    concurrency: int,
    max_new_tokens: int,
    seed: int,
) -> RequestRecord:
    """Send one streaming request and derive timings from chunk arrival times."""
    started = time.perf_counter()
    ttft_ms: float | None = None
    chunk_arrivals: list[float] = []
    itl_ms: float | None = None
    generated_tokens: int | None = None
    error: str | None = None

    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "seed": seed,
    }
    try:
        async with client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                body = line.removeprefix("data:").strip()
                if not body:
                    continue
                event = json.loads(body)
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                arrived = time.perf_counter()
                if event.get("text"):
                    if ttft_ms is None:
                        ttft_ms = (arrived - started) * 1000
                    chunk_arrivals.append(arrived)
                if event.get("generated_tokens") is not None:
                    generated_tokens = int(event["generated_tokens"])
                if event.get("is_last"):
                    break
    except Exception as exc:  # noqa: BLE001 - record failures instead of aborting the run
        error = f"{type(exc).__name__}: {exc}"

    ended = time.perf_counter()
    if len(chunk_arrivals) >= 2:
        gaps = [(later - earlier) * 1000 for earlier, later in zip(chunk_arrivals, chunk_arrivals[1:])]
        itl_ms = statistics.fmean(gaps)

    return RequestRecord(
        prompt_id=prompt_id,
        concurrency=concurrency,
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
        e2e_ms=(ended - started) * 1000,
        generated_tokens=generated_tokens,
        error=error,
    )


async def warmup(
    client: httpx.AsyncClient,
    endpoint: str,
    prompts: list[str],
    *,
    count: int,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Send sequential warm-up requests; their timings are discarded."""
    for index in range(count):
        await _one_request(
            client,
            endpoint,
            prompts[index % len(prompts)],
            prompt_id=index,
            concurrency=1,
            max_new_tokens=max_new_tokens,
            seed=seed + index,
        )


async def run_concurrency_level(
    client: httpx.AsyncClient,
    endpoint: str,
    prompts: list[str],
    *,
    concurrency: int,
    total_requests: int,
    max_new_tokens: int,
    seed: int,
) -> list[RequestRecord]:
    """Run a fixed number of requests with at most ``concurrency`` in flight."""
    order = list(range(total_requests))
    random.Random(seed).shuffle(order)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int) -> RequestRecord:
        async with semaphore:
            return await _one_request(
                client,
                endpoint,
                prompts[order[index] % len(prompts)],
                prompt_id=order[index],
                concurrency=concurrency,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )

    return list(await asyncio.gather(*(one(index) for index in range(total_requests))))
