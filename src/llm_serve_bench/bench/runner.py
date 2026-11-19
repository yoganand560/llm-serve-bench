"""Orchestrate warm-up, repeats and result serialisation for a full benchmark."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..config import AppConfig
from .loadgen import run_concurrency_level, warmup
from .metrics import RunMetrics, aggregate_runs, summarize_records

STREAM_ENDPOINT = "/generate/stream"


async def _read_memory(client: httpx.AsyncClient) -> tuple[float | None, str]:
    response = await client.get("/metrics")
    response.raise_for_status()
    body = response.json()
    return body.get("peak_rss_mb"), body.get("memory_source", "unavailable")


async def run_benchmark(
    config: AppConfig,
    *,
    gateway_url: str,
    backend_label: str,
    quantization: str,
    prompts: list[str],
    output_path: Path | None = None,
) -> dict[str, object]:
    """Run the full workload and return (and optionally save) a result document.

    The gateway must already be running. All numbers in the output come from this run;
    no defaults or estimates are substituted for missing measurements.
    """
    benchmark = config.benchmark
    async with httpx.AsyncClient(base_url=gateway_url, timeout=None) as client:
        await warmup(
            client,
            STREAM_ENDPOINT,
            prompts,
            count=benchmark.warmup_requests,
            max_new_tokens=benchmark.max_new_tokens,
            seed=benchmark.seed,
        )

        summaries: list[RunMetrics] = []
        raw_records: list[dict[str, object]] = []
        for level in benchmark.concurrency:
            for repeat in range(1, benchmark.repeats + 1):
                await client.post("/metrics/reset")
                started = time.perf_counter()
                records = await run_concurrency_level(
                    client,
                    STREAM_ENDPOINT,
                    prompts,
                    concurrency=level,
                    total_requests=benchmark.requests_per_run,
                    max_new_tokens=benchmark.max_new_tokens,
                    seed=benchmark.seed + repeat,
                )
                elapsed = time.perf_counter() - started
                peak_rss_mb, memory_source = await _read_memory(client)
                summary = summarize_records(
                    records,
                    backend=backend_label,
                    quantization=quantization,
                    concurrency=level,
                    repeat=repeat,
                    elapsed_s=elapsed,
                    peak_rss_mb=peak_rss_mb,
                    memory_source=memory_source,
                )
                summaries.append(summary)
                raw_records.extend(
                    {"concurrency": level, "repeat": repeat, **asdict(record)} for record in records
                )

    result: dict[str, object] = {
        "schema_version": 1,
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "backend": backend_label,
            "quantization": quantization,
            "model": config.backend.model,
            "gateway_url": gateway_url,
            "prompt_set_size": len(prompts),
            "requests_per_run": benchmark.requests_per_run,
            "repeats": benchmark.repeats,
            "warmup_requests": benchmark.warmup_requests,
            "seed": benchmark.seed,
            "max_new_tokens": benchmark.max_new_tokens,
            "measurement_notes": [
                "Latencies are client-side, measured at the gateway from streamed chunk arrivals.",
                "ITL is unset for backends that return a single chunk (Hugging Face baseline).",
                "tokens/s is unset when the backend does not report an exact token count.",
                "peak_rss_mb is process RSS (unified-memory proxy on Apple Silicon), not VRAM.",
                "Workload fields above are configuration, not measured results.",
            ],
        },
        "runs": [asdict(summary) for summary in summaries],
        "raw_requests": raw_records,
        "aggregates": {
            f"concurrency_{level}": aggregate_runs(
                [summary for summary in summaries if summary.concurrency == level]
            )
            for level in benchmark.concurrency
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
