"""CLI: score a quantization variant on the fixed QA set and save raw JSON.

The gateway must already be running. Sample size is small by design; the output is a
trade-off sanity check, not a headline accuracy number.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from llm_serve_bench.config import load_config
from llm_serve_bench.quality.check import run_quality_check
from llm_serve_bench.quality.qa_set import load_qa_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed QA quality check against a live gateway.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000", help="Gateway base URL.")
    parser.add_argument("--qa-set", default="benchmarks/qa_set.json", help="QA set JSON path.")
    parser.add_argument("--backend-label", default=None, help="Label stored in the result JSON.")
    parser.add_argument("--quantization", default=None, help="Quantization stored in the result JSON.")
    parser.add_argument("--output", default=None, help="Output JSON path.")
    args = parser.parse_args()

    config = load_config(args.config)
    items = load_qa_set(args.qa_set)
    label = args.backend_label or config.backend.name
    quantization = args.quantization or config.backend.quantization
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output) if args.output else Path("results") / f"quality_{label}_{quantization}_{timestamp}.json"

    async def execute() -> list[dict[str, object]]:
        async with httpx.AsyncClient(base_url=args.gateway_url, timeout=None) as client:
            results = await run_quality_check(
                client,
                items,
                max_new_tokens=config.gateway.max_new_tokens_default,
                seed=config.benchmark.seed,
            )
        return [asdict(result) for result in results]

    raw_results = asyncio.run(execute())
    passed = sum(1 for result in raw_results if result["passed"])
    document: dict[str, object] = {
        "schema_version": 1,
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "backend": label,
            "quantization": quantization,
            "model": config.backend.model,
            "sample_size": len(raw_results),
            "passed": passed,
            "passed_fraction": passed / len(raw_results) if raw_results else None,
            "seed": config.benchmark.seed,
            "notes": [
                "Small fixed QA set with keyword matching; a sanity check, not a formal benchmark.",
                "Greedy decoding (temperature 0) with the configured seed.",
            ],
        },
        "results": raw_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps({"saved_to": str(output), "metadata": document["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
