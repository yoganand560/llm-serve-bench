"""CLI: benchmark an already-running gateway and save raw results.

Example (after starting the gateway):
    python benchmarks/run_benchmark.py --config configs/llama_cpp_q4.yaml \
        --backend-label llama_cpp --quantization Q4_K_M
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from llm_serve_bench.bench.runner import run_benchmark
from llm_serve_bench.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed workload against a live gateway.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000", help="Gateway base URL.")
    parser.add_argument("--backend-label", default=None, help="Label stored in the result JSON.")
    parser.add_argument("--quantization", default=None, help="Quantization stored in the result JSON.")
    parser.add_argument("--output", default=None, help="Output JSON path (default: results/<backend>_<timestamp>.json).")
    args = parser.parse_args()

    config = load_config(args.config)
    prompts = json.loads(Path(config.benchmark.prompt_set_path).read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p for p in prompts):
        raise SystemExit("prompt set must be a non-empty JSON list of strings")

    label = args.backend_label or config.backend.name
    quantization = args.quantization or config.backend.quantization
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output) if args.output else Path("results") / f"{label}_{quantization}_{timestamp}.json"

    result = asyncio.run(
        run_benchmark(
            config,
            gateway_url=args.gateway_url,
            backend_label=label,
            quantization=quantization,
            prompts=prompts,
            output_path=output,
        )
    )
    print(json.dumps({"saved_to": str(output), "metadata": result["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
