"""CLI: profile the gateway and save a .prof trace plus a pstats text summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_serve_bench.bench.profiling import profile_requests
from llm_serve_bench.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the in-process gateway with cProfile.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--requests", type=int, default=20, help="Number of profiled requests.")
    parser.add_argument("--prompt", default="Explain batching in one sentence.", help="Fixed prompt.")
    parser.add_argument("--output", default="results/gateway_profile.prof", help=".prof trace path.")
    parser.add_argument("--summary", default="results/gateway_profile_summary.txt", help="pstats text path.")
    args = parser.parse_args()

    config = load_config(args.config)
    max_new_tokens = min(config.gateway.max_new_tokens_default, config.gateway.max_new_tokens_limit)
    stats = profile_requests(
        config,
        requests=args.requests,
        max_new_tokens=max_new_tokens,
        prompt=args.prompt,
    )

    trace_path = Path(args.output)
    summary_path = Path(args.summary)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    stats.dump_stats(str(trace_path))
    summary_path.write_text(stats.sort_stats("cumulative").stream_to_string(), encoding="utf-8")
    print(f"trace: {trace_path}\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
