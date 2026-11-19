"""CLI: generate benchmark PNGs from real result JSON files only.

Example:
    python scripts/make_plots.py --result results/baseline.json --result results/llama_cpp.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_serve_bench.bench.plots import Series, extract_series

PLOTS = (
    ("throughput", "throughput", "Throughput vs concurrency", "tokens/s", "throughput_vs_concurrency.png"),
    ("latency", "p95_ms", "p95 E2E latency vs concurrency", "ms", "p95_latency_vs_concurrency.png"),
    ("memory", "memory_mb", "Peak memory vs concurrency", "MiB", "memory_vs_concurrency.png"),
)


def _plot(series: list[Series], attribute: str, title: str, ylabel: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    for item in series:
        pairs = [
            (concurrency, value)
            for concurrency, value in zip(item.concurrency, getattr(item, attribute))
            if value is not None
        ]
        if pairs:
            axes.plot([pair[0] for pair in pairs], [pair[1] for pair in pairs], marker="o", label=item.label)
    axes.set_xlabel("concurrency")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate charts from saved benchmark JSON.")
    parser.add_argument("--result", action="append", required=True, help="Result JSON; repeat for each series.")
    parser.add_argument("--output-dir", default="results", help="Directory for generated PNGs.")
    args = parser.parse_args()

    series = [extract_series(json.loads(Path(path).read_text(encoding="utf-8"))) for path in args.result]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, attribute, title, ylabel, filename in PLOTS:
        output = output_dir / filename
        _plot(series, attribute, title, ylabel, output)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
