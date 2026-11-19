"""CLI: log an existing raw result JSON (and optional plots) to MLflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_serve_bench.tracking.mlflow_logger import log_benchmark_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Log a benchmark result to MLflow.")
    parser.add_argument("--result", required=True, help="Raw result JSON produced by run_benchmark.py.")
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI.")
    parser.add_argument("--experiment", default="llm-serve-bench", help="MLflow experiment name.")
    parser.add_argument("--run-name", default=None, help="Optional run name.")
    parser.add_argument("--artifact", action="append", default=[], help="Extra artifact; repeatable.")
    args = parser.parse_args()

    result_path = Path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_id = log_benchmark_result(
        result,
        result_path=result_path,
        extra_artifacts=[Path(path) for path in args.artifact],
        run_name=args.run_name,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
    )
    print(f"logged MLflow run: {run_id}")


if __name__ == "__main__":
    main()
