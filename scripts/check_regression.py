"""CLI: fail if fresh p95 latency exceeds the stored baseline by more than a threshold.

CI context: this reads saved JSON files only. It needs no GPU, no model and no server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_serve_bench.bench.regression import check_regression
from llm_serve_bench.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Check p95 latency regression from saved JSON.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--baseline", default=None, help="Stored baseline JSON (defaults to config).")
    parser.add_argument("--fresh", required=True, help="Fresh result JSON to compare.")
    parser.add_argument("--threshold", type=float, default=None, help="Override p95 ratio threshold.")
    args = parser.parse_args()

    config = load_config(args.config)
    baseline_path = Path(args.baseline or config.regression.baseline_path)
    fresh_path = Path(args.fresh)
    threshold = args.threshold if args.threshold is not None else config.regression.p95_threshold_ratio

    for path in (baseline_path, fresh_path):
        if not path.is_file():
            print(f"ERROR: result file not found: {path}")
            return 2

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    try:
        outcome = check_regression(baseline, fresh, threshold=threshold)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    for item in outcome.comparisons:
        verdict = "PASS" if item.passed else "FAIL"
        print(
            f"{verdict} {item.level}: baseline={item.baseline_p95_ms:.3f}ms "
            f"fresh={item.fresh_p95_ms:.3f}ms ratio={item.ratio:.3f} threshold={item.threshold:.3f}"
        )
    if outcome.skipped_levels:
        print("SKIPPED levels without comparable measured values: " + ", ".join(outcome.skipped_levels))

    print("CI mode: saved result JSON only; no GPU, model or live server used.")
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
