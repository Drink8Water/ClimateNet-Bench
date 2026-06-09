#!/usr/bin/env python
"""Build leaderboard tables from completed benchmark experiments.

Usage
-----
.. code-block:: bash

    python scripts/build_leaderboard.py

    python scripts/build_leaderboard.py \\
        --experiments-dir outputs/benchmark/experiments \\
        --output-dir outputs/benchmark
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from climatenet.benchmark.leaderboard import build_leaderboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_leaderboard")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build benchmark leaderboard from experiment results"
    )
    parser.add_argument(
        "--experiments-dir",
        default="outputs/benchmark/experiments",
        help="Directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmark",
        help="Directory for leaderboard output files",
    )
    args = parser.parse_args()

    logger.info("Scanning experiments in %s …", args.experiments_dir)
    result = build_leaderboard(
        experiments_root=args.experiments_dir,
        output_root=args.output_dir,
    )

    if not result:
        logger.warning("No experiment results found.")
        return

    # ── Print summary ────────────────────────────────────────────
    lb = result.get("leaderboard")
    if lb is not None and not lb.empty:
        print()
        print("=" * 80)
        print("  BENCHMARK LEADERBOARD")
        print("=" * 80)
        cols = [
            c for c in ["rank", "model_name", "split_protocol", "feature_set",
                         "rmse", "mae", "r2"]
            if c in lb.columns
        ]
        print(lb[cols].head(20).to_string(index=False))

    diff = result.get("split_difficulty_analysis")
    if diff is not None and not diff.empty:
        print()
        print("--- Split Difficulty ---")
        print(diff.to_string(index=False))

    print()
    print(f"Output files written to: {args.output_dir}")
    for name in ["all_results", "leaderboard", "split_difficulty_analysis",
                 "uncertainty_calibration", "ablation_results"]:
        p = Path(args.output_dir) / f"{name}.csv"
        if p.exists():
            print(f"  ✅ {p}")


if __name__ == "__main__":
    main()
