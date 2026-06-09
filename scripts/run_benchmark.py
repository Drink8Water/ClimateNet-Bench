#!/usr/bin/env python
"""Run the ClimateNet-Bench benchmark pipeline.

Usage
-----
.. code-block:: bash

    # Smoke test (fast)
    python scripts/run_benchmark.py --config configs/benchmark/smoke_test.yaml

    # Full benchmark
    python scripts/run_benchmark.py --config configs/benchmark/evap_anomaly_v1.yaml

    # Custom output directory
    python scripts/run_benchmark.py --config configs/benchmark/smoke_test.yaml \\
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

from climatenet.training.benchmark_runner import run_benchmark
from climatenet.utils.config import load_yaml
from climatenet.utils.paths import resolve_project_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ClimateNet-Bench benchmark")
    parser.add_argument(
        "--config",
        default="configs/benchmark/smoke_test.yaml",
        help="Path to benchmark YAML config",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmark",
        help="Root directory for benchmark outputs",
    )
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    logger.info("Loading config from %s", config_path)
    config = load_yaml(config_path)

    logger.info("Benchmark: %s", config.get("benchmark_name", "unnamed"))
    logger.info("Models: %d", len(config.get("models", [])))
    logger.info("Output: %s", args.output_dir)

    registry = run_benchmark(config=config, output_root=args.output_dir)

    completed = registry.list_completed()
    failed = registry.list_failed()
    logger.info("=== Benchmark Complete ===")
    logger.info("  Completed: %d", len(completed))
    logger.info("  Failed:    %d", len(failed))
    if failed:
        for r in failed:
            logger.info("    ❌ %s: %s", r.experiment_id, r.error_message)


if __name__ == "__main__":
    main()
