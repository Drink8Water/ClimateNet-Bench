#!/usr/bin/env python
"""Run a guarded small real-data ERA5-Land benchmark dry-run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from climatenet.data.era5_dry_run import run_era5_dry_run
from climatenet.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit, subset and run a small real ERA5-Land benchmark. "
            "This command cannot execute the full benchmark config."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark/era5_land_dry_run.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmark_runs",
    )
    args = parser.parse_args()
    registry = run_era5_dry_run(
        load_yaml(args.config),
        output_root=args.output_dir,
    )
    print(f"Run directory: {registry.path.parent}")
    print(f"Completed: {len(registry.list_completed())}")
    print(f"Failed: {len(registry.list_failed())}")


if __name__ == "__main__":
    main()
