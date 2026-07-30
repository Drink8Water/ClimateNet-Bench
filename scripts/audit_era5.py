#!/usr/bin/env python
"""Audit a bounded real ERA5-Land input without running a benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from climatenet.data.era5_audit import (
    audit_era5_files,
    estimate_benchmark_task_count,
)
from climatenet.data.era5_download import ALL_MONTHS, era5_output_path
from climatenet.data.era5_dry_run import (
    era5_audit_options,
    validate_era5_dry_run_config,
)
from climatenet.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a guarded ERA5-Land dry-run subset"
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark/era5_land_dry_run.yaml",
    )
    parser.add_argument(
        "--data-config",
        default=None,
        help="External full data config; requires --full-preflight",
    )
    parser.add_argument(
        "--benchmark-config",
        default="configs/benchmark/evap_anomaly_v1.yaml",
        help="Used only to estimate full benchmark artifact volume",
    )
    parser.add_argument(
        "--full-preflight",
        action="store_true",
        help="Audit deterministic full files without materialising the CSV",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Audit JSON path; defaults to outputs/data_audit/<timestamp>.json",
    )
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.data_config is not None:
        if not args.full_preflight:
            parser.error("--data-config requires --full-preflight")
        data_config = load_yaml(args.data_config)
        era5 = data_config["era5"]
        years = list(era5["full_years"])
        raw_dir = Path(era5["raw_dir"])
        paths = [
            era5_output_path(raw_dir, region, years, ALL_MONTHS)
            for region in era5["regions"]
        ]
        benchmark = load_yaml(args.benchmark_config)
        input_window = int(str(benchmark.get("input_window", 6)).split()[0])
        output = Path(
            args.output
            or era5.get(
                "audit_path",
                f"outputs/data_audit/era5_full_{timestamp}.json",
            )
        )
        report = audit_era5_files(
            paths,
            start=f"{years[0]}-01",
            end=f"{years[-1]}-12",
            max_total_bytes=2 * 1024 * 1024 * 1024,
            input_window=input_window,
            task_count=estimate_benchmark_task_count(benchmark),
            output_path=output,
        )
    else:
        config = load_yaml(args.config)
        validate_era5_dry_run_config(config)
        output = Path(
            args.output or f"outputs/data_audit/era5_{timestamp}.json"
        )
        report = audit_era5_files(
            **era5_audit_options(config), output_path=output
        )
    print(f"Audit status: {report['status']}")
    print(f"Converted rows: {report['converted_row_count']:,}")
    print(
        "Estimated lag samples: "
        f"{report['lag_sample_estimate']['estimated_available_samples']:,}"
    )
    print(f"Warnings: {len(report['warnings'])}")
    print(
        "Estimated processed CSV bytes: "
        f"{report['storage_estimate']['processed_csv_bytes']}"
    )
    print(
        "Estimated prediction/artifact bytes: "
        f"{report['storage_estimate']['predictions_and_metrics_bytes']}"
    )
    print(f"Saved: {output}")
    if report["warnings"]:
        print(json.dumps(report["warnings"], indent=2))


if __name__ == "__main__":
    main()
