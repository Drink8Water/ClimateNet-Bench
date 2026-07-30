#!/usr/bin/env python
"""Run raw or combined final audit for corrected ERA5-Land artifacts."""

from __future__ import annotations

import argparse
import json

from climatenet.data.era5_audit import audit_era5_files
from climatenet.data.era5_corrected_audit import (
    build_corrected_full_audit,
    write_corrected_full_audit,
)
from climatenet.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/data_config_external_corrected_2019_2023.yaml",
    )
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--raw", action="store_true")
    stage.add_argument("--final", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.raw:
        report = audit_era5_files(
            config["era5"]["input_files"],
            start="2019-01",
            end="2023-12",
            max_total_bytes=2 * 1024 * 1024 * 1024,
            input_window=6,
            output_path=config["era5"]["raw_audit_path"],
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "rows": report["converted_row_count"],
                    "negative_evaporation_count": report[
                        "evaporation_target"
                    ]["negative_count_after_conversion"],
                    "warnings": report["warnings"],
                    "output": config["era5"]["raw_audit_path"],
                },
                indent=2,
            )
        )
        if report["status"] != "ready":
            raise SystemExit(1)
        return
    report = build_corrected_full_audit(config)
    outputs = write_corrected_full_audit(config, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "outputs": [str(path) for path in outputs],
                "blocking_issues": report["blocking_issues"],
            },
            indent=2,
        )
    )
    if report["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
