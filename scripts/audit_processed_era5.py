#!/usr/bin/env python
"""Audit the full processed ERA5 CSV in bounded memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from climatenet.data.era5_audit import audit_processed_era5_csv
from climatenet.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a full processed ERA5-Land CSV"
    )
    parser.add_argument(
        "--data-config",
        default="configs/data_config_external_full.yaml",
    )
    parser.add_argument(
        "--readiness-audit",
        default=None,
        help="Optional readiness JSON supplying the expected retained row count",
    )
    args = parser.parse_args()
    config = load_yaml(args.data_config)
    era5 = config["era5"]
    readiness_path = Path(
        args.readiness_audit or era5["audit_path"]
    )
    expected_rows = None
    if readiness_path.is_file():
        expected_rows = json.loads(
            readiness_path.read_text(encoding="utf-8")
        ).get("converted_row_count")
    report = audit_processed_era5_csv(
        era5["processed_path"],
        output_path=era5["processed_audit_path"],
        expected_rows=expected_rows,
    )
    print(f"Audit status: {report['status']}")
    print(f"Rows: {report['row_count']:,}")
    print(f"Regions: {report['regions']}")
    print(f"Duplicate keys: {report['duplicate_key_rows']:,}")
    print(
        "Non-finite values: "
        f"{sum(report['non_finite_counts'].values()):,}"
    )
    print(f"Saved: {era5['processed_audit_path']}")
    if report["blocking_issues"]:
        print(json.dumps(report["blocking_issues"], indent=2))


if __name__ == "__main__":
    main()
