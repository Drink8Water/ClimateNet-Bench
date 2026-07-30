#!/usr/bin/env python
"""Audit corrected accumulated patch files against known-bad full files."""

from __future__ import annotations

import argparse
import json

from climatenet.data.era5_patch_audit import (
    audit_patch_files,
    write_patch_audit,
)
from climatenet.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/data_config_external_patch_202209_202312.yaml",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    report = audit_patch_files(config)
    paths = write_patch_audit(config, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "outputs": [str(path) for path in paths],
                "comparison_summary": report["comparison_summary"],
                "blocking_issues": report["blocking_issues"],
                "warnings": report["warnings"],
            },
            indent=2,
        )
    )
    if report["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
