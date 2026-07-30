#!/usr/bin/env python
"""Dry-run or execute one region's corrected accumulated merge."""

from __future__ import annotations

import argparse
import json
import sys

from climatenet.data.era5_corrected_merge import (
    execute_corrected_merge,
    plan_corrected_merge,
)
from climatenet.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/data_config_external_corrected_2019_2023.yaml",
    )
    parser.add_argument(
        "--region", required=True, choices=["Sahara", "East China"]
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    try:
        result = (
            plan_corrected_merge(config, args.region)
            if args.dry_run
            else execute_corrected_merge(config, args.region)
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
