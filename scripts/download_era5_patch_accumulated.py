#!/usr/bin/env python
"""Safely print or execute the corrected ERA5-Land accumulated patch."""

from __future__ import annotations

import argparse
import json
import sys

from climatenet.data.era5_patch import (
    download_patch_region,
    request_manifest,
    save_request_manifest,
    validate_patch_config,
)
from climatenet.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/data_config_external_patch_202209_202312.yaml",
    )
    parser.add_argument(
        "--region", required=True, choices=["Sahara", "East China"]
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run-request", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_patch_config(config)
    if args.dry_run_request:
        manifest_path = save_request_manifest(config, args.region)
        payload = request_manifest(config, args.region)
        payload["request_manifest"] = str(manifest_path)
        print(json.dumps(payload, indent=2))
        return
    try:
        result = download_patch_region(config, args.region)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
