#!/usr/bin/env python
"""Mark preserved ERA5 v1 artifacts as invalid due to known source data."""

from __future__ import annotations

import argparse
import json

from climatenet.data.era5_corrected_merge import write_source_data_status
from climatenet.utils.config import load_yaml

RUN_DIRECTORIES = [
    "/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/"
    "era5-land-sahara-eastchina-v1-era5-land-"
    "20260730T012055094403Z-8cfe21a0-4e68b2",
    "/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/"
    "era5-land-sahara-eastchina-v1-multiseed-seed123-era5-land-"
    "20260730T020318399976Z-261e682b-a4a68d",
    "/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/"
    "era5-land-sahara-eastchina-v1-multiseed-seed2026-era5-land-"
    "20260730T021517210737Z-60977264-9b5cf0",
    "/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/"
    "era5_land_v1_multiseed_summary",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/data_config_external_corrected_2019_2023.yaml",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    outputs = write_source_data_status(
        RUN_DIRECTORIES,
        corrected_dataset_paths=[
            *config["era5"]["input_files"],
            config["era5"]["processed_path"],
            config["features_path"],
        ],
        known_issue_url=(
            "https://confluence.ecmwf.int/pages/viewpage.action?"
            "pageId=402639006"
        ),
    )
    print(json.dumps([str(path) for path in outputs], indent=2))


if __name__ == "__main__":
    main()
