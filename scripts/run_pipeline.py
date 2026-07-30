"""Run the ClimateNet feature pipeline from YAML configuration."""

from __future__ import annotations

import argparse

from climatenet.data.loaders import load_csv, save_csv
from climatenet.features.physical import (
    add_physical_features,
    build_physical_features_csv,
)
from climatenet.features.pipeline import build_features
from climatenet.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build ClimateNet features from configured input data.")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--audit-output", default=None)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--formal-benchmark",
        action="store_true",
        help=(
            "Generate only row-wise physical features. Required for formal "
            "benchmark input because split-specific anomalies are fit later."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Build and save feature table."""
    args = parse_args()
    config = load_yaml(args.data_config)
    input_path = args.input or config["input_data_path"]
    output_path = args.output or config["features_path"]
    if args.formal_benchmark:
        report = build_physical_features_csv(
            input_path,
            output_path,
            audit_path=args.audit_output,
            chunksize=args.chunksize,
        )
        print(f"Saved features to {output_path}")
        print("Mode: formal benchmark (streamed row-wise physical features only)")
        print(f"Rows: {report['row_count']:,}, Columns: {len(report['columns'])}")
        print(f"SHA256: {report['sha256']}")
        return
    data = load_csv(
        input_path,
        required_columns=config.get("required_raw_columns"),
    )
    features = build_features(data)
    save_csv(features, output_path)
    print(f"Saved features to {output_path}")
    print(
        "Mode:",
        "formal benchmark (physical features only)"
        if args.formal_benchmark
        else "exploratory (full-table anomalies)",
    )
    print(f"Rows: {len(features):,}, Columns: {len(features.columns)}")


if __name__ == "__main__":
    main()
