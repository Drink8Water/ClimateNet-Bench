"""Run a TCN temporal holdout experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from climatenet.training.train_tcn import run_tcn_experiment
from climatenet.utils.paths import resolve_project_path


def parse_channels(value: str) -> list[int]:
    """Parse comma-separated channel sizes."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train TCN for evaporation anomaly temporal prediction.")
    parser.add_argument("--features-path", default="outputs/experiments/latest/features.csv")
    parser.add_argument("--output-dir", default="outputs/experiments/latest/tcn")
    parser.add_argument("--sequence-length", type=int, default=6)
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--test-year", type=int, default=2023)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--channels", type=parse_channels, default=[32, 32, 32])
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    """Run TCN experiment from CLI arguments."""
    args = parse_args()
    metrics = run_tcn_experiment(
        features_path=resolve_project_path(args.features_path),
        output_dir=resolve_project_path(args.output_dir),
        sequence_length=args.sequence_length,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_year=args.test_year,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        channels=args.channels,
        dropout=args.dropout,
    )
    print("TCN experiment complete.")
    print(metrics)


if __name__ == "__main__":
    main()
