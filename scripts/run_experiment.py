"""Run a configuration-driven ClimateNet experiment."""

from __future__ import annotations

import argparse

from climatenet.training.experiment import run_experiment
from climatenet.utils.config import load_experiment_configs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a ClimateNet experiment.")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--experiment-config", default="configs/experiment_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Load configs and run the experiment."""
    args = parse_args()
    configs = load_experiment_configs(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        experiment_config_path=args.experiment_config,
    )
    output_dir = run_experiment(configs)
    print(f"Experiment complete: {output_dir}")


if __name__ == "__main__":
    main()
