"""Configuration-driven experiment runner."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from climatenet.data.loaders import load_csv
from climatenet.models.factory import build_models
from climatenet.training.plots import (
    plot_ablation,
    plot_feature_importance,
    plot_metrics,
    plot_model_comparison,
    plot_prediction_vs_actual,
)
from climatenet.training.train_tcn import run_tcn_arrays_for_split
from climatenet.training.trainer import train_on_split
from climatenet.training.validation import ValidationSplit, build_validation_splits
from climatenet.utils.config import save_yaml
from climatenet.utils.logging import setup_logger
from climatenet.utils.paths import ensure_directory, resolve_project_path
from climatenet.utils.random import set_random_seed


def validate_feature_table(data: pd.DataFrame, feature_names: list[str], target_column: str) -> None:
    """Validate features and target exist before training."""
    required_columns = feature_names + [target_column, "region", "year", "month", "latitude", "longitude"]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")


def experiment_output_dir(experiment_config: dict[str, Any]) -> Path:
    """Create the output directory for an experiment."""
    experiment_id = experiment_config.get("experiment_id", "default_experiment")
    output_root = resolve_project_path(experiment_config.get("output_root", "outputs/experiments"))
    return ensure_directory(output_root / experiment_id)


def metrics_to_json(metrics: pd.DataFrame) -> dict[str, Any]:
    """Convert metrics dataframe to nested JSON-friendly records."""
    return {
        "results": metrics.to_dict(orient="records"),
        "best_by_strategy": (
            metrics.sort_values("RMSE")
            .groupby("validation_strategy")
            .head(1)
            .to_dict(orient="records")
        ),
    }


def save_outputs(
    output_dir: Path,
    config_snapshot: dict[str, Any],
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    save_plots: bool,
) -> None:
    """Save all experiment artifacts."""
    save_yaml(config_snapshot, output_dir / "config_snapshot.yaml")
    save_yaml(config_snapshot, output_dir / "config.yaml")
    features.to_csv(output_dir / "features.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_to_json(metrics), file, indent=2)

    if save_plots:
        plots_dir = ensure_directory(output_dir / "plots")
        plot_metrics(metrics, plots_dir / "validation_metrics.png")
        plot_prediction_vs_actual(predictions, plots_dir / "prediction_vs_actual.png")
        plot_feature_importance(feature_importance, plots_dir / "feature_importance.png")
        plot_model_comparison(metrics, plots_dir / "model_comparison.png")
        plot_ablation(metrics, plots_dir / "ablation_study.png")


def update_all_experiments(metrics: pd.DataFrame, output_root: Path, experiment_id: str) -> None:
    """Update the project-level experiment summary CSV."""
    summary_path = output_root.parent.parent / "all_experiments.csv"
    current = metrics.copy()
    current["experiment_id"] = experiment_id
    current = current[
        [
            "experiment_id",
            "model_name",
            "validation_strategy",
            "feature_set",
            "train_region",
            "test_region",
            "train_period",
            "test_period",
            "n_train",
            "n_test",
            "MAE",
            "RMSE",
            "R2",
        ]
    ]

    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        previous = previous[previous["experiment_id"] != experiment_id]
        current = pd.concat([previous, current], ignore_index=True)

    current.to_csv(summary_path, index=False)


def save_ablation_study(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Save a compact ablation study table."""
    ablation = (
        metrics.groupby(["feature_set", "model_name", "validation_strategy"], as_index=False)
        .agg(
            mean_mae=("MAE", "mean"),
            mean_rmse=("RMSE", "mean"),
            mean_r2=("R2", "mean"),
            best_rmse=("RMSE", "min"),
        )
        .sort_values(["validation_strategy", "mean_rmse"])
    )
    ablation.to_csv(output_dir / "ablation_study.csv", index=False)


def model_config_for_one(model_config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Return a model config with only one non-TCN model enabled."""
    one_model_config = {"models": {}}
    for configured_name, configured_value in model_config.get("models", {}).items():
        value = dict(configured_value)
        value["enabled"] = configured_name == model_name
        one_model_config["models"][configured_name] = value
    return one_model_config


def split_label(split: ValidationSplit) -> str:
    """Human-readable split label for logs."""
    if split.strategy == "region_transfer":
        return f"{split.train_region} -> {split.test_region}"
    if split.strategy == "temporal_holdout":
        return f"{split.train_period} -> {split.test_period}"
    return split.strategy


def run_tcn_for_temporal_split(
    data: pd.DataFrame,
    feature_names: list[str],
    feature_set_name: str,
    experiment_config: dict[str, Any],
    split: ValidationSplit,
    seed: int,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Run TCN for one temporal split and return unified metrics/predictions."""
    tcn_config = experiment_config.get("tcn", {})
    metrics, predictions = run_tcn_arrays_for_split(
        data=data,
        feature_columns=feature_names,
        sequence_length=int(tcn_config.get("sequence_length", 6)),
        train_start_year=int(experiment_config.get("train_start_year", 2019)),
        train_end_year=int(experiment_config.get("train_end_year", 2022)),
        test_year=int(experiment_config.get("test_year", 2023)),
        seed=seed,
        epochs=int(tcn_config.get("epochs", 10)),
        batch_size=int(tcn_config.get("batch_size", 128)),
        learning_rate=float(tcn_config.get("learning_rate", 1e-3)),
        channels=list(tcn_config.get("channels", [32, 32, 32])),
        dropout=float(tcn_config.get("dropout", 0.2)),
    )

    metrics_row = {
        "validation_strategy": "temporal_holdout",
        "train_region": split.train_region,
        "test_region": split.test_region,
        "train_period": split.train_period,
        "test_period": split.test_period,
        "feature_set": feature_set_name,
        "model": "tcn",
        "model_name": "tcn",
        "n_train": len(split.train_data),
        "n_test": len(split.test_data),
        **metrics,
    }

    predictions["validation_strategy"] = "temporal_holdout"
    predictions["train_region"] = split.train_region
    predictions["test_region"] = split.test_region
    predictions["train_period"] = split.train_period
    predictions["test_period"] = split.test_period
    predictions["feature_set"] = feature_set_name
    predictions["model"] = "tcn"
    predictions["model_name"] = "tcn"
    return [metrics_row], [predictions]


def run_experiment(configs: dict[str, Any]) -> Path:
    """Run one configuration-driven experiment and return its output directory."""
    experiment_config = configs["experiment"]
    model_config = configs["models"]
    seed = int(experiment_config.get("seed", 42))
    set_random_seed(seed)

    output_dir = experiment_output_dir(experiment_config)
    logger = setup_logger("climatenet.experiment", output_dir / "experiment.log")
    logger.info("Starting experiment: %s", experiment_config.get("experiment_id"))

    features_path = experiment_config.get("features_path", configs["data"].get("features_path"))
    data = load_csv(features_path)
    target_column = experiment_config.get("target_column", configs["data"].get("target_column", "evaporation_anomaly"))
    logger.info("Loaded feature table with %s rows and %s columns", len(data), len(data.columns))

    validation_splits = build_validation_splits(data, experiment_config)
    logger.info("Built %s validation splits", len(validation_splits))
    feature_sets = experiment_config.get("feature_sets", {"default": experiment_config["model_features"]})
    batch_models = list(experiment_config.get("batch_models", ["linear_regression", "random_forest", "xgboost"]))

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for feature_set_name, feature_names in feature_sets.items():
        feature_names = list(feature_names)
        validate_feature_table(data, feature_names, target_column)
        logger.info("Running feature set: %s (%s features)", feature_set_name, len(feature_names))

        for split in validation_splits:
            logger.info("Validation split: %s", split_label(split))

            for model_name in batch_models:
                if model_name == "tcn":
                    if split.strategy != "temporal_holdout":
                        continue
                    logger.info("Training model=tcn feature_set=%s split=%s", feature_set_name, split_label(split))
                    split_metrics, split_predictions = run_tcn_for_temporal_split(
                        data=data,
                        feature_names=feature_names,
                        feature_set_name=feature_set_name,
                        experiment_config=experiment_config,
                        split=split,
                        seed=seed,
                    )
                    metrics_rows.extend(split_metrics)
                    prediction_frames.extend(split_predictions)
                    continue

                logger.info("Training model=%s feature_set=%s split=%s", model_name, feature_set_name, split_label(split))
                models = build_models(model_config_for_one(model_config, model_name), seed)
                if model_name not in models:
                    logger.info("Skipping unavailable model: %s", model_name)
                    continue
                split_metrics, split_predictions, split_importance = train_on_split(
                    split,
                    models,
                    feature_names,
                    target_column,
                    feature_set_name=feature_set_name,
                )
                metrics_rows.extend(split_metrics)
                prediction_frames.extend(split_predictions)
                importance_frames.extend(split_importance)

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    feature_importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()

    save_outputs(
        output_dir=output_dir,
        config_snapshot=configs,
        features=data,
        metrics=metrics,
        predictions=predictions,
        feature_importance=feature_importance,
        save_plots=bool(experiment_config.get("save_plots", True)),
    )

    logger.info("Saved experiment outputs to %s", output_dir)
    save_ablation_study(metrics, output_dir)
    update_all_experiments(metrics, output_dir, str(experiment_config.get("experiment_id", "default_experiment")))
    logger.info("Updated ablation and all-experiments summaries")

    latest_dir = output_dir.parent / "latest"
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    shutil.copytree(output_dir, latest_dir)
    logger.info("Updated latest experiment copy at %s", latest_dir)

    return output_dir
