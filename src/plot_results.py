"""Create visual outputs for Phase 3 model results."""

from __future__ import annotations

import os

from config import OUTPUTS_DIR

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS_DIR / "matplotlib-cache"))

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    ALL_METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_IMPORTANCE_PLOT_PATH,
    METRICS_PLOT_PATH,
    PREDICTION_PLOT_PATH,
    PREDICTIONS_PATH,
    ensure_directories,
)


def plot_prediction_vs_actual(predictions: pd.DataFrame) -> None:
    """Plot prediction vs actual for the best random-split model."""
    if {"model", "prediction", "actual", "validation_strategy"}.issubset(predictions.columns):
        random_predictions = predictions[predictions["validation_strategy"] == "random_split"].copy()
        if random_predictions.empty:
            random_predictions = predictions.copy()

        model_rmse = (
            random_predictions.groupby("model")
            .apply(lambda group: ((group["actual"] - group["prediction"]) ** 2).mean() ** 0.5)
            .sort_values()
        )
        best_model = model_rmse.index[0]
        plot_data = random_predictions[random_predictions["model"] == best_model]
        label = best_model
        predicted = plot_data["prediction"]
    else:
        prediction_columns = [column for column in predictions.columns if column.endswith("_prediction")]
        if not prediction_columns:
            raise ValueError("No prediction columns found in predictions.csv.")
        best_model = prediction_columns[0].replace("_prediction", "")
        plot_data = predictions
        label = best_model
        predicted = predictions[prediction_columns[0]]

    plt.figure(figsize=(8, 6))
    plt.scatter(plot_data["actual"], predicted, alpha=0.5, s=18, label=label)
    limits = [
        min(plot_data["actual"].min(), predicted.min()),
        max(plot_data["actual"].max(), predicted.max()),
    ]
    plt.plot(limits, limits, color="black", linestyle="--", linewidth=1, label="perfect prediction")
    plt.xlabel("Actual evaporation anomaly")
    plt.ylabel("Predicted evaporation anomaly")
    plt.title(f"Prediction vs Actual ({label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH, dpi=150)
    plt.close()


def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    """Plot top global feature importance values for tree-based models."""
    tree_models = ["random_forest", "xgboost", "lightgbm"]
    tree_importance = feature_importance[feature_importance["model"].isin(tree_models)].copy()

    if tree_importance.empty:
        raise ValueError("No tree-model feature importance found to plot.")

    if "validation_strategy" in tree_importance.columns:
        tree_importance = tree_importance[tree_importance["validation_strategy"] == "random_split"]

    top_features = (
        tree_importance.sort_values("importance", ascending=False)
        .groupby("model")
        .head(10)
        .sort_values(["model", "importance"])
    )

    models = top_features["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharex=False)
    if len(models) == 1:
        axes = [axes]

    for axis, model_name in zip(axes, models):
        model_data = top_features[top_features["model"] == model_name]
        axis.barh(model_data["feature"], model_data["importance"])
        axis.set_title(model_name)
        axis.set_xlabel("Importance")

    fig.suptitle("Top Random-Split Tree Feature Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=150)
    plt.close()


def plot_validation_metrics(metrics: pd.DataFrame) -> None:
    """Plot RMSE by validation strategy and model."""
    metrics = metrics.copy()
    metrics["validation_label"] = metrics["validation_strategy"]
    transfer_mask = metrics["validation_strategy"] == "region_transfer"
    metrics.loc[transfer_mask, "validation_label"] = (
        metrics.loc[transfer_mask, "train_region"] + " to " + metrics.loc[transfer_mask, "test_region"]
    )

    pivot = metrics.pivot_table(index="validation_label", columns="model", values="RMSE", aggfunc="mean")
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("RMSE")
    plt.xlabel("Validation strategy")
    plt.title("Model RMSE Across Validation Strategies")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(METRICS_PLOT_PATH, dpi=150)
    plt.close()


def main() -> None:
    """Load saved outputs and create PNG plots."""
    ensure_directories()
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions at {PREDICTIONS_PATH}. Run: python src/train.py")
    if not FEATURE_IMPORTANCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature importance at {FEATURE_IMPORTANCE_PATH}. Run: python src/train.py"
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)

    plot_prediction_vs_actual(predictions)
    plot_feature_importance(feature_importance)

    if ALL_METRICS_PATH.exists():
        metrics = pd.read_csv(ALL_METRICS_PATH)
        plot_validation_metrics(metrics)
        print(f"Saved validation metrics plot to {METRICS_PLOT_PATH}")

    print(f"Saved prediction plot to {PREDICTION_PLOT_PATH}")
    print(f"Saved feature importance plot to {FEATURE_IMPORTANCE_PLOT_PATH}")


if __name__ == "__main__":
    main()
