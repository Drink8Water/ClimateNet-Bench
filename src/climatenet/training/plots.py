"""Experiment plotting utilities."""

from __future__ import annotations

import os
from pathlib import Path

from climatenet.utils.paths import resolve_project_path

os.environ.setdefault("MPLCONFIGDIR", str(resolve_project_path("outputs/matplotlib-cache")))

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot RMSE by validation strategy and model."""
    plot_data = metrics.copy()
    plot_data["validation_label"] = plot_data["validation_strategy"]
    transfer_mask = plot_data["validation_strategy"] == "region_transfer"
    plot_data.loc[transfer_mask, "validation_label"] = (
        plot_data.loc[transfer_mask, "train_region"] + " to " + plot_data.loc[transfer_mask, "test_region"]
    )

    pivot = plot_data.pivot_table(index="validation_label", columns="model", values="RMSE", aggfunc="mean")
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("RMSE")
    plt.xlabel("Validation strategy")
    plt.title("ClimateNet Validation RMSE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_prediction_vs_actual(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot actual vs predicted values for the best random-split model."""
    random_predictions = predictions[predictions["validation_strategy"] == "random_split"].copy()
    if random_predictions.empty:
        random_predictions = predictions.copy()

    rmse_by_model = (
        random_predictions.groupby("model")
        .apply(lambda group: ((group["actual"] - group["prediction"]) ** 2).mean() ** 0.5)
        .sort_values()
    )
    best_model = rmse_by_model.index[0]
    plot_data = random_predictions[random_predictions["model"] == best_model]

    plt.figure(figsize=(7, 6))
    plt.scatter(plot_data["actual"], plot_data["prediction"], alpha=0.5, s=18)
    limits = [
        min(plot_data["actual"].min(), plot_data["prediction"].min()),
        max(plot_data["actual"].max(), plot_data["prediction"].max()),
    ]
    plt.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Actual evaporation anomaly")
    plt.ylabel("Predicted evaporation anomaly")
    plt.title(f"Prediction vs Actual: {best_model}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(feature_importance: pd.DataFrame, output_path: Path) -> None:
    """Plot top random-split tree feature importance."""
    if feature_importance.empty or "model" not in feature_importance.columns:
        return
    tree_importance = feature_importance[
        feature_importance["model"].isin(["random_forest", "xgboost", "lightgbm"])
    ].copy()
    tree_importance = tree_importance[tree_importance["validation_strategy"] == "random_split"]
    if tree_importance.empty:
        return

    top_features = (
        tree_importance.sort_values("importance", ascending=False)
        .groupby("model")
        .head(10)
        .sort_values(["model", "importance"])
    )

    models = top_features["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for axis, model_name in zip(axes, models):
        model_data = top_features[top_features["model"] == model_name]
        axis.barh(model_data["feature"], model_data["importance"])
        axis.set_title(model_name)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot model RMSE comparison grouped by feature set."""
    if metrics.empty:
        return
    plot_data = metrics.copy()
    plot_data["label"] = plot_data["validation_strategy"]
    transfer_mask = plot_data["validation_strategy"] == "region_transfer"
    plot_data.loc[transfer_mask, "label"] = (
        plot_data.loc[transfer_mask, "train_region"] + " to " + plot_data.loc[transfer_mask, "test_region"]
    )
    summary = (
        plot_data.groupby(["feature_set", "model_name"], as_index=False)["RMSE"]
        .mean()
        .sort_values("RMSE")
    )
    pivot = summary.pivot_table(index="feature_set", columns="model_name", values="RMSE")
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Mean RMSE")
    plt.xlabel("Feature set")
    plt.title("Model Comparison Across Feature Sets")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ablation(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot best RMSE for each feature set."""
    if metrics.empty:
        return
    summary = (
        metrics.groupby("feature_set", as_index=False)["RMSE"]
        .min()
        .sort_values("RMSE")
    )
    plt.figure(figsize=(8, 4))
    plt.bar(summary["feature_set"], summary["RMSE"])
    plt.ylabel("Best RMSE")
    plt.xlabel("Feature set")
    plt.title("Ablation Study: Best RMSE by Feature Set")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
