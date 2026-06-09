"""Explain the best tree-based model with regional importance and SHAP plots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from config import OUTPUTS_DIR

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS_DIR / "matplotlib-cache"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from config import (
    ALL_METRICS_PATH,
    FEATURE_IMPORTANCE_BY_REGION_PATH,
    FEATURES_PATH,
    MODEL_FEATURES,
    RANDOM_SEED,
    REGIONAL_SHAP_EAST_CHINA_PATH,
    REGIONAL_SHAP_SAHARA_PATH,
    SHAP_SUMMARY_PATH,
    TARGET_COLUMN,
    ensure_directories,
)
from train import build_models, load_features, validate_required_columns
from validation import random_split

TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}


def choose_best_tree_model() -> str:
    """Choose the best available tree model by random split RMSE."""
    if ALL_METRICS_PATH.exists():
        metrics = pd.read_csv(ALL_METRICS_PATH)
        tree_metrics = metrics[
            (metrics["validation_strategy"] == "random_split")
            & (metrics["model"].isin(TREE_MODELS))
        ].copy()
        if not tree_metrics.empty:
            return str(tree_metrics.sort_values("RMSE").iloc[0]["model"])

    available_models = set(build_models())
    for model_name in ["xgboost", "lightgbm", "random_forest"]:
        if model_name in available_models:
            return model_name
    raise ValueError("No tree-based model is available for SHAP explanation.")


def fit_best_tree_model(data: pd.DataFrame, model_name: str) -> tuple[Any, pd.DataFrame, pd.Series]:
    """Fit the chosen tree model on the random split training data."""
    split = random_split(data)
    models = build_models()
    if model_name not in models:
        raise ValueError(f"Chosen model {model_name} is not available. Available: {list(models)}")

    model = models[model_name]
    x_train = split.train_data[MODEL_FEATURES]
    y_train = split.train_data[TARGET_COLUMN]
    model.fit(x_train, y_train)
    return model, split.test_data, split.test_data[TARGET_COLUMN]


def save_regional_permutation_importance(
    model: Any,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Save permutation importance separately for each region.

    Regional importance helps reveal whether the model relies on different climate
    controls in dry Sahara versus wetter East China conditions.
    """
    frames = []
    for region, region_data in test_data.groupby("region"):
        if len(region_data) < 5:
            print(f"Skipping regional importance for {region}: not enough test rows.")
            continue

        result = permutation_importance(
            model,
            region_data[MODEL_FEATURES],
            region_data[TARGET_COLUMN],
            n_repeats=10,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "feature": MODEL_FEATURES,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            ).sort_values("importance_mean", ascending=False)
        )

    if not frames:
        raise ValueError("No regional feature importance could be calculated.")

    importance = pd.concat(frames, ignore_index=True)
    importance.to_csv(FEATURE_IMPORTANCE_BY_REGION_PATH, index=False)
    return importance


def shap_values_for_plot(model: Any, x_data: pd.DataFrame) -> Any:
    """Calculate SHAP values for a tree model."""
    import shap

    explainer = shap.Explainer(model, x_data)
    return explainer(x_data, check_additivity=False)


def save_shap_summary(model: Any, x_data: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save a SHAP beeswarm summary plot."""
    import shap

    if x_data.empty:
        print(f"Skipping {output_path.name}: no rows available.")
        return

    sample = x_data.sample(n=min(500, len(x_data)), random_state=RANDOM_SEED)
    shap_values = shap_values_for_plot(model, sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False, max_display=15)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Generate regional importance and SHAP plots."""
    ensure_directories()

    try:
        import shap  # noqa: F401
    except ImportError:
        print("SHAP is not installed. Install dependencies with: pip install -r requirements.txt")
        return

    data = load_features(FEATURES_PATH)
    validate_required_columns(data)

    model_name = choose_best_tree_model()
    model, test_data, _ = fit_best_tree_model(data, model_name)

    regional_importance = save_regional_permutation_importance(model, test_data)
    print(f"Saved regional feature importance to {FEATURE_IMPORTANCE_BY_REGION_PATH}")

    x_test = test_data[MODEL_FEATURES]
    save_shap_summary(model, x_test, SHAP_SUMMARY_PATH, f"SHAP Summary: {model_name}")
    save_shap_summary(
        model,
        test_data[test_data["region"] == "Sahara"][MODEL_FEATURES],
        REGIONAL_SHAP_SAHARA_PATH,
        f"SHAP Summary for Sahara: {model_name}",
    )
    save_shap_summary(
        model,
        test_data[test_data["region"] == "East China"][MODEL_FEATURES],
        REGIONAL_SHAP_EAST_CHINA_PATH,
        f"SHAP Summary for East China: {model_name}",
    )

    print(f"Best explained tree model: {model_name}")
    print(f"Saved SHAP summary to {SHAP_SUMMARY_PATH}")
    print(f"Saved regional SHAP plot to {REGIONAL_SHAP_SAHARA_PATH}")
    print(f"Saved regional SHAP plot to {REGIONAL_SHAP_EAST_CHINA_PATH}")
    print("Top regional importance rows:")
    print(regional_importance.groupby("region").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
