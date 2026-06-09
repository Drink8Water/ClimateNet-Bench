"""Reproducible benchmark runner for ClimateNet-Bench.

Orchestrates: config → dataset → splits → models → train → evaluate → save.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climatenet.benchmark.split_protocols import (
    SplitResult,
    generate_all_splits,
    validate_split,
)
from climatenet.data.forecasting_dataset import build_forecasting_samples
from climatenet.data.loaders import load_csv
from climatenet.evaluation.metrics import evaluate_regression
from climatenet.evaluation.skill_score import compute_skill_scores
from climatenet.models.model_factory import create_model
from climatenet.training.experiment_registry import (
    ExperimentRecord,
    ExperimentRegistry,
)
from climatenet.utils.config import load_yaml, save_yaml
from climatenet.utils.paths import ensure_directory, resolve_project_path
from climatenet.utils.random import set_random_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-set helpers
# ---------------------------------------------------------------------------


def _get_feature_columns(
    samples_df: pd.DataFrame,
    feature_set_name: str,
    feature_sets: dict,
) -> list[str]:
    """Return the list of column names for a given feature set."""
    if feature_set_name in feature_sets:
        requested = feature_sets[feature_set_name].get("features", [])
    else:
        requested = feature_sets.get("full", {}).get("features", [])

    # Only return columns that actually exist in the data
    available = [c for c in requested if c in samples_df.columns]
    if not available:
        raise ValueError(
            f"No feature columns found for feature_set='{feature_set_name}'. "
            f"Requested: {requested}"
        )
    return available


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_benchmark(
    config: dict[str, Any],
    output_root: str | Path = "outputs/benchmark",
) -> ExperimentRegistry:
    """Run a complete benchmark according to a config dict.

    Parameters
    ----------
    config
        Benchmark configuration (from YAML).  Must contain keys:
        ``benchmark_name``, ``regions``, ``models``, ``split_protocols``,
        ``feature_sets``, ``metrics``.
    output_root
        Root directory for benchmark outputs.

    Returns
    -------
    :class:`ExperimentRegistry` with results for every completed experiment.
    """
    set_random_seed(int(config.get("random_seed", 42)))
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    registry = ExperimentRegistry(root / "experiment_registry.json")

    benchmark_name = config.get("benchmark_name", "unnamed")
    models_cfg = config.get("models", [])
    split_protocols = config.get("split_protocols", ["random"])
    feature_sets = config.get("feature_sets", {})
    time_range = config.get("time_range", {})
    seed = int(config.get("random_seed", 42))

    logger.info("=== Benchmark: %s ===", benchmark_name)
    logger.info("Models: %s", [m.get("name", m) for m in models_cfg])
    logger.info("Splits: %s", split_protocols)
    logger.info("Feature sets: %s", list(feature_sets.keys()))

    # ── Load or build forecasting dataset ──────────────────────────
    samples_path = config.get("forecasting_samples_path",
                              "data/processed/forecasting_samples.csv")
    samples_path = resolve_project_path(samples_path)

    if samples_path.exists():
        logger.info("Loading forecasting samples from %s", samples_path)
        samples_df = load_csv(samples_path)
    else:
        logger.info("Building forecasting samples from features …")
        features_path = config.get("features_path", "data/processed/features.csv")
        features_df = load_csv(resolve_project_path(features_path))
        samples_df, _ = build_forecasting_samples(features_df)

    logger.info("Samples: %d rows, %d grid cells",
                len(samples_df), samples_df["grid_id"].nunique())

    # ── Generate splits ────────────────────────────────────────────
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_results: list[SplitResult] = generate_all_splits(samples_df, splits_dir)
    logger.info("Generated %d split(s)", len(split_results))

    # ── Prepare feature set list ───────────────────────────────────
    feature_set_names = list(feature_sets.keys())
    if not feature_set_names:
        feature_set_names = ["full"]

    # ── Run experiments ────────────────────────────────────────────
    experiments_dir = root / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    for model_cfg in models_cfg:
        if isinstance(model_cfg, str):
            model_name = model_cfg
            model_kwargs = {}
        else:
            model_name = model_cfg.get("name", "")
            model_kwargs: dict[str, Any] = {}
            for k, v in model_cfg.items():
                if k in ("name", "type", "description"):
                    continue
                if k == "params" and isinstance(v, dict):
                    model_kwargs.update(v)  # flatten params
                else:
                    model_kwargs[k] = v

        # Build model once per model type
        try:
            model = create_model(model_name, model_kwargs)
            logger.info("Created model: %s", model.get_model_name())
        except Exception as e:
            logger.error("Failed to create model '%s': %s", model_name, e)
            continue

        # Baselines don't need feature sets
        is_baseline = model_name in ("climatology", "persistence")

        for split_result in split_results:
            # Tree models do temporal splits too now
            for fs_name in feature_set_names:
                if is_baseline and fs_name != feature_set_names[0]:
                    continue  # Baselines only run once per split

                experiment_id = (
                    f"{benchmark_name}_{model.get_model_name()}"
                    f"_{split_result.protocol}_{fs_name}"
                )
                exp_dir = experiments_dir / experiment_id
                exp_dir.mkdir(parents=True, exist_ok=True)

                record = ExperimentRecord(
                    experiment_id=experiment_id,
                    benchmark_name=benchmark_name,
                    model_name=model.get_model_name(),
                    split_protocol=split_result.protocol,
                    feature_set=fs_name,
                    train_regions=sorted(set(
                        samples_df[samples_df["sample_id"].isin(split_result.train_ids)]["region"]
                    )),
                    test_regions=sorted(set(
                        samples_df[samples_df["sample_id"].isin(split_result.test_ids)]["region"]
                    )),
                    train_years=sorted(set(
                        samples_df[samples_df["sample_id"].isin(split_result.train_ids)]["target_year"]
                    )),
                    test_years=sorted(set(
                        samples_df[samples_df["sample_id"].isin(split_result.test_ids)]["target_year"]
                    )),
                    seed=seed,
                )
                registry.add(record)

                try:
                    registry.mark_running(experiment_id)
                    _run_one_experiment(
                        samples_df=samples_df,
                        split_result=split_result,
                        model=model,
                        model_name=model.get_model_name(),
                        feature_set_name=fs_name,
                        feature_sets=feature_sets,
                        is_baseline=is_baseline,
                        exp_dir=exp_dir,
                        config=config,
                        seed=seed,
                    )
                    registry.mark_completed(experiment_id)
                    logger.info("✅ %s", experiment_id)
                except Exception as e:
                    logger.error("❌ %s: %s", experiment_id, e)
                    registry.mark_failed(experiment_id, str(e))

    registry.save()
    logger.info("Registry saved: %d completed, %d failed",
                len(registry.list_completed()), len(registry.list_failed()))
    return registry


# ---------------------------------------------------------------------------
# Single-experiment runner
# ---------------------------------------------------------------------------


def _run_one_experiment(
    samples_df: pd.DataFrame,
    split_result: SplitResult,
    model: Any,
    model_name: str,
    feature_set_name: str,
    feature_sets: dict,
    is_baseline: bool,
    exp_dir: Path,
    config: dict,
    seed: int,
) -> None:
    """Train, predict, evaluate for one (model × split × feature_set)."""
    train_df = samples_df[samples_df["sample_id"].isin(split_result.train_ids)].copy()
    val_df = samples_df[samples_df["sample_id"].isin(split_result.val_ids)].copy()
    test_df = samples_df[samples_df["sample_id"].isin(split_result.test_ids)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Empty train or test set.")

    # ── features ─────────────────────────────────────────────────
    if is_baseline:
        feature_cols = []  # baselines ignore feature_columns
    else:
        feature_cols = _get_feature_columns(samples_df, feature_set_name, feature_sets)

    target_col = config.get("target", "y_true")

    # ── fit ───────────────────────────────────────────────────────
    fit_kwargs: dict = {"train_df": train_df, "target_column": target_col}
    if not is_baseline:
        fit_kwargs["feature_columns"] = feature_cols
    if val_df is not None and not val_df.empty and not is_baseline:
        fit_kwargs["val_df"] = val_df

    model.fit(**fit_kwargs)

    # ── predict ──────────────────────────────────────────────────
    y_pred = model.predict(test_df)
    y_true = test_df[target_col].to_numpy(dtype=np.float64)

    # ── primary metrics ──────────────────────────────────────────
    metrics = evaluate_regression(y_true, y_pred)
    metrics["model_name"] = model_name
    metrics["split_protocol"] = split_result.protocol
    metrics["feature_set"] = feature_set_name
    metrics["n_train"] = len(train_df)
    metrics["n_val"] = len(val_df)
    metrics["n_test"] = len(test_df)

    # Save metrics
    with (exp_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ── predictions ──────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "sample_id": test_df["sample_id"].tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "model_name": model_name,
        "split_protocol": split_result.protocol,
        "feature_set": feature_set_name,
    })
    # Merge metadata from test_df
    for col in ["region", "climate_type", "target_year", "target_month",
                "latitude", "longitude", "grid_id"]:
        if col in test_df.columns:
            pred_df[col] = test_df[col].to_numpy()
    pred_df.to_csv(exp_dir / "predictions.csv", index=False)

    # ── config snapshot ───────────────────────────────────────────
    save_yaml(config, exp_dir / "config.yaml")
