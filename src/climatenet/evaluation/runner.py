"""Evaluation runner for the mini-benchmark pipeline.

Provides :func:`evaluate_model_on_split` — a single function that fits
a model on train data, predicts on test data, and computes regression
metrics, event detection metrics, and grouped metrics.

Grouped metrics are computed by ``region`` and ``climate_zone`` when
those columns are present in the test DataFrame.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from climatenet.evaluation.metrics import evaluate_regression

logger = logging.getLogger(__name__)


def _build_event_label_for_target(
    df: pd.DataFrame,
    target_col: str,
    continuous_values: np.ndarray,
    event_type: str,
    thresholds: dict,
) -> np.ndarray:
    """Construct a boolean event label from continuous values using
    train-fitted percentile thresholds.

    Parameters
    ----------
    df
        Test DataFrame (must contain ``"month"`` column).
    target_col
        The continuous variable name (e.g. ``"evaporation_anomaly"``).
    continuous_values
        1-D array of values for the target variable (can be y_true or y_pred).
    event_type
        Event type key from ``hydroclimate_labels.ALL_EVENT_TYPES``.
    thresholds
        Output of ``fit_event_thresholds`` on the training set.

    Returns
    -------
    np.ndarray of bool, shape ``(n_samples,)``.
    """
    months = df["month"].to_numpy()
    result = np.zeros(len(continuous_values), dtype=bool)

    for month in sorted(thresholds.keys()):
        mask = months == month
        if not mask.any():
            continue
        t = thresholds[month]

        if event_type == "evaporation_deficit":
            result[mask] = continuous_values[mask] < t["evap_p10"]
        elif event_type == "soil_moisture_drought":
            result[mask] = continuous_values[mask] < t["sm_p10"]
        elif event_type == "compound_hot_dry":
            # Cannot construct from a single continuous variable — skip.
            continue

    return result


def evaluate_model_on_split(
    model: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    event_cols: list[str] | None = None,
    event_thresholds: dict | None = None,
    group_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Fit a model on train, predict on test, and compute metrics.

    Parameters
    ----------
    model
        Model object with ``fit``, ``predict``, ``get_model_name`` methods.
    train_df
        Training data.
    val_df
        Optional validation data (passed to ``model.fit()``).
    test_df
        Test data.
    target_col
        Name of the target column.
    feature_cols
        Feature column names.  May be ``None`` for baselines that don't
        use features (e.g. climatology).
    event_cols
        Optional list of event type names (e.g. ``["evaporation_deficit"]``).
        If provided, **true event labels** are constructed from ``y_true``
        and **predicted event labels** from ``y_pred`` using
        ``event_thresholds`` (which must be fitted on the training set).
        **Do NOT pass pre-existing ``{ev}_pred`` columns in test_df** —
        predicted events are always derived from model predictions.
    event_thresholds
        Required when ``event_cols`` is provided.  Output of
        :func:`~climatenet.evaluation.hydroclimate_labels.fit_event_thresholds`.
        Used to construct both true and predicted event labels from the
        continuous target variable.
    group_cols
        Optional grouping columns for per-group metrics.  Defaults to
        ``["region", "climate_zone"]`` when those columns are present.

    Returns
    -------
    dict
        Keys: ``"predictions_df"``, ``"metrics_overall"``,
        ``"metrics_by_group"``.
    """
    model_name = (
        model.get_model_name() if hasattr(model, "get_model_name") else "unknown"
    )

    # --- fit ---
    fit_kwargs: dict[str, Any] = {"target_col": target_col}
    if feature_cols is not None:
        fit_kwargs["feature_cols"] = feature_cols
    if val_df is not None and not val_df.empty:
        fit_kwargs["val_df"] = val_df

    model.fit(train_df, **fit_kwargs)

    # --- predict ---
    y_pred = model.predict(test_df)
    y_true = test_df[target_col].to_numpy(dtype=np.float64)

    # --- regression metrics ---
    metrics_overall = evaluate_regression(y_true, y_pred)
    metrics_overall["model_name"] = model_name
    metrics_overall["n_samples"] = len(test_df)

    # --- predictions DataFrame ---
    pred_cols = {
        "y_true": y_true,
        "y_pred": y_pred,
        "model_name": model_name,
    }
    for meta_col in ["sample_id", "region", "climate_zone", "year", "month"]:
        if meta_col in test_df.columns:
            pred_cols[meta_col] = test_df[meta_col].to_numpy()

    predictions_df = pd.DataFrame(pred_cols)

    # --- event detection metrics ---
    # Construct true_event from y_true and pred_event from y_pred
    # using train-fitted thresholds.  This ensures a real comparison
    # between observed and predicted events — never truth vs itself.
    if event_cols:
        if event_thresholds is None:
            raise ValueError(
                "event_thresholds is required when event_cols is provided. "
                "Use climatenet.evaluation.hydroclimate_labels.fit_event_thresholds() "
                "on the training set."
            )

        from climatenet.evaluation.detection import (
            compute_csi,
            compute_far,
            compute_intensity_bias,
            compute_pod,
        )
        from climatenet.evaluation.hydroclimate_labels import (
            ALL_EVENT_TYPES,
            build_all_event_labels,
        )

        # Determine which event types are derivable from the target variable.
        # For a single-target model (e.g. evaporation_anomaly), only
        # events defined on that variable can be evaluated.
        # ALL_EVENT_TYPES = ["soil_moisture_drought", "evaporation_deficit",
        #                    "compound_hot_dry"]

        # Map event type → which anomaly column is used for the threshold
        _EVENT_TARGET_MAP = {
            "evaporation_deficit": "evaporation_anomaly",
            "soil_moisture_drought": "soil_moisture_anomaly",
            "compound_hot_dry": None,  # requires both temperature + soil moisture
        }

        for ev in event_cols:
            if ev not in ALL_EVENT_TYPES:
                logger.warning("Unknown event type '%s' — skipping.", ev)
                continue

            required_var = _EVENT_TARGET_MAP.get(ev)
            if required_var is None:
                # Compound event — skip unless the model predicts all required vars.
                logger.info(
                    "Skipping '%s': compound event requires multi-target "
                    "predictions (temperature_anomaly + soil_moisture_anomaly).",
                    ev,
                )
                continue

            # Only evaluate if the model's target matches the event's variable.
            if target_col != required_var:
                logger.info(
                    "Skipping '%s': model predicts '%s' but event is "
                    "defined on '%s'.",
                    ev, target_col, required_var,
                )
                continue

            # Construct true_event from y_true (observed).
            # Construct pred_event from y_pred (model prediction).
            # Both use the same train-fitted thresholds.
            true_label = _build_event_label_for_target(
                test_df, target_col, y_true, ev, event_thresholds,
            )
            pred_label = _build_event_label_for_target(
                test_df, target_col, y_pred, ev, event_thresholds,
            )

            pod_r = compute_pod(true_label, pred_label)
            far_r = compute_far(true_label, pred_label)
            csi_r = compute_csi(true_label, pred_label)
            ibias_r = compute_intensity_bias(y_true, y_pred, true_label)

            predictions_df[f"{ev}_true"] = true_label
            predictions_df[f"{ev}_pred"] = pred_label

            metrics_overall[f"pod_{ev}"] = pod_r["value"]
            metrics_overall[f"far_{ev}"] = far_r["value"]
            metrics_overall[f"csi_{ev}"] = csi_r["value"]
            metrics_overall[f"intensity_bias_{ev}"] = ibias_r["value"]
            metrics_overall[f"n_events_{ev}"] = int(true_label.sum())

    # --- grouped metrics ---
    metrics_by_group: list[dict[str, Any]] = []
    if group_cols is None:
        group_cols = []
        for gc in ["region", "climate_zone"]:
            if gc in test_df.columns:
                group_cols.append(gc)

    if group_cols:
        for group_key, group_idx in predictions_df.groupby(group_cols).groups.items():
            if isinstance(group_key, (str, int, float)):
                group_key = (group_key,)
            g_idx = list(group_idx)
            yt_g = y_true[g_idx]
            yp_g = y_pred[g_idx]

            if len(yt_g) == 0:
                continue

            g_metrics = evaluate_regression(yt_g, yp_g)
            g_metrics["model_name"] = model_name
            g_metrics["n_samples"] = len(yt_g)
            for i, col in enumerate(group_cols):
                g_metrics[col] = str(group_key[i])

            metrics_by_group.append(g_metrics)

    return {
        "predictions_df": predictions_df,
        "metrics_overall": metrics_overall,
        "metrics_by_group": metrics_by_group,
    }
