"""Binary event detection metrics for hydroclimate extreme evaluation.

All metrics operate on boolean label arrays (True = event occurred).
When a denominator is zero the metric returns ``np.nan`` and includes
a ``"warning"`` key in the returned dict describing which condition
caused the undefined result.

Metrics
-------
- **POD** (Probability of Detection): TP / (TP + FN)
- **FAR** (False Alarm Ratio): FP / (TP + FP)
- **CSI** (Critical Success Index): TP / (TP + FP + FN)
- **intensity_bias**: mean(predicted value) / mean(observed value)
  computed only over grid points where the **observed** event occurred.

Usage
-----
.. code-block:: python

    from climatenet.evaluation.detection import (
        compute_pod, compute_far, compute_csi,
        compute_intensity_bias, compute_event_detection_table,
    )

    result = compute_pod(y_true_label, y_pred_label)
    # result == {"pod": 0.85}
    # or, when no events observed:
    # result == {"pod": NaN, "warning": "no observed events"}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_bool(arr: np.ndarray) -> np.ndarray:
    """Convert to boolean array with validation."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"Input must be 1-D, got shape {arr.shape}")
    return arr.astype(bool)


def _check_length_match(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> None:
    """Raise ValueError if arrays have different lengths."""
    if len(a) != len(b):
        raise ValueError(
            f"Length mismatch: {label_a} has {len(a)} samples, "
            f"{label_b} has {len(b)} samples."
        )


def _nan_result(warning: str) -> dict[str, Any]:
    """Return a NaN metric dict with a warning string."""
    logger.warning(warning)
    return {"value": float("nan"), "warning": warning}


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------


def compute_pod(
    y_true_label: np.ndarray,
    y_pred_label: np.ndarray,
) -> dict[str, Any]:
    """Probability of Detection (hit rate).

    POD = TP / (TP + FN)

    Returns ``np.nan`` with a warning when no events are observed
    (denominator = 0).

    Parameters
    ----------
    y_true_label : np.ndarray
        Ground-truth event labels (1-D boolean or 0/1).
    y_pred_label : np.ndarray
        Predicted event labels (1-D boolean or 0/1).

    Returns
    -------
    dict with keys ``"value"`` (float) and optionally ``"warning"`` (str).
    """
    y_true = _as_bool(y_true_label)
    y_pred = _as_bool(y_pred_label)
    _check_length_match(y_true, y_pred, "y_true_label", "y_pred_label")

    tp = int((y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    denom = tp + fn

    if denom == 0:
        return _nan_result("no observed events — POD is undefined (TP+FN=0)")

    return {"value": tp / denom}


def compute_far(
    y_true_label: np.ndarray,
    y_pred_label: np.ndarray,
) -> dict[str, Any]:
    """False Alarm Ratio.

    FAR = FP / (TP + FP)

    Returns ``np.nan`` with a warning when no events are predicted
    (denominator = 0).

    Parameters
    ----------
    y_true_label : np.ndarray
        Ground-truth event labels (1-D boolean or 0/1).
    y_pred_label : np.ndarray
        Predicted event labels (1-D boolean or 0/1).

    Returns
    -------
    dict with keys ``"value"`` (float) and optionally ``"warning"`` (str).
    """
    y_true = _as_bool(y_true_label)
    y_pred = _as_bool(y_pred_label)
    _check_length_match(y_true, y_pred, "y_true_label", "y_pred_label")

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    denom = tp + fp

    if denom == 0:
        return _nan_result("no predicted events — FAR is undefined (TP+FP=0)")

    return {"value": fp / denom}


def compute_csi(
    y_true_label: np.ndarray,
    y_pred_label: np.ndarray,
) -> dict[str, Any]:
    """Critical Success Index (threat score).

    CSI = TP / (TP + FP + FN)

    Returns ``np.nan`` with a warning when no events exist in either
    observed or predicted (denominator = 0).

    Parameters
    ----------
    y_true_label : np.ndarray
        Ground-truth event labels (1-D boolean or 0/1).
    y_pred_label : np.ndarray
        Predicted event labels (1-D boolean or 0/1).

    Returns
    -------
    dict with keys ``"value"`` (float) and optionally ``"warning"`` (str).
    """
    y_true = _as_bool(y_true_label)
    y_pred = _as_bool(y_pred_label)
    _check_length_match(y_true, y_pred, "y_true_label", "y_pred_label")

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    denom = tp + fp + fn

    if denom == 0:
        return _nan_result(
            "no events in either observed or predicted — CSI is undefined (TP+FP+FN=0)"
        )

    return {"value": tp / denom}


def compute_intensity_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_label: np.ndarray,
) -> dict[str, Any]:
    """Intensity bias over observed event grid points.

    intensity_bias = mean(y_pred[event]) / mean(y_true[event])

    Only samples where ``y_true_label == True`` are included.
    Returns ``np.nan`` with a warning when no observed events exist.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth continuous values (e.g. evaporation anomaly).
    y_pred : np.ndarray
        Predicted continuous values.
    y_true_label : np.ndarray
        Boolean array — True where the observed event occurred.

    Returns
    -------
    dict with keys ``"value"`` (float) and optionally ``"warning"`` (str).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true_label = _as_bool(y_true_label)

    _check_length_match(y_true, y_pred, "y_true", "y_pred")
    _check_length_match(y_true, y_true_label, "y_true", "y_true_label")

    event_mask = y_true_label
    n_events = int(event_mask.sum())

    if n_events == 0:
        return _nan_result(
            "no observed events — intensity_bias is undefined"
        )

    mean_obs = float(np.mean(y_true[event_mask]))
    mean_pred = float(np.mean(y_pred[event_mask]))

    if mean_obs == 0.0:
        return _nan_result(
            "mean observed value over event points is zero — "
            "intensity_bias is undefined (division by zero)"
        )

    return {"value": mean_pred / mean_obs, "n_event_samples": n_events}


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def compute_event_detection_table(
    results_df: pd.DataFrame,
    event_types: list[str] | None = None,
) -> pd.DataFrame:
    """Compute all detection metrics for multiple event types.

    Expects a DataFrame containing, for each event type *ev*, two boolean
    columns:

    - ``{ev}`` — observed (ground-truth) event labels
    - ``{ev}_pred`` — predicted event labels

    For example, for ``"soil_moisture_drought"`` the columns
    ``soil_moisture_drought`` and ``soil_moisture_drought_pred``
    must both exist.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain ``y_true`` (continuous), ``y_pred`` (continuous),
        and for each event type *ev*: ``{ev}`` (bool observed label)
        and ``{ev}_pred`` (bool predicted label).
    event_types : list[str] or None
        Event type names.  Defaults to ``ALL_EVENT_TYPES``.

    Returns
    -------
    pd.DataFrame
        Columns: ``event_type``, ``pod``, ``far``, ``csi``,
        ``intensity_bias``, ``n_observed_events``, ``n_predicted_events``,
        plus ``*_warning`` columns for each metric.
    """
    if event_types is None:
        from climatenet.evaluation.hydroclimate_labels import ALL_EVENT_TYPES

        event_types = list(ALL_EVENT_TYPES)

    # Build required column set: y_true, y_pred, plus {ev} and {ev}_pred
    required = {"y_true", "y_pred"}
    for ev in event_types:
        required.add(ev)
        required.add(f"{ev}_pred")
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(
            f"results_df is missing required columns: {sorted(missing)}"
        )

    rows: list[dict[str, Any]] = []

    for event_type in event_types:
        y_true_label = results_df[event_type].to_numpy()
        y_pred_label = results_df[f"{event_type}_pred"].to_numpy()

        pod_result = compute_pod(y_true_label, y_pred_label)
        far_result = compute_far(y_true_label, y_pred_label)
        csi_result = compute_csi(y_true_label, y_pred_label)
        ibias_result = compute_intensity_bias(
            results_df["y_true"].to_numpy(),
            results_df["y_pred"].to_numpy(),
            y_true_label,
        )

        rows.append(
            {
                "event_type": event_type,
                "pod": pod_result["value"],
                "far": far_result["value"],
                "csi": csi_result["value"],
                "intensity_bias": ibias_result["value"],
                "n_observed_events": int(y_true_label.sum()),
                "n_predicted_events": int(y_pred_label.sum()),
                "pod_warning": pod_result.get("warning"),
                "far_warning": far_result.get("warning"),
                "csi_warning": csi_result.get("warning"),
                "ibias_warning": ibias_result.get("warning"),
            }
        )

    return pd.DataFrame(rows)
