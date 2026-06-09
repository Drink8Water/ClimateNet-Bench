"""Calibration evaluation for ClimateNet-Bench.

Computes calibration reports — coverage and interval width broken down by
region, split protocol, and climate type.  Writes the standard benchmark
output file ``intervals.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climatenet.evaluation.conformal import (
    build_prediction_intervals,
    evaluate_by_group,
    evaluate_coverage,
    evaluate_interval_width,
    fit_conformal_quantile,
)


# ---------------------------------------------------------------------------
# Intervals CSV builder
# ---------------------------------------------------------------------------


def build_intervals_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    meta_df: pd.DataFrame | None = None,
    model_name: str = "",
    split_id: str = "",
    experiment_id: str = "",
) -> pd.DataFrame:
    """Build the standard ``intervals.csv`` table.

    Parameters
    ----------
    y_true, y_pred
        Test-set targets and predictions.
    lower, upper
        Prediction interval bounds.
    meta_df
        Optional DataFrame with metadata columns.  Must have the same
        row count as ``y_true``.
    model_name, split_id, experiment_id
        Labels added to every row.

    Returns
    -------
    DataFrame with columns:
    ``experiment_id, model_name, split_id, region, climate_type,
    year, month, latitude, longitude, y_true, y_pred, lower, upper,
    covered, interval_width``.
    """
    n = len(y_true)
    covered = (y_true >= lower) & (y_true <= upper)
    width = upper - lower

    table = pd.DataFrame(
        {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "split_id": split_id,
            "y_true": y_true,
            "y_pred": y_pred,
            "lower": lower,
            "upper": upper,
            "covered": covered,
            "interval_width": width,
        }
    )

    if meta_df is not None:
        if len(meta_df) != n:
            raise ValueError(
                f"meta_df has {len(meta_df)} rows but y_true has {n}"
            )
        for col in [
            "region",
            "climate_type",
            "year",
            "month",
            "latitude",
            "longitude",
            "split_protocol",
        ]:
            if col in meta_df.columns:
                table[col] = meta_df[col].to_numpy()

        # If split_protocol is present as meta but not as an arg, use it
        if "split_protocol" in meta_df.columns and not split_id:
            table["split_id"] = meta_df["split_protocol"]

    # Ensure standard columns exist (NaN-filled if missing)
    for col in [
        "region",
        "climate_type",
        "year",
        "month",
        "latitude",
        "longitude",
    ]:
        if col not in table.columns:
            table[col] = float("nan")

    return table


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------


def build_calibration_report(
    y_calib: np.ndarray,
    pred_calib: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    alpha: float = 0.1,
    test_df: pd.DataFrame | None = None,
    group_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Build a complete calibration report.

    Returns a dict suitable for serialisation to JSON or inclusion in
    ``benchmark_metadata.json``.
    """
    q = fit_conformal_quantile(y_calib, pred_calib, alpha=alpha)
    lower, upper = build_prediction_intervals(pred_test, q)
    coverage = evaluate_coverage(y_test, lower, upper)
    width = evaluate_interval_width(lower, upper)

    report: dict[str, Any] = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "conformal_quantile": q,
        "n_calibration": int(len(y_calib)),
        "n_test": int(len(y_test)),
        "coverage": coverage,
        "mean_interval_width": width,
    }

    if test_df is not None and group_cols:
        for col in group_cols:
            if col in test_df.columns:
                gdf = test_df.copy()
                gdf["y_true"] = y_test
                gdf["lower"] = lower
                gdf["upper"] = upper
                report[f"by_{col}"] = (
                    evaluate_by_group(gdf, col)
                    .to_dict(orient="records")
                )

    return report


def save_calibration_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Write a calibration report as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
