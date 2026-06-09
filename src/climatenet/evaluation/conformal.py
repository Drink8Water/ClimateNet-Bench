"""Split conformal prediction for regression uncertainty.

Implements split (inductive) conformal prediction for ClimateNet-Bench.
The calibration set must be disjoint from both the training set and the
test set — we calibrate the quantile on the **validation** split.

Method
------
1. Train model on training set.
2. Compute absolute residuals on calibration (validation) set:
   ``r_i = |y_i − ŷ_i|``
3. For a target coverage ``1 − α``, compute the empirical quantile:
   ``q̂ = quantile({r_i}, 1 − α)`` (with finite-sample correction)
4. For each test point, build a constant-width interval:
   ``[ŷ − q̂,  ŷ + q̂]``
5. The interval is guaranteed (under exchangeability) to achieve
   coverage ≥ ``1 − α`` in expectation.

References
----------
Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World.*
Angelopoulos & Bates (2021). *A Gentle Introduction to Conformal Prediction.*
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core conformal functions
# ---------------------------------------------------------------------------


def fit_conformal_quantile(
    y_calib: np.ndarray,
    pred_calib: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Compute the conformal quantile from calibration residuals.

    Parameters
    ----------
    y_calib
        True target values for the calibration set, shape ``(n_calib,)``.
    pred_calib
        Model predictions on the calibration set, same shape.
    alpha
        Significance level.  ``0.1`` → 90 % target coverage.

    Returns
    -------
    ``q̂`` — the scalar quantile used to build prediction intervals
    ``[ŷ − q̂, ŷ + q̂]``.

    Raises
    ------
    ValueError
        If ``alpha`` is not in ``(0, 1)`` or calibration arrays are empty.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if len(y_calib) == 0:
        raise ValueError("Calibration set is empty.")
    if len(y_calib) != len(pred_calib):
        raise ValueError(
            f"Length mismatch: y_calib={len(y_calib)}, pred_calib={len(pred_calib)}"
        )

    residuals = np.abs(np.asarray(y_calib, dtype=np.float64)
                       - np.asarray(pred_calib, dtype=np.float64))

    # Finite-sample correction: (1 − α) × (n + 1) / n
    n = len(residuals)
    level = (1.0 - alpha) * (n + 1) / n

    # Clamp to [0, 1] for small n edge cases
    level = max(0.0, min(1.0, level))

    return float(np.quantile(residuals, level))


def build_prediction_intervals(
    y_pred: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric prediction intervals around each prediction.

    Parameters
    ----------
    y_pred
        Model predictions on the test set.
    q
        Conformal quantile from ``fit_conformal_quantile``.

    Returns
    -------
    ``(lower, upper)`` — each is a 1-D array of same length as ``y_pred``.
    ``lower[i] = y_pred[i] − q``, ``upper[i] = y_pred[i] + q``.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if q < 0:
        raise ValueError(f"Quantile q must be ≥ 0, got {q}")
    lower = y_pred - q
    upper = y_pred + q
    return lower, upper


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of true values that fall within the prediction intervals.

    Parameters
    ----------
    y_true
        True target values.
    lower
        Lower bound of prediction intervals, same length as ``y_true``.
    upper
        Upper bound of prediction intervals, same length as ``y_true``.

    Returns
    -------
    Coverage ∈ ``[0, 1]``.  Target is ``1 − α`` (e.g. 0.90 for α = 0.1).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    _check_lengths(y_true, lower, "y_true", "lower")
    _check_lengths(y_true, upper, "y_true", "upper")

    if len(y_true) == 0:
        return float("nan")

    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def evaluate_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of prediction intervals.

    For split conformal with constant ``q``, all intervals have the same
    width ``2q``.  This function returns the mean width, which equals ``2q``
    for the constant-width case but generalises to variable-width methods.

    Parameters
    ----------
    lower
        Lower bound of prediction intervals.
    upper
        Upper bound of prediction intervals.

    Returns
    -------
    Mean interval width ≥ 0.  Smaller is better, for a given coverage level.
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    _check_lengths(lower, upper, "lower", "upper")
    if len(lower) == 0:
        return float("nan")
    return float(np.mean(upper - lower))


# ---------------------------------------------------------------------------
# Grouped evaluation
# ---------------------------------------------------------------------------


def evaluate_by_group(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Compute coverage and interval width per group.

    Parameters
    ----------
    df
        DataFrame with columns ``y_true``, ``lower``, ``upper``,
        and the column named by ``group_col``.
    group_col
        Column to group by (e.g. ``"region"``, ``"split_protocol"``,
        ``"climate_type"``).

    Returns
    -------
    DataFrame with columns ``group_col``, ``n``, ``coverage``,
    ``mean_interval_width``.
    """
    required = {"y_true", "lower", "upper", group_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {sorted(missing)}"
        )

    rows = []
    for name, group in df.groupby(group_col):
        n = len(group)
        cov = evaluate_coverage(
            group["y_true"].to_numpy(),
            group["lower"].to_numpy(),
            group["upper"].to_numpy(),
        )
        width = evaluate_interval_width(
            group["lower"].to_numpy(),
            group["upper"].to_numpy(),
        )
        rows.append(
            {
                group_col: name,
                "n": n,
                "coverage": cov,
                "mean_interval_width": width,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


def run_conformal_pipeline(
    y_calib: np.ndarray,
    pred_calib: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    alpha: float = 0.1,
    test_df: pd.DataFrame | None = None,
    group_cols: list[str] | None = None,
) -> dict:
    """Run the complete split-conformal pipeline and return all results.

    Parameters
    ----------
    y_calib, pred_calib
        Calibration (validation) set targets and predictions.
    y_test, pred_test
        Test set targets and predictions.
    alpha
        Significance level (default 0.1 → 90 % intervals).
    test_df
        Optional DataFrame with metadata columns (region, climate_type,
        split_protocol, etc.) for grouped evaluation.
    group_cols
        Columns to compute grouped coverage for.

    Returns
    -------
    Dict with keys:

    - ``q`` — conformal quantile
    - ``coverage`` — test-set coverage
    - ``mean_interval_width`` — mean interval width
    - ``lower``, ``upper`` — prediction interval arrays
    - ``covered`` — boolean array
    - ``by_group`` — dict of DataFrames (one per group_col)
    """
    q = fit_conformal_quantile(y_calib, pred_calib, alpha=alpha)
    lower, upper = build_prediction_intervals(pred_test, q)
    coverage = evaluate_coverage(y_test, lower, upper)
    width = evaluate_interval_width(lower, upper)
    covered = (y_test >= lower) & (y_test <= upper)

    result: dict = {
        "q": q,
        "coverage": coverage,
        "mean_interval_width": width,
        "lower": lower,
        "upper": upper,
        "covered": covered,
        "by_group": {},
    }

    if test_df is not None and group_cols:
        for col in group_cols:
            if col in test_df.columns:
                gdf = test_df.copy()
                gdf["y_true"] = y_test
                gdf["lower"] = lower
                gdf["upper"] = upper
                result["by_group"][col] = evaluate_by_group(gdf, col)

    return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _check_lengths(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"Length mismatch: {name_a}={len(a)}, {name_b}={len(b)}"
        )
