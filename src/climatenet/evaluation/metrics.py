"""Primary regression metrics for ClimateNet-Bench.

Provides standalone functions and a batch evaluator that mirrors
scikit-learn conventions but adds input validation and NaN handling.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true
        Ground-truth values, shape ``(n_samples,)``.
    y_pred
        Predicted values, same shape as ``y_true``.

    Returns
    -------
    MAE ≥ 0.  Lower is better.
    """
    _validate_inputs(y_true, y_pred)
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true
        Ground-truth values, shape ``(n_samples,)``.
    y_pred
        Predicted values, same shape as ``y_true``.

    Returns
    -------
    RMSE ≥ 0.  Lower is better.  Same unit as the target variable.
    """
    _validate_inputs(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R².

    Parameters
    ----------
    y_true
        Ground-truth values, shape ``(n_samples,)``.
    y_pred
        Predicted values, same shape as ``y_true``.

    Returns
    -------
    R² ≤ 1.  Higher is better.  Can be negative when the model
    performs worse than a constant mean prediction.
    """
    _validate_inputs(y_true, y_pred)
    return float(r2_score(y_true, y_pred))


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE, RMSE, and R² in a single call.

    Parameters
    ----------
    y_true
        Ground-truth values.
    y_pred
        Predicted values.

    Returns
    -------
    Dict with keys ``"mae"``, ``"rmse"``, ``"r2"``.
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Check that inputs are compatible and finite."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1-D arrays, "
            f"got shapes {y_true.shape} and {y_pred.shape}"
        )
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("Input arrays are empty.")

    # NaN check — produce a clear message rather than silent propagation.
    nan_in_true = np.any(np.isnan(y_true))
    nan_in_pred = np.any(np.isnan(y_pred))
    if nan_in_true or nan_in_pred:
        locs: list[str] = []
        if nan_in_true:
            idx = np.where(np.isnan(y_true))[0]
            locs.append(f"y_true at indices {idx[:5].tolist()}")
        if nan_in_pred:
            idx = np.where(np.isnan(y_pred))[0]
            locs.append(f"y_pred at indices {idx[:5].tolist()}")
        raise ValueError(f"NaN values found in: {'; '.join(locs)}")
