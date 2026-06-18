"""Hydroclimate stress event label construction.

Event labels are built using **train-only percentile thresholds**, grouped
by calendar month to preserve seasonality.  The test set must never affect
threshold computation — this is the core anti-leakage guarantee.

Three event types are defined:

========  ===========================================  ================================
Event     Definition                                    Threshold source
========  ===========================================  ================================
drought   soil_moisture_anomaly < train P10[month]      Train set, by calendar month
deficit   evaporation_anomaly < train P10[month]        Train set, by calendar month
compound  temperature_anomaly > train P90[month]        Train set, by calendar month
          AND soil_moisture_anomaly < train P10[month]
========  ===========================================  ================================

Usage
-----
.. code-block:: python

    from climatenet.evaluation.hydroclimate_labels import (
        fit_event_thresholds,
        build_all_event_labels,
    )

    thresholds = fit_event_thresholds(train_df)
    labels = build_all_event_labels(test_df, thresholds)
    # labels["soil_moisture_drought"]  → boolean np.ndarray
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column names used for threshold computation.
_SM_COL = "soil_moisture_anomaly"
_EVAP_COL = "evaporation_anomaly"
_TEMP_COL = "temperature_anomaly"
_MONTH_COL = "month"

# Percentile values.
_SM_PERCENTILE = 10   # P10 for soil moisture drought
_EVAP_PERCENTILE = 10  # P10 for evaporation deficit
_TEMP_PERCENTILE = 90  # P90 for compound hot-dry (temperature component)

# Event type keys returned by build_all_event_labels.
EVENT_SOIL_MOISTURE_DROUGHT = "soil_moisture_drought"
EVENT_EVAPORATION_DEFICIT = "evaporation_deficit"
EVENT_COMPOUND_HOT_DRY = "compound_hot_dry"

ALL_EVENT_TYPES = [
    EVENT_SOIL_MOISTURE_DROUGHT,
    EVENT_EVAPORATION_DEFICIT,
    EVENT_COMPOUND_HOT_DRY,
]

# Columns required in the input DataFrame.
_REQUIRED_COLS = [_SM_COL, _EVAP_COL, _TEMP_COL, _MONTH_COL]


# ---------------------------------------------------------------------------
# Threshold fitting (train-only)
# ---------------------------------------------------------------------------


def fit_event_thresholds(train_df: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Compute train-only percentile thresholds for each calendar month
    present in the training data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.  Must contain columns:
        ``soil_moisture_anomaly``, ``evaporation_anomaly``,
        ``temperature_anomaly``, ``month``.

    Returns
    -------
    dict[int, dict[str, float]]
        Nested dict: ``thresholds[month][threshold_key] = float``.
        ``month`` covers only months present in *train_df*.
        ``threshold_key`` is one of ``"sm_p10"``, ``"evap_p10"``, ``"temp_p90"``.

    Raises
    ------
    ValueError
        If *train_df* has zero rows for any month that appears in the data.
    """
    _validate_columns(train_df, "train_df")

    months_present = sorted(train_df[_MONTH_COL].unique())
    if len(months_present) == 0:
        raise ValueError("train_df contains no rows.")

    thresholds: dict[int, dict[str, float]] = {}

    for month in months_present:
        month_data = train_df[train_df[_MONTH_COL] == month]
        if len(month_data) == 0:
            raise ValueError(
                f"Training data has no samples for calendar month {month}. "
                f"Cannot fit percentile thresholds."
            )

        thresholds[int(month)] = {
            "sm_p10": float(np.percentile(month_data[_SM_COL], _SM_PERCENTILE)),
            "evap_p10": float(np.percentile(month_data[_EVAP_COL], _EVAP_PERCENTILE)),
            "temp_p90": float(np.percentile(month_data[_TEMP_COL], _TEMP_PERCENTILE)),
        }

    return thresholds


# ---------------------------------------------------------------------------
# Label builders (apply thresholds to any DataFrame)
# ---------------------------------------------------------------------------


def build_soil_moisture_drought_label(
    df: pd.DataFrame,
    thresholds: dict[int, dict[str, float]],
) -> np.ndarray:
    """Build boolean label: soil_moisture_anomaly < train P10 by calendar month.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing ``soil_moisture_anomaly`` and ``month`` columns.
    thresholds : dict
        Output of :func:`fit_event_thresholds`.

    Returns
    -------
    np.ndarray of bool, shape ``(n_samples,)``.
    """
    _validate_columns(df, "df")
    _validate_thresholds(thresholds)

    result = np.zeros(len(df), dtype=bool)
    for month in sorted(thresholds.keys()):
        mask = df[_MONTH_COL] == month
        if mask.any():
            result[mask] = (
                df.loc[mask, _SM_COL].to_numpy()
                < thresholds[month]["sm_p10"]
            )
    return result


def build_evaporation_deficit_label(
    df: pd.DataFrame,
    thresholds: dict[int, dict[str, float]],
) -> np.ndarray:
    """Build boolean label: evaporation_anomaly < train P10 by calendar month.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing ``evaporation_anomaly`` and ``month`` columns.
    thresholds : dict
        Output of :func:`fit_event_thresholds`.

    Returns
    -------
    np.ndarray of bool, shape ``(n_samples,)``.
    """
    _validate_columns(df, "df")
    _validate_thresholds(thresholds)

    result = np.zeros(len(df), dtype=bool)
    for month in sorted(thresholds.keys()):
        mask = df[_MONTH_COL] == month
        if mask.any():
            result[mask] = (
                df.loc[mask, _EVAP_COL].to_numpy()
                < thresholds[month]["evap_p10"]
            )
    return result


def build_compound_hot_dry_label(
    df: pd.DataFrame,
    thresholds: dict[int, dict[str, float]],
) -> np.ndarray:
    """Build boolean label: temperature_anomaly > train P90 AND
    soil_moisture_anomaly < train P10, by calendar month.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing ``temperature_anomaly``, ``soil_moisture_anomaly``,
        and ``month`` columns.
    thresholds : dict
        Output of :func:`fit_event_thresholds`.

    Returns
    -------
    np.ndarray of bool, shape ``(n_samples,)``.
    """
    _validate_columns(df, "df")
    _validate_thresholds(thresholds)

    result = np.zeros(len(df), dtype=bool)
    for month in sorted(thresholds.keys()):
        mask = df[_MONTH_COL] == month
        if mask.any():
            hot = (
                df.loc[mask, _TEMP_COL].to_numpy()
                > thresholds[month]["temp_p90"]
            )
            dry = (
                df.loc[mask, _SM_COL].to_numpy()
                < thresholds[month]["sm_p10"]
            )
            result[mask] = hot & dry
    return result


def build_all_event_labels(
    df: pd.DataFrame,
    thresholds: dict[int, dict[str, float]],
) -> dict[str, np.ndarray]:
    """Build all three event labels from pre-fitted thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the required columns.
    thresholds : dict
        Output of :func:`fit_event_thresholds`.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``"soil_moisture_drought"``, ``"evaporation_deficit"``,
        ``"compound_hot_dry"``.  Values are boolean arrays.
    """
    return {
        EVENT_SOIL_MOISTURE_DROUGHT: build_soil_moisture_drought_label(df, thresholds),
        EVENT_EVAPORATION_DEFICIT: build_evaporation_deficit_label(df, thresholds),
        EVENT_COMPOUND_HOT_DRY: build_compound_hot_dry_label(df, thresholds),
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_columns(df: pd.DataFrame, name: str) -> None:
    """Raise ValueError if required columns are missing."""
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. "
            f"Required: {_REQUIRED_COLS}"
        )


def _validate_thresholds(thresholds: dict[int, dict[str, float]]) -> None:
    """Raise ValueError if thresholds dict is malformed.

    Accepts any non-empty set of months; each month must contain
    ``sm_p10``, ``evap_p10``, and ``temp_p90`` keys.
    """
    if not thresholds:
        raise ValueError("Thresholds dict is empty.")
    expected_keys = {"sm_p10", "evap_p10", "temp_p90"}
    for month in sorted(thresholds.keys()):
        if not isinstance(month, int) or month < 1 or month > 12:
            raise ValueError(f"Invalid month key: {month}")
        actual = set(thresholds[month].keys())
        if not expected_keys.issubset(actual):
            raise ValueError(
                f"Thresholds for month {month} missing keys: "
                f"{expected_keys - actual}"
            )
