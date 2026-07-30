"""Train-only climatology and anomaly computation.

.. warning::

   The existing ``climatenet.features.anomalies.add_monthly_climatology_and_anomalies``
   computes climatology from the **entire** dataset, including validation and test
   rows.  That leaks future information into the training features.

   This module provides functions that **fit climatology exclusively on train data**
   and then apply it to any DataFrame, so validation and test anomalies never
   influence the training distribution.

Anti-leakage guarantees
-----------------------
- ``compute_monthly_climatology`` only ever reads ``train_df``.
- ``apply_monthly_anomaly`` performs a pure left-join; it never computes
  statistics from the DataFrame it transforms.
- ``build_train_only_anomaly`` is the convenience wrapper that chains the
  two steps together.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climatenet.evaluation.hydroclimate_labels import fit_event_thresholds


# ---------------------------------------------------------------------------
# compute_monthly_climatology
# ---------------------------------------------------------------------------


def compute_monthly_climatology(
    train_df: pd.DataFrame,
    value_col: str,
    group_cols: list[str] | None = None,
    *,
    group_by_climate_zone: bool = False,
    require_all_months: bool = True,
) -> pd.DataFrame:
    """Compute monthly climatology (mean) from **training data only**.

    Parameters
    ----------
    train_df
        Training DataFrame.  Must contain ``value_col``, ``"month"``, and
        ``"region"`` columns.  If ``group_by_climate_zone=True`` it must
        also contain ``"climate_zone"``.
    value_col
        Name of the column whose monthly mean is computed (e.g.
        ``"evaporation"``).
    group_cols
        Columns to group by for the climatology computation.
        Defaults to ``["month"]``, or ``["climate_zone", "month"]`` when
        ``group_by_climate_zone=True``.
    group_by_climate_zone
        When ``True``, climatologies are computed per **(climate_zone,
        month)** instead of **(region, month)**.  This is useful for
        climate-zone transfer splits where the test climate zone must not
        influence the climatology.
    require_all_months
        Preserve the historical strict behavior by default. Split-aware
        preprocessing sets this to ``False`` and uses train-only fallbacks
        when a group or calendar month is absent.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ``[*group_cols, value_col]`` where
        ``value_col`` holds the long-term mean of ``value_col`` for each
        group.  The column is renamed to ``{value_col}_climatology``.

    Raises
    ------
    ValueError
        If any calendar month (1–12) is missing from ``train_df``, or if
        ``group_by_climate_zone=True`` but ``"climate_zone"`` is not in
        the DataFrame.
    """
    # --- validate inputs --------------------------------------------------
    if value_col not in train_df.columns:
        raise ValueError(
            f"Column '{value_col}' not found in train_df. "
            f"Available columns: {sorted(train_df.columns)}"
        )
    if "month" not in train_df.columns:
        raise ValueError("train_df must contain a 'month' column (1–12).")

    if group_cols is None:
        if group_by_climate_zone:
            if "climate_zone" not in train_df.columns:
                raise ValueError(
                    "group_by_climate_zone=True requires a 'climate_zone' column "
                    "in train_df."
                )
            group_cols = ["climate_zone", "month"]
        else:
            group_cols = ["month"]

    # --- check that every calendar month is present -----------------------
    present_months = set(train_df["month"].unique())
    expected_months = set(range(1, 13))
    missing = expected_months - present_months
    if missing and require_all_months:
        raise ValueError(
            f"train_df is missing data for calendar month(s): "
            f"{sorted(missing)}.  Climatology requires all 12 months."
        )

    # --- compute ----------------------------------------------------------
    clim_col = f"{value_col}_climatology"
    climatology = (
        train_df.groupby(group_cols, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: clim_col})
    )

    return climatology


# ---------------------------------------------------------------------------
# apply_monthly_anomaly
# ---------------------------------------------------------------------------


def apply_monthly_anomaly(
    df: pd.DataFrame,
    climatology: pd.DataFrame,
    value_col: str,
    output_col: str | None = None,
    *,
    group_by_climate_zone: bool = False,
    group_cols: list[str] | None = None,
    fallback_climatology: pd.DataFrame | None = None,
    global_fallback: float | None = None,
) -> pd.DataFrame:
    """Apply a pre-computed climatology to produce anomaly values.

    The ``climatology`` DataFrame is left-joined onto ``df`` on the group
    columns (``"month"`` or ``["climate_zone", "month"]``).  The anomaly
    is computed as ``df[value_col] - climatology[{value_col}_climatology]``.

    Parameters
    ----------
    df
        DataFrame to transform (may be train, val, or test — the
        climatology is always applied, never refit).
    climatology
        Climatology DataFrame produced by :func:`compute_monthly_climatology`.
        Must contain the group columns and ``{value_col}_climatology``.
    value_col
        The raw value column in ``df``.
    output_col
        Name for the anomaly column.  Defaults to ``{value_col}_anomaly``.
    group_by_climate_zone
        Must match the setting used when ``climatology`` was computed.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with a new ``{output_col}`` column.
    """
    if output_col is None:
        output_col = f"{value_col}_anomaly"

    clim_col = f"{value_col}_climatology"

    if clim_col not in climatology.columns:
        raise ValueError(
            f"Climatology DataFrame missing expected column '{clim_col}'. "
            f"Available columns: {sorted(climatology.columns)}"
        )

    if group_cols is None:
        if group_by_climate_zone:
            group_cols = ["climate_zone", "month"]
        else:
            group_cols = ["month"]

    for col in group_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' missing from input DataFrame. "
                f"Available columns: {sorted(df.columns)}"
            )
        if col not in climatology.columns:
            raise ValueError(
                f"Column '{col}' missing from climatology DataFrame. "
                f"Available columns: {sorted(climatology.columns)}"
            )

    result = df.copy()
    # Raw feature tables may contain legacy full-dataset climatology columns.
    # They must never participate in split-aware transformation.
    if clim_col in result.columns:
        result = result.drop(columns=[clim_col])
    result = result.merge(climatology, on=group_cols, how="left")
    if fallback_climatology is not None:
        fallback_col = f"{clim_col}__fallback"
        fallback = fallback_climatology.rename(columns={clim_col: fallback_col})
        result = result.merge(fallback, on=["month"], how="left")
        result[clim_col] = result[clim_col].fillna(result[fallback_col])
        result = result.drop(columns=[fallback_col])
    if global_fallback is not None:
        result[clim_col] = result[clim_col].fillna(float(global_fallback))
    if (
        fallback_climatology is not None or global_fallback is not None
    ) and result[clim_col].isna().any():
        missing_groups = result.loc[result[clim_col].isna(), group_cols].drop_duplicates()
        raise ValueError(
            "No train-fitted climatology or fallback for transformed rows: "
            f"{missing_groups.to_dict(orient='records')}"
        )
    result[output_col] = result[value_col] - result[clim_col]

    # Drop the intermediate climatology column so the output stays clean.
    result = result.drop(columns=[clim_col])

    return result


# ---------------------------------------------------------------------------
# build_train_only_anomaly  (convenience)
# ---------------------------------------------------------------------------


def build_train_only_anomaly(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    value_col: str,
    *,
    group_by_climate_zone: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit climatology on train, apply to train/val/test.

    Convenience wrapper that calls :func:`compute_monthly_climatology` on
    ``train_df`` and then :func:`apply_monthly_anomaly` on each of
    ``train_df``, ``val_df``, and ``test_df``.

    Parameters
    ----------
    train_df
        Training data.  Climatology is computed **exclusively** from this
        DataFrame.
    val_df
        Validation data.
    test_df
        Test data.
    value_col
        Column to compute anomalies for.
    group_by_climate_zone
        Forwarded to both functions.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_anomaly_df, val_anomaly_df, test_anomaly_df, climatology)``.
        Each returned DataFrame is a copy with an additional
        ``{value_col}_anomaly`` column.
    """
    climatology = compute_monthly_climatology(
        train_df,
        value_col=value_col,
        group_by_climate_zone=group_by_climate_zone,
    )

    anomaly_col = f"{value_col}_anomaly"

    train_out = apply_monthly_anomaly(
        train_df, climatology, value_col, output_col=anomaly_col,
        group_by_climate_zone=group_by_climate_zone,
    )
    val_out = apply_monthly_anomaly(
        val_df, climatology, value_col, output_col=anomaly_col,
        group_by_climate_zone=group_by_climate_zone,
    )
    test_out = apply_monthly_anomaly(
        test_df, climatology, value_col, output_col=anomaly_col,
        group_by_climate_zone=group_by_climate_zone,
    )

    return train_out, val_out, test_out, climatology


# ---------------------------------------------------------------------------
# Split-aware preprocessing
# ---------------------------------------------------------------------------


@dataclass
class TrainOnlyClimatePreprocessor:
    """Fit climate statistics on train rows and apply them without refitting.

    Region-month climatologies are preferred. An unseen region (as in region
    transfer) falls back to the train-only global monthly climatology, then to
    the train-only global mean when that month is absent.
    """

    anomaly_columns: dict[str, str]
    regional_climatologies: dict[str, pd.DataFrame] = field(default_factory=dict)
    monthly_fallbacks: dict[str, pd.DataFrame] = field(default_factory=dict)
    global_fallbacks: dict[str, float] = field(default_factory=dict)
    event_thresholds: dict[int, dict[str, float]] = field(default_factory=dict)
    fallback_usage: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=dict
    )
    _fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> TrainOnlyClimatePreprocessor:
        """Fit every climatology and event threshold from ``train_df`` only."""
        for value_col in self.anomaly_columns:
            self.regional_climatologies[value_col] = compute_monthly_climatology(
                train_df,
                value_col,
                group_cols=["region", "month"],
                require_all_months=False,
            )
            self.monthly_fallbacks[value_col] = compute_monthly_climatology(
                train_df,
                value_col,
                group_cols=["month"],
                require_all_months=False,
            )
            self.global_fallbacks[value_col] = float(train_df[value_col].mean())

        self._fitted = True
        transformed_train = self.transform(
            train_df, partition="train_fit", track_fallback=False
        )
        event_columns = {
            "temperature_anomaly",
            "soil_moisture_anomaly",
            "evaporation_anomaly",
            "month",
        }
        if event_columns.issubset(transformed_train.columns):
            self.event_thresholds = fit_event_thresholds(transformed_train)
        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        partition: str = "unspecified",
        track_fallback: bool = True,
    ) -> pd.DataFrame:
        """Apply frozen train climatologies and optionally track fallbacks."""
        if not self._fitted:
            raise RuntimeError("TrainOnlyClimatePreprocessor must be fit before transform")
        result = df.copy()
        for value_col, output_col in self.anomaly_columns.items():
            regional = self.regional_climatologies[value_col]
            monthly = self.monthly_fallbacks[value_col]
            row_pairs = pd.MultiIndex.from_frame(result[["region", "month"]])
            fitted_pairs = pd.MultiIndex.from_frame(regional[["region", "month"]])
            regional_missing = ~row_pairs.isin(fitted_pairs)
            monthly_available = result["month"].isin(monthly["month"])
            global_monthly_used = regional_missing & monthly_available
            global_mean_used = regional_missing & ~monthly_available
            if track_fallback:
                usage = self.fallback_usage.setdefault(partition, {}).setdefault(
                    output_col,
                    {
                        "rows": 0,
                        "region_monthly_rows": 0,
                        "global_monthly_fallback_rows": 0,
                        "global_mean_fallback_rows": 0,
                    },
                )
                usage["rows"] += int(len(result))
                usage["region_monthly_rows"] += int((~regional_missing).sum())
                usage["global_monthly_fallback_rows"] += int(
                    global_monthly_used.sum()
                )
                usage["global_mean_fallback_rows"] += int(global_mean_used.sum())
            result = apply_monthly_anomaly(
                result,
                regional,
                value_col,
                output_col=output_col,
                group_cols=["region", "month"],
                fallback_climatology=self.monthly_fallbacks[value_col],
                global_fallback=self.global_fallbacks[value_col],
            )
        return result

    def record_fallback_usage(
        self, df: pd.DataFrame, *, partition: str
    ) -> None:
        """Record which frozen fallback each row would use, without copying."""
        if not self._fitted:
            raise RuntimeError("TrainOnlyClimatePreprocessor must be fit before transform")
        for value_col, output_col in self.anomaly_columns.items():
            regional = self.regional_climatologies[value_col]
            monthly = self.monthly_fallbacks[value_col]
            row_pairs = pd.MultiIndex.from_frame(df[["region", "month"]])
            fitted_pairs = pd.MultiIndex.from_frame(
                regional[["region", "month"]]
            )
            regional_missing = ~row_pairs.isin(fitted_pairs)
            monthly_available = df["month"].isin(monthly["month"]).to_numpy()
            usage = self.fallback_usage.setdefault(partition, {}).setdefault(
                output_col,
                {
                    "rows": 0,
                    "region_monthly_rows": 0,
                    "global_monthly_fallback_rows": 0,
                    "global_mean_fallback_rows": 0,
                },
            )
            usage["rows"] += int(len(df))
            usage["region_monthly_rows"] += int((~regional_missing).sum())
            usage["global_monthly_fallback_rows"] += int(
                (regional_missing & monthly_available).sum()
            )
            usage["global_mean_fallback_rows"] += int(
                (regional_missing & ~monthly_available).sum()
            )

    def metadata(self) -> dict[str, Any]:
        """Return serialisable strategy and fitted-parameter provenance."""
        payload = {
            value_col: {
                "regional": self.regional_climatologies[value_col].to_dict(
                    orient="records"
                ),
                "monthly_fallback": self.monthly_fallbacks[value_col].to_dict(
                    orient="records"
                ),
                "global_fallback": self.global_fallbacks[value_col],
            }
            for value_col in self.anomaly_columns
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "fit_scope": "train_only",
            "climatology": "region_month_mean",
            "unseen_region_fallback": (
                "train_global_month_mean_then_train_global_mean"
            ),
            "climatology_fingerprint": fingerprint,
            "event_thresholds": self.event_thresholds,
            "event_threshold_fit_scope": "train_only",
            "fallback_usage_by_partition": self.fallback_usage,
            "validation_used_for_fit": False,
            "test_used_for_fit": False,
        }

    def save_climatology_tables(self, output_dir: str | Path) -> list[str]:
        """Save train-fitted regional/global climatologies for audit."""
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for value_col in self.anomaly_columns:
            regional_path = root / f"climatology_{value_col}_region_month.csv"
            monthly_path = root / f"climatology_{value_col}_global_month.csv"
            self.regional_climatologies[value_col].to_csv(
                regional_path, index=False
            )
            self.monthly_fallbacks[value_col].to_csv(monthly_path, index=False)
            paths.extend([regional_path.name, monthly_path.name])
        return paths


@dataclass
class TrainOnlyStandardizer:
    """Column-wise standardizer whose parameters are fitted on train only."""

    columns: list[str]
    means: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    _fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> TrainOnlyStandardizer:
        missing = [column for column in self.columns if column not in train_df.columns]
        if missing:
            raise ValueError(f"Cannot standardize missing columns: {missing}")
        for column in self.columns:
            values = train_df[column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Cannot fit standardizer: train column {column!r} "
                    "contains missing or non-finite values. Formal benchmark "
                    "does not fit an imputer."
                )
            self.means[column] = float(np.mean(values))
            scale = float(np.std(values))
            self.scales[column] = scale if np.isfinite(scale) and scale > 0 else 1.0
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("TrainOnlyStandardizer must be fit before transform")
        result = df.copy()
        for column in self.columns:
            values = result[column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Cannot transform column {column!r}: validation/test "
                    "contains missing or non-finite values; no refitting or "
                    "test-derived imputation is allowed."
                )
            result[column] = (
                (values - self.means[column]) / self.scales[column]
            ).astype(np.float32)
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "standardization": "zscore",
            "standardization_fit_scope": "train_only",
            "standardization_parameters": {
                column: {
                    "mean": self.means[column],
                    "scale": self.scales[column],
                }
                for column in self.columns
            },
        }
