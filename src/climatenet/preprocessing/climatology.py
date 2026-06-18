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

import pandas as pd


# ---------------------------------------------------------------------------
# compute_monthly_climatology
# ---------------------------------------------------------------------------


def compute_monthly_climatology(
    train_df: pd.DataFrame,
    value_col: str,
    group_cols: list[str] | None = None,
    *,
    group_by_climate_zone: bool = False,
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
    if missing:
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

    group_cols: list[str]
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
    result = result.merge(climatology, on=group_cols, how="left")
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
