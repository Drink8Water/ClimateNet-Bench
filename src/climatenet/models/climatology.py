"""Climatology baseline model.

Predicts the mean target value from the training set, optionally grouped
by region and month.  This is the simplest possible forecast and serves
as the "no-skill" reference point.

Two variants are supported:

- **global_monthly** — mean target per calendar month across all regions.
- **region_monthly** — mean target per (region, calendar month).

The region-monthly variant is the default because it is a stronger
baseline in the presence of regional climate differences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from climatenet.models.base import ClimateModel


class ClimatologyBaseline(ClimateModel):
    """Climatological-mean baseline.

    Parameters
    ----------
    variant
        ``"region_monthly"`` (default) or ``"global_monthly"``.
    """

    def __init__(self, variant: str = "region_monthly") -> None:
        if variant not in ("region_monthly", "global_monthly"):
            raise ValueError(
                f"variant must be 'region_monthly' or 'global_monthly', got '{variant}'"
            )
        self.variant = variant
        self._climatology: pd.DataFrame | None = None
        self._global_fallback: float | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str] | None = None,  # ignored — no features needed
        target_column: str = "y_true",
        val_df: pd.DataFrame | None = None,
    ) -> ClimatologyBaseline:
        """Compute climatological means from the training set only.

        No test or validation data may be used.
        """
        required_cols = [target_column, "target_month"]
        if self.variant == "region_monthly" and "region" in train_df.columns:
            required_cols.append("region")
        missing = [c for c in required_cols if c not in train_df.columns]
        if missing:
            raise ValueError(f"ClimatologyBaseline.fit missing columns: {missing}")

        if self.variant == "region_monthly" and "region" in train_df.columns:
            self._climatology = (
                train_df.groupby(["region", "target_month"])[target_column]
                .mean()
                .reset_index()
                .rename(columns={target_column: "clim_mean"})
            )
        else:
            self._climatology = (
                train_df.groupby("target_month")[target_column]
                .mean()
                .reset_index()
                .rename(columns={target_column: "clim_mean"})
            )

        self._global_fallback = float(train_df[target_column].mean())
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return climatological predictions.

        If a (region, month) pair is missing from the training climatology,
        falls back to the global mean for that month, and ultimately to the
        overall global mean.
        """
        if self._climatology is None:
            raise RuntimeError("ClimatologyBaseline must be fit before predict.")

        # Build a lookup from the test data
        if self.variant == "region_monthly" and "region" in test_df.columns:
            predictions = test_df[["region", "target_month"]].merge(
                self._climatology,
                on=["region", "target_month"],
                how="left",
            )["clim_mean"]
        else:
            predictions = test_df[["target_month"]].merge(
                self._climatology,
                on="target_month",
                how="left",
            )["clim_mean"]

        # Fallback for any NaN (unseen months / regions in training)
        nan_mask = predictions.isna()
        if nan_mask.any():
            predictions.loc[nan_mask] = self._global_fallback or 0.0

        return predictions.to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return f"climatology_{self.variant}"
