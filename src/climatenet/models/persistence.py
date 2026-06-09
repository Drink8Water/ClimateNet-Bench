"""Persistence baseline model.

Predicts that next month's evaporation anomaly equals the current month's
anomaly:  ŷ_t = y_{t-1}.

This baseline is often surprisingly strong in forecasting tasks because
month-to-month climate anomalies are highly autocorrelated.  It sets a
higher bar than climatology — any ML model must beat persistence to
demonstrate added value.

The model reads ``evaporation_anomaly_lag_1`` from the forecasting
samples table.  This column is automatically included when building
the dataset with ``build_forecasting_samples``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from climatenet.models.base import ClimateModel


class PersistenceBaseline(ClimateModel):
    """Persistence forecast: ŷ_t = y_{t-1}.

    Does not require training.  ``fit()`` is a no-op that validates the
    lag column exists.
    """

    def __init__(self) -> None:
        self._lag_column: str = "evaporation_anomaly_lag_1"

    # ------------------------------------------------------------------
    # fit (no-op validation)
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str] | None = None,  # ignored
        target_column: str = "y_true",
        val_df: pd.DataFrame | None = None,
    ) -> PersistenceBaseline:
        """Validate that the required lag column exists in the training data.

        No actual training is performed — this is a zero-parameter model.
        """
        if self._lag_column not in train_df.columns:
            raise ValueError(
                f"PersistenceBaseline requires column '{self._lag_column}' "
                f"in the data.  Rebuild your forecasting dataset with "
                f"a recent version of build_forecasting_samples."
            )
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return y_{t-1} as the prediction for y_t.

        If the lag column has NaN values (e.g. at the start of a time
        series), those samples will have NaN predictions.
        """
        if self._lag_column not in test_df.columns:
            raise ValueError(
                f"PersistenceBaseline requires column '{self._lag_column}' "
                f"in the test data."
            )
        return test_df[self._lag_column].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return "persistence"

    @property
    def lag_column(self) -> str:
        """The column name used for persistence prediction."""
        return self._lag_column
