"""Lightweight baseline models for the mini-benchmark pipeline.

Implements three baseline models with a uniform interface:

1. **ClimatologyBaseline** — predicts train-set monthly mean (or zero for anomalies).
2. **PersistenceBaseline** — predicts using the latest lag feature.
3. **LightGBMBaseline** — gradient-boosted tree model (optional dependency).

Each model supports: ``fit()``, ``predict()``, ``get_params()``, ``get_model_name()``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. ClimatologyBaseline
# ==============================================================================


class ClimatologyBaseline:
    """Predicts the train-set monthly climatological mean.

    For **anomaly** target columns (e.g. ``evaporation_anomaly``), the
    default prediction is **zero** because anomalies are defined as
    deviations from the climatological mean.

    For **raw** target columns, the model learns the monthly mean from
    the training data (grouped by ``month``, and optionally ``region``)
    and predicts that mean for each test sample.

    Parameters
    ----------
    target_col : str, optional
        Target column name.  Used to configure ``get_params()`` metadata.
    predict_zero : bool, default=True
        When True, always predict 0 (suitable for anomaly targets).
        When False, learn and predict the monthly mean.
    """

    def __init__(
        self,
        target_col: str | None = None,
        predict_zero: bool = True,
    ) -> None:
        self.target_col = target_col or "y_true"
        self.predict_zero = predict_zero
        self._climatology: pd.DataFrame | None = None
        self._global_mean: float = 0.0

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        target_col: str | None = None,
        feature_cols: list[str] | None = None,
    ) -> ClimatologyBaseline:
        """Compute climatological means from the training set only.

        Parameters
        ----------
        train_df
            Training data.  Must contain ``target_col`` and ``"month"``.
        val_df
            Ignored — climatology never uses validation data.
        target_col
            Column name for the target variable.
        feature_cols
            Ignored — climatology does not use features.
        """
        if target_col is not None:
            self.target_col = target_col

        if self.target_col not in train_df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in train_df. "
                f"Available columns: {sorted(train_df.columns)}"
            )
        if "month" not in train_df.columns:
            raise ValueError("train_df must contain a 'month' column (1–12).")

        if not self.predict_zero:
            # Learn monthly means.
            group_cols = ["month"]
            if "region" in train_df.columns:
                group_cols = ["region", "month"]

            self._climatology = (
                train_df.groupby(group_cols)[self.target_col]
                .mean()
                .reset_index()
                .rename(columns={self.target_col: "clim_mean"})
            )
            self._global_mean = float(train_df[self.target_col].mean())

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return predictions.

        When ``predict_zero=True``, returns a zero array.
        Otherwise, looks up monthly means (with fallback to global mean).
        """
        n = len(test_df)

        if self.predict_zero:
            return np.zeros(n, dtype=np.float64)

        if self._climatology is None:
            raise RuntimeError("ClimatologyBaseline must be fit before predict.")

        if "region" in test_df.columns and "region" in self._climatology.columns:
            preds = test_df[["region", "month"]].merge(
                self._climatology, on=["region", "month"], how="left"
            )["clim_mean"]
        else:
            preds = test_df[["month"]].merge(
                self._climatology, on="month", how="left"
            )["clim_mean"]

        nan_mask = preds.isna()
        if nan_mask.any():
            preds.loc[nan_mask] = self._global_mean

        return preds.to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return "climatology"

    def get_params(self) -> dict[str, Any]:
        return {
            "model_name": self.get_model_name(),
            "target_col": self.target_col,
            "predict_zero": self.predict_zero,
            "n_climatology_rows": (
                len(self._climatology) if self._climatology is not None else 0
            ),
        }


# ==============================================================================
# 2. PersistenceBaseline
# ==============================================================================


class PersistenceBaseline:
    """Predicts the next target using the latest available lag feature.

    By default, reads ``{target_col}_lag1`` from the test DataFrame.
    The lag column name can be configured via the ``lag_col`` parameter.

    If the required lag column is missing, ``fit()`` raises a clear
    ``ValueError``.

    Parameters
    ----------
    target_col : str, optional
        Target column name.  Used to derive the default lag column name.
    lag_col : str, optional
        Explicit lag column name override.
    """

    def __init__(
        self,
        target_col: str | None = None,
        lag_col: str | None = None,
    ) -> None:
        self.target_col = target_col or "y_true"
        self._lag_col = lag_col or f"{self.target_col}_lag1"

    # ------------------------------------------------------------------
    # fit (no-op validation)
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        target_col: str | None = None,
        feature_cols: list[str] | None = None,
    ) -> PersistenceBaseline:
        """Validate that the lag column exists in the training data.

        No actual training is performed.
        """
        if target_col is not None:
            self.target_col = target_col
            if self._lag_col == f"y_true_lag1":  # only update if default was used
                self._lag_col = f"{target_col}_lag1"

        if self._lag_col not in train_df.columns:
            raise ValueError(
                f"PersistenceBaseline requires lag column '{self._lag_col}' "
                f"in the training data.  Available columns: "
                f"{sorted(train_df.columns)}.  "
                f"Ensure your dataset includes lag features."
            )
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return the lag-column values as predictions."""
        if self._lag_col not in test_df.columns:
            raise ValueError(
                f"PersistenceBaseline requires column '{self._lag_col}' "
                f"in the test data."
            )
        return test_df[self._lag_col].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return "persistence"

    def get_params(self) -> dict[str, Any]:
        return {
            "model_name": self.get_model_name(),
            "target_col": self.target_col,
            "lag_col": self._lag_col,
        }


# ==============================================================================
# 3. LightGBMBaseline
# ==============================================================================


class LightGBMBaseline:
    """LightGBM gradient boosting regressor.

    Uses small, CI-friendly default hyperparameters suitable for smoke
    tests and mini-benchmark runs.

    If LightGBM is not installed, construction raises ``ImportError``.
    Tests should use ``pytest.importorskip("lightgbm")`` to skip gracefully.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds (default 50 — small for CI).
    learning_rate : float
        Step size shrinkage (default 0.05).
    num_leaves : int
        Maximum tree leaves (default 31).
    random_state : int
        Random seed (default 42).
    **kwargs
        Extra keyword arguments forwarded to ``LGBMRegressor``.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )
        self._lgb = lgb
        self._model: Any = None
        self._params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "random_state": random_state,
            **kwargs,
        }
        self._feature_cols: list[str] = []
        self.target_col: str = "y_true"

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        target_col: str | None = None,
        feature_cols: list[str] | None = None,
    ) -> LightGBMBaseline:
        """Train the LightGBM model.

        Parameters
        ----------
        train_df
            Training data.
        val_df
            Optional validation data (passed as ``eval_set``).
        target_col
            Target column name.
        feature_cols
            Feature column names.
        """
        if target_col is not None:
            self.target_col = target_col
        if feature_cols is not None:
            self._feature_cols = list(feature_cols)

        if not self._feature_cols:
            raise ValueError(
                "feature_cols must be provided and non-empty for LightGBMBaseline."
            )

        for col in self._feature_cols:
            if col not in train_df.columns:
                raise ValueError(
                    f"Feature column '{col}' not found in train_df."
                )
        if self.target_col not in train_df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in train_df."
            )

        X_train = train_df[self._feature_cols].to_numpy(dtype=np.float64)
        y_train = train_df[self.target_col].to_numpy(dtype=np.float64)

        eval_set = None
        if val_df is not None and not val_df.empty:
            X_val = val_df[self._feature_cols].to_numpy(dtype=np.float64)
            y_val = val_df[self.target_col].to_numpy(dtype=np.float64)
            eval_set = [(X_val, y_val)]

        self._model = self._lgb.LGBMRegressor(
            n_estimators=self._params["n_estimators"],
            learning_rate=self._params["learning_rate"],
            num_leaves=self._params["num_leaves"],
            random_state=self._params["random_state"],
            verbose=-1,
            **{k: v for k, v in self._params.items()
               if k not in ("n_estimators", "learning_rate", "num_leaves", "random_state")},
        )
        self._model.fit(X_train, y_train, eval_set=eval_set)
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return LightGBM predictions."""
        if self._model is None:
            raise RuntimeError("LightGBMBaseline must be fit before predict.")
        X = test_df[self._feature_cols].to_numpy(dtype=np.float64)
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return "lightgbm"

    def get_params(self) -> dict[str, Any]:
        return {
            "model_name": self.get_model_name(),
            "n_estimators": self._params["n_estimators"],
            "learning_rate": self._params["learning_rate"],
            "num_leaves": self._params["num_leaves"],
            "random_state": self._params["random_state"],
            "feature_cols": self._feature_cols,
            "target_col": self.target_col,
        }
