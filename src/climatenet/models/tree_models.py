"""Tree-based ensemble models for ClimateNet-Bench.

Provides wrappers for scikit-learn RandomForest, XGBoost, and LightGBM
with a uniform ``ClimateModel`` interface.

Optional dependencies (XGBoost, LightGBM) are imported lazily and fail
with a clear error message when not installed.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from climatenet.models.base import ClimateModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


class RandomForestModel(ClimateModel):
    """scikit-learn RandomForestRegressor wrapper.

    Parameters
    ----------
    n_estimators
        Number of trees (default 250).
    min_samples_leaf
        Minimum samples per leaf (default 3).
    n_jobs
        Parallel workers (default -1 = all cores).
    random_state
        Random seed (default 42).
    **kwargs
        Extra keyword arguments forwarded to ``RandomForestRegressor``.
    """

    def __init__(
        self,
        n_estimators: int = 250,
        min_samples_leaf: int = 3,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        from sklearn.ensemble import RandomForestRegressor

        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )
        self._feature_columns: list[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "y_true",
        val_df: pd.DataFrame | None = None,
    ) -> RandomForestModel:
        self._feature_columns = list(feature_columns)
        X = train_df[self._feature_columns].to_numpy(dtype=np.float64)
        y = train_df[target_column].to_numpy(dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._model.fit(X, y)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = test_df[self._feature_columns].to_numpy(dtype=np.float64)
        return self._model.predict(X)

    def get_model_name(self) -> str:
        return "random_forest"

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def _is_xgboost_available() -> bool:
    """Check whether xgboost is importable (including native library)."""
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


class XGBoostModel(ClimateModel):
    """XGBoost regressor wrapper.

    Gracefully raises ``ImportError`` if xgboost is not installed.

    Parameters
    ----------
    n_estimators
        Number of boosting rounds (default 300).
    learning_rate
        Step size shrinkage (default 0.05).
    max_depth
        Maximum tree depth (default 4).
    subsample
        Row subsample ratio (default 0.9).
    colsample_bytree
        Column subsample ratio per tree (default 0.9).
    random_state
        Random seed (default 42).
    **kwargs
        Extra keyword arguments forwarded to ``XGBRegressor``.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        if not _is_xgboost_available():
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )
        from xgboost import XGBRegressor

        self._model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=kwargs.pop("n_jobs", 1),
            verbosity=0,
            **kwargs,
        )
        self._feature_columns: list[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "y_true",
        val_df: pd.DataFrame | None = None,
    ) -> XGBoostModel:
        self._feature_columns = list(feature_columns)
        X = train_df[self._feature_columns].to_numpy(dtype=np.float64)
        y = train_df[target_column].to_numpy(dtype=np.float64)

        eval_set = None
        if val_df is not None:
            X_val = val_df[self._feature_columns].to_numpy(dtype=np.float64)
            y_val = val_df[target_column].to_numpy(dtype=np.float64)
            eval_set = [(X_val, y_val)]

        self._model.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = test_df[self._feature_columns].to_numpy(dtype=np.float64)
        return self._model.predict(X)

    def get_model_name(self) -> str:
        return "xgboost"

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


def _is_lightgbm_available() -> bool:
    """Check whether lightgbm is importable (including native library)."""
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


class LightGBMModel(ClimateModel):
    """LightGBM regressor wrapper.

    Gracefully raises ``ImportError`` if lightgbm is not installed.

    Parameters
    ----------
    n_estimators
        Number of boosting rounds (default 300).
    learning_rate
        Step size shrinkage (default 0.05).
    random_state
        Random seed (default 42).
    **kwargs
        Extra keyword arguments forwarded to ``LGBMRegressor``.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        if not _is_lightgbm_available():
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )
        from lightgbm import LGBMRegressor

        self._model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=-1,
            **kwargs,
        )
        self._feature_columns: list[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "y_true",
        val_df: pd.DataFrame | None = None,
    ) -> LightGBMModel:
        self._feature_columns = list(feature_columns)
        X = train_df[self._feature_columns].to_numpy(dtype=np.float64)
        y = train_df[target_column].to_numpy(dtype=np.float64)

        eval_set = None
        eval_names = None
        if val_df is not None:
            X_val = val_df[self._feature_columns].to_numpy(dtype=np.float64)
            y_val = val_df[target_column].to_numpy(dtype=np.float64)
            eval_set = [(X_val, y_val)]
            eval_names = ["val"]

        self._model.fit(
            X, y,
            eval_set=eval_set,
            eval_names=eval_names,
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = test_df[self._feature_columns].to_numpy(dtype=np.float64)
        return self._model.predict(X)

    def get_model_name(self) -> str:
        return "lightgbm"

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_
