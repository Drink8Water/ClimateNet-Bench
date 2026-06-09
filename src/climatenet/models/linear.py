"""Linear regression model (ridge-regularised).

Wraps scikit-learn's ``Ridge`` for a stable linear baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from climatenet.models.base import ClimateModel


class LinearRegressionModel(ClimateModel):
    """Ridge-regressed linear model.

    Parameters
    ----------
    alpha
        L2 regularisation strength (default 1.0).
    **kwargs
        Extra keyword arguments forwarded to ``sklearn.linear_model.Ridge``.
    """

    def __init__(self, alpha: float = 1.0, **kwargs) -> None:
        self.alpha = alpha
        self._model = Ridge(alpha=alpha, random_state=kwargs.pop("random_state", 42), **kwargs)
        self._feature_columns: list[str] = []

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "y_true",
        val_df: pd.DataFrame | None = None,
    ) -> LinearRegressionModel:
        self._feature_columns = list(feature_columns)
        X = train_df[self._feature_columns].to_numpy(dtype=np.float64)
        y = train_df[target_column].to_numpy(dtype=np.float64)
        self._model.fit(X, y)
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = test_df[self._feature_columns].to_numpy(dtype=np.float64)
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return "linear_regression"

    @property
    def coefficients(self) -> np.ndarray:
        """Return the fitted linear coefficients."""
        return self._model.coef_

    @property
    def intercept(self) -> float:
        """Return the fitted intercept."""
        return float(self._model.intercept_)
