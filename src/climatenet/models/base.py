"""Abstract base class for ClimateNet-Bench models.

All models (baselines, ML, DL) must implement this interface so they can
be used interchangeably in the benchmark pipeline.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ClimateModel(ABC):
    """Abstract base for all ClimateNet-Bench models.

    Subclasses must implement:

    - :meth:`fit`
    - :meth:`predict`
    - :meth:`get_model_name`

    Default implementations are provided for :meth:`save` and :meth:`load`
    using pickle.  Override if your model has its own serialization format.
    """

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "y_true",
        val_df: pd.DataFrame | None = None,
    ) -> ClimateModel:
        """Train the model.

        Parameters
        ----------
        train_df
            Training data with feature columns and ``target_column``.
        feature_columns
            List of column names to use as predictors.
        target_column
            Name of the column containing the ground truth (default ``"y_true"``).
        val_df
            Optional validation data for models that support early stopping.
        """
        ...

    @abstractmethod
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return predictions for the test set as a 1-D numpy array.

        Parameters
        ----------
        test_df
            Test data. Must contain the same ``feature_columns`` passed
            to :meth:`fit`.
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return a short unique model name (e.g. ``"climatology"``)."""
        ...

    def save(self, path: str | Path) -> None:
        """Persist the trained model to disk via pickle."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> ClimateModel:
        """Load a trained model from disk via pickle."""
        with Path(path).open("rb") as f:
            return pickle.load(f)
