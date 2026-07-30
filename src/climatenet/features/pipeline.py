"""Feature engineering pipeline."""

from __future__ import annotations

import pandas as pd

from climatenet.features.anomalies import add_monthly_climatology_and_anomalies
from climatenet.features.physical import add_physical_features


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create an exploratory full-table feature table.

    This compatibility pipeline computes full-table anomalies and must not be
    used by the formal benchmark runner. The runner starts from raw physical
    features and fits split-specific preprocessing on train only.
    """
    features = add_physical_features(data)
    features = add_monthly_climatology_and_anomalies(features)
    return features
