"""Feature engineering pipeline."""

from __future__ import annotations

import pandas as pd

from climatenet.features.anomalies import add_monthly_climatology_and_anomalies
from climatenet.features.physical import add_physical_features


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the full ClimateNet feature table."""
    features = add_physical_features(data)
    features = add_monthly_climatology_and_anomalies(features)
    return features
