"""Monthly climatology and anomaly features."""

from __future__ import annotations

import pandas as pd

DEFAULT_ANOMALY_COLUMNS = ["temperature", "precipitation", "radiation", "soil_moisture", "evaporation"]


def add_monthly_climatology_and_anomalies(
    data: pd.DataFrame,
    anomaly_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Add monthly climatology and anomaly columns by region and month."""
    features = data.copy()
    columns = anomaly_columns or DEFAULT_ANOMALY_COLUMNS
    group_keys = ["region", "month"]

    for column in columns:
        climatology_column = f"{column}_climatology"
        anomaly_column = f"{column}_anomaly"
        features[climatology_column] = features.groupby(group_keys)[column].transform("mean")
        features[anomaly_column] = features[column] - features[climatology_column]

    return features
