"""Legacy full-table monthly climatology and anomaly features.

This module is retained for exploratory analysis and backwards compatibility.
It fits statistics on every row passed to it and therefore MUST NOT be used by
the formal benchmark runner, where validation/test rows would leak into the
climatology. Formal experiments use
``climatenet.preprocessing.climatology.TrainOnlyClimatePreprocessor``.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_ANOMALY_COLUMNS = ["temperature", "precipitation", "radiation", "soil_moisture", "evaporation"]


def add_monthly_climatology_and_anomalies(
    data: pd.DataFrame,
    anomaly_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Add full-table monthly anomalies for exploratory use only.

    .. warning::
       This function is not split-aware and is unsuitable for formal
       benchmark evaluation.
    """
    features = data.copy()
    columns = anomaly_columns or DEFAULT_ANOMALY_COLUMNS
    group_keys = ["region", "month"]

    for column in columns:
        climatology_column = f"{column}_climatology"
        anomaly_column = f"{column}_anomaly"
        features[climatology_column] = features.groupby(group_keys)[column].transform("mean")
        features[anomaly_column] = features[column] - features[climatology_column]

    return features
