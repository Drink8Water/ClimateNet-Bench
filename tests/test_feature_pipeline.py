"""Basic tests for the package feature pipeline."""

from __future__ import annotations

import pandas as pd

from climatenet.features.pipeline import build_features


def test_build_features_adds_expected_columns() -> None:
    """The feature pipeline should add physical and anomaly columns."""
    data = pd.DataFrame(
        {
            "region": ["Sahara", "Sahara"],
            "year": [2020, 2021],
            "month": [1, 1],
            "latitude": [20.0, 20.0],
            "longitude": [10.0, 10.0],
            "temperature": [20.0, 22.0],
            "precipitation": [1.0, 2.0],
            "radiation": [200.0, 220.0],
            "soil_moisture": [0.1, 0.2],
            "u_wind": [3.0, 4.0],
            "v_wind": [4.0, 3.0],
            "evaporation": [5.0, 7.0],
        }
    )

    features = build_features(data)

    assert "wind_speed" in features.columns
    assert "saturation_vapor_pressure" in features.columns
    assert "evaporation_anomaly" in features.columns
    assert features["wind_speed"].iloc[0] == 5.0
