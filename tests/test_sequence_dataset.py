"""Tests for sequence dataset construction."""

from __future__ import annotations

import pandas as pd

from climatenet.data.sequence_dataset import build_sequence_arrays


def test_build_sequence_arrays_next_month_target() -> None:
    """Sliding windows should target the next month after the input window."""
    rows = []
    for month in range(1, 9):
        rows.append(
            {
                "region": "Sahara",
                "year": 2020,
                "month": month,
                "latitude": 20.0,
                "longitude": 10.0,
                "temperature_anomaly": float(month),
                "precipitation_anomaly": float(month),
                "radiation_anomaly": float(month),
                "soil_moisture_anomaly": float(month),
                "wind_speed": float(month),
                "dryness_proxy": float(month),
                "saturation_vapor_pressure": float(month),
                "evaporation_anomaly": float(month * 10),
            }
        )

    data = pd.DataFrame(rows)
    features = [
        "temperature_anomaly",
        "precipitation_anomaly",
        "radiation_anomaly",
        "soil_moisture_anomaly",
        "wind_speed",
        "dryness_proxy",
        "saturation_vapor_pressure",
    ]
    sequences, targets, metadata = build_sequence_arrays(
        data,
        feature_columns=features,
        target_column="evaporation_anomaly",
        sequence_length=6,
    )

    assert sequences.shape == (2, 6, 7)
    assert targets.tolist() == [70.0, 80.0]
    assert metadata[0].month == 7
