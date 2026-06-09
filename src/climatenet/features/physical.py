"""Physically informed climate feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def saturation_vapor_pressure(temperature_celsius: pd.Series) -> pd.Series:
    """Calculate saturation vapor pressure in kPa from Celsius temperature."""
    return 0.6108 * np.exp((17.27 * temperature_celsius) / (temperature_celsius + 237.3))


def add_physical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add physical predictors used by ClimateNet models."""
    features = data.copy()
    features["wind_speed"] = np.sqrt(features["u_wind"] ** 2 + features["v_wind"] ** 2)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    features["dryness_proxy"] = features["radiation"] / (features["precipitation"] + 1e-6)
    features["saturation_vapor_pressure"] = saturation_vapor_pressure(features["temperature"])
    return features
