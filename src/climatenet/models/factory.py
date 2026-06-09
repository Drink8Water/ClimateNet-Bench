"""Model factory for configuration-driven experiments."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def build_models(model_config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Build enabled models from YAML configuration."""
    configured_models = model_config.get("models", {})
    models: dict[str, Any] = {}

    if configured_models.get("linear_regression", {}).get("enabled", False):
        models["linear_regression"] = LinearRegression()

    random_forest_config = configured_models.get("random_forest", {})
    if random_forest_config.get("enabled", False):
        params = dict(random_forest_config.get("params", {}))
        params.setdefault("random_state", seed)
        models["random_forest"] = RandomForestRegressor(**params)

    xgboost_config = configured_models.get("xgboost", {})
    if xgboost_config.get("enabled", False):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is enabled but not installed.") from exc
        params = dict(xgboost_config.get("params", {}))
        params.setdefault("random_state", seed)
        models["xgboost"] = XGBRegressor(**params)

    lightgbm_config = configured_models.get("lightgbm", {})
    if lightgbm_config.get("enabled", False):
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            if lightgbm_config.get("skip_if_missing", True):
                print("LightGBM is enabled but not installed; skipping.")
            else:
                raise
        else:
            params = dict(lightgbm_config.get("params", {}))
            params.setdefault("random_state", seed)
            models["lightgbm"] = LGBMRegressor(**params)

    if not models:
        raise ValueError("No models are enabled in model_config.yaml")
    return models
