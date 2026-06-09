"""Unified model factory for ClimateNet-Bench.

Creates models by name, with graceful handling of optional dependencies.
"""

from __future__ import annotations

import logging
from typing import Any

from climatenet.models.base import ClimateModel
from climatenet.models.climatology import ClimatologyBaseline
from climatenet.models.linear import LinearRegressionModel
from climatenet.models.persistence import PersistenceBaseline
from climatenet.models.tree_models import (
    LightGBMModel,
    RandomForestModel,
    XGBoostModel,
    _is_lightgbm_available,
    _is_xgboost_available,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry of supported model names → (class, description)
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, tuple[type[ClimateModel], str]] = {
    "climatology": (ClimatologyBaseline, "Climatological monthly mean baseline"),
    "persistence": (PersistenceBaseline, "Previous-month persistence baseline"),
    "linear_regression": (LinearRegressionModel, "Ridge-regularized linear regression"),
    "random_forest": (RandomForestModel, "scikit-learn RandomForestRegressor"),
    "xgboost": (XGBoostModel, "XGBoost gradient boosting (optional dependency)"),
    "lightgbm": (LightGBMModel, "LightGBM gradient boosting (optional dependency)"),
    "tcn": (None, "PyTorch TCN — use create_tcn_model() directly (requires 3D arrays)"),
}


def list_available_models() -> dict[str, dict[str, Any]]:
    """Return a dict of model_name → {class, description, available}."""
    result: dict[str, dict[str, Any]] = {}
    for name, (cls, desc) in _MODEL_REGISTRY.items():
        available = True
        if name == "xgboost" and not _is_xgboost_available():
            available = False
        elif name == "lightgbm" and not _is_lightgbm_available():
            available = False
        elif name == "tcn":
            try:
                import torch  # noqa: F401
            except ImportError:
                available = False
        result[name] = {
            "class": cls,
            "description": desc,
            "available": available,
        }
    return result


def create_model(
    model_name: str,
    config: dict[str, Any] | None = None,
) -> ClimateModel:
    """Create a model instance by name.

    Parameters
    ----------
    model_name
        One of: ``"climatology"``, ``"persistence"``, ``"linear_regression"``,
        ``"random_forest"``, ``"xgboost"``, ``"lightgbm"``.
    config
        Optional keyword arguments passed to the model constructor.

    Returns
    -------
    A :class:`ClimateModel` instance.

    Raises
    ------
    ValueError
        If ``model_name`` is unknown.
    ImportError
        If an optional dependency (XGBoost, LightGBM) is not installed.
    """
    if config is None:
        config = {}

    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {sorted(_MODEL_REGISTRY.keys())}"
        )

    cls, _desc = _MODEL_REGISTRY[model_name]

    # TCN requires special handling (3D array input).
    if model_name == "tcn":
        raise ValueError(
            "TCN requires 3D sequence arrays. Use create_tcn_model() directly "
            "or train via scripts/run_tcn_experiment.py."
        )

    # Pass config as kwargs to constructor.
    # Filter out non-constructor keys.
    model_kwargs = {k: v for k, v in config.items() if k not in ("type", "description")}
    return cls(**model_kwargs)


def create_tcn_model(
    num_features: int,
    channels: list[int] | None = None,
    kernel_size: int = 3,
    dropout: float = 0.2,
    **kwargs,
) -> Any:
    """Create a TCNRegressor (PyTorch) instance.

    The TCN does NOT implement ``ClimateModel`` because it requires 3-D
    sequence arrays ``(batch, sequence_length, num_features)`` rather than
    a 2-D tabular DataFrame.  Use the dedicated training scripts for TCN.
    """
    from climatenet.models.tcn import TCNRegressor

    channels = channels or [32, 32, 32]
    return TCNRegressor(
        num_features=num_features,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )
