"""Lightweight model registry for the mini-benchmark pipeline.

Supports registering model classes by name, listing available models,
and creating model instances with configurable hyper-parameters.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry storage
# ---------------------------------------------------------------------------

_registry: dict[str, type] = {}


def register_model(name: str, cls: type) -> None:
    """Register a model class under a name.

    Parameters
    ----------
    name
        Unique model name (e.g. ``"climatology"``).
    cls
        Model class (must support ``fit``, ``predict``, ``get_model_name``,
        ``get_params``).
    """
    _registry[name] = cls


def get_model(name: str, **kwargs) -> Any:
    """Create a model instance by name.

    Parameters
    ----------
    name
        Registered model name.
    **kwargs
        Forwarded to the model constructor.

    Returns
    -------
    Model instance.

    Raises
    ------
    ValueError
        If *name* is not registered.
    """
    if name not in _registry:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(_registry.keys())}"
        )
    return _registry[name](**kwargs)


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(_registry.keys())


def is_registered(name: str) -> bool:
    """Check whether a model name is registered."""
    return name in _registry


# ---------------------------------------------------------------------------
# Auto-register known models
# ---------------------------------------------------------------------------


def _auto_register() -> None:
    """Register the baseline models if their modules are importable."""
    try:
        from climatenet.models.baselines import (
            ClimatologyBaseline,
            LightGBMBaseline,
            PersistenceBaseline,
        )
        register_model("climatology", ClimatologyBaseline)
        register_model("persistence", PersistenceBaseline)
        register_model("lightgbm", LightGBMBaseline)
    except ImportError as e:
        logger.debug("Could not auto-register baseline models: %s", e)


_auto_register()
