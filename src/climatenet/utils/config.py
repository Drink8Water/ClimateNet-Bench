"""YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from climatenet.utils.paths import resolve_project_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    resolved_path = resolve_project_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {resolved_path}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as YAML."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def load_experiment_configs(
    data_config_path: str | Path,
    model_config_path: str | Path,
    experiment_config_path: str | Path,
) -> dict[str, Any]:
    """Load and combine the three ClimateNet config files."""
    return {
        "data": load_yaml(data_config_path),
        "models": load_yaml(model_config_path),
        "experiment": load_yaml(experiment_config_path),
    }
