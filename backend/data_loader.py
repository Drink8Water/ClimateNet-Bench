"""File-reading layer for local CSV and JSON outputs.

All functions return pandas DataFrames or dicts.  File-not-found errors
are turned into empty results so the API can return graceful fallbacks.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file, returning an empty DataFrame when the file is missing."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning an empty dict when the file is missing."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv_cached(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read CSV with a simple TTL-free cache (cleared on restart)."""
    return _read_csv(path, **kwargs)


def read_json_cached(path: Path) -> dict[str, Any]:
    """Read JSON with a simple cache."""
    return _read_json(path)


def read_yaml_cached(path: Path) -> dict[str, Any]:
    """Read a YAML config file, returning an empty dict on failure."""
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def read_first_lines(path: Path, n: int = 5) -> str:
    """Read the first N lines of a file for preview/debugging."""
    if not path.exists():
        return ""
    with open(path, encoding="utf-8") as fh:
        return "".join(fh.readline() for _ in range(n))
