"""Mini-leaderboard for the benchmark pipeline.

Stores and ranks model evaluation results by RMSE (ascending).
Supports regression and event detection metrics.

Leaderboard columns
-------------------
- **rank** — integer rank (1 = best RMSE).
- **model** — model name.
- **split** — split name (e.g. ``"temporal_holdout"``).
- **rmse** — Root Mean Squared Error.
- **mae** — Mean Absolute Error.
- **acc** — R² (coefficient of determination).
- **bias** — mean signed error (y_pred - y_true).
- **skill_score** — optional skill score (0 = no-skill, 1 = perfect).
- **pod** — Probability of Detection (NaN when no event metrics).
- **far** — False Alarm Ratio.
- **csi** — Critical Success Index.
- **intensity_bias** — intensity ratio over event points.
- **n_samples** — number of test samples.
- **n_events** — number of observed event samples.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Leaderboard column specification
# ---------------------------------------------------------------------------

LEADERBOARD_COLUMNS = [
    "rank",
    "model",
    "split",
    "rmse",
    "mae",
    "acc",
    "bias",
    "skill_score",
    "pod",
    "far",
    "csi",
    "intensity_bias",
    "n_samples",
    "n_events",
]

# ---------------------------------------------------------------------------
# Default precision for float columns
# ---------------------------------------------------------------------------

_FLOAT_PRECISION = 6


def _safe_round(value: float | None, precision: int = _FLOAT_PRECISION) -> float | None:
    """Round a float, returning None if NaN."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), precision)


# ---------------------------------------------------------------------------
# update_leaderboard
# ---------------------------------------------------------------------------


def update_leaderboard(
    metrics: dict[str, Any],
    model_name: str,
    split_name: str,
    output_path: str | Path,
) -> pd.DataFrame:
    """Add (or replace) a row in the leaderboard CSV for ``(model_name, split_name)``.

    Parameters
    ----------
    metrics
        Dict returned by :func:`~climatenet.evaluation.runner.evaluate_model_on_split`
        ``metrics_overall``.
    model_name
        Model name (e.g. ``"climatology"``).
    split_name
        Split name (e.g. ``"temporal_holdout"``).
    output_path
        Path to the leaderboard CSV.  Created if it does not exist.

    Returns
    -------
    pd.DataFrame
        The full leaderboard sorted by RMSE ascending.
    """
    output_path = Path(output_path)

    # Load existing
    existing = load_leaderboard(output_path)

    # Build new row
    y_true_arr = metrics.get("y_true")
    y_pred_arr = metrics.get("y_pred")

    if y_true_arr is not None and y_pred_arr is not None:
        bias_val = float(np.mean(np.asarray(y_pred_arr) - np.asarray(y_true_arr)))
    else:
        bias_val = None

    row = {
        "model": model_name,
        "split": split_name,
        "rmse": _safe_round(metrics.get("rmse")),
        "mae": _safe_round(metrics.get("mae")),
        "acc": _safe_round(metrics.get("r2")),
        "bias": _safe_round(bias_val) if bias_val is not None else None,
        "skill_score": _safe_round(metrics.get("skill_score")),
        "pod": _safe_round(_pick_event_metric(metrics, "pod")),
        "far": _safe_round(_pick_event_metric(metrics, "far")),
        "csi": _safe_round(_pick_event_metric(metrics, "csi")),
        "intensity_bias": _safe_round(_pick_event_metric(metrics, "intensity_bias")),
        "n_samples": metrics.get("n_samples"),
        "n_events": _pick_event_metric(metrics, "n_events"),
    }

    # Upsert: remove existing row for (model, split), then append
    if not existing.empty:
        existing = existing[
            ~((existing["model"] == model_name) & (existing["split"] == split_name))
        ]

    new_row_df = pd.DataFrame([row])
    leaderboard = pd.concat([existing, new_row_df], ignore_index=True)

    # Rank by RMSE ascending
    leaderboard = leaderboard.sort_values("rmse", ascending=True, na_position="last")
    leaderboard["rank"] = range(1, len(leaderboard) + 1)

    # Enforce column order
    for col in LEADERBOARD_COLUMNS:
        if col not in leaderboard.columns:
            leaderboard[col] = None
    leaderboard = leaderboard[LEADERBOARD_COLUMNS]

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(output_path, index=False)

    return leaderboard


# ---------------------------------------------------------------------------
# load_leaderboard
# ---------------------------------------------------------------------------


def load_leaderboard(path: str | Path) -> pd.DataFrame:
    """Load a leaderboard CSV, returning an empty DataFrame with correct
    columns if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    df = pd.read_csv(path)
    # Ensure all expected columns exist
    for col in LEADERBOARD_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_event_metric(metrics: dict[str, Any], key: str) -> float | None:
    """Extract the first event metric matching ``{key}_*`` from metrics."""
    for k, v in metrics.items():
        if k.startswith(f"{key}_"):
            if isinstance(v, float) and not math.isnan(v):
                return v
            if isinstance(v, (int, np.integer)):
                return float(v)
    return None
