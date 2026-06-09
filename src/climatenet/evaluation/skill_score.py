"""Skill scores for ClimateNet-Bench.

A **skill score** quantifies how much a model improves over a reference
baseline.  The standard formulation used in meteorology and climate
science is::

    skill = 1 − RMSE_model / RMSE_baseline

- ``skill > 0`` → model is better than the baseline.
- ``skill = 0`` → model ties the baseline.
- ``skill < 0`` → model is worse than the baseline.
- ``skill = 1`` → perfect model (RMSE = 0).

Reference baselines
-------------------
- **climatology** — long-term monthly mean per (region, month).
- **persistence** — previous-month value (ŷ_t = y_{t-1}).

Both are defined in ``climatenet.models``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """Compute skill score of a model relative to a baseline.

    Parameters
    ----------
    model_rmse
        RMSE of the model being evaluated.
    baseline_rmse
        RMSE of the reference baseline.

    Returns
    -------
    Skill score in ``(−∞, 1]``.

    Edge cases
    ----------
    - ``baseline_rmse == 0`` → returns ``np.nan`` (baseline is perfect;
      skill is undefined).
    - ``model_rmse < 0`` → raises ``ValueError``.
    - ``baseline_rmse < 0`` → raises ``ValueError``.
    """
    if model_rmse < 0:
        raise ValueError(f"model_rmse must be ≥ 0, got {model_rmse}")
    if baseline_rmse < 0:
        raise ValueError(f"baseline_rmse must be ≥ 0, got {baseline_rmse}")

    if baseline_rmse == 0.0:
        # Baseline is perfect — skill is undefined.
        # Return NaN rather than ±inf so downstream aggregations can skip it.
        return float("nan")

    return float(1.0 - model_rmse / baseline_rmse)


def compute_skill_scores(
    results_df: "pd.DataFrame",
    baseline_names: list[str] | None = None,
    model_col: str = "model_name",
    split_col: str = "split_protocol",
    rmse_col: str = "rmse",
) -> "pd.DataFrame":
    """Compute skill scores for every (model, split) against every baseline.

    Parameters
    ----------
    results_df
        DataFrame with at minimum columns ``model_col``, ``split_col``,
        ``rmse_col``.  Must contain both model rows and baseline rows.
    baseline_names
        List of model names to treat as baselines.
        Default: ``["climatology", "persistence"]``.
    model_col
        Column identifying model / baseline name.
    split_col
        Column identifying the split protocol.
    rmse_col
        Column with RMSE values.

    Returns
    -------
    DataFrame with columns:

    - ``model_name``
    - ``split_protocol``
    - ``baseline``
    - ``model_rmse``
    - ``baseline_rmse``
    - ``skill_score``
    """
    import pandas as pd

    if baseline_names is None:
        baseline_names = ["climatology", "persistence"]

    required = {model_col, split_col, rmse_col}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []

    # Get baseline RMSEs keyed by (split_protocol, baseline_name)
    baseline_rmse_map: dict[tuple[str, str], float] = {}
    for _, row in results_df.iterrows():
        name = str(row[model_col])
        if name in baseline_names:
            key = (str(row[split_col]), name)
            baseline_rmse_map[key] = float(row[rmse_col])

    # Compute skill for every non-baseline model
    for _, row in results_df.iterrows():
        model_name = str(row[model_col])
        if model_name in baseline_names:
            continue  # skip baselines themselves

        split = str(row[split_col])
        model_rmse_val = float(row[rmse_col])

        for baseline in baseline_names:
            base_rmse = baseline_rmse_map.get((split, baseline))
            if base_rmse is None:
                # Baseline result missing for this split — skip.
                rows.append(
                    {
                        "model_name": model_name,
                        "split_protocol": split,
                        "baseline": baseline,
                        "model_rmse": model_rmse_val,
                        "baseline_rmse": float("nan"),
                        "skill_score": float("nan"),
                        "note": f"baseline '{baseline}' not found for split '{split}'",
                    }
                )
                continue

            if model_name == baseline:
                # Same model as baseline → skill = 0 by definition.
                ss = 0.0
            else:
                ss = skill_score(model_rmse_val, base_rmse)

            rows.append(
                {
                    "model_name": model_name,
                    "split_protocol": split,
                    "baseline": baseline,
                    "model_rmse": model_rmse_val,
                    "baseline_rmse": base_rmse,
                    "skill_score": ss,
                }
            )

    return pd.DataFrame(rows)
