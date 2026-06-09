"""Out-of-distribution (OOD) degradation metrics.

Measures how much model performance drops when moving from an
easy / in-distribution split (typically **random split**) to a
harder / OOD split (spatial, temporal, region transfer, etc.).

Formula
-------
::

    degradation = (RMSE_ood − RMSE_random) / RMSE_random

- ``degradation > 0`` → model performs worse under OOD (expected).
- ``degradation ≈ 0`` → model generalises equally well.
- ``degradation < 0`` → model performs *better* under OOD (unusual;
  warrants investigation).

The random split serves as the reference because it is the most
optimistic baseline — it leaks spatial and temporal information.
OOD degradation quantifies the **cost of honest evaluation**.
"""

from __future__ import annotations

from typing import Any


def compute_ood_degradation(
    rmse_ood: float,
    rmse_random: float,
) -> float:
    """Compute OOD degradation for one (model, ood_split) pair.

    Parameters
    ----------
    rmse_ood
        RMSE on the out-of-distribution split.
    rmse_random
        RMSE on the random (in-distribution) split for the same model.

    Returns
    -------
    Degradation as a fraction.  ``0.5`` means RMSE increased by 50 %.

    Edge cases
    ----------
    - ``rmse_random == 0`` → returns ``np.nan`` (reference is perfect).
    - Negative RMSEs → raises ``ValueError``.
    """
    if rmse_ood < 0:
        raise ValueError(f"rmse_ood must be ≥ 0, got {rmse_ood}")
    if rmse_random < 0:
        raise ValueError(f"rmse_random must be ≥ 0, got {rmse_random}")
    if rmse_random == 0.0:
        return float("nan")

    return float((rmse_ood - rmse_random) / rmse_random)


def compute_ood_degradation_table(
    results_df: "pd.DataFrame",
    reference_split: str = "random",
    model_col: str = "model_name",
    split_col: str = "split_protocol",
    rmse_col: str = "rmse",
) -> "pd.DataFrame":
    """Compute OOD degradation for every non-random split.

    Parameters
    ----------
    results_df
        DataFrame with columns ``model_col``, ``split_col``, ``rmse_col``.
    reference_split
        The in-distribution reference split (default ``"random"``).
    model_col
        Column identifying the model name.
    split_col
        Column identifying the split protocol name.
    rmse_col
        Column with RMSE values.

    Returns
    -------
    DataFrame with columns:

    - ``model_name``
    - ``ood_split``
    - ``rmse_random``
    - ``rmse_ood``
    - ``ood_degradation``
    """
    import pandas as pd

    required = {model_col, split_col, rmse_col}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df missing required columns: {sorted(missing)}")

    # Index random-split RMSEs by model name.
    random_rmse: dict[str, float] = {}
    random_mask = results_df[split_col] == reference_split
    for _, row in results_df[random_mask].iterrows():
        random_rmse[str(row[model_col])] = float(row[rmse_col])

    rows: list[dict[str, Any]] = []
    ood_mask = results_df[split_col] != reference_split
    for _, row in results_df[ood_mask].iterrows():
        model = str(row[model_col])
        ood_split = str(row[split_col])
        rmse_ood_val = float(row[rmse_col])
        rmse_ref = random_rmse.get(model)

        if rmse_ref is None:
            rows.append(
                {
                    "model_name": model,
                    "ood_split": ood_split,
                    "rmse_random": float("nan"),
                    "rmse_ood": rmse_ood_val,
                    "ood_degradation": float("nan"),
                    "note": f"no random-split result for model '{model}'",
                }
            )
            continue

        deg = compute_ood_degradation(rmse_ood_val, rmse_ref)
        rows.append(
            {
                "model_name": model,
                "ood_split": ood_split,
                "rmse_random": rmse_ref,
                "rmse_ood": rmse_ood_val,
                "ood_degradation": deg,
            }
        )

    return pd.DataFrame(rows)
