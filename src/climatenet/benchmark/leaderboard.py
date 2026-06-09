"""Leaderboard generator for ClimateNet-Bench.

Reads all completed experiments and produces ranked tables.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Columns expected in leaderboard output
LEADERBOARD_COLUMNS = [
    "rank",
    "model_name",
    "split_protocol",
    "feature_set",
    "rmse",
    "mae",
    "r2",
    "skill_vs_climatology",
    "skill_vs_persistence",
    "coverage_90",
    "mean_interval_width",
    "ood_degradation",
    "experiment_id",
]


def build_leaderboard(
    experiments_root: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Scan experiment directories and produce benchmark tables.

    Parameters
    ----------
    experiments_root
        Directory containing experiment subdirectories (each with
        ``metrics.json``, ``predictions.csv``, etc.).
    output_root
        If provided, CSV files are written here.

    Returns
    -------
    Dict mapping table name → DataFrame:
    ``"all_results"``, ``"leaderboard"``,
    ``"split_difficulty_analysis"``, ``"uncertainty_calibration"``,
    ``"ablation_results"``.
    """
    root = Path(experiments_root)
    if not root.exists():
        raise FileNotFoundError(f"Experiments directory not found: {root}")

    # ── 1. Collect all metrics ────────────────────────────────────
    rows: list[dict[str, Any]] = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            continue

        m["experiment_id"] = exp_dir.name

        # check for intervals
        intervals_path = exp_dir / "intervals.csv"
        if intervals_path.exists():
            try:
                idf = pd.read_csv(intervals_path)
                if "covered" in idf.columns:
                    m["coverage_90"] = float(idf["covered"].mean())
                if "interval_width" in idf.columns:
                    m["mean_interval_width"] = float(idf["interval_width"].mean())
            except Exception:
                pass

        rows.append(m)

    if not rows:
        logger.warning("No experiment metrics found in %s", root)
        return {}

    all_results = pd.DataFrame(rows)

    # ── 2. Compute skill scores ───────────────────────────────────
    from climatenet.evaluation.skill_score import compute_skill_scores

    # Temporarily remap columns to what compute_skill_scores expects
    if "model_name" in all_results.columns and "split_protocol" in all_results.columns:
        skill_df = compute_skill_scores(
            all_results,
            baseline_names=["climatology", "persistence"],
            model_col="model_name",
            split_col="split_protocol",
            rmse_col="rmse",
        )
        # Pivot skill scores into columns
        for baseline in ["climatology", "persistence"]:
            sub = skill_df[skill_df["baseline"] == baseline]
            skill_map = dict(zip(
                sub["model_name"] + "|" + sub["split_protocol"],
                sub["skill_score"],
            ))
            col_name = f"skill_vs_{baseline}"
            all_results[col_name] = all_results.apply(
                lambda r: skill_map.get(
                    f"{r['model_name']}|{r['split_protocol']}"
                ),
                axis=1,
            )

    # ── 3. OOD degradation ────────────────────────────────────────
    from climatenet.evaluation.ood_degradation import compute_ood_degradation_table

    if "split_protocol" in all_results.columns:
        try:
            ood_df = compute_ood_degradation_table(
                all_results,
                reference_split="random",
                model_col="model_name",
                split_col="split_protocol",
                rmse_col="rmse",
            )
            ood_map = dict(zip(
                ood_df["model_name"] + "|" + ood_df["ood_split"],
                ood_df["ood_degradation"],
            ))
            all_results["ood_degradation"] = all_results.apply(
                lambda r: ood_map.get(
                    f"{r['model_name']}|{r['split_protocol']}"
                ),
                axis=1,
            )
        except Exception:
            pass

    # ── 4. Build leaderboard ──────────────────────────────────────
    leaderboard = all_results.copy()

    # Sort: by split_protocol, then rmse ascending
    leaderboard = leaderboard.sort_values(
        ["split_protocol", "rmse"],
        ascending=[True, True],
    ).reset_index(drop=True)
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))

    # ── 5. Split difficulty analysis ──────────────────────────────
    try:
        difficulty = (
            all_results.groupby("split_protocol")
            .agg(
                mean_rmse=("rmse", "mean"),
                std_rmse=("rmse", "std"),
                min_rmse=("rmse", "min"),
                max_rmse=("rmse", "max"),
                n_models=("model_name", "nunique"),
            )
            .sort_values("mean_rmse")
            .reset_index()
        )
    except Exception:
        difficulty = pd.DataFrame()

    # ── 6. Ablation results ───────────────────────────────────────
    try:
        ablation = (
            all_results.groupby(["feature_set", "model_name", "split_protocol"])
            .agg(
                mean_rmse=("rmse", "mean"),
                mean_mae=("mae", "mean"),
                mean_r2=("r2", "mean"),
                best_rmse=("rmse", "min"),
            )
            .sort_values(["split_protocol", "mean_rmse"])
            .reset_index()
        )
    except Exception:
        ablation = pd.DataFrame()

    # ── 7. Uncertainty calibration ────────────────────────────────
    if "coverage_90" in all_results.columns:
        calib = all_results[
            ["model_name", "split_protocol", "feature_set",
             "coverage_90", "mean_interval_width"]
        ].dropna(subset=["coverage_90"])
    else:
        calib = pd.DataFrame()

    result = {
        "all_results": all_results,
        "leaderboard": leaderboard,
        "split_difficulty_analysis": difficulty,
        "uncertainty_calibration": calib,
        "ablation_results": ablation,
    }

    # ── Write output ──────────────────────────────────────────────
    if output_root is not None:
        out = Path(output_root)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in result.items():
            if not df.empty:
                df.to_csv(out / f"{name}.csv", index=False)
        logger.info("Leaderboard written to %s", out)

    return result
