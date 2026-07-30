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
    "run_id",
    "data_source",
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
    run_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Scan experiment directories and produce benchmark tables.

    Parameters
    ----------
    experiments_root
        Directory containing experiment subdirectories (each with
        ``metrics.json``, ``predictions.csv``, etc.).
    output_root
        If provided, CSV files are written here.
    run_id
        Run to aggregate. When omitted and multiple run IDs are present,
        only the most recently created run is selected. Legacy metrics
        without a run ID remain supported when no run-aware metrics exist.

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
    for metrics_path in sorted(root.rglob("metrics.json")):
        exp_dir = metrics_path.parent
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

    known_run_ids = sorted(
        {
            str(row["run_id"])
            for row in rows
            if row.get("run_id") not in (None, "")
        }
    )
    selected_run_id = run_id
    if selected_run_id is not None:
        rows = [
            row for row in rows
            if str(row.get("run_id", "")) == selected_run_id
        ]
        if not rows:
            raise ValueError(f"No experiment metrics found for run_id={selected_run_id!r}")
    elif known_run_ids:
        # Never mix run-aware results with history. ISO UTC timestamps sort
        # chronologically, with run_id as a deterministic tie-breaker.
        selected_run_id = max(
            known_run_ids,
            key=lambda candidate: max(
                (
                    str(row.get("run_created_at", "")),
                    candidate,
                )
                for row in rows
                if str(row.get("run_id", "")) == candidate
            ),
        )
        rows = [
            row for row in rows
            if str(row.get("run_id", "")) == selected_run_id
        ]
        if len(known_run_ids) > 1:
            logger.info(
                "Selected latest run %s; ignored %d historical run(s)",
                selected_run_id,
                len(known_run_ids) - 1,
            )

    data_sources = {
        str(row["data_source"])
        for row in rows
        if row.get("data_source") not in (None, "")
    }
    if len(data_sources) > 1:
        raise ValueError(
            "Leaderboard cannot mix data sources within one run: "
            f"{sorted(data_sources)}"
        )

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
            computed = all_results.apply(
                lambda r: skill_map.get(
                    f"{r['model_name']}|{r['split_protocol']}"
                ),
                axis=1,
            )
            if col_name in all_results.columns:
                all_results[col_name] = computed.combine_first(
                    pd.to_numeric(all_results[col_name], errors="coerce")
                )
            else:
                all_results[col_name] = computed

    # ── 3. OOD degradation ────────────────────────────────────────
    if "split_protocol" in all_results.columns:
        try:
            # Compare like with like. Feature-set variants of the same model
            # must not overwrite each other's random-split reference.
            comparison_columns = ["model_name"]
            if "feature_set" in all_results.columns:
                comparison_columns.append("feature_set")
            references = (
                all_results[all_results["split_protocol"] == "random"]
                .groupby(comparison_columns, dropna=False)["rmse"]
                .mean()
            )
            def degradation(row: pd.Series) -> float:
                if row["split_protocol"] == "random":
                    return float("nan")
                key: Any = (
                    tuple(row[column] for column in comparison_columns)
                    if len(comparison_columns) > 1
                    else row[comparison_columns[0]]
                )
                reference = references.get(key)
                if reference is None or pd.isna(reference) or reference == 0:
                    return float("nan")
                return float((row["rmse"] - reference) / reference)

            all_results["ood_degradation"] = all_results.apply(
                degradation, axis=1
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
