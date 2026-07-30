"""Multi-run aggregation for bounded benchmark robustness checks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRICS = ["mae", "rmse", "r2", "skill_vs_climatology", "ood_degradation"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def _run_seed(run_dir: Path) -> int:
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    if metadata.get("status") != "completed":
        raise ValueError(f"Run is not completed: {run_dir}")
    return int(metadata["seed"])


def summarize_multiseed_runs(
    run_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    expected_seeds: list[int] | None = None,
    feature_set: str = "full",
) -> dict[str, Any]:
    """Aggregate isolated seed runs without mixing unrelated history."""
    roots = [Path(path) for path in run_dirs]
    seed_to_root: dict[int, Path] = {}
    frames: list[pd.DataFrame] = []
    regional_rows: list[dict[str, Any]] = []
    for root in roots:
        seed = _run_seed(root)
        if seed in seed_to_root:
            raise ValueError(f"Duplicate seed in multi-seed inputs: {seed}")
        seed_to_root[seed] = root
        leaderboard = pd.read_csv(root / "leaderboard.csv")
        selected = leaderboard[leaderboard["feature_set"] == feature_set].copy()
        if len(selected) != 6:
            raise ValueError(
                f"Expected six {feature_set!r} tasks for seed {seed}, "
                f"found {len(selected)}"
            )
        selected["seed"] = seed
        frames.append(selected)
        for metrics_path in sorted((root / "metrics").glob("*.json")):
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("feature_set") != feature_set:
                continue
            for region, values in metrics.get("regional_metrics", {}).items():
                regional_rows.append(
                    {
                        "seed": seed,
                        "model_name": metrics["model_name"],
                        "split_protocol": metrics["split_protocol"],
                        "feature_set": feature_set,
                        "region": region,
                        **values,
                    }
                )
    if expected_seeds is not None and set(seed_to_root) != set(expected_seeds):
        raise ValueError(
            f"Seed mismatch: expected {sorted(expected_seeds)}, "
            f"found {sorted(seed_to_root)}"
        )

    results = pd.concat(frames, ignore_index=True)
    group_keys = ["model_name", "feature_set", "split_protocol"]
    aggregations: dict[str, tuple[str, str]] = {}
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    mean_std = (
        results.groupby(group_keys, as_index=False)
        .agg(**aggregations)
        .sort_values(["split_protocol", "model_name"])
    )
    extrema_rows: list[dict[str, Any]] = []
    for keys, rows in results.groupby(group_keys):
        best = rows.loc[rows["rmse"].idxmin()]
        worst = rows.loc[rows["rmse"].idxmax()]
        extrema_rows.append(
            {
                **dict(zip(group_keys, keys)),
                "best_seed": int(best["seed"]),
                "best_rmse": float(best["rmse"]),
                "worst_seed": int(worst["seed"]),
                "worst_rmse": float(worst["rmse"]),
            }
        )
    best_worst = pd.DataFrame(extrema_rows)

    indexed = results.set_index(["seed", "model_name", "split_protocol"])
    checks: dict[str, list[dict[str, Any]]] = {
        "lightgbm_beats_linear_random": [],
        "lightgbm_beats_linear_spatial": [],
        "temporal_harder_than_random": [],
        "spatial_harder_than_random": [],
        "lightgbm_temporal_degradation_positive": [],
    }
    for seed in sorted(seed_to_root):
        for split, key in [
            ("random", "lightgbm_beats_linear_random"),
            ("spatial_block", "lightgbm_beats_linear_spatial"),
        ]:
            lightgbm = indexed.loc[(seed, "lightgbm", split), "rmse"]
            linear = indexed.loc[(seed, "linear_regression", split), "rmse"]
            checks[key].append(
                {
                    "seed": seed,
                    "holds": bool(lightgbm < linear),
                    "lightgbm_rmse": float(lightgbm),
                    "linear_rmse": float(linear),
                }
            )
        for model in ["linear_regression", "lightgbm"]:
            random_rmse = indexed.loc[(seed, model, "random"), "rmse"]
            temporal_rmse = indexed.loc[(seed, model, "temporal"), "rmse"]
            spatial_rmse = indexed.loc[(seed, model, "spatial_block"), "rmse"]
            checks["temporal_harder_than_random"].append(
                {
                    "seed": seed,
                    "model_name": model,
                    "holds": bool(temporal_rmse > random_rmse),
                    "relative_degradation": float(
                        (temporal_rmse - random_rmse) / random_rmse
                    ),
                }
            )
            checks["spatial_harder_than_random"].append(
                {
                    "seed": seed,
                    "model_name": model,
                    "holds": bool(spatial_rmse > random_rmse),
                    "relative_degradation": float(
                        (spatial_rmse - random_rmse) / random_rmse
                    ),
                }
            )
        lgb_ood = indexed.loc[
            (seed, "lightgbm", "temporal"), "ood_degradation"
        ]
        checks["lightgbm_temporal_degradation_positive"].append(
            {"seed": seed, "holds": bool(lgb_ood > 0), "value": float(lgb_ood)}
        )

    regional = pd.DataFrame(regional_rows)
    east_china_checks: list[dict[str, Any]] = []
    if not regional.empty:
        pivot = regional.pivot_table(
            index=["seed", "model_name", "split_protocol"],
            columns="region",
            values="rmse",
            aggfunc="first",
        )
        if {"East China", "Sahara"}.issubset(pivot.columns):
            for keys, row in pivot.iterrows():
                east_china_checks.append(
                    {
                        "seed": int(keys[0]),
                        "model_name": keys[1],
                        "split_protocol": keys[2],
                        "holds": bool(row["East China"] > row["Sahara"]),
                        "east_china_rmse": float(row["East China"]),
                        "sahara_rmse": float(row["Sahara"]),
                    }
                )
    checks["east_china_rmse_exceeds_sahara"] = east_china_checks
    stability = {
        name: {
            "holds_count": sum(int(item["holds"]) for item in values),
            "comparison_count": len(values),
            "stable": bool(values) and all(item["holds"] for item in values),
            "details": values,
        }
        for name, values in checks.items()
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    results.to_csv(destination / "all_seed_results.csv", index=False)
    mean_std.to_csv(destination / "mean_std.csv", index=False)
    best_worst.to_csv(destination / "best_worst_seed.csv", index=False)
    if not regional.empty:
        regional.to_csv(destination / "regional_metrics.csv", index=False)
    for seed, root in sorted(seed_to_root.items()):
        shutil.copyfile(
            root / "leaderboard.csv",
            destination / f"leaderboard_seed{seed}.csv",
        )
    manifest = {
        "status": "completed",
        "seeds": sorted(seed_to_root),
        "run_dirs": {
            str(seed): str(root.resolve())
            for seed, root in sorted(seed_to_root.items())
        },
        "total_tasks": int(len(results)),
        "completed_tasks": int(len(results)),
        "failed_tasks": 0,
        "feature_set": feature_set,
        "stability": stability,
    }
    (destination / "multiseed_summary.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_safe),
        encoding="utf-8",
    )
    return manifest
