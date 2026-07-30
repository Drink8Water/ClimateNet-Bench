"""Tests for isolated multi-seed benchmark aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from climatenet.benchmark.multiseed import summarize_multiseed_runs


def _write_seed_run(root: Path, seed: int, offset: float) -> None:
    root.mkdir()
    (root / "metrics").mkdir()
    (root / "run_metadata.json").write_text(
        json.dumps({"status": "completed", "seed": seed}),
        encoding="utf-8",
    )
    rows = []
    for model, model_delta in [
        ("linear_regression", 2.0),
        ("lightgbm", 0.0),
    ]:
        random_rmse = 5.0 + model_delta + offset
        for split, split_delta in [
            ("random", 0.0),
            ("spatial_block", 1.0),
            ("temporal", 2.0),
        ]:
            rmse = random_rmse + split_delta
            rows.append(
                {
                    "model_name": model,
                    "feature_set": "full",
                    "split_protocol": split,
                    "mae": rmse - 1,
                    "rmse": rmse,
                    "r2": 0.5 - split_delta / 10,
                    "skill_vs_climatology": 0.4 - split_delta / 10,
                    "ood_degradation": (
                        float("nan")
                        if split == "random"
                        else split_delta / random_rmse
                    ),
                }
            )
            metrics = {
                **rows[-1],
                "regional_metrics": {
                    "East China": {"rmse": rmse + 2, "mae": rmse},
                    "Sahara": {"rmse": rmse, "mae": rmse - 1},
                },
            }
            (root / "metrics" / f"{model}_{split}.json").write_text(
                json.dumps(metrics),
                encoding="utf-8",
            )
    pd.DataFrame(rows).to_csv(root / "leaderboard.csv", index=False)


def test_multiseed_summary_writes_mean_std_and_stability(tmp_path: Path) -> None:
    runs = []
    for seed, offset in [(42, 0.0), (123, 0.2), (2026, -0.1)]:
        run = tmp_path / f"run-{seed}"
        _write_seed_run(run, seed, offset)
        runs.append(run)
    output = tmp_path / "summary"

    manifest = summarize_multiseed_runs(
        runs, output, expected_seeds=[42, 123, 2026]
    )

    assert manifest["total_tasks"] == 18
    assert manifest["failed_tasks"] == 0
    assert manifest["stability"]["lightgbm_beats_linear_random"]["stable"]
    assert manifest["stability"]["lightgbm_beats_linear_spatial"]["stable"]
    assert manifest["stability"]["temporal_harder_than_random"]["stable"]
    assert manifest["stability"]["spatial_harder_than_random"]["stable"]
    assert manifest["stability"]["east_china_rmse_exceeds_sahara"]["stable"]
    mean_std = pd.read_csv(output / "mean_std.csv")
    assert len(mean_std) == 6
    assert mean_std["rmse_std"].gt(0).all()
    assert len(list(output.glob("leaderboard_seed*.csv"))) == 3
