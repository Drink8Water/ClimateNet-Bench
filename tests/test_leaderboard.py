"""Tests for experiment registry and leaderboard generator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climatenet.benchmark.leaderboard import build_leaderboard
from climatenet.training.experiment_registry import (
    ExperimentRecord,
    ExperimentRegistry,
)


# ---------------------------------------------------------------------------
# Experiment registry tests
# ---------------------------------------------------------------------------


class TestExperimentRegistry:
    def test_add_and_get(self) -> None:
        reg = ExperimentRegistry(Path("/tmp/test_registry.json"))
        rec = ExperimentRecord(experiment_id="exp_001", model_name="rf")
        reg.add(rec)
        assert reg.get("exp_001") is not None
        assert reg.get("exp_001").model_name == "rf"

    def test_list_all(self) -> None:
        reg = ExperimentRegistry(Path("/tmp/test_registry.json"))
        reg.add(ExperimentRecord(experiment_id="a"))
        reg.add(ExperimentRecord(experiment_id="b"))
        assert len(reg.list_all()) == 2

    def test_mark_running_completed_failed(self) -> None:
        reg = ExperimentRegistry(Path("/tmp/test_registry.json"))
        rec = ExperimentRecord(experiment_id="exp_001")
        reg.add(rec)
        reg.mark_running("exp_001")
        assert reg.get("exp_001").status == "running"
        reg.mark_completed("exp_001")
        assert reg.get("exp_001").status == "completed"
        reg.mark_failed("exp_001", "out of memory")
        assert reg.get("exp_001").status == "failed"
        assert "memory" in reg.get("exp_001").error_message

    def test_list_completed_and_failed(self) -> None:
        reg = ExperimentRegistry(Path("/tmp/test_registry.json"))
        reg.add(ExperimentRecord(experiment_id="ok", status="completed"))
        reg.add(ExperimentRecord(experiment_id="bad", status="failed"))
        reg.add(ExperimentRecord(experiment_id="run", status="running"))
        assert len(reg.list_completed()) == 1
        assert len(reg.list_failed()) == 1

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            reg = ExperimentRegistry(path)
            reg.add(ExperimentRecord(
                experiment_id="exp_001",
                model_name="random_forest",
                split_protocol="spatial_block",
                train_regions=["Sahara"],
                test_regions=["Amazon"],
                seed=42,
                status="completed",
            ))
            reg.save()

            loaded = ExperimentRegistry.load(path)
            rec = loaded.get("exp_001")
            assert rec.model_name == "random_forest"
            assert rec.train_regions == ["Sahara"]
            assert rec.status == "completed"

    def test_missing_file_returns_empty_registry(self) -> None:
        reg = ExperimentRegistry.load("/tmp/nonexistent_registry.json")
        assert len(reg.list_all()) == 0

    def test_nonexistent_experiment_returns_none(self) -> None:
        reg = ExperimentRegistry(Path("/tmp/test.json"))
        assert reg.get("nope") is None


# ---------------------------------------------------------------------------
# Leaderboard tests
# ---------------------------------------------------------------------------


def _make_fake_experiment_dir(
    root: Path,
    exp_id: str,
    model_name: str,
    split: str,
    feature_set: str = "full",
    rmse: float = 0.5,
    mae: float = 0.35,
    r2: float = 0.7,
) -> None:
    """Create a minimal experiment directory for leaderboard testing."""
    exp_dir = root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model_name": model_name,
        "split_protocol": split,
        "feature_set": feature_set,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_train": 500,
        "n_test": 100,
    }
    with (exp_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f)


class TestLeaderboard:
    def test_build_from_fake_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fake_experiment_dir(root, "clim_random_full", "climatology", "random", "full", rmse=0.80)
            _make_fake_experiment_dir(root, "pers_random_full", "persistence", "random", "full", rmse=0.72)
            _make_fake_experiment_dir(root, "rf_random_full", "random_forest", "random", "full", rmse=0.45)
            _make_fake_experiment_dir(root, "rf_spatial_full", "random_forest", "spatial_block", "full", rmse=0.55)
            _make_fake_experiment_dir(root, "xg_random_full", "xgboost", "random", "full", rmse=0.42)

            result = build_leaderboard(root)

            assert "leaderboard" in result
            lb = result["leaderboard"]
            assert len(lb) >= 3  # at least non-baseline models

            # Ranking: within each split, lowest RMSE gets lowest rank
            random_rows = lb[lb["split_protocol"] == "random"]
            # xgboost (0.42) should be ranked before rf (0.45)
            xg_rank = random_rows[random_rows["model_name"] == "xgboost"]["rank"].iloc[0]
            rf_rank = random_rows[random_rows["model_name"] == "random_forest"]["rank"].iloc[0]
            assert xg_rank < rf_rank

    def test_missing_optional_metrics_handled(self) -> None:
        """Leaderboard should not crash if skill scores / OOD can't be computed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Only one model, no baselines → skill scores will be NaN
            _make_fake_experiment_dir(root, "rf_random_base", "random_forest", "random", "base", rmse=0.50)

            result = build_leaderboard(root)
            assert "leaderboard" in result
            assert len(result["leaderboard"]) == 1

    def test_empty_experiments_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = build_leaderboard(tmp)
            assert result == {}

    def test_split_difficulty_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, rmse in [("random", 0.40), ("spatial_block", 0.55), ("temporal", 0.50)]:
                _make_fake_experiment_dir(root, f"rf_{split}", "random_forest", split, rmse=rmse)

            result = build_leaderboard(root)
            diff = result.get("split_difficulty_analysis")
            assert diff is not None
            assert not diff.empty
            # spatial_block should have higher mean RMSE than random
            rand_mean = diff[diff["split_protocol"] == "random"]["mean_rmse"].iloc[0]
            spat_mean = diff[diff["split_protocol"] == "spatial_block"]["mean_rmse"].iloc[0]
            assert spat_mean > rand_mean

    def test_experiment_id_present_in_leaderboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fake_experiment_dir(root, "my_experiment_v1", "rf", "random", "full", rmse=0.5)

            result = build_leaderboard(root)
            lb = result["leaderboard"]
            assert "experiment_id" in lb.columns
            assert lb["experiment_id"].iloc[0] == "my_experiment_v1"

    def test_output_files_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiments"
            out = Path(tmp) / "output"
            root.mkdir()
            _make_fake_experiment_dir(root, "rf_random", "random_forest", "random", "full", rmse=0.5)
            _make_fake_experiment_dir(root, "clim_random", "climatology", "random", "full", rmse=0.8)

            result = build_leaderboard(experiments_root=root, output_root=out)

            assert (out / "all_results.csv").exists()
            assert (out / "leaderboard.csv").exists()
            assert (out / "split_difficulty_analysis.csv").exists()
