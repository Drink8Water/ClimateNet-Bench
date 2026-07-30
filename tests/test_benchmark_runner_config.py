"""Configuration execution tests for the benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climatenet.benchmark.split_protocols import SplitResult, generate_all_splits
from climatenet.benchmark.leaderboard import build_leaderboard
from climatenet.data.forecasting_dataset import build_forecasting_samples
from climatenet.preprocessing.climatology import (
    TrainOnlyClimatePreprocessor,
    TrainOnlyStandardizer,
)
from climatenet.training import benchmark_runner
from climatenet.training.benchmark_runner import (
    _get_feature_columns,
    _prepare_train_only_split,
    _resolve_split_protocols,
    run_benchmark,
)
from climatenet.utils.config import load_yaml


DYNAMIC_FEATURES = [
    "temperature_anomaly",
    "precipitation_anomaly",
    "radiation_anomaly",
    "soil_moisture_anomaly",
    "wind_speed",
    "dryness_proxy",
    "saturation_vapor_pressure",
]


def _feature_frame(input_window: int = 6) -> pd.DataFrame:
    data: dict[str, list[float]] = {
        "latitude": [20.0],
        "longitude": [10.0],
        "month_sin": [0.5],
        "month_cos": [0.8],
    }
    for feature in DYNAMIC_FEATURES:
        for lag in range(1, input_window + 1):
            data[f"{feature}_lag_{lag}"] = [float(lag)]
    return pd.DataFrame(data)


def _split_frame() -> pd.DataFrame:
    rows = []
    for block, (lat, lon) in enumerate([(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]):
        for month in range(1, 5):
            rows.append(
                {
                    "sample_id": f"sample-{block}-{month}",
                    "grid_id": f"grid-{block}",
                    "region": f"region-{block}",
                    "climate_type": f"zone-{block}",
                    "target_year": 2020,
                    "target_month": month,
                    "latitude": lat,
                    "longitude": lon,
                }
            )
    return pd.DataFrame(rows)


def test_smoke_config_generates_only_its_two_splits(tmp_path: Path) -> None:
    config = load_yaml("configs/benchmark/smoke_test.yaml")
    protocols = _resolve_split_protocols(config["split_protocols"])

    results = generate_all_splits(
        _split_frame(),
        tmp_path,
        protocols=protocols,
    )

    assert [result.protocol for result in results] == ["random", "spatial_block"]
    assert {path.name for path in tmp_path.iterdir()} == {"random", "spatial_block"}


def test_unknown_split_is_an_error() -> None:
    with pytest.raises(ValueError, match="Unknown split protocol"):
        _resolve_split_protocols(["random_split", "typo_holdout"])


def test_base_and_full_feature_sets_resolve_differently() -> None:
    config = load_yaml("configs/benchmark/smoke_test.yaml")
    samples = _feature_frame()

    base = _get_feature_columns(samples, "base", config["feature_sets"], 6)
    full = _get_feature_columns(samples, "full", config["feature_sets"], 6)

    assert base == ["latitude", "longitude", "month_sin", "month_cos"]
    assert full != base


def test_era5_v1_is_bounded_to_twelve_tasks_and_stable_dryness_feature() -> None:
    config = load_yaml("configs/benchmark/era5_land_v1.yaml")
    models = [
        model if isinstance(model, str) else model["name"]
        for model in config["models"]
    ]
    full_features = config["feature_sets"]["full"]["features"]

    assert config["synthetic"] is False
    assert "synthetic" not in config["features_path"].lower()
    assert models == ["linear_regression", "lightgbm"]
    assert config["split_protocols"] == [
        "random_split",
        "temporal_holdout",
        "spatial_block_holdout",
    ]
    assert set(config["feature_sets"]) == {"base", "full"}
    assert len(models) * len(config["split_protocols"]) * len(
        config["feature_sets"]
    ) == 12
    assert "dryness_proxy_log1p" in full_features
    assert "dryness_proxy" not in full_features
    assert config["artifacts"]["compatibility_predictions"] is False


def test_corrected_era5_v1_is_bounded_and_uses_only_corrected_data() -> None:
    config = load_yaml("configs/benchmark/era5_land_v1_corrected.yaml")
    models = [
        model if isinstance(model, str) else model["name"]
        for model in config["models"]
    ]
    path = config["features_path"]
    full_features = config["feature_sets"]["full"]["features"]

    assert "corrected" in config["benchmark_name"].lower()
    assert "corrected" in config["dataset_name"].lower()
    assert config["data_source"] == "ERA5-Land corrected"
    assert config["synthetic"] is False
    assert path == config["input_data_path"]
    assert path.endswith(
        "era5_physical_features_full_2019_2023_corrected.csv"
    )
    assert "synthetic" not in path.lower()
    assert models == ["linear_regression", "lightgbm"]
    assert config["split_protocols"] == [
        "random_split",
        "temporal_holdout",
        "spatial_block_holdout",
    ]
    assert set(config["feature_sets"]) == {"base", "full"}
    assert len(models) * len(config["split_protocols"]) * len(
        config["feature_sets"]
    ) == 12
    assert config["expected_features_sha256"] == (
        "90067630ca1a8f11b6005026313e1d7f2"
        "cb343ddd21fcbf8744bc38806808db7"
    )
    assert "dryness_proxy_log1p" in full_features
    assert "dryness_proxy" not in full_features
    assert config["artifacts"]["compatibility_predictions"] is False
    assert (
        config["artifacts"]["estimated_total_run_bytes"]
        < config["artifacts"]["estimated_max_bytes"]
    )


def test_repeated_spatial_lite_config_is_ready_and_exactly_ten_tasks() -> None:
    config = load_yaml(
        "configs/benchmark/"
        "era5_land_corrected_repeated_spatial_lite.yaml"
    )
    models = [
        model if isinstance(model, str) else model["name"]
        for model in config["models"]
    ]
    assert config["execution_status"] == "benchmarked"
    assert config["result_status"] == "final_corrected"
    assert config["runner_integration"]
    assert config["synthetic"] is False
    assert "corrected" in config["features_path"]
    assert config["split_protocols"] == [
        "repeated_region_stratified_spatial"
    ]
    assert config["fold_count"] == 5
    assert models == ["linear_regression", "lightgbm"]
    assert set(config["feature_sets"]) == {"full"}
    assert config["expected_task_count"] == (
        config["fold_count"] * len(models)
    ) == 10
    assert config["repeated_spatial"]["folds"] == [0, 1, 2, 3, 4]
    assert config["artifacts"]["compatibility_predictions"] is False
    assert config["artifacts"]["predictions"] == "canonical_only"


def test_runner_consumes_manifest_backed_repeated_fold(tmp_path: Path) -> None:
    samples = _runner_samples()
    samples_path = tmp_path / "samples.csv"
    samples.to_csv(samples_path, index=False)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    grid_rows = samples[
        ["region", "latitude", "longitude"]
    ].drop_duplicates().sort_values("latitude")
    partitions = ["train", "train", "validation", "validation", "test", "test"]
    manifest = grid_rows.copy()
    manifest["spatial_block_id"] = [
        f"block_lat{int(latitude)}_lon{int(longitude)}"
        for latitude, longitude in manifest[
            ["latitude", "longitude"]
        ].itertuples(index=False, name=None)
    ]
    manifest["partition"] = partitions
    manifest.to_csv(
        manifest_dir / "fold_0_block_assignments.csv", index=False
    )
    audit_path = tmp_path / "fold_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "audit_passed": True,
                "folds": [{"fold": 0, "fold_passed": True}],
            }
        ),
        encoding="utf-8",
    )
    config = _runner_config(samples_path)
    config["split_protocols"] = ["repeated_region_stratified_spatial"]
    config["repeated_spatial"] = {
        "audit_path": str(audit_path),
        "manifest_dir": str(manifest_dir),
        "required_audit_status": "ready",
        "folds": [0],
        "block_size_deg": 5.0,
        "generation_seed": 42,
    }
    config["artifacts"] = {"compatibility_predictions": False}

    registry = run_benchmark(
        config, output_root=tmp_path / "benchmark_runs"
    )
    run_dir = registry.path.parent
    assert len(registry.list_completed()) == 1
    metrics = json.loads(next((run_dir / "metrics").glob("*.json")).read_text())
    predictions = pd.read_csv(next((run_dir / "predictions").glob("*.csv")))
    assert metrics["fold"] == 0
    assert predictions["fold"].unique().tolist() == [0]
    assert (run_dir / "across_fold_summary.csv").exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["generalisation"]["reference_split"] is None
    assert summary["across_fold_summary"][0]["fold_count"] == 1


def test_era5_v1_multiseed_matrix_is_exactly_eighteen_full_tasks() -> None:
    config = load_yaml("configs/benchmark/era5_land_v1_multiseed.yaml")
    models = [
        model if isinstance(model, str) else model["name"]
        for model in config["models"]
    ]
    features = config["feature_sets"]["full"]["features"]

    assert config["random_seeds"] == [42, 123, 2026]
    assert models == ["linear_regression", "lightgbm"]
    assert set(config["feature_sets"]) == {"full"}
    assert len(config["random_seeds"]) * len(models) * len(
        config["feature_sets"]
    ) * len(config["split_protocols"]) == 18
    assert "dryness_proxy_log1p" in features
    assert "dryness_proxy" not in features
    assert config["artifacts"]["compatibility_predictions"] is False
    assert (
        config["artifacts"]["estimated_aggregate_bytes_including_reused_seed"]
        <= config["artifacts"]["estimated_max_bytes"]
    )


def test_corrected_era5_v1_lite_multiseed_uses_corrected_source_and_reuses_seed42() -> None:
    config = load_yaml("configs/benchmark/era5_land_v1_corrected_multiseed_lite.yaml")
    models = [
        model if isinstance(model, str) else model["name"]
        for model in config["models"]
    ]
    features = config["feature_sets"]["full"]["features"]

    assert "corrected" in config["benchmark_name"].lower()
    assert "lite" in config["benchmark_name"].lower()
    assert config["data_source"] == "ERA5-Land corrected"
    assert config["synthetic"] is False
    assert config["input_data_path"] == config["features_path"]
    assert config["features_path"].endswith(
        "era5_physical_features_full_2019_2023_corrected.csv"
    )
    assert "synthetic" not in config["features_path"].lower()
    assert config["expected_features_sha256"] == (
        "90067630ca1a8f11b6005026313e1d7f2"
        "cb343ddd21fcbf8744bc38806808db7"
    )
    assert config["preflight_audit_path"].endswith(
        "era5_corrected_full_2019_2023_audit.json"
    )
    assert config["required_preflight_audit_status"] == "ready"
    assert config["random_seeds"] == [42, 123, 2026]
    assert config["existing_seed_runs"].keys() == {"42"}
    assert "corrected-v1" in config["existing_seed_runs"]["42"]
    assert models == ["linear_regression", "lightgbm"]
    assert set(config["feature_sets"]) == {"full"}
    assert len(config["random_seeds"]) * len(models) * len(
        config["feature_sets"]
    ) * len(config["split_protocols"]) == 18
    assert "dryness_proxy_log1p" in features
    assert "dryness_proxy" not in features
    assert config["artifacts"]["compatibility_predictions"] is False
    assert (
        config["artifacts"]["estimated_aggregate_bytes_including_reused_seed"]
        <= config["artifacts"]["estimated_max_bytes"]
    )


def test_full_feature_set_contains_every_lag() -> None:
    config = load_yaml("configs/benchmark/smoke_test.yaml")
    full = _get_feature_columns(
        _feature_frame(), "full", config["feature_sets"], input_window=6
    )

    for feature in DYNAMIC_FEATURES:
        assert [f"{feature}_lag_{lag}" for lag in range(1, 7)] == [
            column for column in full if column.startswith(f"{feature}_lag_")
        ]


def test_missing_lag_column_is_an_error() -> None:
    config = load_yaml("configs/benchmark/smoke_test.yaml")
    samples = _feature_frame().drop(columns=["temperature_anomaly_lag_4"])

    with pytest.raises(ValueError, match="temperature_anomaly_lag_4"):
        _get_feature_columns(samples, "full", config["feature_sets"], 6)


def test_missing_target_column_fails_before_model_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "grid_id": ["grid-1"],
            "region": ["Sahara"],
            "target_year": [2020],
            "target_month": [1],
            "latitude": [20.0],
            "longitude": [10.0],
            "month_sin": [0.5],
            "month_cos": [0.8],
        }
    )
    samples_path = tmp_path / "samples.csv"
    samples.to_csv(samples_path, index=False)

    model_created = False

    def fail_if_model_created(*args: object, **kwargs: object) -> None:
        nonlocal model_created
        model_created = True
        raise AssertionError("model creation must not happen before schema validation")

    monkeypatch.setattr(benchmark_runner, "create_model", fail_if_model_created)
    config = {
        "benchmark_name": "missing-target",
        "forecasting_samples_path": str(samples_path),
        "input_window": 6,
        "target": "evaporation_anomaly",
        "target_column": "y_true",
        "split_protocols": ["random_split"],
        "feature_sets": {
            "base": {
                "features": ["latitude", "longitude", "month_sin", "month_cos"]
            }
        },
        "models": [{"name": "linear_regression"}],
    }

    with pytest.raises(ValueError, match="y_true"):
        run_benchmark(config, output_root=tmp_path / "output")

    assert model_created is False


def _raw_region_transfer_data() -> pd.DataFrame:
    rows = []
    for region_index, region in enumerate(["Sahara", "Amazon"]):
        for offset in range(24):
            year = 2019 + offset // 12
            month = offset % 12 + 1
            seasonal = float(month + region_index * 20)
            rows.append(
                {
                    "region": region,
                    "year": year,
                    "month": month,
                    "latitude": 20.0 - region_index * 25.0,
                    "longitude": 10.0 - region_index * 70.0,
                    "temperature": seasonal + 10.0,
                    "precipitation": seasonal / 10.0,
                    "radiation": seasonal * 2.0,
                    "soil_moisture": seasonal / 100.0,
                    "evaporation": seasonal / 20.0,
                    "wind_speed": seasonal / 5.0,
                    "dryness_proxy": seasonal / 7.0,
                    "saturation_vapor_pressure": seasonal / 3.0,
                    "month_sin": float(np.sin(2 * np.pi * month / 12)),
                    "month_cos": float(np.cos(2 * np.pi * month / 12)),
                }
            )
    return pd.DataFrame(rows)


def _region_transfer_inputs() -> tuple[pd.DataFrame, SplitResult, dict]:
    raw = _raw_region_transfer_data()
    basis, _ = build_forecasting_samples(
        raw,
        feature_columns=["temperature"],
        target_column="evaporation",
        sequence_length=6,
    )
    sahara_ids = basis.loc[basis["region"] == "Sahara", "sample_id"].tolist()
    amazon_ids = basis.loc[basis["region"] == "Amazon", "sample_id"].tolist()
    split = SplitResult(
        split_id="region_transfer_test",
        protocol="region_transfer",
        train_ids=sahara_ids[:12],
        val_ids=sahara_ids[12:],
        test_ids=amazon_ids,
    )
    feature_sets = {
        "full": {
            "features": [
                "latitude",
                "longitude",
                "month_sin",
                "month_cos",
                "temperature_anomaly",
                "soil_moisture_anomaly",
            ]
        }
    }
    return raw, split, feature_sets


def test_test_and_validation_changes_do_not_affect_train_preprocessing() -> None:
    raw, split, feature_sets = _region_transfer_inputs()

    prepared_1, resolved_1, metadata_1 = _prepare_train_only_split(
        raw,
        split,
        feature_sets,
        input_window=6,
        target="evaporation_anomaly",
        target_column="y_true",
    )

    mutated = raw.copy()
    held_out_mask = (mutated["region"] == "Amazon") | (
        (mutated["region"] == "Sahara")
        & (
            (mutated["year"] > 2020)
            | ((mutated["year"] == 2020) & (mutated["month"] >= 7))
        )
    )
    numeric_sources = [
        "temperature",
        "precipitation",
        "radiation",
        "soil_moisture",
        "evaporation",
        "wind_speed",
        "dryness_proxy",
        "saturation_vapor_pressure",
    ]
    mutated.loc[held_out_mask, numeric_sources] += 1_000_000.0

    prepared_2, resolved_2, metadata_2 = _prepare_train_only_split(
        mutated,
        split,
        feature_sets,
        input_window=6,
        target="evaporation_anomaly",
        target_column="y_true",
    )

    train_1 = prepared_1[
        prepared_1["sample_id"].isin(split.train_ids)
    ].sort_values("sample_id")
    train_2 = prepared_2[
        prepared_2["sample_id"].isin(split.train_ids)
    ].sort_values("sample_id")
    compare_columns = resolved_1["full"] + ["y_true"]

    pd.testing.assert_frame_equal(
        train_1[compare_columns].reset_index(drop=True),
        train_2[compare_columns].reset_index(drop=True),
    )
    assert resolved_1 == resolved_2
    assert (
        metadata_1["climatology_fingerprint"]
        == metadata_2["climatology_fingerprint"]
    )
    assert metadata_1["event_thresholds"] == metadata_2["event_thresholds"]
    assert (
        metadata_1["standardization_parameters"]
        == metadata_2["standardization_parameters"]
    )
    assert metadata_1["unseen_region_fallback"] == (
        "train_global_month_mean_then_train_global_mean"
    )
    assert prepared_1.loc[
        prepared_1["sample_id"].isin(split.test_ids),
        "temperature_anomaly_lag_1",
    ].notna().all()


def test_test_mutation_does_not_change_train_or_validation_features() -> None:
    raw, split, feature_sets = _region_transfer_inputs()
    prepared_1, resolved, metadata_1 = _prepare_train_only_split(
        raw, split, feature_sets, 6, "evaporation_anomaly", "y_true"
    )
    mutated = raw.copy()
    source_columns = [
        "temperature",
        "precipitation",
        "radiation",
        "soil_moisture",
        "evaporation",
    ]
    mutated.loc[mutated["region"] == "Amazon", source_columns] = -9_999_999.0
    prepared_2, _, metadata_2 = _prepare_train_only_split(
        mutated, split, feature_sets, 6, "evaporation_anomaly", "y_true"
    )
    unaffected_ids = split.train_ids + split.val_ids
    columns = ["sample_id", "y_true", *resolved["full"]]
    left = prepared_1[prepared_1["sample_id"].isin(unaffected_ids)][columns]
    right = prepared_2[prepared_2["sample_id"].isin(unaffected_ids)][columns]
    pd.testing.assert_frame_equal(
        left.sort_values("sample_id").reset_index(drop=True),
        right.sort_values("sample_id").reset_index(drop=True),
    )
    assert (
        metadata_1["climatology_fingerprint"]
        == metadata_2["climatology_fingerprint"]
    )


def test_validation_mutation_does_not_change_train_features() -> None:
    raw, split, feature_sets = _region_transfer_inputs()
    prepared_1, resolved, metadata_1 = _prepare_train_only_split(
        raw, split, feature_sets, 6, "evaporation_anomaly", "y_true"
    )
    mutated = raw.copy()
    validation_raw = (
        (mutated["region"] == "Sahara")
        & (mutated["year"] == 2020)
        & (mutated["month"] >= 7)
    )
    mutated.loc[validation_raw, "temperature"] = 5_000_000.0
    mutated.loc[validation_raw, "evaporation"] = 5_000_000.0
    prepared_2, _, metadata_2 = _prepare_train_only_split(
        mutated, split, feature_sets, 6, "evaporation_anomaly", "y_true"
    )
    columns = ["sample_id", "y_true", *resolved["full"]]
    left = prepared_1[prepared_1["sample_id"].isin(split.train_ids)][columns]
    right = prepared_2[prepared_2["sample_id"].isin(split.train_ids)][columns]
    pd.testing.assert_frame_equal(
        left.sort_values("sample_id").reset_index(drop=True),
        right.sort_values("sample_id").reset_index(drop=True),
    )
    assert metadata_1["event_thresholds"] == metadata_2["event_thresholds"]
    assert (
        metadata_1["standardization_parameters"]
        == metadata_2["standardization_parameters"]
    )


def test_unseen_region_uses_train_global_monthly_fallback() -> None:
    raw, split, feature_sets = _region_transfer_inputs()
    prepared, _, metadata = _prepare_train_only_split(
        raw,
        split,
        feature_sets,
        6,
        "evaporation_anomaly",
        "y_true",
        standardize=False,
    )
    test_rows = prepared[prepared["sample_id"].isin(split.test_ids)]

    # Amazon raw temperature is exactly 20 above Sahara for every month.
    assert np.allclose(test_rows["temperature_anomaly_lag_1"], 20.0)
    fallback = metadata["fallback_sample_counts"]["test"]
    assert fallback["fallback_samples"] == fallback["total_samples"]
    assert fallback["global_monthly_fallback_samples"] == fallback["total_samples"]
    usage = metadata["fallback_usage_by_partition"]["test"][
        "temperature_anomaly"
    ]
    assert usage["region_monthly_rows"] == 0
    assert usage["global_monthly_fallback_rows"] > 0
    assert metadata["validation_used_for_fit"] is False
    assert metadata["test_used_for_fit"] is False


def test_month_absent_from_train_uses_train_global_mean_without_refit() -> None:
    train = pd.DataFrame(
        {
            "region": ["Sahara"] * 11,
            "month": list(range(1, 12)),
            "temperature": np.arange(1.0, 12.0),
        }
    )
    preprocessor = TrainOnlyClimatePreprocessor(
        {"temperature": "temperature_anomaly"}
    ).fit(train)
    validation = train[train["month"].isin([1, 2])].copy()
    test = pd.DataFrame(
        {"region": ["Unseen"], "month": [12], "temperature": [1000.0]}
    )

    validation_out = preprocessor.transform(
        validation, partition="validation"
    )
    test_out = preprocessor.transform(test, partition="test")

    assert len(validation_out) == 2
    assert test_out["temperature_anomaly"].iloc[0] == pytest.approx(
        1000.0 - train["temperature"].mean()
    )
    assert preprocessor.metadata()["fallback_usage_by_partition"]["test"][
        "temperature_anomaly"
    ]["global_mean_fallback_rows"] == 1
    assert preprocessor.metadata()["validation_used_for_fit"] is False
    assert preprocessor.metadata()["test_used_for_fit"] is False


def test_nonfinite_validation_feature_errors_without_fitting_imputer() -> None:
    train = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    validation = pd.DataFrame({"feature": [np.nan]})
    standardizer = TrainOnlyStandardizer(["feature"]).fit(train)

    with pytest.raises(ValueError, match="no refitting.*imputation"):
        standardizer.transform(validation)

    assert standardizer.means["feature"] == pytest.approx(2.0)


def test_formal_runner_avoids_legacy_full_table_anomaly_and_saves_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _raw_region_transfer_data()
    raw_path = tmp_path / "raw_features.csv"
    raw.to_csv(raw_path, index=False)

    def forbidden_legacy_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("formal runner called full-table anomaly fit")

    monkeypatch.setattr(
        "climatenet.features.anomalies.add_monthly_climatology_and_anomalies",
        forbidden_legacy_fit,
    )
    config = {
        "benchmark_name": "formal-preprocessing-test",
        "data_source": "synthetic",
        "features_path": str(raw_path),
        "input_window": 6,
        "target": "evaporation_anomaly",
        "target_column": "y_true",
        "random_seed": 42,
        "preprocessing": {
            "train_only": True,
            "standardize_features": True,
        },
        "split_protocols": ["random_split"],
        "feature_sets": {
            "base": {
                "features": ["latitude", "longitude", "month_sin", "month_cos"]
            }
        },
        "models": [],
    }

    registry = run_benchmark(config, output_root=tmp_path / "outputs")
    artifact_dir = registry.path.parent / "preprocessing" / "random"
    metadata_path = artifact_dir / "preprocessing_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["climatology_group_keys"] == ["region", "month"]
    assert metadata["validation_used_for_fit"] is False
    assert metadata["test_used_for_fit"] is False
    assert metadata["preprocessing_config_hash"]
    assert metadata["training_years"] == [2019, 2020]
    assert metadata["training_regions"] == ["Amazon", "Sahara"]
    assert metadata["input_variables"] == [
        "temperature",
        "precipitation",
        "radiation",
        "soil_moisture",
        "evaporation",
    ]
    assert "fallback_sample_counts" in metadata
    assert "fallback_usage_by_partition" in metadata
    assert metadata["climatology_artifacts"]
    assert (artifact_dir / metadata["climatology_artifacts"][0]).exists()


def _runner_samples(n_grids: int = 6, n_per_grid: int = 12) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for grid in range(n_grids):
        for index in range(n_per_grid):
            month = index % 12 + 1
            rows.append(
                {
                    "sample_id": f"sample-{grid}-{index}",
                    "grid_id": f"grid-{grid}",
                    "region": f"region-{grid % 2}",
                    "climate_type": f"zone-{grid % 2}",
                    "target_year": 2020 + index // 12,
                    "target_month": month,
                    "latitude": float(grid * 10),
                    "longitude": float(grid * 10),
                    "month_sin": float(np.sin(2 * np.pi * month / 12)),
                    "month_cos": float(np.cos(2 * np.pi * month / 12)),
                    "y_true": float(grid + month + rng.normal(0, 0.1)),
                }
            )
    return pd.DataFrame(rows)


def _runner_config(
    samples_path: Path,
    *,
    data_source: str = "synthetic",
    model_name: str = "linear_regression",
) -> dict:
    model: dict = {"name": model_name}
    if model_name == "random_forest":
        model["params"] = {
            "n_estimators": 12,
            "min_samples_leaf": 1,
            "n_jobs": 1,
        }
    return {
        "benchmark_name": "isolation-test",
        "data_source": data_source,
        "forecasting_samples_path": str(samples_path),
        "input_window": 6,
        "target": "evaporation_anomaly",
        "target_column": "y_true",
        "random_seed": 17,
        "split_protocols": ["random_split"],
        "feature_sets": {
            "base": {
                "features": ["latitude", "longitude", "month_sin", "month_cos"]
            }
        },
        "models": [model],
    }


def test_each_experiment_gets_a_fresh_model_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples_path = tmp_path / "samples.csv"
    _runner_samples().to_csv(samples_path, index=False)
    config = _runner_config(samples_path)
    config["split_protocols"] = ["random_split", "spatial_block_holdout"]
    config["feature_sets"]["coordinates"] = {
        "features": ["latitude", "longitude"]
    }

    created_instances: list[object] = []
    fitted_instance_ids: list[int] = []
    received_configs: list[dict] = []

    class SpyModel:
        def __init__(self) -> None:
            self.mean = 0.0

        def get_model_name(self) -> str:
            return "spy"

        def fit(self, train_df: pd.DataFrame, **kwargs: object) -> "SpyModel":
            fitted_instance_ids.append(id(self))
            self.mean = float(train_df["y_true"].mean())
            return self

        def predict(self, test_df: pd.DataFrame) -> np.ndarray:
            return np.full(len(test_df), self.mean)

    def create_spy(model_name: str, model_config: dict) -> SpyModel:
        received_configs.append(model_config)
        instance = SpyModel()
        created_instances.append(instance)
        return instance

    monkeypatch.setattr(benchmark_runner, "create_model", create_spy)
    registry = run_benchmark(config, output_root=tmp_path / "outputs")

    assert len(registry.list_completed()) == 4
    assert len(created_instances) == 4
    assert len(set(fitted_instance_ids)) == 4
    assert all(model_config["random_state"] == 17 for model_config in received_configs)


def test_same_seed_produces_identical_predictions_and_distinct_runs(
    tmp_path: Path,
) -> None:
    samples_path = tmp_path / "samples.csv"
    _runner_samples().to_csv(samples_path, index=False)
    config = _runner_config(samples_path, model_name="random_forest")
    output_root = tmp_path / "outputs"

    first = run_benchmark(config, output_root=output_root)
    second = run_benchmark(config, output_root=output_root)

    assert first.path.parent != second.path.parent
    first_prediction = next((first.path.parent / "experiments").glob("*/predictions.csv"))
    second_prediction = next((second.path.parent / "experiments").glob("*/predictions.csv"))
    pd.testing.assert_frame_equal(
        pd.read_csv(first_prediction),
        pd.read_csv(second_prediction),
    )

    first_metadata = json.loads(
        (first.path.parent / "run_metadata.json").read_text(encoding="utf-8")
    )
    second_metadata = json.loads(
        (second.path.parent / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert first_metadata["run_id"] != second_metadata["run_id"]
    for key in [
        "data_source",
        "synthetic",
        "config_hash",
        "seed",
        "target",
        "target_column",
        "data_files",
        "git_commit",
    ]:
        assert first_metadata[key] == second_metadata[key]

    first_metrics = json.loads(
        next((first.path.parent / "experiments").glob("*/metrics.json"))
        .read_text(encoding="utf-8")
    )
    second_metrics = json.loads(
        next((second.path.parent / "experiments").glob("*/metrics.json"))
        .read_text(encoding="utf-8")
    )
    for key in ["rmse", "mae", "r2"]:
        assert first_metrics[key] == pytest.approx(second_metrics[key])
    assert first_metrics["split_id"] == second_metrics["split_id"]
    assert first_metrics["seed"] == second_metrics["seed"]


def test_data_sources_are_isolated_and_leaderboard_selects_one_run(
    tmp_path: Path,
) -> None:
    samples_path = tmp_path / "samples.csv"
    _runner_samples().to_csv(samples_path, index=False)
    output_root = tmp_path / "outputs"

    synthetic_registry = run_benchmark(
        _runner_config(samples_path, data_source="synthetic"),
        output_root=output_root,
    )
    era_registry = run_benchmark(
        _runner_config(samples_path, data_source="ERA5-Land"),
        output_root=output_root,
    )

    assert "-synthetic-" in synthetic_registry.path.parent.name
    assert "-era5-land-" in era_registry.path.parent.name
    assert synthetic_registry.path.parent != era_registry.path.parent

    latest = build_leaderboard(output_root)["all_results"]
    assert latest["run_id"].nunique() == 1
    assert latest["data_source"].nunique() == 1
    assert latest["data_source"].iloc[0] == "ERA5-Land"

    synthetic_run_id = synthetic_registry.list_all()[0].run_id
    synthetic_only = build_leaderboard(
        output_root, run_id=synthetic_run_id
    )["all_results"]
    assert set(synthetic_only["data_source"]) == {"synthetic"}


def test_run_writes_resolved_config_audit_and_canonical_artifacts(
    tmp_path: Path,
) -> None:
    samples_path = tmp_path / "samples.csv"
    _runner_samples().to_csv(samples_path, index=False)

    registry = run_benchmark(
        _runner_config(samples_path),
        output_root=tmp_path / "benchmark_runs",
    )
    run_dir = registry.path.parent
    resolved = load_yaml(run_dir / "config_resolved.yaml")
    metadata = json.loads(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )

    assert resolved["split_protocols_resolved"] == ["random"]
    assert resolved["feature_sets_selected"] == ["base"]
    assert resolved["models_selected"] == ["linear_regression"]
    assert resolved["random_seeds"] == [17]
    assert resolved["input_window"] == 6
    assert resolved["output_directory"] == str(run_dir.resolve())
    assert isinstance(resolved["git"]["dirty"], (bool, type(None)))
    assert resolved["environment"]["python"]
    assert resolved["environment"]["packages"]["numpy"]

    assert metadata["status"] == "completed"
    assert metadata["started_at"]
    assert metadata["finished_at"]
    assert metadata["total_models"] == 1
    assert metadata["total_splits"] == 1
    assert metadata["total_feature_sets"] == 1
    assert metadata["total_tasks"] == 1
    assert metadata["completed_task_count"] == 1
    assert metadata["failed_task_count"] == 0
    assert metadata["input_data_row_count"] == 72
    assert metadata["region_count"] == 2
    assert metadata["climate_zone_count"] == 2
    assert metadata["target_summary"]["missing_count"] == 0
    assert metadata["feature_non_finite_check"]["total_non_finite"] == 0
    assert "preprocessing_fallback_summary" in metadata

    prediction_path = next((run_dir / "predictions").glob("*.csv"))
    metrics_path = next((run_dir / "metrics").glob("*.json"))
    predictions = pd.read_csv(prediction_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert {
        "sample_id",
        "y_true",
        "y_pred",
        "partition",
        "split_id",
        "region",
        "target_year",
        "target_month",
    }.issubset(predictions.columns)
    assert set(predictions["partition"]) == {"test"}
    assert metrics["run_id"] == metadata["run_id"]
    assert (run_dir / "leaderboard.csv").exists()
    assert summary["run_id"] == metadata["run_id"]
    assert summary["split_performance"]["random"]["task_count"] == 1


def test_one_failed_task_preserves_completed_task_and_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples_path = tmp_path / "samples.csv"
    _runner_samples().to_csv(samples_path, index=False)
    config = _runner_config(samples_path)
    config["models"] = [
        {"name": "linear_regression"},
        {"name": "deliberately_broken"},
    ]
    real_create_model = benchmark_runner.create_model

    def create_with_failure(model_name: str, model_config: dict) -> object:
        if model_name == "deliberately_broken":
            raise RuntimeError("injected task failure")
        return real_create_model(model_name, model_config)

    monkeypatch.setattr(benchmark_runner, "create_model", create_with_failure)
    registry = run_benchmark(config, output_root=tmp_path / "outputs")
    run_dir = registry.path.parent
    metadata = json.loads(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    )

    assert len(registry.list_completed()) == 1
    assert len(registry.list_failed()) == 1
    assert metadata["status"] == "partial"
    assert metadata["total_tasks"] == 2
    assert metadata["completed_task_count"] == 1
    assert metadata["failed_task_count"] == 1
    assert len(list((run_dir / "predictions").glob("*.csv"))) == 1
    task_artifacts = list((run_dir / "metrics").glob("*.json"))
    assert len(task_artifacts) == 2
    failed = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in task_artifacts
        if json.loads(path.read_text(encoding="utf-8")).get("status") == "failed"
    ]
    assert failed[0]["error_message"] == "injected task failure"
    assert "RuntimeError" in failed[0]["traceback_summary"]
    assert (run_dir / "leaderboard.csv").exists()


def test_run_metadata_aggregates_unseen_region_fallback(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.csv"
    _raw_region_transfer_data().to_csv(raw_path, index=False)
    config = {
        "benchmark_name": "fallback-audit",
        "data_source": "synthetic",
        "features_path": str(raw_path),
        "input_window": 6,
        "target": "evaporation_anomaly",
        "target_column": "y_true",
        "random_seed": 23,
        "preprocessing": {
            "train_only": True,
            "standardize_features": True,
        },
        "split_protocols": ["region_transfer"],
        "region_transfer_pairs": [
            {
                "train_regions": ["Sahara"],
                "test_regions": ["Amazon"],
            }
        ],
        "feature_sets": {
            "base": {
                "features": ["latitude", "longitude", "month_sin", "month_cos"]
            }
        },
        "models": [],
    }

    registry = run_benchmark(config, output_root=tmp_path / "outputs")
    metadata = json.loads(
        (registry.path.parent / "run_metadata.json").read_text(encoding="utf-8")
    )
    totals = metadata["preprocessing_fallback_summary"]["totals"]

    assert totals["rows_checked"] > 0
    assert totals["global_monthly_fallback_rows"] > 0
    assert totals["fallback_samples"] > 0
    assert metadata["preprocessing_fallback_summary"]["by_split"]
