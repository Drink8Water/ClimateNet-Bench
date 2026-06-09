"""Tests for benchmark baselines and model factory."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climatenet.models.base import ClimateModel
from climatenet.models.climatology import ClimatologyBaseline
from climatenet.models.linear import LinearRegressionModel
from climatenet.models.model_factory import create_model, list_available_models
from climatenet.models.persistence import PersistenceBaseline
from climatenet.models.tree_models import (
    LightGBMModel,
    RandomForestModel,
    XGBoostModel,
    _is_lightgbm_available,
    _is_xgboost_available,
)


# ---------------------------------------------------------------------------
# Tiny test data
# ---------------------------------------------------------------------------


def make_tiny_train_test(
    n_train: int = 30, n_test: int = 10, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build tiny train and test DataFrames for baseline tests."""
    rng = np.random.default_rng(seed)
    features = [
        "temperature_anomaly_lag_1",
        "temperature_anomaly_lag_2",
        "precipitation_anomaly_lag_1",
        "precipitation_anomaly_lag_2",
        "month_sin",
        "month_cos",
    ]

    def _make_df(n: int, base_month: int) -> pd.DataFrame:
        rows = []
        for i in range(n):
            month = ((base_month + i - 1) % 12) + 1
            year = 2020 + (base_month + i - 1) // 12
            row = {
                "region": "Sahara" if i < n // 2 else "East China",
                "target_year": year,
                "target_month": month,
                "y_true": 3.0 + 0.5 * month + rng.normal(0, 0.5),
            }
            for f in features:
                row[f] = rng.normal(0, 1)
            # Add lag target for persistence
            row["evaporation_anomaly_lag_1"] = row["y_true"] + rng.normal(0, 0.3)
            rows.append(row)
        return pd.DataFrame(rows)

    train = _make_df(n_train, base_month=1)
    test = _make_df(n_test, base_month=n_train + 1)
    return train, test, features


# ---------------------------------------------------------------------------
# Climatology baseline
# ---------------------------------------------------------------------------


class TestClimatologyBaseline:
    def test_fit_and_predict_region_monthly(self) -> None:
        train, test, _ = make_tiny_train_test()
        model = ClimatologyBaseline(variant="region_monthly")
        model.fit(train)
        preds = model.predict(test)
        assert len(preds) == len(test)
        assert not np.any(np.isnan(preds))

    def test_fit_and_predict_global_monthly(self) -> None:
        train, test, _ = make_tiny_train_test()
        model = ClimatologyBaseline(variant="global_monthly")
        model.fit(train)
        preds = model.predict(test)
        assert len(preds) == len(test)
        assert not np.any(np.isnan(preds))

    def test_uses_train_only(self) -> None:
        """Climatology means should be computed from training data only."""
        train, test, _ = make_tiny_train_test(n_train=50, n_test=10)
        # Create train with y_true = month * 10, test with y_true = month * 0
        train_alt = train.copy()
        train_alt["y_true"] = train_alt["target_month"] * 10.0
        test_alt = test.copy()
        test_alt["y_true"] = test_alt["target_month"] * 0.0

        model = ClimatologyBaseline(variant="global_monthly")
        model.fit(train_alt)
        preds = model.predict(test_alt)

        # Predictions should be close to month*10 (from train), not month*0
        for i, row in test_alt.iterrows():
            expected = row["target_month"] * 10.0
            assert abs(preds[i] - expected) < 1.0  # tight because train data is clean

    def test_missing_target_month_raises(self) -> None:
        train, _, _ = make_tiny_train_test()
        train.drop(columns=["target_month"], inplace=True)
        with pytest.raises(ValueError, match="target_month"):
            ClimatologyBaseline().fit(train)

    def test_unseen_region_fallback(self) -> None:
        """A region not in training should fall back to global."""
        train, test, _ = make_tiny_train_test()
        train = train[train["region"] == "Sahara"]
        test = test.copy()
        test["region"] = "Mars"  # unseen
        model = ClimatologyBaseline(variant="region_monthly")
        model.fit(train)
        preds = model.predict(test)
        assert len(preds) == len(test)
        assert not np.any(np.isnan(preds))

    def test_invalid_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="variant"):
            ClimatologyBaseline(variant="daily")

    def test_get_model_name(self) -> None:
        assert ClimatologyBaseline(variant="region_monthly").get_model_name() == "climatology_region_monthly"
        assert ClimatologyBaseline(variant="global_monthly").get_model_name() == "climatology_global_monthly"


# ---------------------------------------------------------------------------
# Persistence baseline
# ---------------------------------------------------------------------------


class TestPersistenceBaseline:
    def test_fit_validates_lag_column(self) -> None:
        train, _, _ = make_tiny_train_test()
        model = PersistenceBaseline()
        model.fit(train)  # should not raise

    def test_fit_missing_lag_column_raises(self) -> None:
        train, _, _ = make_tiny_train_test()
        train.drop(columns=["evaporation_anomaly_lag_1"], inplace=True)
        with pytest.raises(ValueError, match="evaporation_anomaly_lag_1"):
            PersistenceBaseline().fit(train)

    def test_predict_equals_lag_1(self) -> None:
        _, test, _ = make_tiny_train_test(n_test=5)
        model = PersistenceBaseline()
        preds = model.predict(test)
        expected = test["evaporation_anomaly_lag_1"].to_numpy()
        np.testing.assert_array_almost_equal(preds, expected)

    def test_predict_length_matches_input(self) -> None:
        train, test, _ = make_tiny_train_test(n_test=15)
        model = PersistenceBaseline()
        model.fit(train)
        preds = model.predict(test)
        assert len(preds) == len(test)

    def test_no_nan_predictions(self) -> None:
        _, test, _ = make_tiny_train_test(n_test=10)
        model = PersistenceBaseline()
        preds = model.predict(test)
        assert not np.any(np.isnan(preds))

    def test_get_model_name(self) -> None:
        assert PersistenceBaseline().get_model_name() == "persistence"

    def test_predict_without_fit_still_works(self) -> None:
        """Fit is a no-op validation; predict uses lag column directly."""
        _, test, _ = make_tiny_train_test(n_test=5)
        model = PersistenceBaseline()
        preds = model.predict(test)
        assert len(preds) == len(test)


# ---------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------


class TestLinearRegressionModel:
    def test_fit_and_predict(self) -> None:
        train, test, features = make_tiny_train_test()
        model = LinearRegressionModel()
        model.fit(train, feature_columns=features)
        preds = model.predict(test)
        assert len(preds) == len(test)
        assert not np.any(np.isnan(preds))

    def test_coefficients_match_features(self) -> None:
        train, _, features = make_tiny_train_test()
        model = LinearRegressionModel()
        model.fit(train, feature_columns=features)
        assert len(model.coefficients) == len(features)

    def test_get_model_name(self) -> None:
        assert LinearRegressionModel().get_model_name() == "linear_regression"


# ---------------------------------------------------------------------------
# Tree models
# ---------------------------------------------------------------------------


class TestRandomForestModel:
    def test_fit_and_predict(self) -> None:
        train, test, features = make_tiny_train_test()
        model = RandomForestModel(n_estimators=20, random_state=42)
        model.fit(train, feature_columns=features)
        preds = model.predict(test)
        assert len(preds) == len(test)
        assert not np.any(np.isnan(preds))

    def test_feature_importances(self) -> None:
        train, _, features = make_tiny_train_test()
        model = RandomForestModel(n_estimators=20)
        model.fit(train, feature_columns=features)
        assert len(model.feature_importances) == len(features)

    def test_get_model_name(self) -> None:
        assert RandomForestModel().get_model_name() == "random_forest"


_XGB_SKIP = not _is_xgboost_available()


class TestXGBoostModel:
    @pytest.mark.skipif(_XGB_SKIP, reason="xgboost not installed or segfaults on Python 3.13")
    def test_fit_and_predict(self) -> None:
        train, test, features = make_tiny_train_test()
        model = XGBoostModel(n_estimators=10, random_state=42)
        model.fit(train, feature_columns=features)
        preds = model.predict(test)
        assert len(preds) == len(test)
        assert not np.any(np.isnan(preds))

    @pytest.mark.skipif(_XGB_SKIP, reason="xgboost not installed or segfaults on Python 3.13")
    def test_validation_set(self) -> None:
        train, test, features = make_tiny_train_test(n_train=30)
        val = test.iloc[:5].copy()
        model = XGBoostModel(n_estimators=10)
        model.fit(train, feature_columns=features, val_df=val)
        preds = model.predict(test)
        assert len(preds) == len(test)

    @pytest.mark.skipif(_XGB_SKIP, reason="xgboost segfaults on Python 3.13")
    def test_import_error_when_missing(self, monkeypatch) -> None:
        # Only runs when xgboost IS installed; simulates missing xgboost.
        import importlib
        monkeypatch.setitem(importlib.import_module("climatenet.models.tree_models").__dict__,
                            "_is_xgboost_available", lambda: False)
        with pytest.raises(ImportError, match="xgboost"):
            XGBoostModel()

    @pytest.mark.skipif(_XGB_SKIP, reason="xgboost segfaults on Python 3.13")
    def test_get_model_name(self) -> None:
        assert XGBoostModel().get_model_name() == "xgboost"


class TestLightGBMModel:
    @pytest.mark.skipif(not _is_lightgbm_available(), reason="lightgbm not installed")
    def test_fit_and_predict(self) -> None:
        train, test, features = make_tiny_train_test()
        model = LightGBMModel(n_estimators=10, random_state=42)
        model.fit(train, feature_columns=features)
        preds = model.predict(test)
        assert len(preds) == len(test)
        assert not np.any(np.isnan(preds))

    @pytest.mark.skipif(_is_lightgbm_available(), reason="lightgbm IS installed — skip the skip-if-missing test")
    def test_import_error_when_missing(self, monkeypatch) -> None:
        import importlib
        monkeypatch.setitem(importlib.import_module("climatenet.models.tree_models").__dict__,
                            "_is_lightgbm_available", lambda: False)
        with pytest.raises(ImportError, match="lightgbm"):
            LightGBMModel()

    def test_get_model_name(self) -> None:
        if _is_lightgbm_available():
            assert LightGBMModel().get_model_name() == "lightgbm"


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


class TestModelFactory:
    def test_create_climatology(self) -> None:
        m = create_model("climatology")
        assert isinstance(m, ClimatologyBaseline)

    def test_create_persistence(self) -> None:
        m = create_model("persistence")
        assert isinstance(m, PersistenceBaseline)

    def test_create_linear_regression(self) -> None:
        m = create_model("linear_regression")
        assert isinstance(m, LinearRegressionModel)

    def test_create_random_forest(self) -> None:
        m = create_model("random_forest", {"n_estimators": 50})
        assert isinstance(m, RandomForestModel)

    def test_create_xgboost(self) -> None:
        if not _XGB_SKIP:
            m = create_model("xgboost", {"n_estimators": 10})
            assert isinstance(m, XGBoostModel)

    def test_create_lightgbm(self) -> None:
        if _is_lightgbm_available():
            m = create_model("lightgbm", {"n_estimators": 10})
            assert isinstance(m, LightGBMModel)

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("neural_ode_transformer_v7")

    def test_create_tcn_raises_with_hint(self) -> None:
        with pytest.raises(ValueError, match="TCN requires 3D"):
            create_model("tcn")

    def test_list_available_models(self) -> None:
        models = list_available_models()
        assert "climatology" in models
        assert "persistence" in models
        assert "random_forest" in models
        assert "linear_regression" in models


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_climatology_save_load(self) -> None:
        train, test, _ = make_tiny_train_test()
        model = ClimatologyBaseline()
        model.fit(train)
        preds_before = model.predict(test)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clim.pkl"
            model.save(path)
            loaded = ClimatologyBaseline.load(path)
            preds_after = loaded.predict(test)
            np.testing.assert_array_equal(preds_before, preds_after)

    def test_persistence_save_load(self) -> None:
        train, test, _ = make_tiny_train_test()
        model = PersistenceBaseline()
        model.fit(train)
        preds_before = model.predict(test)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pers.pkl"
            model.save(path)
            loaded = PersistenceBaseline.load(path)
            preds_after = loaded.predict(test)
            np.testing.assert_array_equal(preds_before, preds_after)


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------


class TestInterfaceCompliance:
    def test_all_models_are_climate_models(self) -> None:
        for name in ["climatology", "persistence", "linear_regression", "random_forest"]:
            m = create_model(name, config={"n_estimators": 10} if "forest" in name else {})
            assert isinstance(m, ClimateModel), f"{name} is not a ClimateModel"
            assert hasattr(m, "fit")
            assert hasattr(m, "predict")
            assert hasattr(m, "get_model_name")
