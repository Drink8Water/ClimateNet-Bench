"""Tests for baseline models in climatenet.models.baselines.

Covers:

- ClimatologyBaseline: predicts zero for anomaly targets.
- PersistenceBaseline: uses lag column, raises on missing lag.
- LightGBMBaseline: trains and predicts (skipped when lightgbm absent).
- model_registry: registers, lists, and creates models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climatenet.models.baselines import (
    ClimatologyBaseline,
    LightGBMBaseline,
    PersistenceBaseline,
)
from climatenet.models.model_registry import (
    get_model,
    is_registered,
    list_models,
    register_model,
)


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_anomaly_data(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """Build a small DataFrame with anomaly target and lag features."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_samples):
        month = (i % 12) + 1
        year = 2020 + i // 12
        region = "Sahara" if i < n_samples // 2 else "East China"
        zone = "arid" if region == "Sahara" else "monsoon"
        target = rng.normal(0, 1.0)  # anomaly ~ N(0, 1)
        rows.append(
            {
                "sample_id": f"{region}_{i:04d}",
                "year": year,
                "month": month,
                "region": region,
                "climate_zone": zone,
                "lat": 25.0,
                "lon": 10.0,
                "y_true": round(float(target), 4),
                "y_true_lag1": round(float(target + rng.normal(0, 0.5)), 4),
                "feature_a": rng.normal(0, 1),
                "feature_b": rng.normal(0, 1),
                "feature_c": rng.normal(0, 1),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ClimatologyBaseline
# ---------------------------------------------------------------------------


class TestClimatologyBaseline:
    def test_predicts_zero_for_anomaly_default(self) -> None:
        train = _make_anomaly_data(60)
        test = _make_anomaly_data(20, seed=99)

        model = ClimatologyBaseline(target_col="y_true", predict_zero=True)
        model.fit(train, target_col="y_true")
        preds = model.predict(test)

        assert len(preds) == len(test)
        np.testing.assert_array_equal(preds, np.zeros(len(test)))

    def test_predicts_zero_for_anomaly_explicit(self) -> None:
        train = _make_anomaly_data(60)
        test = _make_anomaly_data(20, seed=99)

        model = ClimatologyBaseline(target_col="y_true", predict_zero=True)
        model.fit(train)
        preds = model.predict(test)

        assert len(preds) == len(test)
        assert (preds == 0.0).all()

    def test_get_model_name(self) -> None:
        model = ClimatologyBaseline()
        assert model.get_model_name() == "climatology"

    def test_get_params(self) -> None:
        train = _make_anomaly_data(60)
        model = ClimatologyBaseline(target_col="y_true", predict_zero=True)
        model.fit(train)
        params = model.get_params()
        assert params["model_name"] == "climatology"
        assert params["predict_zero"] is True
        assert params["target_col"] == "y_true"

    def test_fit_uses_train_only(self) -> None:
        """fit() should never reference val_df or test_df."""
        train = _make_anomaly_data(50)
        model = ClimatologyBaseline(predict_zero=True)
        # Should not raise even if val_df has wrong columns
        model.fit(train, val_df=pd.DataFrame({"bogus": [1, 2, 3]}))
        preds = model.predict(train)
        assert (preds == 0.0).all()

    def test_raises_when_target_col_missing(self) -> None:
        train = _make_anomaly_data(10)
        train = train.drop(columns=["y_true"])
        model = ClimatologyBaseline(target_col="y_true")
        with pytest.raises(ValueError, match="not found in train_df"):
            model.fit(train)

    def test_learns_monthly_mean_when_predict_zero_false(self) -> None:
        """When predict_zero=False, should predict monthly means."""
        rows = []
        for year in [2020, 2021, 2022]:
            for month in range(1, 13):
                rows.append(
                    {
                        "sample_id": f"s_{year}_{month}",
                        "year": year,
                        "month": month,
                        "region": "Test",
                        "climate_zone": "test",
                        "lat": 0.0,
                        "lon": 0.0,
                        "y_true": float(month * 10),  # month 1=10, month 7=70, etc.
                    }
                )
        train = pd.DataFrame(rows)
        model = ClimatologyBaseline(target_col="y_true", predict_zero=False)
        model.fit(train)
        preds = model.predict(train)
        # Each month's prediction should be close to month*10
        for i, row in train.iterrows():
            expected = row["month"] * 10.0
            assert abs(preds[i] - expected) < 0.01


# ---------------------------------------------------------------------------
# PersistenceBaseline
# ---------------------------------------------------------------------------


class TestPersistenceBaseline:
    def test_predicts_lag_values(self) -> None:
        train = _make_anomaly_data(60)
        test = _make_anomaly_data(20, seed=99)

        model = PersistenceBaseline(target_col="y_true", lag_col="y_true_lag1")
        model.fit(train)
        preds = model.predict(test)

        expected = test["y_true_lag1"].to_numpy(dtype=np.float64)
        np.testing.assert_array_almost_equal(preds, expected)

    def test_raises_when_lag_column_missing(self) -> None:
        train = _make_anomaly_data(10)
        train = train.drop(columns=["y_true_lag1"])

        model = PersistenceBaseline(target_col="y_true", lag_col="y_true_lag1")
        with pytest.raises(ValueError, match="requires lag column"):
            model.fit(train)

    def test_get_model_name(self) -> None:
        model = PersistenceBaseline()
        assert model.get_model_name() == "persistence"

    def test_get_params(self) -> None:
        model = PersistenceBaseline(target_col="my_target", lag_col="my_target_lag1")
        params = model.get_params()
        assert params["model_name"] == "persistence"
        assert params["target_col"] == "my_target"
        assert params["lag_col"] == "my_target_lag1"

    def test_auto_derives_lag_col(self) -> None:
        model = PersistenceBaseline(target_col="evaporation_anomaly")
        # When fit with a different target_col, lag_col should update
        train = _make_anomaly_data(10)
        train["evaporation_anomaly_lag1"] = train["y_true_lag1"]
        model.fit(train, target_col="evaporation_anomaly")
        assert model._lag_col == "evaporation_anomaly_lag1"

    def test_predict_length_matches_input(self) -> None:
        train = _make_anomaly_data(50)
        test = _make_anomaly_data(30, seed=77)

        model = PersistenceBaseline(target_col="y_true", lag_col="y_true_lag1")
        model.fit(train)
        preds = model.predict(test)
        assert len(preds) == len(test)

    def test_no_nan_in_predictions(self) -> None:
        train = _make_anomaly_data(60)
        test = _make_anomaly_data(20, seed=99)

        model = PersistenceBaseline(target_col="y_true", lag_col="y_true_lag1")
        model.fit(train)
        preds = model.predict(test)
        assert not np.any(np.isnan(preds))


# ---------------------------------------------------------------------------
# LightGBMBaseline
# ---------------------------------------------------------------------------


class TestLightGBMBaseline:
    def test_fit_and_predict(self) -> None:
        pytest.importorskip("lightgbm")
        train = _make_anomaly_data(80)
        test = _make_anomaly_data(20, seed=99)

        model = LightGBMBaseline(n_estimators=20, learning_rate=0.05, random_state=42)
        model.fit(
            train,
            target_col="y_true",
            feature_cols=["y_true_lag1", "feature_a", "feature_b", "feature_c"],
        )
        preds = model.predict(test)
        assert len(preds) == len(test)
        assert not np.any(np.isnan(preds))

    def test_get_model_name(self) -> None:
        pytest.importorskip("lightgbm")
        model = LightGBMBaseline()
        assert model.get_model_name() == "lightgbm"

    def test_get_params_returns_feature_cols(self) -> None:
        pytest.importorskip("lightgbm")
        train = _make_anomaly_data(50)
        fc = ["y_true_lag1", "feature_a", "feature_b"]

        model = LightGBMBaseline(n_estimators=20)
        model.fit(train, target_col="y_true", feature_cols=fc)
        params = model.get_params()
        assert params["model_name"] == "lightgbm"
        assert params["n_estimators"] == 20
        assert params["feature_cols"] == fc

    def test_raises_when_feature_cols_empty(self) -> None:
        pytest.importorskip("lightgbm")
        train = _make_anomaly_data(20)
        model = LightGBMBaseline()
        with pytest.raises(ValueError, match="feature_cols must be provided"):
            model.fit(train, target_col="y_true", feature_cols=[])

    def test_raises_when_feature_missing_from_train(self) -> None:
        pytest.importorskip("lightgbm")
        train = _make_anomaly_data(20)
        model = LightGBMBaseline()
        with pytest.raises(ValueError, match="not found in train_df"):
            model.fit(train, target_col="y_true", feature_cols=["nonexistent_feature"])

    def test_uses_validation_set(self) -> None:
        pytest.importorskip("lightgbm")
        train = _make_anomaly_data(60)
        val = _make_anomaly_data(20, seed=77)

        model = LightGBMBaseline(n_estimators=20)
        model.fit(
            train,
            val_df=val,
            target_col="y_true",
            feature_cols=["y_true_lag1", "feature_a", "feature_b", "feature_c"],
        )
        # Should not raise
        preds = model.predict(train.head(5))
        assert len(preds) == 5


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_auto_registers_baselines(self) -> None:
        models = list_models()
        assert "climatology" in models
        assert "persistence" in models
        assert "lightgbm" in models

    def test_is_registered(self) -> None:
        assert is_registered("climatology") is True
        assert is_registered("unknown_model_xyz") is False

    def test_get_model_creates_instance(self) -> None:
        m = get_model("climatology", target_col="y_true", predict_zero=True)
        assert isinstance(m, ClimatologyBaseline)

        m2 = get_model("persistence", target_col="y_true")
        assert isinstance(m2, PersistenceBaseline)

    def test_register_and_get_custom_model(self) -> None:
        class DummyModel:
            def fit(self, train_df, **kwargs):
                return self

            def predict(self, test_df):
                return np.zeros(len(test_df))

            def get_model_name(self):
                return "dummy"

            def get_params(self):
                return {"model_name": "dummy"}

        register_model("dummy", DummyModel)
        assert is_registered("dummy")
        m = get_model("dummy")
        assert isinstance(m, DummyModel)

        # Clean up
        from climatenet.models.model_registry import _registry

        _registry.pop("dummy", None)

    def test_get_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("not_a_real_model_42")
