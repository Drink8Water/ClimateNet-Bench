-- PostgreSQL schema for the Phase 4 climate ML engineering layer.
-- The backend is read-only; this schema is populated by src/load_to_db.py.

CREATE TABLE IF NOT EXISTS climate_features (
    id BIGSERIAL PRIMARY KEY,
    region TEXT NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    temperature DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    radiation DOUBLE PRECISION,
    soil_moisture DOUBLE PRECISION,
    u_wind DOUBLE PRECISION,
    v_wind DOUBLE PRECISION,
    evaporation DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    month_sin DOUBLE PRECISION,
    month_cos DOUBLE PRECISION,
    dryness_proxy DOUBLE PRECISION,
    saturation_vapor_pressure DOUBLE PRECISION,
    temperature_climatology DOUBLE PRECISION,
    temperature_anomaly DOUBLE PRECISION,
    precipitation_climatology DOUBLE PRECISION,
    precipitation_anomaly DOUBLE PRECISION,
    radiation_climatology DOUBLE PRECISION,
    radiation_anomaly DOUBLE PRECISION,
    soil_moisture_climatology DOUBLE PRECISION,
    soil_moisture_anomaly DOUBLE PRECISION,
    evaporation_climatology DOUBLE PRECISION,
    evaporation_anomaly DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id BIGSERIAL PRIMARY KEY,
    region TEXT NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    validation_strategy TEXT NOT NULL,
    train_region TEXT,
    test_region TEXT,
    model TEXT NOT NULL,
    actual DOUBLE PRECISION,
    prediction DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS feature_importance (
    id BIGSERIAL PRIMARY KEY,
    region TEXT,
    validation_strategy TEXT,
    train_region TEXT,
    test_region TEXT,
    model TEXT,
    feature TEXT NOT NULL,
    importance DOUBLE PRECISION,
    importance_type TEXT,
    importance_mean DOUBLE PRECISION,
    importance_std DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id BIGSERIAL PRIMARY KEY,
    validation_strategy TEXT NOT NULL,
    train_region TEXT,
    test_region TEXT,
    model TEXT NOT NULL,
    n_train INTEGER,
    n_test INTEGER,
    mae DOUBLE PRECISION,
    rmse DOUBLE PRECISION,
    r2 DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_climate_features_region_time
    ON climate_features (region, year, month);

CREATE INDEX IF NOT EXISTS idx_predictions_strategy_model
    ON model_predictions (validation_strategy, model);

CREATE INDEX IF NOT EXISTS idx_feature_importance_region_feature
    ON feature_importance (region, feature);

CREATE INDEX IF NOT EXISTS idx_metrics_strategy_model
    ON model_metrics (validation_strategy, model);
