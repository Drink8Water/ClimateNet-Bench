# Benchmark Protocol

## Overview

ClimateNet-Bench evaluates models under multiple data split protocols to test generalization across space, time, and climate zones.

## Split Protocols

The benchmark runner executes only the entries listed in
`config["split_protocols"]`. Configuration names are mapped explicitly:
`random_split` → `random`, `spatial_block_holdout` → `spatial_block`,
`temporal_holdout` → `temporal`, and `spatial_temporal_holdout` →
`spatiotemporal`. Unknown names are configuration errors.

Feature sets declare static features (`latitude`, `longitude`, `month_sin`,
`month_cos`) by their column names. Other feature names are time-series
features and expand to all lag columns implied by `input_window`; for example,
`temperature_anomaly` with a six-month window requires
`temperature_anomaly_lag_1` through `temperature_anomaly_lag_6`. Missing
requested columns fail schema validation before training.

`target` identifies the forecast task (`evaporation_anomaly`), while
`target_column` identifies the label column in the forecasting samples
(`y_true`).

For formal benchmark configs, preprocessing is performed independently for
each split: fit on train, then transform train/validation/test with the frozen
parameters. Region-month climatologies, anomaly baselines, feature
standardization parameters, and monthly event thresholds are train-only.
During region transfer, a region absent from train uses the train-wide monthly
climatology; if that calendar month is also absent, it uses the train-wide
global mean. The fitted strategy and parameter provenance are stored in each
experiment's registry metadata. Precomputed-feature mode remains available
for compatibility when `preprocessing.train_only` is false or omitted.

Each `preprocessing/<split-id>/` directory contains
`preprocessing_metadata.json`. Formal
train-only runs also export the fitted region-month and global-month
climatology tables. Metadata records training regions/years, variables,
fallback strategy and counts, standardization/event-threshold provenance, and
explicitly states that validation and test were not used for fitting. The
legacy `features.anomalies` full-table transform remains available only for
exploration and is not called by the formal runner.

### 1. Random Split (Baseline)

Samples are randomly assigned to train/val/test (70/15/15). This is the weakest test of generalization and serves as an upper bound on expected performance. Random splitting leaks spatial and temporal information because nearby grid cells and adjacent months are highly correlated.

**Use this split for:** debugging, hyperparameter tuning, and establishing an optimistic performance ceiling.

### 2. Spatial Block Holdout

Entire grid cells are held out from training. All time steps for a given (latitude, longitude) pair go to exactly one split.

**Implementation:** Group samples by unique (latitude, longitude), then randomly split groups 70/15/15.

**Tests:** Can the model generalize to completely unseen locations?

### 3. Temporal Holdout

The earliest years form the training set, and the most recent years form the test set. No future information leaks into training.

**Implementation:** Split by year. Example: train on 2010–2019, test on 2020–2023.

**Tests:** Can the model generalize to future climate conditions?

### 4. Region Transfer

Train on one or more regions, test on a completely disjoint region.

**Implementation:** Hold out entire regions from training. Example: train on Sahara, East China, Amazon, Central Europe; test on Western US.

**Tests:** Can the model transfer across different climate regimes?

### 5. Climate-Zone Transfer

Train on regions from one climate zone (e.g., arid), test on regions from a different climate zone (e.g., tropical).

**Implementation:** Map each region to a climate zone (Köppen classification), then split by zone.

**Tests:** Can the model generalize across fundamentally different climate physics?

### 6. Spatial-Temporal Holdout

The strictest protocol. Hold out both specific grid cells AND future years simultaneously.

**Implementation:** Hold out a spatial block AND the most recent years from training. Test on the intersection.

**Tests:** The hardest generalization test — new locations in new climate conditions.

## Models

| Model | Type | Description |
|---|---|---|
| Climatology Baseline | Baseline | Predicts the long-term monthly mean for each grid cell |
| Persistence Baseline | Baseline | Predicts that next month's anomaly equals this month's anomaly |
| Linear Regression | Linear | Ridge-regularized linear model |
| Random Forest | Tree Ensemble | scikit-learn RandomForestRegressor |
| XGBoost | Gradient Boosting | xgboost.XGBRegressor |
| LightGBM | Gradient Boosting | lightgbm.LGBMRegressor |
| TCN | Deep Learning | PyTorch Temporal Convolutional Network |

## Metrics

### Primary Metrics

| Metric | Description | Range |
|---|---|---|
| MAE | Mean Absolute Error | [0, ∞), lower is better |
| RMSE | Root Mean Squared Error | [0, ∞), lower is better |
| R² | Coefficient of Determination | (-∞, 1], higher is better |

### Skill Scores

| Metric | Formula | Description |
|---|---|---|
| Skill vs Climatology | 1 − RMSE_model / RMSE_climatology | Improvement over climatological mean |
| Skill vs Persistence | 1 − RMSE_model / RMSE_persistence | Improvement over persistence forecast |

### OOD (Out-of-Distribution) Metrics

| Metric | Description |
|---|---|
| OOD Degradation | ΔRMSE = RMSE_strict_split − RMSE_random_split | Performance drop under strict splits |

### Uncertainty Metrics

| Metric | Description |
|---|---|
| Conformal Coverage | Fraction of test targets falling within prediction intervals |
| Average Interval Width | Mean width of prediction intervals |

## Random Seed

The configured `random_seed` controls split generation, Python/NumPy state,
and supported sklearn, XGBoost, and LightGBM model seeds. The resolved value
is recorded in both the config snapshot and run metadata.

## Experiment Output Structure

```
outputs/benchmark_runs/
└── <benchmark>-<data-source>-<utc>-<hash>-<suffix>/
    ├── config_resolved.yaml
    ├── run_metadata.json
    ├── experiment_registry.json
    ├── splits/
    ├── preprocessing/<split-id>/
    ├── metrics/<experiment-id>.json
    ├── predictions/<experiment-id>.csv
    ├── leaderboard.csv
    ├── summary.json
    └── experiments/<experiment-id>/
        ├── config.yaml
        ├── metrics.json
        └── predictions.csv
```

The final `experiments/` subtree is retained as a compatibility view for
existing result readers. Root-level metrics and predictions are the canonical
audit artifacts. A failed task writes its error and bounded traceback under
`metrics/`; completed tasks and the registry are saved incrementally.

## ERA5-Land Readiness and Dry-Run

The real-data entry path is:

```text
CDS monthly ERA5-Land NetCDF
→ scripts/audit_era5.py
→ bounded NetCDF subset and unit conversion
→ row-wise physical features (no full-table anomaly)
→ formal benchmark split
→ train-only climatology/anomaly/standardization
→ lag samples
→ train/evaluate
```

Use `configs/benchmark/era5_land_dry_run.yaml` with
`scripts/run_era5_dry_run.py`. The dry-run uses only a small configured time
window and bounding box and is protected by explicit file-size, grid-cell,
month, split, model and tree-count limits. It cannot be interpreted as the
full five-region benchmark and its scores are not scientific conclusions.

ERA5-Land monthly averaged accumulation fields are converted as follows:

- `t2m`: K to degrees Celsius.
- `tp`: monthly mean daily accumulation in metres to mm/month.
- `ssrd`: monthly mean daily accumulation in J m-2 to MJ m-2/month.
- `e`: normally negative upward evaporation accumulation, inverted to
  positive mm/month.
- soil moisture and wind components retain their source units.

The audit must pass before a benchmark run. It checks required variables,
unit metadata, time and coordinate bounds, grid size, region mapping,
optional climate-zone behavior, NaN/Inf, min/mean/max, evaporation sign,
continuous months, per-region/month counts and consecutive lag availability.
Formal configs point to `era5_physical_features.csv`; the repository
synthetic `features.csv` is never a valid substitute.

For the external-disk full-data preflight, use
`configs/data_config_external_full.yaml`. It names the Sahara and East China
2019--2023 NetCDF files explicitly; directory globbing is intentionally
avoided because the raw directory can also contain a shorter dry-run file.
Run `scripts/audit_era5.py --full-preflight` before conversion and
`scripts/audit_processed_era5.py` afterward. The first report estimates lag
yield and disk usage and treats all-variable land/sea mask cells as excluded
source cells, while partial missingness or Inf is blocking. The second report
checks the complete CSV row count, region/month grid stability, duplicate
`region/year/month/latitude/longitude` keys and non-finite values in bounded
chunks.

### ERA5-Land Sahara-East China v1

`configs/benchmark/era5_land_v1.yaml` is the first bounded formal matrix:
linear regression and LightGBM, base/full features, and random, temporal, and
spatial-block splits (12 tasks). TCN, XGBoost, random forest, region transfer,
climate-zone transfer, and spatiotemporal holdout are intentionally excluded.

The unbounded `dryness_proxy` ratio remains in the physical-feature CSV for
audit, but is not a formal v1 model input. Models use
`dryness_proxy_log1p = log1p(radiation / (precipitation + 1e-6))`, a row-wise
transformation that has no fitted statistics and cannot leak held-out data.
Negative converted evaporation values are retained unchanged and counted in
the physical-feature audit; they may represent condensation/dew or ERA5
flux-sign and accumulation nuances and must not be silently clipped.

The v1 artifact policy writes one canonical prediction CSV per task and omits
the duplicate compatibility prediction copy. Per-task metrics, preprocessing
metadata, leaderboard, summary, resolved config, and experiment registry are
still retained.

The bounded multi-seed extension is configured in
`configs/benchmark/era5_land_v1_multiseed.yaml`. It evaluates only the full
feature set for seeds 42, 123, and 2026, giving 18 aggregate tasks. Seed 42
reuses the audited full-feature rows from the original v1 run; seeds 123 and
2026 receive independent run IDs. The summary reports across-seed mean/sample
standard deviation, best/worst seed, matched OOD degradation, and regional
RMSE consistency. It must not aggregate unrelated historical runs.

## Physical Consistency Audit

Beyond statistical metrics, predictions are audited for physical plausibility:

1. **Energy conservation:** Evaporation requires energy. Predicted evaporation should not exceed available net radiation.
2. **Water limitation:** In water-limited regions, evaporation cannot exceed precipitation plus soil moisture drawdown.
3. **Sign consistency:** In arid regions, a positive temperature anomaly combined with a negative precipitation anomaly should not lead to a large positive evaporation anomaly.
4. **Spatial smoothness:** Predicted evaporation fields should exhibit realistic spatial autocorrelation.

## Anti-Data-Leakage Rules

1. **No future information:** All features at time `t` must be computable from data available before month `t`.
2. **Spatial split by grid cell:** Spatial splits must group by grid cell, not by random row.
3. **Temporal split by year:** Temporal splits must not allow any month from a test year into training.
4. **Region split by region label:** Region transfer must use disjoint sets of regions.
5. **No target leakage:** The target variable (evaporation) must never appear as an input feature, even in lagged form.
