# Benchmark Protocol

## Overview

ClimateNet-Bench evaluates models under multiple data split protocols to test generalization across space, time, and climate zones.

## Split Protocols

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

All experiments use **random seed 42** for reproducibility.

## Experiment Output Structure

```
outputs/benchmark/
├── dataset_card.md
├── benchmark_metadata.json
├── all_results.csv
├── leaderboard.csv
├── split_difficulty_analysis.csv
├── uncertainty_calibration.csv
├── ablation_results.csv
├── physical_consistency_report.md
└── experiments/{experiment_id}/
    ├── config.yaml
    ├── metrics.json
    ├── predictions.csv
    ├── intervals.csv
    ├── feature_importance.csv
    └── plots/
```

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
