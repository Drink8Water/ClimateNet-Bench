# Mini Benchmark

A lightweight, self-contained benchmark pipeline for fast iteration and
CI validation.  Runs on **synthetic data only** — no ERA5-Land download
required.

## Quick Start

```bash
# Install core dependencies
pip install -r requirements.txt
pip install -e .

# Optional: install LightGBM for full model zoo
pip install -r requirements-ml.txt

# Run the mini benchmark (~5 seconds)
python scripts/run_mini_benchmark.py
```

Outputs are written to `outputs/mini_benchmark/`:

```
outputs/mini_benchmark/
├── predictions_climatology.csv
├── predictions_persistence.csv
├── predictions_lightgbm.csv        # if LightGBM installed
├── metrics_climatology.json
├── metrics_persistence.json
├── metrics_lightgbm.json
└── leaderboard/
    └── v1_mini.csv
```

## What It Does

1. **Creates a synthetic dataset** — 3 regions × 4 grid points × 5 years of
   monthly data with realistic seasonal cycles.
2. **Engineers features** — anomalies (by region × month), 6 lag features,
   and physical variables.
3. **Fits event thresholds** on the training set (P10 for evaporation
   deficit).
4. **Runs three baselines**:
   - `climatology` — predicts zero anomaly (the "no-skill" reference).
   - `persistence` — predicts `y_{t-1}` (the "hard-to-beat" reference).
   - `lightgbm` — gradient-boosted tree model (optional; skipped if not
     installed).
5. **Evaluates each model** on a temporal holdout split (train 2020–2022,
   val 2023, test 2024).
6. **Writes a leaderboard** ranked by RMSE.

## Split Protocol

The mini benchmark uses a **temporal holdout** split because:

- It is the most intuitive "real-world" test — can the model predict
  **future** months it has never seen?
- It requires no spatial blocking, region metadata, or climate zone
  classification (keeps the synthetic data simple).
- It is strict enough to reveal overfitting but fast enough for CI.

## Event Metrics

The mini benchmark evaluates **evaporation deficit** detection.

| Metric | Meaning |
|--------|---------|
| `pod_evaporation_deficit` | Fraction of true deficit events detected |
| `far_evaporation_deficit` | Fraction of predicted events that were false alarms |
| `csi_evaporation_deficit` | Balanced score (penalises both misses and false alarms) |

### How Event Labels Are Constructed

1. **True event** = `y_true < P10` (the 10th percentile of the target
   variable on the training set, computed per calendar month).
2. **Predicted event** = `y_pred < P10` (same threshold, applied to model
   predictions).

Both labels use the **same** train-fitted thresholds.  This is the only
correct way to compare predicted events against observed events — the
threshold never sees test data.

### Why Not Compound Events?

The mini benchmark models predict a single target (`evaporation_anomaly`).
Compound events (e.g. *compound hot-dry* = high temperature **AND** low
soil moisture) require multi-target predictions and are therefore excluded
from the mini benchmark.  They will be evaluated in the full benchmark
when models that output multiple variables (temperature, soil moisture,
evaporation) are added.

## Leaderboard Columns

| Column | Description |
|--------|-------------|
| `rank` | Position by RMSE (1 = best) |
| `model` | Model name (`climatology`, `persistence`, `lightgbm`) |
| `split` | Split type (`temporal_holdout`) |
| `rmse` | Root Mean Squared Error (primary ranking metric) |
| `mae` | Mean Absolute Error |
| `acc` | R² (coefficient of determination) |
| `bias` | Mean signed error (positive = overprediction) |
| `skill_score` | 1 − RMSE_model / RMSE_climatology (when available) |
| `pod` | Probability of Detection for event type |
| `far` | False Alarm Ratio |
| `csi` | Critical Success Index |
| `intensity_bias` | Mean predicted / mean observed over event points |
| `n_samples` | Number of test samples |
| `n_events` | Number of observed events in the test set |

## Smoke vs Real Benchmark

| | Mini Benchmark | Full Benchmark |
|---|---|---|
| **Data** | Synthetic (generated on-the-fly) | ERA5-Land reanalysis |
| **Run time** | ~5 seconds | 10–60 minutes |
| **Purpose** | CI validation, quick iteration | Scientific results |
| **Reported in README?** | ❌ No (smoke test only) | ✅ Yes |

The mini benchmark leaderboard must **never** be presented as a real
benchmark result.  It uses synthetic data with known seasonal patterns
and is only useful for catching pipeline regressions.

## Adding a New Model

1. Implement `fit(train_df, val_df=None, target_col=None, feature_cols=None)`,
   `predict(test_df)`, `get_model_name()`, and `get_params()`.
2. Instantiate the model and add it to `models_to_run` in
   `scripts/run_mini_benchmark.py`.
3. Run `python scripts/run_mini_benchmark.py` to verify it appears in
   the leaderboard.
