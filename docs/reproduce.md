# Reproducing Benchmark Results

This guide explains how to reproduce ClimateNet-Bench results from scratch.

## Prerequisites

- Python ≥ 3.10 (recommended: 3.11)
- pip + conda (recommended)
- Node.js ≥ 18 (for frontend)
- Git

## Step 1: Clone and Set Up

```bash
git clone https://github.com/Drink8Water/ClimateNet-Bench
cd ClimateNet-Bench

# Create conda environment
conda create -n climatenet-py311 python=3.11 -y
conda activate climatenet-py311

# Install Python dependencies
pip install -r requirements.txt
pip install -e .
pip install lightgbm  # optional
```

## Step 2: Run Tests

```bash
python -m pytest tests/ -v
# Expect: 198+ passed
```

## Step 3: Smoke Test (Synthetic Data, ~15 seconds)

```bash
# Build forecasting dataset from synthetic data
python scripts/build_forecasting_dataset.py --synthetic

# Run smoke test benchmark (4 models × 2 splits × 2 feature sets = 48 exps)
python scripts/run_benchmark.py \
    --config configs/benchmark/smoke_test.yaml \
    --output-dir outputs/benchmark

# Build leaderboard
python scripts/build_leaderboard.py \
    --experiments-dir outputs/benchmark/experiments \
    --output-dir outputs/benchmark

# Verify
python -c "
import pandas as pd
lb = pd.read_csv('outputs/benchmark/leaderboard.csv')
print(f'{len(lb)} rows, columns: {list(lb.columns)}')
print(f'Best RMSE: {lb[\"rmse\"].min():.4f} by {lb.loc[lb[\"rmse\"].idxmin(), \"model_name\"]}')
"
```

## Step 4: Full Benchmark (Real ERA5-Land Data)

First, configure CDS API access (see `docs/development_history.md` Phase 2).

```bash
# Download ERA5-Land data
python scripts/download_era5.py --full-request

# Preprocess
python scripts/preprocess_era5.py

# Build features
python scripts/run_pipeline.py --data-config configs/data_config.yaml

# Build forecasting dataset
python scripts/build_forecasting_dataset.py \
    --input data/processed/features.csv

# Run full benchmark (all models × all splits × all feature sets)
python scripts/run_benchmark.py \
    --config configs/benchmark/evap_anomaly_v1.yaml \
    --output-dir outputs/benchmark

# Build leaderboard
python scripts/build_leaderboard.py
```

## Step 5: Run Backend

```bash
PYTHONPATH=$(pwd) uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

API docs at http://127.0.0.1:8000/docs

## Step 6: Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard at http://localhost:5173

## Expected Outputs

After a successful benchmark run:

```
outputs/benchmark/
├── experiment_registry.json
├── leaderboard.csv
├── all_results.csv
├── split_difficulty_analysis.csv
├── uncertainty_calibration.csv
├── ablation_results.csv
├── splits/                           # 6+ split protocols
│   ├── random/
│   ├── spatial_block/
│   ├── temporal/
│   └── ...
└── experiments/                      # One per (model × split × feature_set)
    ├── EvapAnomaly-Forecast-v1_climatology_region_monthly_random_base/
    │   ├── config.yaml
    │   ├── metrics.json
    │   └── predictions.csv
    └── ...
```

## Quick Verification

1. `leaderboard.csv` — should have rows for every (model, split, feature_set) combination
2. `split_difficulty_analysis.csv` — `spatiotemporal` should have highest mean RMSE
3. `experiment_registry.json` — status should be "completed" for all experiments
4. API `/leaderboard?limit=5` should return JSON with ranked models
5. Frontend http://localhost:5173/leaderboard should show the leaderboard table

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: backend` | Set `PYTHONPATH=$(pwd)` before running uvicorn |
| xgboost segfault on Python 3.13 | Use Python 3.11 (conda env). xgboost C extension is unstable on 3.13. |
| xgboost ImportError (libomp) | `brew install libomp` and symlink into conda env |
| CDS API timeout | Retry; the CDS queue can take minutes to hours |
| PostgreSQL connection refused | Backend uses file-based storage by default; PostgreSQL is optional |
| Frontend blank page | Check that the API base URL in Vite proxy matches your backend port |
