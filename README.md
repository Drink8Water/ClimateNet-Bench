# Domain-informed Spatio-temporal Machine Learning for Land Evaporation Anomaly Analysis

Phase 1 MVP uses synthetic monthly climate-like data for two regions: Sahara and East China. The goal is to prove the end-to-end machine learning workflow before working with real ERA5-Land NetCDF data.

## Project Structure

```text
climate-ml-system/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
├── src/
│   ├── config.py
│   ├── make_sample_data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── plot_results.py
└── notebooks/
    └── 01_exploration.ipynb
```

## Setup

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Phase 1 Pipeline

```bash
python src/make_sample_data.py
python src/features.py
python src/train.py
python src/evaluate.py
python src/plot_results.py
```

## Run Phase 2 With ERA5-Land Data

First configure CDS API credentials. Create an account at the Copernicus Climate Data Store, accept the ERA5-Land monthly means dataset licence, then create `~/.cdsapirc`:

```text
url: https://cds.climate.copernicus.eu/api
key: <your-personal-api-key>
```

Install the Phase 2 dependencies:

```bash
pip install -r requirements.txt
```

Download a small safe ERA5-Land sample, preprocess it, and create features:

```bash
python src/download_era5.py
python src/preprocess_era5.py
python src/features.py --input data/processed/era5_climate_data.csv --output data/processed/features.csv
python src/train.py
python src/evaluate.py
python src/plot_results.py
```

The default ERA5 download requests only January 2019 for both regions. When you are ready for the full 2019-2023 monthly request:

```bash
python src/download_era5.py --full-request
```

## Run Phase 3 Validation and Explainability

After `data/processed/features.csv` has been created from either synthetic data or ERA5-Land data:

```bash
python src/train.py
python src/plot_results.py
python src/explain.py
```

Phase 3 adds:

- physically informed `saturation_vapor_pressure`
- random row split baseline
- spatial holdout by unique latitude-longitude grid cells
- cross-region transfer between Sahara and East China
- optional LightGBM if installed
- regional permutation importance
- SHAP summary plots for the best tree-based model

Spatial holdout is important because gridded climate records from the same grid cell are strongly related through time. Keeping whole grid cells out of training gives a more honest estimate of spatial generalization.

## Run Phase 4 Database, API, and Dashboard

Phase 4 adds a lightweight engineering layer around saved ML outputs:

- PostgreSQL database tables
- read-only FastAPI backend
- Streamlit dashboard
- local CSV/PNG fallback if the backend is unavailable

Create a local PostgreSQL database:

```bash
createdb climate_ml
```

Create `.env` from the example and edit the connection string:

```bash
cp .env.example .env
```

Example `.env`:

```text
DATABASE_URL=postgresql+psycopg2://climate_user:climate_password@localhost:5432/climate_ml
API_BASE_URL=http://127.0.0.1:8000
```

Load the saved CSV outputs into PostgreSQL:

```bash
python src/load_to_db.py
```

Run the read-only API:

```bash
uvicorn backend.main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Dashboard URL:

```text
http://localhost:8501
```

Verify API endpoints:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
curl http://127.0.0.1:8000/feature-importance
curl http://127.0.0.1:8000/timeseries
curl "http://127.0.0.1:8000/predictions?model=random_forest&limit=20"
curl http://127.0.0.1:8000/regional-summary
```

## ClimateNet Package Refactor

The project also includes a package-style experiment runner under `src/climatenet/`.
This keeps reusable logic importable while preserving the original scripts.

Install the local package in editable mode:

```bash
pip install -e .
```

Build features from YAML config:

```bash
python scripts/run_pipeline.py
```

Run a configuration-driven experiment:

```bash
python scripts/run_experiment.py
```

Default configs:

- `configs/data_config.yaml`
- `configs/model_config.yaml`
- `configs/experiment_config.yaml`

Experiment outputs are saved to:

```text
outputs/experiments/{experiment_id}/
```

Each experiment saves:

- `config_snapshot.yaml`
- `experiment.log`
- `metrics.json`
- `metrics.csv`
- `predictions.csv`
- `feature_importance.csv`
- `plots/validation_metrics.png`
- `plots/prediction_vs_actual.png`
- `plots/feature_importance.png`

Run tests:

```bash
pytest tests
```

ERA5-Land data can now also be handled through the package entry points:

```bash
python scripts/download_era5.py
python scripts/preprocess_era5.py
python scripts/run_pipeline.py --data-config configs/data_config.yaml
```

The older commands still work as compatibility wrappers:

```bash
python src/download_era5.py
python src/preprocess_era5.py
```

The database loader and dashboard now prefer the latest experiment directory:

```text
outputs/experiments/latest/
```

For database loading, you can override this with:

```text
EXPERIMENT_OUTPUT_DIR=outputs/experiments/my_experiment
```

## Run TCN Temporal Model

ClimateNet includes a PyTorch Temporal Convolutional Network (TCN) for next-month
evaporation anomaly prediction from historical climate anomaly windows.

Default ERA5-style temporal holdout:

```bash
python scripts/run_tcn_experiment.py
```

Default split:

```text
train: 2019-2022
test: 2023
sequence_length: 6 months
```

For the synthetic sample data currently generated by the project, use 2020 as
the holdout year:

```bash
python scripts/run_tcn_experiment.py \
  --train-start-year 2010 \
  --train-end-year 2019 \
  --test-year 2020 \
  --epochs 30
```

Fast smoke test:

```bash
python scripts/run_tcn_experiment.py \
  --train-start-year 2010 \
  --train-end-year 2019 \
  --test-year 2020 \
  --epochs 3
```

Outputs:

```text
outputs/experiments/latest/tcn/tcn_metrics.json
outputs/experiments/latest/tcn/tcn_predictions.csv
outputs/experiments/latest/tcn/training_loss.png
```

## Run Batch Experiments and Ablation Studies

The main experiment runner supports feature-set ablations, multiple models, and
multiple validation strategies from `configs/experiment_config.yaml`.

Run the configured batch experiment:

```bash
python scripts/run_experiment.py
```

The default feature sets are:

- `base`: latitude, longitude, month_sin, month_cos
- `meteorological`: anomaly variables plus wind_speed
- `physical`: dryness_proxy and saturation_vapor_pressure
- `full`: base + meteorological + physical

Default validation strategies:

- random split
- spatial holdout
- temporal holdout
- region transfer

Outputs:

```text
outputs/experiments/{experiment_id}/config.yaml
outputs/experiments/{experiment_id}/metrics.json
outputs/experiments/{experiment_id}/metrics.csv
outputs/experiments/{experiment_id}/predictions.csv
outputs/experiments/{experiment_id}/feature_importance.csv
outputs/experiments/{experiment_id}/ablation_study.csv
outputs/experiments/{experiment_id}/plots/model_comparison.png
outputs/experiments/{experiment_id}/plots/ablation_study.png
outputs/all_experiments.csv
```

To include TCN in the unified batch runner, add `tcn` to `batch_models` in
`configs/experiment_config.yaml`. TCN runs only for `temporal_holdout` because it
uses sliding historical windows.

## Expected Outputs

Processed data:

- `data/processed/sample_climate_data.csv`
- `data/processed/features.csv`

Model outputs:

- `data/outputs/model_metrics.json`
- `data/outputs/predictions.csv`
- `data/outputs/feature_importance.csv`
- `data/outputs/prediction_vs_actual.png`
- `data/outputs/feature_importance.png`

## How to Check the Pipeline

1. Confirm all expected files exist.
2. Open `model_metrics.json` and check that all three models have MAE, RMSE, and R2 values.
3. Open `predictions.csv` and check that it includes actual and predicted evaporation anomaly values.
4. Open the PNG plots to visually inspect model fit and feature importance.
