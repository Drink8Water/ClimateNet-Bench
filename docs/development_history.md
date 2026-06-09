# Development History

This document records the phase-by-phase development history of ClimateNet-Bench.
It is preserved for reference and is not required reading for new users.

---

## Phase 1 — Synthetic Data MVP

Phase 1 used synthetic monthly climate-like data for two regions (Sahara and East China) to prove the end-to-end ML workflow before working with real ERA5-Land NetCDF data.

**Scripts:**
```bash
python src/make_sample_data.py
python src/features.py
python src/train.py
python src/evaluate.py
python src/plot_results.py
```

**Outputs:**
- `data/processed/sample_climate_data.csv`
- `data/processed/features.csv`
- `data/outputs/model_metrics.json`
- `data/outputs/predictions.csv`
- `data/outputs/feature_importance.csv`
- `data/outputs/prediction_vs_actual.png`
- `data/outputs/feature_importance.png`

---

## Phase 2 — ERA5-Land Data Integration

Phase 2 integrated real ERA5-Land monthly means via the Copernicus Climate Data Store (CDS) API.

**Setup:**
1. Create a CDS account and accept the ERA5-Land monthly means dataset licence.
2. Create `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: <your-personal-api-key>
```

**Scripts:**
```bash
python src/download_era5.py
python src/preprocess_era5.py
python src/features.py --input data/processed/era5_climate_data.csv --output data/processed/features.csv
python src/train.py
python src/evaluate.py
python src/plot_results.py
```

The default ERA5 download requests only January 2019 for both regions. For the full 2019–2023 monthly request:
```bash
python src/download_era5.py --full-request
```

---

## Phase 3 — Validation and Explainability

Phase 3 added:
- Physically informed `saturation_vapor_pressure`
- Random row split baseline
- Spatial holdout by unique latitude-longitude grid cells
- Cross-region transfer between Sahara and East China
- Optional LightGBM if installed
- Regional permutation importance
- SHAP summary plots for the best tree-based model

**Scripts:**
```bash
python src/train.py
python src/plot_results.py
python src/explain.py
```

---

## Phase 4 — Database, API, and Dashboard

Phase 4 added a lightweight engineering layer:
- PostgreSQL database tables
- Read-only FastAPI backend
- Streamlit dashboard
- Local CSV/PNG fallback if the backend is unavailable

**Setup:**
```bash
createdb climate_ml
cp .env.example .env
# Edit .env with your database connection string
python src/load_to_db.py
uvicorn backend.main:app --reload
streamlit run dashboard/app.py
```

---

## ClimateNet Package Refactor

The project was refactored into a package-style experiment runner under `src/climatenet/` to keep reusable logic importable.

```bash
pip install -e .
python scripts/run_pipeline.py
python scripts/run_experiment.py
```

**Config files:**
- `configs/data_config.yaml`
- `configs/model_config.yaml`
- `configs/experiment_config.yaml`

---

## TCN Temporal Model

A PyTorch Temporal Convolutional Network (TCN) was added for next-month evaporation anomaly prediction from historical climate anomaly windows.

```bash
python scripts/run_tcn_experiment.py
```

Default split: train 2019–2022, test 2023, sequence_length 6 months.

---

## Batch Experiments and Ablation Studies

The experiment runner supports feature-set ablations, multiple models, and multiple validation strategies.

**Feature sets:**
- `base`: latitude, longitude, month_sin, month_cos
- `meteorological`: anomaly variables plus wind_speed
- `physical`: dryness_proxy and saturation_vapor_pressure
- `full`: base + meteorological + physical

**Validation strategies:**
- random split
- spatial holdout
- temporal holdout
- region transfer

---

## Current State (June 2026)

The project is being refactored into **ClimateNet-Bench**, a reproducible spatio-temporal ML benchmark.
See the main [README.md](../README.md) for current usage.
