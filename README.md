# ClimateNet-Bench

*A reproducible spatio-temporal machine learning benchmark for next-month land evaporation anomaly forecasting using ERA5-Land reanalysis data.*

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-198%20passed-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)](https://github.com/Drink8Water/ClimateNet-Bench)

---

> **Central Question:** Do machine learning models trained on climate data truly generalize across unseen grid cells, future years, and different climate regions?

---

## Why ClimateNet-Bench?

Most ML-for-climate papers evaluate models with random train/test splits. But climate data is spatio-temporally autocorrelated — nearby grid cells and adjacent months are highly correlated. Random splits leak information and produce overconfident performance estimates.

ClimateNet-Bench provides a rigorous benchmark that tests generalization under **six split protocols** of increasing difficulty, from random splits to joint spatial-temporal holdouts.

**This is a research benchmark, not an operational forecasting system.** It is designed for:
- ML researchers who want a well-defined climate prediction task with rigorous evaluation
- Climate scientists curious about ML model generalization
- Students building ML portfolio projects with real geospatial data

---

## Quick Start

```bash
git clone https://github.com/Drink8Water/ClimateNet-Bench
cd ClimateNet-Bench

# Setup
conda create -n climatenet-py311 python=3.11 -y
conda activate climatenet-py311
pip install -r requirements.txt
pip install -e .

# Run tests (198 passing)
python -m pytest tests/ -v

# Smoke test benchmark (~15 seconds)
python scripts/build_forecasting_dataset.py --synthetic
python scripts/run_benchmark.py --config configs/benchmark/smoke_test.yaml
python scripts/build_leaderboard.py

# Start dashboard
PYTHONPATH=$(pwd) uvicorn backend.main:app --port 8000 --reload &
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

---

## Benchmark Task

| Aspect | Specification |
|---|---|
| **Input** | 6 months of climate anomaly features (`X_{t-6 : t-1}`) |
| **Target** | `evaporation_anomaly` at month `t` |
| **Spatial unit** | Grid cell (0.1° × 0.1°) |
| **Temporal unit** | Monthly |
| **Data source** | ERA5-Land reanalysis |

### Input Features

| Category | Features |
|---|---|
| **Physical anomalies** | temperature_anomaly, precipitation_anomaly, radiation_anomaly, soil_moisture_anomaly |
| **Physical variables** | wind_speed, dryness_proxy, saturation_vapor_pressure |
| **Temporal encoding** | month_sin, month_cos |
| **Spatial encoding** | latitude, longitude |

---

## Dataset & Regions

Five geographically and climatically distinct regions from ERA5-Land:

| Region | Climate | Rationale |
|---|---|---|
| **Sahara** | Arid | Low evaporation, extreme temperatures |
| **East China** | Humid subtropical | Strong monsoonal seasonality |
| **Amazon** | Tropical rainforest | High evaporation, vegetation feedback |
| **Central Europe** | Temperate | Moderate signal, anthropogenic influence |
| **Western US** | Mediterranean / Semi-arid | Water-limited, interannual variability |

---

## Split Protocols

Models are tested under six protocols of increasing difficulty:

| # | Protocol | What It Tests |
|---|---|---|
| 1 | **Random Split** | Optimistic baseline (information leaks across space/time) |
| 2 | **Spatial Block Holdout** | Generalization to unseen locations |
| 3 | **Temporal Holdout** | Generalization to future years |
| 4 | **Region Transfer** | Generalization to unseen climate regions |
| 5 | **Climate-Zone Transfer** | Generalization across climate classifications |
| 6 | **Spatial-Temporal Holdout** | Joint generalization to new places AND future times |

---

## Models

| Model | Type | Implementation |
|---|---|---|
| **Climatology** | Baseline | Long-term monthly mean per grid cell |
| **Persistence** | Baseline | ŷ_t = y_{t-1} |
| **Linear Regression** | Linear | Ridge-regularized (scikit-learn) |
| **Random Forest** | Tree ensemble | scikit-learn |
| **XGBoost** | Gradient boosting | xgboost |
| **LightGBM** | Gradient boosting | lightgbm |
| **TCN** | Deep learning | PyTorch Temporal Convolutional Network |

---

## Metrics

### Primary
| Metric | Range | Target |
|---|---|---|
| MAE | [0, ∞) | Lower |
| RMSE | [0, ∞) | Lower |
| R² | (−∞, 1] | Higher |

### Skill Scores
| Metric | Formula |
|---|---|
| Skill vs Climatology | 1 − RMSE / RMSE_climatology |
| Skill vs Persistence | 1 − RMSE / RMSE_persistence |

### Generalization & Uncertainty
| Metric | Description |
|---|---|
| OOD Degradation | Performance drop from random split → strictest split |
| Conformal Coverage | Fraction of targets inside 90% prediction intervals |
| Average Interval Width | Mean width of prediction intervals |
| Physical Consistency Score | Fraction of features with physically plausible monotonic trends |

---

## Project Structure

```
ClimateNet-Bench/
├── src/climatenet/              # Core Python package
│   ├── benchmark/               # Region registry, split protocols, leaderboard
│   ├── data/                    # Data download, preprocessing, forecasting dataset
│   ├── evaluation/              # Metrics, skill scores, OOD, conformal, physical audit
│   ├── features/                # Feature engineering pipeline
│   ├── models/                  # Model zoo (base class, baselines, tree models, TCN)
│   └── training/                # Benchmark runner, experiment registry
├── configs/                     # YAML configs (benchmark, splits, models, data)
├── scripts/                     # CLI entry points
├── backend/                     # FastAPI read-only API (34 endpoints)
├── frontend/                    # Vue 3 benchmark portal (7 pages)
├── tests/                       # 198+ tests (pytest)
├── docs/                        # Documentation (8 docs)
├── outputs/benchmark/           # Benchmark outputs (leaderboard, experiments, splits)
├── .github/workflows/ci.yml     # CI pipeline (tests + smoke test)
├── CITATION.cff                 # Citation metadata
├── CONTRIBUTING.md              # Contribution guide
└── CHANGELOG.md                 # Version history
```

---

## Dashboard

```
http://localhost:5173
```

| Page | Description |
|---|---|
| **Overview** | Benchmark KPIs, task definition, regions, pipeline |
| **Leaderboard** | Ranked results table with filters |
| **Split Difficulty** | RMSE by split protocol (bar chart) |
| **Forecast Explorer** | Predicted vs actual scatter + time series |
| **Uncertainty** | Coverage vs interval width calibration |
| **Physical Audit** | Consistency score + regional sensitivity |
| **Spatial Diagnostics** | Grid-cell time series |

---

## API

```
http://127.0.0.1:8000/docs
```

34 endpoints covering benchmark info, leaderboard, experiments, predictions, intervals, uncertainty, physical consistency, and spatial data. See [docs/api.md](docs/api.md).

---

## Documentation

| Document | Content |
|---|---|
| [Task Definition](docs/task_definition.md) | Input/output specification, features, regions |
| [Benchmark Protocol](docs/benchmark_protocol.md) | Split protocols, models, metrics, anti-leakage rules |
| [Reproducing Results](docs/reproduce.md) | Step-by-step reproduction guide |
| [API Reference](docs/api.md) | Complete endpoint reference |
| [Uncertainty Quantification](docs/uncertainty.md) | Conformal prediction methods and interpretation |
| [Physical Consistency](docs/physical_consistency.md) | Model behaviour audit methodology |
| [Limitations](docs/limitations.md) | Data, task, model, and benchmark limitations |
| [Development History](docs/development_history.md) | Archive of phase-by-phase development |

---

## Citation

```bibtex
@software{climatenet_bench,
  author       = {ClimateNet-Bench Contributors},
  title        = {ClimateNet-Bench: A Spatio-Temporal ML Benchmark for
                  Land Evaporation Anomaly Forecasting},
  year         = {2026},
  url          = {https://github.com/Drink8Water/ClimateNet-Bench},
  note         = {Open-source benchmark project; 198 tests, 6 split protocols,
                  7 models, 5 benchmark regions}
}
```

See [CITATION.cff](CITATION.cff) for structured metadata.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, coding style, and how to add new models, split protocols, or benchmark regions.

---

## License

MIT License.

---

*This benchmark does not replace physical climate models. It evaluates the generalization capability of statistical ML models on a well-defined climate prediction task.*
