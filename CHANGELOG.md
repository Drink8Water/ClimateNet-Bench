# Changelog

## v0.3.0 (2026-06-09) — Benchmark Portal Release

### Added
- **Benchmark runner** (`scripts/run_benchmark.py`) — reproducible CLI workflow
- **Leaderboard generator** (`scripts/build_leaderboard.py`) — ranked results tables
- **Experiment registry** — track experiment metadata + status
- **FastAPI benchmark API** — 34 endpoints exposing leaderboard, splits, uncertainty, physical audit
- **Vue 3 Benchmark Portal** — 7-page dashboard (overview, leaderboard, split difficulty, forecast explorer, uncertainty, physical audit, spatial diagnostics)
- **Physical consistency audit** — feature sensitivity curves, monotonic trend checks, regional breakdowns
- **Conformal prediction** — split-conformal intervals with coverage/width evaluation
- **CI pipeline** (`.github/workflows/ci.yml`) — tests + smoke test on push/PR
- **Project documentation** — CONTRIBUTING.md, CITATION.cff, docs/api.md

### Changed
- Python version: 3.13 → 3.11 (xgboost segfault mitigation)
- `xgboost` pinned to `>=2.1,<2.2`
- Backend reads from `outputs/benchmark/` instead of `outputs/experiments/`
- Frontend refactored from generic dashboard to benchmark portal
- README rewritten as benchmark-style project overview

---

## v0.2.0 (2026-05) — Model Zoo & Split Protocols

### Added
- **Model zoo**: `ClimateModel` base class, climatology, persistence, linear regression, random forest, XGBoost, LightGBM, TCN
- **Split protocols**: random, spatial_block, temporal, region_transfer, climate_zone_transfer, spatiotemporal
- **Forecasting dataset constructor**: lagged features, sliding windows, anti-leakage validation
- **Evaluation metrics**: MAE, RMSE, R², skill scores, OOD degradation

---

## v0.1.0 (2026-04) — Foundation

### Added
- **Region registry**: 5 fixed benchmark regions (Sahara, East China, Amazon, Central Europe, Western US)
- **ERA5-Land data pipeline**: download, preprocess, feature engineering
- **Synthetic data generator** for smoke testing
- **Initial FastAPI + Streamlit dashboard**
- **Project README** with task definition

---

## Roadmap — Benchmark V1

- [x] Milestone 1: README & project positioning
- [x] Milestone 2: Region registry & benchmark config
- [x] Milestone 3: Forecasting dataset constructor
- [x] Milestone 4: Split protocols
- [x] Milestone 5: Baselines & model zoo
- [x] Milestone 6: Evaluation metrics
- [x] Milestone 7: Conformal prediction
- [x] Milestone 8: Physical consistency audit
- [x] Milestone 9: Benchmark runner & leaderboard
- [x] Milestone 10: Benchmark API
- [x] Milestone 11: Benchmark portal (frontend)
- [x] Milestone 12: Tests, CI, open-source hygiene

### Future
- [ ] Real ERA5-Land data benchmark run
- [ ] Additional deep learning architectures (LSTM, Transformer, GNN)
- [ ] Probabilistic forecasting metrics (CRPS, pinball loss)
- [ ] Adaptive conformal prediction (CQR, Mondrian)
- [ ] Benchmark website deployment
- [ ] Community leaderboard (public submission)
- [ ] DOI via Zenodo
