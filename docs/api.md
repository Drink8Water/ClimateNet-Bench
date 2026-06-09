# ClimateNet-Bench API Reference

Base URL: `http://127.0.0.1:8000`

Interactive docs: http://127.0.0.1:8000/docs

## Endpoints

### Health

```
GET /health
```

```json
{"status": "ok", "database": "file-based"}
```

### Benchmark Info

```
GET /benchmark/summary
```

```json
{
  "benchmark_name": "EvapAnomaly-Forecast-v1",
  "total_experiments": 36,
  "completed": 36,
  "failed": 0,
  "n_models": 4,
  "n_split_protocols": 6,
  "best_rmse": 1.017,
  "best_model": "linear_regression"
}
```

```
GET /benchmark/task
```

```json
{
  "task_name": "Next-Month Land Evaporation Anomaly Forecasting",
  "input_window": "X_{t-6 : t-1}",
  "target": "evaporation_anomaly at month t",
  "forecast_horizon": "1 month"
}
```

```
GET /benchmark/regions
```

```json
[{
  "name": "Sahara",
  "lat_min": 15.0, "lat_max": 30.0,
  "lon_min": -20.0, "lon_max": 30.0,
  "climate_type": "arid"
}]
```

```
GET /benchmark/splits
```

Returns metadata for all available split protocols.

### Leaderboard

```
GET /leaderboard?limit=100&model_name=random_forest&split_protocol=spatial_block
```

| Param | Type | Default | Description |
|---|---|---|---|
| `limit` | int | 100 | Max rows (1–1000) |
| `model_name` | str | — | Filter by model |
| `split_protocol` | str | — | Filter by split |
| `feature_set` | str | — | Filter by feature set |

```
GET /leaderboard/{split_protocol}
```

Filter leaderboard by a specific split protocol.

```
GET /split-difficulty
```

Returns mean/std/min/max RMSE per split protocol (6 rows).

```
GET /ablation-study?model_name=random_forest&split_protocol=spatial_block
```

Returns ablation study results (feature set comparisons).

### Experiments

```
GET /experiments?model_name=rf&validation_strategy=spatial_block&limit=100
```

```
GET /experiments/summary
```

Returns aggregated KPIs (total experiments, best R², best RMSE).

```
GET /experiments/{experiment_id}
```

Returns config + metrics summary for one experiment.

```
GET /experiments/{experiment_id}/predictions?limit=500
```

```
GET /experiments/{experiment_id}/intervals?limit=500
```

Returns prediction intervals (lower, upper, covered, interval_width).

### Uncertainty

```
GET /uncertainty/calibration?model_name=rf&split_protocol=spatial_block
```

Returns coverage and interval width per model/split.

### Physical Consistency

```
GET /physical-consistency/summary
```

```json
{
  "model_name": "random_forest",
  "consistency_score": 0.67,
  "n_features_audited": 6,
  "n_physically_consistent": 4,
  "feature_results": [...]
}
```

```
GET /physical-consistency/regional-sensitivity?feature=soil_moisture_anomaly_lag_1&region=Sahara
```

### Spatial

```
GET /spatial-grid?region=Sahara&variable=evaporation_anomaly&year=2020&month=7
```

```
GET /timeseries?region=Sahara&limit=200
```

### Legacy

```
GET /project-summary
GET /dataset-summary
GET /model-comparison
```

## Running the API

```bash
cd ClimateNet-Bench
PYTHONPATH=$(pwd) uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

## Running the Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

The Vite dev server proxies `/api/*` → `http://127.0.0.1:8000`.
