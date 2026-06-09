# Physical Consistency Audit

## What It Is

The physical consistency audit checks whether a trained ML model's predictions
respond to input perturbations in ways that are broadly consistent with
atmospheric-science expectations.

This is **NOT**:
- A causal discovery tool
- A replacement for physics-based models
- A guarantee of physical correctness

This **IS**:
- A diagnostic to catch models that learned spurious correlations
- A credibility check before reporting benchmark results
- A way to compare models on scientific plausibility, not just RMSE

## Audit Questions

For each audited feature, ClimateNet-Bench asks:

| Feature | Expected Sign | Rationale |
|---|---|---|
| `radiation_anomaly` | **positive** | More solar radiation → more energy for evaporation |
| `soil_moisture_anomaly` | **positive** | Wetter soil → more water available for evaporation. Strongest in arid regions. |
| `temperature_anomaly` | **positive** | Higher T → higher saturation vapour pressure → more evaporative demand |
| `precipitation_anomaly` | **positive** | More precipitation → wetter surface → more evaporation |
| `dryness_proxy` | **negative** | Higher dryness (radiation/precip) → water stress → less evaporation |
| `wind_speed` | **positive** | Stronger winds → enhanced turbulent transport |

## Method

### Feature Sensitivity Curve

1. Take a baseline sample of observations (up to 200 rows).
2. For each feature, vary it across its observed range (5th–95th percentile).
3. Keep all other features at their observed values.
4. Compute the mean and standard deviation of model predictions.
5. Plot the response curve and compute the Spearman rank correlation.

### Monotonic Trend Check

- Spearman ρ between feature value and mean prediction.
- ρ ≈ +1 → monotonically increasing (may be physically consistent).
- ρ ≈ −1 → monotonically decreasing.
- ρ ≈ 0 → no clear trend.
- Physical consistency = direction matches expectation AND p < 0.05.

### Regional Breakdown

For features where regional contrast is diagnostic (soil moisture,
precipitation, radiation, dryness proxy), the audit computes separate
sensitivity curves per benchmark region.  Key expectations:

| Region | Climate | Key Diagnostic |
|---|---|---|
| Sahara | arid | Soil moisture should dominate (water-limited) |
| Amazon | tropical_humid | Radiation should dominate (energy-limited) |
| East China | monsoon | Precipitation seasonality should matter |
| Central Europe | temperate | Moderate sensitivity across all features |
| Western US | semi_arid | Similar to Sahara but less extreme |

## Outputs

```
outputs/benchmark/physical_consistency/
├── physical_consistency_report.md   ← human-readable report
├── consistency_summary.json         ← machine-readable results
├── pdp_radiation_anomaly_lag_1.png
├── pdp_soil_moisture_anomaly_lag_1.png
├── pdp_temperature_anomaly_lag_1.png
├── pdp_precipitation_anomaly_lag_1.png
├── pdp_dryness_proxy_lag_1.png
├── pdp_wind_speed_lag_1.png
├── pdp_soil_moisture_anomaly_lag_1_by_region.png
├── pdp_precipitation_anomaly_lag_1_by_region.png
├── pdp_radiation_anomaly_lag_1_by_region.png
├── pdp_dryness_proxy_lag_1_by_region.png
└── regional_sensitivity.csv
```

## Consistency Score

The **consistency score** is the fraction of audited features whose monotonic
trend direction matches the physically expected sign with statistical
significance (p < 0.05).

- Score = 1.0 → all features behave as expected.
- Score = 0.5 → half the features show physically plausible behaviour.
- Score = 0.0 → the model may have learned spurious correlations.

A low score does not mean the model is "wrong" — it means the model's
behaviour under single-feature perturbations does not align with simple
physical expectations.  This warrants investigation.

## Limitations

1. **One-feature-at-a-time.**  Real climate variables co-vary.  Varying one
   feature while holding others fixed may produce responses that would not
   occur in nature.

2. **No interaction effects.**  The audit does not test whether the model
   captures interactions (e.g. radiation × soil moisture synergy).

3. **ERA5-Land is reanalysis.**  "Physically consistent" means consistent
   with the reanalysis model, not necessarily with real-world measurements.

4. **Lag-1 only.**  Only the most recent month's features are audited.
   Longer lags may show different behaviour.

5. **Not a substitute for expert review.**  Domain experts should review
   the regional sensitivity plots and assess whether the patterns are
   plausible for the specific regions and seasons.

## How This Creates Atmospheric-Science Credibility

1. **Transparency.**  The audit exposes what the model has learned, not just
   how well it predicts.  A model with RMSE = 0.4 that predicts evaporation
   *decreases* when soil moisture increases is suspicious.

2. **Reproducibility.**  The audit runs automatically as part of the
   benchmark pipeline.  Every model gets the same checks.

3. **Bridging ML and climate science.**  Climate scientists trust models
   that respect known physical relationships.  The audit provides a shared
   language between ML practitioners and domain experts.

4. **Catching spurious correlations.**  A model trained on random splits
   may achieve high R² by memorising spatial patterns.  The physical audit
   reveals whether the model's internal logic makes physical sense.
