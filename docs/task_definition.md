# Benchmark Task Definition

## Problem Statement

Predict next-month land evaporation anomaly from the previous 6 months of climate variables.

## Core Task

For each grid cell and month, use the previous 6 months of climate anomaly features to predict `evaporation_anomaly` at month `t`.

### Input Window

```
X_{t-6 : t-1}
```

The model sees 6 months of historical climate data (a sliding window of length 6) and must predict the target variable for the next month.

### Target Variable

```
y_t = evaporation_anomaly at month t
```

Evaporation anomaly is defined as the deviation of monthly total evaporation from the long-term climatological mean for that grid cell and calendar month.

### Spatial Unit

Each sample corresponds to a single **grid cell** (latitude-longitude point) at a single **monthly** time step.

### Temporal Unit

**Monthly.** All variables are aggregated or averaged to monthly resolution.

## Input Features

| Feature | Description | Type |
|---|---|---|
| `temperature_anomaly` | Deviation of 2m air temperature from climatology | Physical |
| `precipitation_anomaly` | Deviation of total precipitation from climatology | Physical |
| `radiation_anomaly` | Deviation of surface solar radiation from climatology | Physical |
| `soil_moisture_anomaly` | Deviation of volumetric soil water from climatology | Physical |
| `wind_speed` | 10m wind speed | Physical |
| `dryness_proxy` | Derived index combining temperature and precipitation deficits | Derived |
| `saturation_vapor_pressure` | Physically derived from temperature via Clausius-Clapeyron | Derived |
| `month_sin` | sin(2π × month / 12) | Temporal encoding |
| `month_cos` | cos(2π × month / 12) | Temporal encoding |
| `latitude` | Grid cell latitude (degrees north) | Spatial |
| `longitude` | Grid cell longitude (degrees east) | Spatial |
| `region` | Categorical region label | Metadata |

## Data Source

**ERA5-Land** monthly averaged reanalysis data (1950–present), accessed via the Copernicus Climate Data Store (CDS).

- Spatial resolution: 0.1° × 0.1° (~9 km at equator)
- Temporal coverage: monthly means from 1950 onward
- Variables: 2m temperature, total precipitation, surface solar radiation, volumetric soil water, evaporation, 10m wind components

## Benchmark Regions

Five geographically and climatically distinct regions:

| Region | Climate Zone | Rationale |
|---|---|---|
| Sahara | Arid / Hyper-arid | Low evaporation signal, extreme temperatures |
| East China | Humid subtropical / Monsoonal | Strong seasonality, high precipitation |
| Amazon | Tropical rainforest | High evaporation, dense vegetation feedback |
| Central Europe | Temperate continental | Moderate signal, strong anthropogenic influence |
| Western US | Mediterranean / Semi-arid | Water-limited, strong interannual variability |

## Why This Task Matters

1. **Land evaporation** is a key component of the terrestrial water cycle, linking energy, water, and carbon budgets.
2. **Anomaly forecasting** is directly useful for drought early warning, irrigation planning, and water resource management.
3. **Spatio-temporal generalization** is an open problem — models that perform well on random splits often fail when tested on unseen locations or future years.
4. **Benchmarking** with rigorous split protocols reveals whether ML models learn physical relationships or simply memorize spatial-temporal correlations.
