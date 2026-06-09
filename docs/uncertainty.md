# Uncertainty Quantification in ClimateNet-Bench

## Why Uncertainty Matters for Climate Forecasting

Climate predictions are inherently uncertain.  Sources of uncertainty include:

1. **Data uncertainty** — ERA5-Land is a reanalysis product, not direct observation. Variables like evaporation are model-derived.
2. **Model uncertainty** — ML models approximate complex physical processes with finite data.
3. **Spatio-temporal variability** — Some regions (e.g. Amazon) have more predictable evaporation than others (e.g. Sahara).

A point forecast (single number) is misleading without an accompanying uncertainty estimate.  Two models can have identical RMSE but very different prediction intervals.

## What ClimateNet-Bench Measures

ClimateNet-Bench uses **split (inductive) conformal prediction** to produce
distribution-free prediction intervals with formal coverage guarantees.

### Split Conformal Prediction

1. **Train** a model on the training set as usual.
2. **Calibrate** on a held-out validation set:
   - Compute absolute residuals: `r_i = |y_i − ŷ_i|`
   - For target coverage `1 − α`, compute quantile `q̂ = quantile({r_i}, (1 − α)(n+1)/n)`
3. **Predict** intervals for test points:
   - `lower = ŷ − q̂`
   - `upper = ŷ + q̂`
4. **Guarantee:** Under exchangeability, `P(y_test ∈ [lower, upper]) ≥ 1 − α`

### Key Properties

- **Distribution-free** — no Gaussian assumption required.
- **Constant-width intervals** — all predictions get the same `±q̂` band.
- **Finite-sample valid** — the coverage guarantee holds for any sample size.
- **Calibration set must be independent of training and test** — ClimateNet-Bench
  uses the validation split for calibration, never the test set.

## Output Metrics

| Metric | Definition | Target |
|---|---|---|
| `coverage_90` | Fraction of test targets inside 90 % prediction intervals | ≈ 0.90 |
| `mean_interval_width` | Mean width of prediction intervals (`2q̂`) | As small as possible for given coverage |
| `coverage_by_region` | Coverage broken down by benchmark region | Check for regional fairness |
| `coverage_by_split` | Coverage per split protocol | OOD splits may show under-coverage |
| `coverage_by_climate_type` | Coverage per climate zone | Check calibration across climates |

### Interpreting Coverage and Width

- **Coverage close to target (0.90)** — the model is well-calibrated for the given α.
- **Coverage below target** — the model is under-confident; predictions are too tight. This often happens under OOD splits (spatial_block, temporal, region_transfer) because the exchangeability assumption is violated.
- **Coverage above target** — intervals are too wide; the model is over-cautious.
- **Width** — a model with coverage 0.90 and width 0.5 is better than one with coverage 0.90 and width 2.0. Width should be interpreted alongside coverage.

## `intervals.csv` Schema

| Column | Type | Description |
|---|---|---|
| `experiment_id` | str | Unique experiment identifier |
| `model_name` | str | Model name (e.g. "random_forest") |
| `split_id` | str | Split protocol name |
| `region` | str | Benchmark region |
| `climate_type` | str | Climate classification |
| `year` | int | Target year |
| `month` | int | Target month |
| `latitude` | float | Grid cell latitude |
| `longitude` | float | Grid cell longitude |
| `y_true` | float | True evaporation anomaly |
| `y_pred` | float | Model point prediction |
| `lower` | float | Lower bound of 90 % prediction interval |
| `upper` | float | Upper bound of 90 % prediction interval |
| `covered` | bool | Whether y_true ∈ [lower, upper] |
| `interval_width` | float | upper − lower |

## Limitations

1. **Exchangeability assumption** — Conformal prediction guarantees marginal
   coverage only when calibration and test data are exchangeable.  Under OOD
   splits (spatial_block, temporal, region_transfer), this assumption is
   violated, and coverage may degrade.

2. **Marginal, not conditional coverage** — The guarantee is `P(Y_test ∈
   interval) ≥ 1 − α` on average across all test points.  It does NOT
   guarantee `P(Y_test ∈ interval | region=Sahara) ≥ 1 − α`.  The grouped
   coverage metrics (`coverage_by_region`, etc.) help diagnose conditional
   coverage failures.

3. **Constant-width intervals** — All predictions share the same `±q̂` band.
   This is simple and robust but cannot capture heteroscedasticity (e.g.
   higher uncertainty in summer vs winter).  Future milestones may add
   adaptive conformal methods (Conformalized Quantile Regression, Mondrian
   conformal).

4. **Single α** — The current benchmark reports results for a single
   significance level (α = 0.1 → 90 % intervals).  A full calibration curve
   (coverage vs α) would be more informative.

5. **Calibration set size** — Small validation splits produce noisy quantile
   estimates.  ClimateNet-Bench uses a minimum validation set size of 100
   samples for reliable calibration.

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
- Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511.
- Shafer, G., & Vovk, V. (2008). *A Tutorial on Conformal Prediction.* Journal of Machine Learning Research, 9, 371–421.

## How the Dashboard Can Use Coverage and Width

The ClimateNet-Bench dashboard can present uncertainty information through:

1. **Coverage bar chart** — horizontal bars showing actual vs target coverage per (model, split) pair.  A red line at 0.90 makes under-coverage immediately visible.
2. **Width vs coverage scatter plot** — one point per (model, split), with coverage on the x-axis and width on the y-axis.  The ideal region is top-left (high coverage, narrow width).
3. **Regional coverage heatmap** — a grid of (region × model) cells coloured by coverage.  Highlights which regions have unreliable predictions.
4. **Time-series with uncertainty bands** — plot `y_pred` as a line with `[lower, upper]` shading.  Shows when the model is confident vs uncertain.
5. **OOD degradation of coverage** — table comparing coverage under random split vs spatial/temporal/region splits.  Quantifies how much OOD conditions degrade uncertainty calibration.
