# ClimateNet-Bench Benchmark Report

## Scope

This report summarizes the corrected ERA5-Land Sahara-East China v1,
corrected multi-seed v1-lite, and repeated region-stratified spatial
benchmark. These corrected artifacts are the current formal evidence. Smoke
and sample benchmarks remain useful for CI and onboarding, but they are not
scientific results. See `docs/FINAL_RESULTS.md` for the compact result table
and `docs/ARTIFACT_INDEX.md` for provenance.

## Data and Provenance

The corrected benchmark uses ERA5-Land derived physical features for Sahara
and East China, covering 2019-2023.

An audit traced the original 2022-09 onward discontinuity to the
ECMWF-documented affected accumulated variables in the monthly-averaged
product. Corrected `ssrd`, `tp`, and `e` values were obtained from
monthly-averaged reanalysis by hour of day at 00:00, merged by
region/year/month/grid, and used to regenerate both tabular datasets. The old
runs were preserved and marked `source_data_invalid`.

| Item | Value |
|---|---|
| Corrected feature CSV | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/data/processed/era5_physical_features_full_2019_2023_corrected.csv` |
| SHA256 | `90067630ca1a8f11b6005026313e1d7f2cb343ddd21fcbf8744bc38806808db7` |
| Audit status | `ready` |
| Corrected v1 config | `configs/benchmark/era5_land_v1_corrected.yaml` |
| Corrected multi-seed lite config | `configs/benchmark/era5_land_v1_corrected_multiseed_lite.yaml` |

The corrected v1 run executed 12 tasks:

- 2 models: Linear Regression and LightGBM.
- 2 feature sets: `base` and `full`.
- 3 split protocols: random, temporal holdout, and spatial-block holdout.

The corrected multi-seed v1-lite run executed 18 tasks:

- 3 seeds: 42, 123, and 2026.
- 2 models: Linear Regression and LightGBM.
- 1 feature set: `full`.
- 3 split protocols: random, temporal holdout, and spatial-block holdout.

## Artifact Locations

| Artifact | Path |
|---|---|
| Corrected v1 run | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5-land-sahara-eastchina-corrected-v1-era5-land-corrected-20260730T064523304383Z-348c8f24-2b3cf0` |
| Multi-seed summary | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5_land_v1_corrected_multiseed_lite_summary` |
| Multi-seed `mean_std.csv` | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5_land_v1_corrected_multiseed_lite_summary/mean_std.csv` |
| Multi-seed `multiseed_summary.json` | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5_land_v1_corrected_multiseed_lite_summary/multiseed_summary.json` |
| Multi-seed regional metrics | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5_land_v1_corrected_multiseed_lite_summary/regional_metrics.csv` |
| Repeated spatial run | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5-land-sahara-eastchina-corrected-repeated-spatial-lite-era5-land-corrected-20260730T092850273890Z-0332be26-07b3a2` |

## Corrected v1 Results

Corrected v1 completed all 12 tasks with 0 failures. The table below reports
the main RMSE/R2 results from the corrected source run.

| Model | Features | Split | RMSE | R2 | Skill vs climatology | OOD degradation |
|---|---|---|---:|---:|---:|---:|
| Linear Regression | base | random | 10.412 | 0.082 | 0.042 | - |
| Linear Regression | base | spatial_block | 11.764 | 0.091 | 0.055 | +13.0% |
| Linear Regression | base | temporal | 8.687 | 0.023 | 0.012 | -16.6% |
| Linear Regression | full | random | 9.308 | 0.267 | 0.144 | - |
| Linear Regression | full | spatial_block | 12.110 | 0.036 | 0.027 | +30.1% |
| Linear Regression | full | temporal | 8.172 | 0.135 | 0.070 | -12.2% |
| LightGBM | base | random | 7.098 | 0.573 | 0.347 | - |
| LightGBM | base | spatial_block | 9.079 | 0.458 | 0.270 | +27.9% |
| LightGBM | base | temporal | 6.560 | 0.443 | 0.254 | -7.6% |
| LightGBM | full | random | 5.769 | 0.718 | 0.469 | - |
| LightGBM | full | spatial_block | 7.869 | 0.593 | 0.368 | +36.4% |
| LightGBM | full | temporal | 6.299 | 0.486 | 0.283 | +9.2% |

## Source Correction Audit Case

The values in this section are shown only to document why the source was
invalidated. The old run directories contain
`status: source_data_invalid`; none of their metrics are formal results or
CV-ready claims.

The previous invalid v1 temporal collapse was primarily caused by incorrect
accumulated source data, not by the temporal split or train-only preprocessing
implementation.

| Check | Invalid v1 | Corrected v1 | Change |
|---|---:|---:|---:|
| LightGBM full temporal RMSE | 12.142 | 6.299 | -48.1% |
| LightGBM full temporal R2 | -0.996 | 0.486 | recovered |
| Linear full temporal RMSE | 8.654 | 8.172 | improved |
| Linear full temporal R2 | -0.014 | 0.135 | recovered |
| LightGBM full random RMSE | 5.623 | 5.769 | +2.6% |
| LightGBM full spatial RMSE | 7.755 | 7.869 | +1.5% |

The random and spatial LightGBM changes are small, while temporal performance
recovers dramatically. This isolates the failure mode to the old accumulated
source data rather than a broad modeling or split-protocol issue.

## Corrected Multi-Seed v1-Lite

The corrected multi-seed v1-lite run completed all 18 tasks with 0 failures.
It reuses the corrected v1 seed-42 run and adds seeds 123 and 2026.

| Model | Features | Split | RMSE mean | RMSE std | R2 mean | Skill mean | OOD degradation mean |
|---|---|---|---:|---:|---:|---:|---:|
| LightGBM | full | random | 5.766 | 0.020 | 0.718 | 0.469 | - |
| Linear Regression | full | random | 9.296 | 0.028 | 0.267 | 0.144 | - |
| LightGBM | full | spatial_block | 7.089 | 1.368 | 0.615 | 0.388 | +23.0% |
| Linear Regression | full | spatial_block | 10.346 | 1.652 | 0.182 | 0.105 | +11.3% |
| LightGBM | full | temporal | 6.309 | 0.015 | 0.485 | 0.282 | +9.4% |
| Linear Regression | full | temporal | 8.172 | 0.000 | 0.135 | 0.070 | -12.1% |

## Stability Findings

| Finding | Result | Evidence |
|---|---|---|
| LightGBM beats Linear on random split | Stable | 3/3 seeds |
| LightGBM beats Linear on spatial split | Stable | 3/3 seeds |
| LightGBM temporal degradation is positive | Stable | 3/3 seeds, about +9-10% relative RMSE |
| East China RMSE exceeds Sahara RMSE | Stable | 18/18 model-split-seed comparisons |
| Temporal is always harder than random | Not stable | 3/6 model-seed comparisons |
| Spatial is always harder than random | Not stable | 4/6 model-seed comparisons |

The most important corrected multi-seed conclusion is that LightGBM temporal
recovery is stable across seeds. Temporal performance is no longer a collapse:
LightGBM full temporal RMSE is tightly concentrated around 6.31.

## Regional Error Hotspot

East China remains a consistent high-error region. For the corrected v1
LightGBM full model:

| Split | East China RMSE | Sahara RMSE |
|---|---:|---:|
| random | 9.645 | 3.927 |
| spatial_block | 12.261 | 3.982 |
| temporal | 10.823 | 4.059 |

The multi-seed validation confirms this is not a seed artifact: East China RMSE
is higher than Sahara RMSE in all 18 model-split-seed comparisons.

## Spatial Split Composition Diagnostic

The read-only composition diagnostic is stored at
`/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/diagnostics/era5_land_corrected_spatial_split_composition`.
It uses the saved spatial assignments, prepared samples, train-only
preprocessing parameters, and completed metrics; it does not retrain models.

| Seed | Test samples | Test grids | East China share | Target std | Zero baseline RMSE | Mean abs. feature SMD | LightGBM RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 874,260 | 16,190 | 34.3% | 12.335 | 12.443 | 0.275 | 7.869 |
| 123 | 690,228 | 12,782 | 35.5% | 11.600 | 11.719 | 0.197 | 7.888 |
| 2026 | 710,694 | 13,161 | 3.9% | 10.320 | 10.375 | 0.236 | 5.508 |

Seed 2026 is easier primarily because its test blocks are 96.1% Sahara and
have lower target dispersion: its zero-anomaly and lag-1 persistence baseline
RMSEs are 10.375 and 8.595, versus 12.443/11.008 for seed 42 and
11.719/11.337 for seed 123. Its aggregate feature shift is not uniquely
smaller; the main shift instead changes family, with stronger dryness and
saturation-vapour-pressure shifts but almost no precipitation or soil-moisture
mean shift. Region mix and held-out block assignment therefore explain the
instability better than a uniformly closer test distribution.

Spatial generalisation should be reported using repeated, preferably
region-stratified spatial folds with mean/std, not a single holdout. A future
Split Diagnostics view can show test region share, sample/grid counts, target
standard deviation, naive-baseline RMSE, mean/max feature SMD, and model RMSE
for each fold.

## Repeated Spatial Fold Design

A five-fold, region-stratified 5-degree block design now has a passing
composition audit at
`/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/diagnostics/era5_land_corrected_repeated_spatial_folds_v2`.
Blocks are geographically interleaved within each region without using target
values; every block is test once, validation once, and train three times.
All folds contain both regions, cover all 12 months, and have zero grid
leakage. Test East China share ranges from 17.7% to 33.2%, and
zero-baseline RMSE CV is 0.102.

The audited folds have now been benchmarked with the 10-task
Linear/LightGBM configuration in
`configs/benchmark/era5_land_corrected_repeated_spatial_lite.yaml`. The
canonical run is
`/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5-land-sahara-eastchina-corrected-repeated-spatial-lite-era5-land-corrected-20260730T092850273890Z-0332be26-07b3a2`.
Across five folds, Linear Regression RMSE is 9.782 ± 0.773 and LightGBM RMSE
is 7.066 ± 1.093 (sample standard deviation). LightGBM reduces mean RMSE by
27.8% and wins on every fold.

Fold 0 is the hardest LightGBM fold (RMSE 8.871). It combines the largest
East China test share (33.2%), the highest tested feature-shift score, and a
high target standard deviation. Fold 4 is easiest (RMSE 6.212). Across these
five folds, LightGBM RMSE correlates more strongly with target standard
deviation (r=0.87) and mean absolute feature shift (r=0.92) than with East
China share alone (r=0.79); these correlations are descriptive because n=5.
The single spatial holdout is therefore superseded by repeated-fold
mean/std for spatial-generalisation reporting. Full fold metrics and
provenance are in the run artifacts.

## Interpretation

The corrected source data fixes the temporal collapse observed in the invalid
v1 run. LightGBM now outperforms Linear Regression across the tested random and
spatial protocols, and the temporal LightGBM result is stable across seeds.

The main unresolved scientific issue is not temporal failure anymore. It is
regional sensitivity:

- East China is a stable error hotspot.
- Single spatial-block difficulty has high variance across seeds because of
  held-out composition.
- The completed repeated region-stratified result supersedes any single
  spatial holdout as the headline spatial-generalisation result.

## Limitations

- Coverage is limited to Sahara and East China during 2019–2023; this is not a
  global ERA5-Land claim.
- Only Linear Regression and LightGBM are compared in the final corrected
  study.
- Climate-zone mapping, region transfer, and joint spatiotemporal transfer
  remain outside the final matrix.
- Temporal robustness uses one deterministic holdout and three model/split
  seeds; longer periods and rolling temporal folds would strengthen evidence.

## Dashboard Implications

The frontend dashboard should be driven by the formal benchmark artifacts
above, not by smoke or sample benchmarks. Smoke remains a health check; sample
bench remains an onboarding demo. The main dashboard should surface:

- formal run status and data provenance;
- corrected v1 leaderboard;
- multi-seed mean/std stability;
- split difficulty diagnostics;
- East China vs Sahara regional error comparison;
- links to canonical artifacts and run directories.

## Next Steps

1. Freeze the corrected data/run hashes and commit the consolidation patch.
2. If desired, connect the dashboard to corrected `mean_std.csv`,
   `multiseed_summary.json`, `regional_metrics.csv`, and
   `across_fold_summary.json`; never ingest invalid historical runs.
3. Prepare a concise application-facing CV version with the two-region scope
   and limitations intact.
