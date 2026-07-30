# Final Corrected Results

This page is the compact source of project-level quantitative claims. All
numbers come from corrected ERA5-Land runs. Historical runs marked
`source_data_invalid` are excluded.

## Dataset and protocol

| Item | Value |
|---|---|
| Regions | Sahara and East China |
| Period | 2019–2023, 60 continuous months |
| Monthly grid records | 5,318,160 |
| Six-month lag samples | 4,786,344 |
| Target | Next-month `evaporation_anomaly` |
| Corrected variables | `ssrd`, `tp`, `e` for 2022-09–2023-12 |
| Source readiness | `ready`; no tabular NaN/Inf or duplicate grid-month keys |
| Preprocessing | Per-split train-only climatology, event thresholds, and feature standardization |
| Models | Linear Regression and LightGBM |

The correction uses the ERA5-Land monthly-by-hour product at 00:00 for the
affected accumulated variables. Row-wise physical features are computed
before splitting; anomaly climatology and all fitted statistics are learned
inside each training partition.

## Corrected multi-seed v1-lite

Three seeds (`42`, `123`, `2026`), full features, and random, temporal, and
single spatial-block splits produced 18/18 successful tasks. Values are
mean ± sample standard deviation.

| Model | Split | MAE | RMSE | R² | Skill vs climatology | Relative RMSE vs random |
|---|---|---:|---:|---:|---:|---:|
| LightGBM | random | 2.816 ± 0.011 | 5.766 ± 0.020 | 0.718 ± 0.001 | 0.469 ± 0.001 | reference |
| Linear | random | 5.488 ± 0.010 | 9.296 ± 0.028 | 0.267 ± 0.001 | 0.144 ± 0.000 | reference |
| LightGBM | temporal | 3.194 ± 0.012 | 6.309 ± 0.015 | 0.485 ± 0.002 | 0.282 ± 0.002 | +9.4% ± 0.6% |
| Linear | temporal | 5.202 ± 0.000 | 8.172 ± 0.000 | 0.135 ± 0.000 | 0.070 ± 0.000 | −12.1% ± 0.3% |
| LightGBM | single spatial | 3.493 ± 1.273 | 7.089 ± 1.368 | 0.615 ± 0.091 | 0.388 ± 0.073 | +23.0% ± 24.0% |
| Linear | single spatial | 5.961 ± 1.657 | 10.346 ± 1.652 | 0.182 ± 0.127 | 0.105 ± 0.068 | +11.3% ± 17.8% |

The single-spatial rows are a composition diagnostic, not the primary spatial
generalisation result. Seed 2026 held out a test set containing only 3.9%
East China, making that holdout materially easier.

## Repeated region-stratified spatial result

The primary spatial result uses five audited 5-degree folds. Every fold
contains test and validation blocks from both regions, carries all months with
each grid cell, and has zero grid leakage.

| Model | Folds | MAE mean ± std | RMSE mean ± std | R² mean ± std | Skill mean ± std |
|---|---:|---:|---:|---:|---:|
| LightGBM | 5 | 3.453 ± 0.746 | **7.066 ± 1.093** | 0.594 ± 0.066 | 0.371 ± 0.045 |
| Linear | 5 | 5.770 ± 0.626 | **9.782 ± 0.773** | 0.217 ± 0.086 | 0.125 ± 0.051 |

LightGBM has lower RMSE in 5/5 folds and lowers mean RMSE by 27.8% relative
to Linear. Fold 0 is hardest for LightGBM (RMSE 8.871); it combines the
largest East China test share, the largest aggregate feature shift, and high
target variance. This is why spatial performance is reported as repeated-fold
mean/std rather than one holdout.

## Supported conclusions

- LightGBM consistently outperforms the Linear baseline on random and spatial
  evaluation within this two-region dataset.
- Corrected LightGBM temporal RMSE is stable across seeds. Its degradation
  relative to random is positive but moderate: 8.9%–10.1%.
- East China has higher RMSE than Sahara in all 18 multi-seed
  model/split/seed regional comparisons.
- Spatial difficulty depends strongly on held-out block composition.
  Region-stratified repeated folds provide more credible evidence than a
  single seed-dependent spatial split.
- The invalid-source temporal collapse was a data-product failure, not a
  valid model-generalisation result.

## Limitations

- The study covers two regions, not global ERA5-Land; results must not be
  generalized worldwide.
- The period is limited to five years and the temporal holdout is deterministic.
- Only traditional Linear and LightGBM baselines are part of the final
  corrected comparison.
- Climate-zone labels and climate-zone transfer are not evaluated.
- The three random seeds quantify split/model variability only partially;
  the repeated spatial design is stronger evidence specifically for spatial
  generalisation.
- Source correction depends on ECMWF/CDS product semantics and preserved
  external-disk artifacts; the hashes in `ARTIFACT_INDEX.md` are part of the
  reproducibility record.

See [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) for interpretation and
[ARTIFACT_INDEX.md](ARTIFACT_INDEX.md) for the exact run paths.
