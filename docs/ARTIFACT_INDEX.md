# Final Artifact Index

This index separates result-bearing corrected artifacts from diagnostics,
provenance, and source-data-invalid historical runs. Paths under
`/media/drink8water/拯救者PSSD` require the project external disk.

Status meanings:

- `final`: may support the final project results.
- `diagnostic`: supports interpretation, but is not itself a model result.
- `provenance`: records source correction or reproducibility evidence.
- `invalid-for-results`: retained for audit only and must not be quoted as a
  performance result.

## Corrected data

| Artifact | Path | Purpose | Status | Size | SHA256 | Formal conclusions |
|---|---|---|---|---:|---|---|
| Sahara corrected raw | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/data/raw/era5_land_corrected/era5_land_sahara_2019_2023_corrected_accumulated.nc` | Full 2019–2023 raw data with `ssrd/tp/e` replaced for 2022-09–2023-12 | final | 49.1 MiB | `e35404dacdc9963d5da81c2626a74c5d8b2571d4d3b3acedcc75028e5762c764` | Yes |
| East China corrected raw | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/data/raw/era5_land_corrected/era5_land_east_china_2019_2023_corrected_accumulated.nc` | Same correction for East China | final | 18.8 MiB | `5a77b78b6f003626794015d00a0a10173c34dc75643629bd96aa49cf9a20602a` | Yes |
| Corrected processed records | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/data/processed/era5_climate_data_full_2019_2023_corrected.csv` | Unit-converted monthly tabular records | final | 464 MiB | `8465ad99a38879a96f7445b606ddffff370fc7204e5955dcdea0b73986f71cb6` | Yes |
| Corrected physical features | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/data/processed/era5_physical_features_full_2019_2023_corrected.csv` | Row-wise physical features; formal runner fits anomalies per split | final | 0.99 GiB | `90067630ca1a8f11b6005026313e1d7f2cb343ddd21fcbf8744bc38806808db7` | Yes |
| Corrected full audit JSON | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/data/audit/era5_corrected_full_2019_2023_audit.json` | Combined raw, processed, and feature readiness audit | provenance | 91 KiB | `d78efb7f75b238c11b990a33e92ed6a63ae83ab23ea86cc0d52aa8fce441e86c` | Supports provenance |
| Corrected full audit summary | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/data/audit/era5_corrected_full_2019_2023_audit.md` | Human-readable readiness summary | provenance | 2.5 KiB | `416bc400536c3c3ac632c69437885ae9234d9e5a12355d674cd1dc67693ee9c6` | Supports provenance |

The merge manifests beside each corrected NetCDF record old/patch/output
hashes, replacement months, variables, grid shape, and year-month matching.

## Final benchmark results

| Artifact | Path | Purpose | Status | Size | Formal conclusions |
|---|---|---|---|---:|---|
| Corrected v1, seed 42 | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5-land-sahara-eastchina-corrected-v1-era5-land-corrected-20260730T064523304383Z-348c8f24-2b3cf0` | 12-task base/full sanity and seed-42 reference | final | 4.6 GiB | Yes |
| Corrected multi-seed summary | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5_land_v1_corrected_multiseed_lite_summary` | Three-seed full-feature mean/std and regional stability | final | 124 KiB | Yes |
| Multi-seed seed 123 | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5-land-sahara-eastchina-corrected-v1-lite-multiseed-seed123-era5-land-corrected-20260730T071040084392Z-98335dc3-40812a` | Canonical metrics and predictions | final | 4.2 GiB | Via aggregate summary |
| Multi-seed seed 2026 | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5-land-sahara-eastchina-corrected-v1-lite-multiseed-seed2026-era5-land-corrected-20260730T072218388306Z-98357658-df5b2d` | Canonical metrics and predictions | final | 4.2 GiB | Via aggregate summary |
| Repeated spatial benchmark | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5-land-sahara-eastchina-corrected-repeated-spatial-lite-era5-land-corrected-20260730T092850273890Z-0332be26-07b3a2` | Five region-stratified spatial folds, 10 tasks | final | 7.0 GiB | Primary spatial result |

Use `mean_std.csv` and `multiseed_summary.json` in the multi-seed summary,
and `across_fold_summary.csv/json` in the repeated-spatial run. Per-run
`run_metadata.json`, `config_resolved.yaml`, preprocessing metadata, metrics,
and predictions are the audit trail.

## Diagnostics and source correction

| Artifact | Path | Purpose | Status | Size/hash | Formal conclusions |
|---|---|---|---|---|---|
| Radiation consistency audit | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/diagnostics/era5_radiation_consistency_audit` | Isolated the ECMWF accumulated-variable source issue from pipeline conversion | diagnostic | 376 KiB | Interpretation only |
| Accumulated patch audit | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/data/audit/era5_accumulated_patch_202209_202312_audit.json` | Validates `mnth`, 00:00 patch, variables, time, grid, and old-vs-patch changes | provenance | SHA256 `667dea4ec828b9731027828456ca6c5d313157d03961fcdfea72120719cff20f` | Supports corrected data |
| Single-spatial composition | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/diagnostics/era5_land_corrected_spatial_split_composition` | Explains seed-sensitive single holdouts | diagnostic | 104 KiB | Not a primary score |
| Repeated-fold audit v2 | `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/diagnostics/era5_land_corrected_repeated_spatial_folds_v2` | Accepted five leakage-free, region-stratified folds | provenance | 132 KiB; audit SHA256 `4ee6c02219e20be6d64ed3e1eb1f17e79c66ddf63d40475c25f45000c2fc2750` | Supports repeated spatial |

Patch NetCDF provenance:

- Sahara: `/media/drink8water/拯救者PSSD/ClimateNet-Bench/data/raw/era5_land_patch_hourly_monthly_202209_202312/era5_land_sahara_202209_202312_accumulated_patch_hourly_monthly_00.nc`,
  6.3 MiB, SHA256
  `a60947a4e69b284e30a5d275c5c5690f97e28ca20f0d4152a594b13792ed1b5c`.
- East China: `/media/drink8water/拯救者PSSD/ClimateNet-Bench/data/raw/era5_land_patch_hourly_monthly_202209_202312/era5_land_east_china_202209_202312_accumulated_patch_hourly_monthly_00.nc`,
  2.6 MiB,
  SHA256
  `fa14641beaffe49a7cc512a720c9686f4dc18cedb754b1934a89f435af476b2f`.

## Invalid historical runs

The following paths contain `source_data_status.json` with
`status: source_data_invalid`. They are retained only to document how the
source issue was detected:

- `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5-land-sahara-eastchina-v1-era5-land-20260730T012055094403Z-8cfe21a0-4e68b2`
- `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5-land-sahara-eastchina-v1-multiseed-seed123-era5-land-20260730T020318399976Z-261e682b-a4a68d`
- `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5-land-sahara-eastchina-v1-multiseed-seed2026-era5-land-20260730T021517210737Z-60977264-9b5cf0`
- `/media/drink8water/拯救者PSSD/ClimateNet-Bench/outputs/benchmark_runs/era5_land_v1_multiseed_summary`

Their metrics are `invalid-for-results`: do not use them in the README,
dashboard, CV, or model comparisons.
