# Configuration Index

`final` configurations point at corrected data and may reproduce reported
results. `diagnostic` configurations inspect data or split composition.
`invalid-source` configurations point at the original affected accumulated
variables and must not be used for reported performance.

## Data configurations

| Config | Status | Purpose and main I/O |
|---|---|---|
| `configs/data_config_external_corrected_2019_2023.yaml` | **final** | Explicit corrected Sahara/East China NetCDF inputs → corrected processed and physical CSVs; includes old/patch provenance |
| `configs/data_config_external_patch_202209_202312.yaml` | provenance | Controlled three-variable CDS patch requests for 2022-09–2023-12 |
| `configs/data_config_external_full.yaml` | **invalid-source for results** | Original 2019–2023 full NetCDF → old processed/physical CSV; retain for source audit only |
| `configs/data_config_external_dryrun.yaml` | diagnostic | Bounded Sahara 2019–2021 real-data plumbing check |
| `configs/data_config_external.yaml` | development | Generic external-disk template; still contains sample feature defaults |
| `configs/data_config.yaml` | development | Repository-local synthetic/sample pipeline |

## Benchmark configurations

| Config | Status | Matrix and input |
|---|---|---|
| `configs/benchmark/era5_land_v1_corrected.yaml` | **final** | Corrected seed 42; 2 models × 2 feature sets × 3 splits = 12 tasks |
| `configs/benchmark/era5_land_v1_corrected_multiseed_lite.yaml` | **final** | Corrected full features; 3 seeds × 2 models × 3 splits = 18 tasks |
| `configs/benchmark/era5_land_corrected_repeated_spatial_lite.yaml` | **final** | Corrected full features; 5 audited folds × 2 models = 10 tasks |
| `configs/benchmark/era5_land_dry_run.yaml` | diagnostic | Restricted real-data plumbing validation; never a performance conclusion |
| `configs/benchmark/smoke_test.yaml` | development | Synthetic smoke test with random and spatial splits |
| `configs/benchmark/era5_land_v1.yaml` | **invalid-source** | Original affected physical CSV; old v1 results are audit cases |
| `configs/benchmark/era5_land_v1_multiseed.yaml` | **invalid-source** | Original affected multi-seed matrix |
| `configs/benchmark/evap_anomaly_v1.yaml` | template | General formal protocol template; not the final corrected run config |

## Diagnostic configurations

| Config | Status | Purpose |
|---|---|---|
| `configs/diagnostics/era5_radiation_consistency_audit.yaml` | provenance | Traces the accumulated-variable discontinuity and unit semantics |
| `configs/diagnostics/era5_land_v1_temporal_failure.yaml` | invalid-source diagnostic | Investigates the old temporal collapse; not a final model result |
| `configs/diagnostics/era5_land_corrected_spatial_split_composition.yaml` | diagnostic | Explains seed-dependent single-spatial composition |
| `configs/diagnostics/era5_land_repeated_spatial_folds_audit.yaml` | **final provenance** | Generates/audits the five accepted region-stratified folds |

## Split configurations

| Config | Status | Use |
|---|---|---|
| `configs/splits/repeated_region_stratified_spatial.yaml` | **final spatial protocol** | Five region-stratified, manifest-backed 5-degree folds |
| `configs/splits/random.yaml` | baseline | Optimistic shuffled reference, not primary generalisation evidence |
| `configs/splits/temporal.yaml` | final protocol | Future-period holdout |
| `configs/splits/spatial_block.yaml` | diagnostic/reference | Single seed-sensitive spatial holdout |
| `configs/splits/region_transfer.yaml` | not evaluated in final results | Held-out region protocol |
| `configs/splits/climate_zone_transfer.yaml` | not evaluated | Requires climate-zone mapping |
| `configs/splits/spatiotemporal.yaml` | not evaluated | Joint spatial-temporal holdout |

## Selection rule

For any result intended for the README, CV, or dashboard:

1. require `data_source: ERA5-Land corrected`;
2. require the corrected physical SHA256
   `90067630ca1a8f11b6005026313e1d7f2cb343ddd21fcbf8744bc38806808db7`;
3. require the preflight audit status `ready`;
4. reject any run or summary containing `source_data_status.json` with
   `status: source_data_invalid`;
5. use the repeated-spatial run—not a single spatial seed—for the headline
   spatial result.
