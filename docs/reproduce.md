# Reproducing ClimateNet-Bench

This guide distinguishes data audit, corrected benchmark runs, and optional
application components. You do not need to rerun every large experiment to
inspect or verify the final results.

## Environment and tests

Recommended environment: Python 3.11 with the project requirements and
LightGBM installed.

```bash
conda create -n climatenet-py311 python=3.11 -y
conda activate climatenet-py311
pip install -r requirements.txt
pip install -e .
pip install lightgbm

env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

The synthetic smoke path does not require external ERA5-Land data:

```bash
python scripts/build_forecasting_dataset.py --synthetic
PYTHONPATH=src python scripts/run_benchmark.py \
  --config configs/benchmark/smoke_test.yaml
```

Smoke results validate plumbing only and are not scientific conclusions.

## Corrected data flow

The final data lineage is:

```text
original 2019–2023 ERA5-Land NetCDF
  + corrected accumulated patch (2022-09–2023-12, monthly-by-hour 00:00)
  -> corrected full NetCDF per region
  -> corrected processed CSV
  -> row-wise corrected physical features
  -> split
  -> train-only climatology/anomalies/thresholds/scaling
  -> lag samples
  -> corrected benchmark runs
```

Large NetCDF, CSV, prediction, and prepared-split artifacts live under:

```text
/media/drink8water/拯救者PSSD/ClimateNet-Bench
```

Without that disk, use the hashes and summaries in
[ARTIFACT_INDEX.md](ARTIFACT_INDEX.md); do not silently substitute repository
sample data.

## Data audit and correction

### 1. Patch request and audit

The controlled downloader only permits `ssrd`, `tp`, and `e` for
2022-09–2023-12. Dry-run the CDS request before any download:

```bash
PYTHONPATH=src python scripts/download_era5_patch_accumulated.py \
  --config configs/data_config_external_patch_202209_202312.yaml \
  --region Sahara --dry-run-request

PYTHONPATH=src python scripts/download_era5_patch_accumulated.py \
  --config configs/data_config_external_patch_202209_202312.yaml \
  --region "East China" --dry-run-request
```

Only use `--execute` when reproducing the download with configured CDS access.
The retained patch can be re-audited with:

```bash
PYTHONPATH=src python scripts/audit_era5_patch_accumulated.py \
  --config configs/data_config_external_patch_202209_202312.yaml
```

### 2. Merge corrected accumulated variables

The merge matches by region, calendar year/month, latitude, and longitude—not
exact timestamps—and refuses to overwrite an existing corrected file.

```bash
PYTHONPATH=src python scripts/merge_era5_corrected_accumulated.py \
  --config configs/data_config_external_corrected_2019_2023.yaml \
  --region Sahara --dry-run

PYTHONPATH=src python scripts/merge_era5_corrected_accumulated.py \
  --config configs/data_config_external_corrected_2019_2023.yaml \
  --region "East China" --dry-run

# Replace --dry-run with --execute only when the output paths do not exist.
```

### 3. Audit and build corrected tables

```bash
# Corrected raw NetCDF audit
PYTHONPATH=src python scripts/audit_era5_corrected_full.py \
  --config configs/data_config_external_corrected_2019_2023.yaml --raw

# Explicit corrected NetCDF inputs -> processed CSV
PYTHONPATH=src python scripts/preprocess_era5.py \
  --data-config configs/data_config_external_corrected_2019_2023.yaml

# Chunked processed-table audit
PYTHONPATH=src python scripts/audit_processed_era5.py \
  --data-config configs/data_config_external_corrected_2019_2023.yaml

# Row-wise physics only; no full-table climatology/anomaly fit
PYTHONPATH=src python scripts/run_pipeline.py \
  --data-config configs/data_config_external_corrected_2019_2023.yaml \
  --formal-benchmark \
  --audit-output /media/drink8water/拯救者PSSD/ClimateNet-Bench/data/audit/era5_corrected_physical_2019_2023_audit.json

# Combined final readiness audit
PYTHONPATH=src python scripts/audit_era5_corrected_full.py \
  --config configs/data_config_external_corrected_2019_2023.yaml --final
```

Proceed only when the final audit reports `status: ready` and the physical
feature SHA256 is:

```text
90067630ca1a8f11b6005026313e1d7f2cb343ddd21fcbf8744bc38806808db7
```

## Corrected benchmark entry points

These commands are documented for reproducibility. The final runs already
exist; inspecting their resolved configs, hashes, and metrics is normally
preferable to rerunning multi-gigabyte experiments.

### Corrected v1

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --config configs/benchmark/era5_land_v1_corrected.yaml \
  --dry-check

PYTHONPATH=src python scripts/run_benchmark.py \
  --config configs/benchmark/era5_land_v1_corrected.yaml
```

Dry-check must report exactly 12 tasks and the corrected feature hash.

### Corrected multi-seed v1-lite

```bash
PYTHONPATH=src python scripts/run_multiseed_benchmark.py \
  --config configs/benchmark/era5_land_v1_corrected_multiseed_lite.yaml \
  --dry-check

PYTHONPATH=src python scripts/run_multiseed_benchmark.py \
  --config configs/benchmark/era5_land_v1_corrected_multiseed_lite.yaml
```

This configuration aggregates 18 full-feature tasks across seeds 42, 123,
and 2026. It reuses the configured completed seed-42 corrected run and writes
a separate summary directory.

### Repeated region-stratified spatial benchmark

The accepted fold manifests already exist in the v2 diagnostic directory.
To reproduce their composition audit from scratch, choose a new output
directory or explicitly authorize overwrite:

```bash
PYTHONPATH=src python scripts/audit_repeated_spatial_folds.py \
  --config configs/diagnostics/era5_land_repeated_spatial_folds_audit.yaml
```

The formal benchmark entry point is:

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --config configs/benchmark/era5_land_corrected_repeated_spatial_lite.yaml \
  --dry-check

PYTHONPATH=src python scripts/run_benchmark.py \
  --config configs/benchmark/era5_land_corrected_repeated_spatial_lite.yaml
```

Dry-check must report 5 folds, 2 models, one feature set, 10 tasks, a ready
fold audit, corrected input hash, and disabled compatibility predictions.
Each fold independently fits train-only preprocessing.

## Output verification

Every benchmark invocation creates a new collision-resistant run directory:

```text
outputs/benchmark_runs/<run-id>/
├── config_resolved.yaml
├── run_metadata.json
├── experiment_registry.json
├── splits/
├── preprocessing/
├── predictions/
├── metrics/
├── leaderboard.csv
├── summary.json
└── experiments/
```

Multi-seed summaries add `mean_std.csv`, `regional_metrics.csv`, and
`multiseed_summary.json`. Repeated spatial adds
`across_fold_summary.csv/json`. Verify completed/failed task counts,
data/config hashes, git dirty state, fallback counts, and the absence of
compatibility prediction copies where disabled.

## Invalid historical results

The original v1 and multi-seed runs used affected accumulated variables.
They contain `source_data_status.json` with
`status: source_data_invalid`. Keep them for the source-data audit story, but
never combine their metrics with corrected leaderboards or quote them as
model performance.

The authoritative corrected and invalid paths are listed in
[ARTIFACT_INDEX.md](ARTIFACT_INDEX.md). Configuration status is listed in
[CONFIG_INDEX.md](CONFIG_INDEX.md).

## Optional application

The API/dashboard are not required to verify the corrected scientific result:

```bash
PYTHONPATH=$(pwd) uvicorn backend.main:app --host 127.0.0.1 --port 8000

cd frontend
npm install
npm run dev
```

If the dashboard is connected to formal results, use only the corrected
multi-seed and repeated-spatial summaries.
