# Contributing to ClimateNet-Bench

Thank you for your interest in contributing! ClimateNet-Bench is an open-source
spatio-temporal ML benchmark project. Contributions of all kinds are welcome:
bug reports, model implementations, new split protocols, documentation
improvements, and benchmark result submissions.

## Setup

```bash
git clone https://github.com/Drink8Water/ClimateNet-Bench
cd ClimateNet-Bench

# Create a conda environment (recommended)
conda create -n climatenet-py311 python=3.11 -y
conda activate climatenet-py311

# Install dependencies
pip install -r requirements.txt
pip install -e .
pip install lightgbm  # optional

# Run tests to verify
python -m pytest tests/ -v
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_split_protocols.py -v

# Smoke test benchmark (end-to-end)
python scripts/build_forecasting_dataset.py --synthetic
python scripts/run_benchmark.py --config configs/benchmark/smoke_test.yaml
python scripts/build_leaderboard.py
```

## Adding a New Model

1. Create a new class in `src/climatenet/models/` inheriting from `ClimateModel` (`src/climatenet/models/base.py`).
2. Implement `fit(train_df, feature_columns, target_column)`, `predict(test_df)`, and `get_model_name()`.
3. Register the model in `src/climatenet/models/model_factory.py` in `_MODEL_REGISTRY`.
4. Add tests in `tests/` verifying fit/predict shapes and no NaN outputs.
5. Add the model name to the benchmark config (`configs/benchmark/evap_anomaly_v1.yaml`).

Example model skeleton:

```python
from climatenet.models.base import ClimateModel

class MyModel(ClimateModel):
    def fit(self, train_df, feature_columns, target_column="y_true", val_df=None):
        # train your model
        return self

    def predict(self, test_df):
        # return np.ndarray of predictions
        return predictions

    def get_model_name(self):
        return "my_model"
```

## Adding a New Split Protocol

1. Add a function in `src/climatenet/benchmark/split_protocols.py` following the `make_*_split(df, ...) -> SplitResult` signature.
2. Ensure the function enforces the appropriate disjointness constraint (e.g. no grid_id overlap for spatial splits).
3. Register it in `generate_all_splits()` and `_default_split_configs()`.
4. Add a YAML config in `configs/splits/`.
5. Add tests verifying the leakage check passes and the constraint holds.

## Adding a New Benchmark Region

1. Add the region definition to `configs/benchmark/evap_anomaly_v1.yaml` under `regions:`.
2. Update `src/climatenet/benchmark/region_registry.py` if new climate types are introduced.
3. Update `backend/config.py` `REGIONS` list.
4. Ensure the region sits within ERA5-Land coverage (land points at 0.1° resolution).

## Coding Style

- Python 3.10+ with `from __future__ import annotations`.
- Follow the existing code style: dataclasses for config objects, type annotations on public functions, Google-style docstrings.
- Use `pathlib.Path` for file paths, not `os.path`.
- Keep functions small and testable. Avoid global mutable state.
- Random seed: use `seed=42` for reproducibility.
- No data leakage: spatial splits must group by grid cell, temporal splits must not use future years in training.
- Run `python -m pytest tests/ -v` before submitting a PR.

## Benchmark Result Submission

To submit benchmark results for the leaderboard:

1. Run the full benchmark suite:
   ```bash
   python scripts/run_benchmark.py --config configs/benchmark/evap_anomaly_v1.yaml
   ```

2. Build the leaderboard:
   ```bash
   python scripts/build_leaderboard.py
   ```

3. Submit a PR with:
   - Your model implementation (if new)
   - The `outputs/benchmark/` directory (excluding large prediction CSVs)
   - A brief description of your model and any hyperparameters

## Questions?

Open an issue on GitHub. We're happy to help!
