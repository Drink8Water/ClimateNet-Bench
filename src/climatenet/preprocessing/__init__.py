"""Preprocessing utilities for ClimateNet-Bench.

Provides train-only climatology computation and anomaly transformation
with strict anti-leakage guarantees.
"""

from climatenet.preprocessing.climatology import (
    apply_monthly_anomaly,
    build_train_only_anomaly,
    compute_monthly_climatology,
)
