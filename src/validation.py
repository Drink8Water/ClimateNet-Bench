"""Validation strategies for gridded spatio-temporal climate data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED


@dataclass(frozen=True)
class ValidationSplit:
    """Container for one explicit validation split."""

    strategy: str
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    train_region: str = "mixed"
    test_region: str = "mixed"


def add_grid_cell_id(data: pd.DataFrame) -> pd.DataFrame:
    """Create a stable grid-cell identifier from latitude and longitude."""
    output = data.copy()
    output["grid_cell"] = (
        output["latitude"].round(6).astype(str) + "_" + output["longitude"].round(6).astype(str)
    )
    return output


def random_split(data: pd.DataFrame, test_size: float = 0.2) -> ValidationSplit:
    """Random row split baseline.

    This is useful as a simple reference, but it is optimistic for gridded climate
    data because records from the same latitude-longitude cell can appear in both
    train and test sets.
    """
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=data["region"] if "region" in data.columns else None,
    )
    return ValidationSplit("random_split", train_data.copy(), test_data.copy())


def spatial_holdout(data: pd.DataFrame, test_size: float = 0.2) -> ValidationSplit:
    """Split by unique grid cells so train and test locations do not overlap.

    Spatial holdout is important because gridded climate records are autocorrelated:
    adjacent months at the same grid cell often look similar. If the same grid cell
    appears in both train and test sets, metrics can overstate spatial generalization.
    """
    data_with_cells = add_grid_cell_id(data)
    unique_cells = data_with_cells[["grid_cell", "region"]].drop_duplicates()
    train_cells, test_cells = train_test_split(
        unique_cells,
        test_size=test_size,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=unique_cells["region"] if unique_cells["region"].nunique() > 1 else None,
    )

    train_data = data_with_cells[data_with_cells["grid_cell"].isin(train_cells["grid_cell"])].copy()
    test_data = data_with_cells[data_with_cells["grid_cell"].isin(test_cells["grid_cell"])].copy()

    overlap = set(train_data["grid_cell"]).intersection(set(test_data["grid_cell"]))
    if overlap:
        raise ValueError(f"Spatial holdout leakage: {len(overlap)} grid cells appear in both sets.")

    return ValidationSplit("spatial_holdout", train_data, test_data)


def cross_region_transfer(data: pd.DataFrame) -> list[ValidationSplit]:
    """Create cross-region transfer splits for Sahara and East China."""
    required_regions = {"Sahara", "East China"}
    available_regions = set(data["region"].unique())
    missing_regions = required_regions - available_regions
    if missing_regions:
        raise ValueError(f"Cross-region validation needs both regions. Missing: {sorted(missing_regions)}")

    return [
        ValidationSplit(
            strategy="region_transfer",
            train_data=data[data["region"] == "Sahara"].copy(),
            test_data=data[data["region"] == "East China"].copy(),
            train_region="Sahara",
            test_region="East China",
        ),
        ValidationSplit(
            strategy="region_transfer",
            train_data=data[data["region"] == "East China"].copy(),
            test_data=data[data["region"] == "Sahara"].copy(),
            train_region="East China",
            test_region="Sahara",
        ),
    ]


def build_validation_splits(data: pd.DataFrame) -> list[ValidationSplit]:
    """Return all Phase 3 validation splits."""
    return [random_split(data), spatial_holdout(data), *cross_region_transfer(data)]
