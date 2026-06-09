"""Validation split strategies for spatio-temporal climate data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ValidationSplit:
    """One explicit train/test split."""

    strategy: str
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    train_region: str = "mixed"
    test_region: str = "mixed"
    train_period: str = "mixed"
    test_period: str = "mixed"


def add_grid_cell_id(data: pd.DataFrame) -> pd.DataFrame:
    """Create a stable grid-cell identifier from latitude and longitude."""
    output = data.copy()
    output["grid_cell"] = (
        output["latitude"].round(6).astype(str) + "_" + output["longitude"].round(6).astype(str)
    )
    return output


def random_split(data: pd.DataFrame, test_size: float, seed: int) -> ValidationSplit:
    """Random row split baseline."""
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=data["region"] if "region" in data.columns else None,
    )
    return ValidationSplit("random_split", train_data.copy(), test_data.copy())


def spatial_holdout(data: pd.DataFrame, test_size: float, seed: int) -> ValidationSplit:
    """Hold out full grid cells to test spatial generalization."""
    data_with_cells = add_grid_cell_id(data)
    unique_cells = data_with_cells[["grid_cell", "region"]].drop_duplicates()
    train_cells, test_cells = train_test_split(
        unique_cells,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=unique_cells["region"] if unique_cells["region"].nunique() > 1 else None,
    )

    train_data = data_with_cells[data_with_cells["grid_cell"].isin(train_cells["grid_cell"])].copy()
    test_data = data_with_cells[data_with_cells["grid_cell"].isin(test_cells["grid_cell"])].copy()
    overlap = set(train_data["grid_cell"]).intersection(set(test_data["grid_cell"]))
    if overlap:
        raise ValueError(f"Spatial holdout leakage: {len(overlap)} overlapping grid cells.")
    return ValidationSplit("spatial_holdout", train_data, test_data)


def temporal_holdout(
    data: pd.DataFrame,
    train_start_year: int,
    train_end_year: int,
    test_year: int,
) -> ValidationSplit:
    """Train on an earlier period and test on a future holdout year.

    Temporal holdout avoids leakage from future climate conditions into the
    training set and is the clearest validation strategy for forecasting use cases.
    """
    train_data = data[(data["year"] >= train_start_year) & (data["year"] <= train_end_year)].copy()
    test_data = data[data["year"] == test_year].copy()
    if train_data.empty or test_data.empty:
        raise ValueError(
            f"Temporal holdout split is empty: train={train_start_year}-{train_end_year}, test={test_year}"
        )
    return ValidationSplit(
        strategy="temporal_holdout",
        train_data=train_data,
        test_data=test_data,
        train_period=f"{train_start_year}-{train_end_year}",
        test_period=str(test_year),
    )


def region_transfer(data: pd.DataFrame, train_region: str, test_region: str) -> ValidationSplit:
    """Train on one region and test on another."""
    train_data = data[data["region"] == train_region].copy()
    test_data = data[data["region"] == test_region].copy()
    if train_data.empty or test_data.empty:
        raise ValueError(f"Region transfer split is empty: {train_region} -> {test_region}")
    train_period = f"{int(train_data['year'].min())}-{int(train_data['year'].max())}"
    test_period = f"{int(test_data['year'].min())}-{int(test_data['year'].max())}"
    return ValidationSplit("region_transfer", train_data, test_data, train_region, test_region, train_period, test_period)


def build_validation_splits(data: pd.DataFrame, experiment_config: dict) -> list[ValidationSplit]:
    """Build validation splits from experiment YAML config."""
    strategies = experiment_config.get("validation_strategies", ["random_split"])
    test_size = float(experiment_config.get("test_size", 0.2))
    seed = int(experiment_config.get("seed", 42))
    splits: list[ValidationSplit] = []

    if "random_split" in strategies:
        splits.append(random_split(data, test_size=test_size, seed=seed))

    if "spatial_holdout" in strategies:
        splits.append(spatial_holdout(data, test_size=test_size, seed=seed))

    if "temporal_holdout" in strategies:
        splits.append(
            temporal_holdout(
                data,
                train_start_year=int(experiment_config.get("train_start_year", 2019)),
                train_end_year=int(experiment_config.get("train_end_year", 2022)),
                test_year=int(experiment_config.get("test_year", 2023)),
            )
        )

    if "region_transfer" in strategies:
        for pair in experiment_config.get("region_transfer_pairs", []):
            splits.append(region_transfer(data, pair["train_region"], pair["test_region"]))

    return splits
