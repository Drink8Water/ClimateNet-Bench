"""Sequence dataset construction for temporal climate models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - exercised only before torch install
    raise ImportError("PyTorch is required for SequenceDataset. Install with: pip install -r requirements.txt") from exc


@dataclass(frozen=True)
class SequenceMetadata:
    """Metadata attached to one sequence sample."""

    region: str
    year: int
    month: int
    latitude: float
    longitude: float


class SequenceDataset(Dataset):
    """PyTorch dataset for sliding windows over grid-cell climate time series.

    Each sample uses the past `sequence_length` rows of climate features from one
    grid cell to predict the next month's evaporation anomaly.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        metadata: list[SequenceMetadata],
    ) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
        self.metadata = metadata

    def __len__(self) -> int:
        """Return number of sequence samples."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one sequence and target."""
        return self.sequences[index], self.targets[index]


def build_sequence_arrays(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "evaporation_anomaly",
    sequence_length: int = 6,
) -> tuple[np.ndarray, np.ndarray, list[SequenceMetadata]]:
    """Build sequence arrays from a tabular climate feature table.

    The dataframe is grouped by grid cell, sorted by year/month, and converted to
    sliding windows. For a window ending at time t, the target is t+1.
    """
    required_columns = [
        "region",
        "year",
        "month",
        "latitude",
        "longitude",
        *feature_columns,
        target_column,
    ]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Cannot build sequences; missing columns: {missing_columns}")

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[SequenceMetadata] = []
    group_columns = ["region", "latitude", "longitude"]

    for (region, latitude, longitude), group in data.groupby(group_columns):
        group = group.sort_values(["year", "month"]).reset_index(drop=True)
        if len(group) <= sequence_length:
            continue

        feature_values = group[feature_columns].to_numpy(dtype=np.float32)
        target_values = group[target_column].to_numpy(dtype=np.float32)

        for start_index in range(0, len(group) - sequence_length):
            target_index = start_index + sequence_length
            window = feature_values[start_index:target_index]
            target = target_values[target_index]
            target_row = group.iloc[target_index]

            sequences.append(window)
            targets.append(float(target))
            metadata.append(
                SequenceMetadata(
                    region=str(region),
                    year=int(target_row["year"]),
                    month=int(target_row["month"]),
                    latitude=float(latitude),
                    longitude=float(longitude),
                )
            )

    if not sequences:
        raise ValueError(
            "No sequence samples were created. Check sequence_length and make sure each grid cell "
            "has enough monthly records."
        )

    return np.stack(sequences), np.asarray(targets, dtype=np.float32), metadata


def build_sequence_dataset(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "evaporation_anomaly",
    sequence_length: int = 6,
) -> SequenceDataset:
    """Build a SequenceDataset directly from a dataframe."""
    sequences, targets, metadata = build_sequence_arrays(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
    )
    return SequenceDataset(sequences, targets, metadata)


def metadata_to_dataframe(metadata: list[SequenceMetadata]) -> pd.DataFrame:
    """Convert sequence metadata objects to a dataframe."""
    return pd.DataFrame([item.__dict__ for item in metadata])
