"""Validation split tests."""

from __future__ import annotations

import pandas as pd

from climatenet.training.validation import spatial_holdout


def test_spatial_holdout_has_no_grid_overlap() -> None:
    """Spatial holdout should not share grid cells between train and test."""
    rows = []
    for region, lon in [("Sahara", 10.0), ("East China", 110.0)]:
        for cell in range(5):
            for month in range(1, 4):
                rows.append(
                    {
                        "region": region,
                        "year": 2020,
                        "month": month,
                        "latitude": 20.0 + cell,
                        "longitude": lon + cell,
                    }
                )
    data = pd.DataFrame(rows)
    split = spatial_holdout(data, test_size=0.4, seed=42)

    train_cells = set(split.train_data["grid_cell"])
    test_cells = set(split.test_data["grid_cell"])
    assert not train_cells.intersection(test_cells)
