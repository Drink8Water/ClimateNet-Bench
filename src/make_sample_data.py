"""Generate synthetic monthly climate-like data for the Phase 1 MVP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import GRID_POINTS_PER_REGION, RANDOM_SEED, REGIONS, SAMPLE_DATA_PATH, YEARS, ensure_directories


def seasonal_cycle(month: int, peak_month: int, amplitude: float) -> float:
    """Return a smooth seasonal signal with a chosen peak month."""
    return amplitude * np.cos(2 * np.pi * (month - peak_month) / 12)


def build_region_grid(region_name: str, rng: np.random.Generator) -> list[tuple[float, float]]:
    """Create random grid points within a region bounding box."""
    region = REGIONS[region_name]
    latitudes = rng.uniform(*region["lat_range"], size=GRID_POINTS_PER_REGION)
    longitudes = rng.uniform(*region["lon_range"], size=GRID_POINTS_PER_REGION)
    return list(zip(latitudes, longitudes))


def generate_record(
    region: str,
    year: int,
    month: int,
    latitude: float,
    longitude: float,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    """Generate one physically plausible synthetic climate record."""
    if region == "Sahara":
        temperature = 30 + seasonal_cycle(month, peak_month=7, amplitude=8) + rng.normal(0, 1.5)
        precipitation = max(0, 5 + seasonal_cycle(month, peak_month=8, amplitude=4) + rng.normal(0, 4))
        radiation = 280 + seasonal_cycle(month, peak_month=6, amplitude=45) + rng.normal(0, 10)
        soil_moisture = max(0.02, 0.08 + 0.0015 * precipitation - 0.0002 * radiation + rng.normal(0, 0.01))
        u_wind = rng.normal(3.5, 1.2)
        v_wind = rng.normal(0.5, 1.0)
    else:
        temperature = 17 + seasonal_cycle(month, peak_month=7, amplitude=12) + rng.normal(0, 1.8)
        monsoon = max(0, seasonal_cycle(month, peak_month=7, amplitude=75))
        precipitation = max(0, 55 + monsoon + rng.normal(0, 18))
        radiation = 190 + seasonal_cycle(month, peak_month=6, amplitude=35) - 0.12 * precipitation + rng.normal(0, 12)
        soil_moisture = max(0.08, 0.24 + 0.0018 * precipitation - 0.00035 * radiation + rng.normal(0, 0.025))
        u_wind = rng.normal(1.8, 1.0)
        v_wind = rng.normal(1.2, 1.0)

    wind_speed = float(np.sqrt(u_wind**2 + v_wind**2))
    evaporation = (
        0.018 * radiation
        + 0.10 * temperature
        + 16.0 * soil_moisture
        + 0.28 * wind_speed
        - 0.003 * precipitation
        + rng.normal(0, 1.2)
    )
    evaporation = max(0, evaporation)

    return {
        "region": region,
        "year": year,
        "month": month,
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "temperature": round(float(temperature), 3),
        "precipitation": round(float(precipitation), 3),
        "radiation": round(float(radiation), 3),
        "soil_moisture": round(float(soil_moisture), 4),
        "u_wind": round(float(u_wind), 3),
        "v_wind": round(float(v_wind), 3),
        "evaporation": round(float(evaporation), 3),
    }


def create_sample_data() -> pd.DataFrame:
    """Create a synthetic spatio-temporal climate dataset."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for region in REGIONS:
        grid_points = build_region_grid(region, rng)
        for latitude, longitude in grid_points:
            for year in YEARS:
                for month in range(1, 13):
                    rows.append(generate_record(region, year, month, latitude, longitude, rng))

    return pd.DataFrame(rows)


def main() -> None:
    """Generate and save synthetic climate data."""
    ensure_directories()
    data = create_sample_data()
    data.to_csv(SAMPLE_DATA_PATH, index=False)
    print(f"Saved sample data to {SAMPLE_DATA_PATH}")
    print(f"Rows: {len(data):,}, Columns: {len(data.columns)}")


if __name__ == "__main__":
    main()

