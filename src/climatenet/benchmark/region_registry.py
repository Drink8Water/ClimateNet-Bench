"""Benchmark region registry.

Defines the ``Region`` dataclass and ``RegionRegistry`` for managing
fixed geographic benchmark regions.

Longitude convention
--------------------
**Degrees east** (negative = western hemisphere, positive = eastern hemisphere).

CDS API note
------------
The Copernicus Climate Data Store expects area bounds in the order
``[North, West, South, East]`` = ``[lat_max, lon_min, lat_min, lon_max]``.
Use :func:`Region.to_cds_area` to convert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from climatenet.utils.paths import resolve_project_path

# ---------------------------------------------------------------------------
# Region dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A fixed geographic benchmark region.

    Attributes
    ----------
    name
        Unique region identifier (e.g. ``"Sahara"``).
    lat_min
        Minimum latitude in degrees north (-90 to 90).
    lat_max
        Maximum latitude in degrees north (-90 to 90).
    lon_min
        Minimum longitude in **degrees east** (-180 to +180).
    lon_max
        Maximum longitude in **degrees east** (-180 to +180).
    climate_type
        Broad climate classification (``"arid"``, ``"monsoon"``, etc.).
    description
        Optional human-readable description.
    """

    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    climate_type: str
    description: str = ""

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate bounds on construction."""
        if not (-90.0 <= self.lat_min <= 90.0):
            raise ValueError(
                f"lat_min must be in [-90, 90], got {self.lat_min} for region '{self.name}'"
            )
        if not (-90.0 <= self.lat_max <= 90.0):
            raise ValueError(
                f"lat_max must be in [-90, 90], got {self.lat_max} for region '{self.name}'"
            )
        if self.lat_min >= self.lat_max:
            raise ValueError(
                f"lat_min ({self.lat_min}) must be < lat_max ({self.lat_max}) for region '{self.name}'"
            )
        if not (-180.0 <= self.lon_min <= 180.0):
            raise ValueError(
                f"lon_min must be in [-180, 180], got {self.lon_min} for region '{self.name}'"
            )
        if not (-180.0 <= self.lon_max <= 180.0):
            raise ValueError(
                f"lon_max must be in [-180, 180], got {self.lon_max} for region '{self.name}'"
            )
        if self.lon_min >= self.lon_max:
            raise ValueError(
                f"lon_min ({self.lon_min}) must be < lon_max ({self.lon_max}) for region '{self.name}'"
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @property
    def lat_span(self) -> float:
        """Return latitude range in degrees."""
        return self.lat_max - self.lat_min

    @property
    def lon_span(self) -> float:
        """Return longitude range in degrees."""
        return self.lon_max - self.lon_min

    @property
    def center(self) -> tuple[float, float]:
        """Return (center_lat, center_lon) of the region."""
        return (
            (self.lat_min + self.lat_max) / 2.0,
            (self.lon_min + self.lon_max) / 2.0,
        )

    def contains(self, lat: float, lon: float) -> bool:
        """Check whether a point lies inside this region."""
        return (
            self.lat_min <= lat <= self.lat_max
            and self.lon_min <= lon <= self.lon_max
        )

    def to_cds_area(self) -> list[float]:
        """Return bounds in CDS API format ``[North, West, South, East]``.

        The Copernicus Climate Data Store order is different from the
        conventional lat/lon order used internally by this module.
        """
        return [self.lat_max, self.lon_min, self.lat_min, self.lon_max]


# ---------------------------------------------------------------------------
# RegionRegistry
# ---------------------------------------------------------------------------


class RegionRegistry:
    """An immutable registry of benchmark regions loaded from a YAML definition.

    Typical usage::

        registry = RegionRegistry.from_yaml("configs/benchmark/evap_anomaly_v1.yaml")
        sahara = registry.get("Sahara")
        arid = registry.filter_by_climate_type("arid")
    """

    def __init__(self, regions: list[Region]) -> None:
        """Create a registry from a list of Region objects."""
        seen: set[str] = set()
        for r in regions:
            if r.name in seen:
                raise ValueError(f"Duplicate region name in registry: {r.name}")
            seen.add(r.name)
        self._regions: dict[str, Region] = {r.name: r for r in regions}

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------

    def get(self, name: str) -> Region:
        """Return the Region with the given name.

        Raises ``KeyError`` if the name is not found.
        """
        if name not in self._regions:
            raise KeyError(
                f"Region '{name}' not found. Available: {list(self._regions.keys())}"
            )
        return self._regions[name]

    def list_regions(self) -> list[str]:
        """Return sorted list of all registered region names."""
        return sorted(self._regions.keys())

    def list_all(self) -> list[Region]:
        """Return all Region objects (sorted by name)."""
        return sorted(self._regions.values(), key=lambda r: r.name)

    def filter_by_climate_type(self, climate_type: str) -> list[Region]:
        """Return all regions matching a given climate type."""
        return [
            r
            for r in self._regions.values()
            if r.climate_type == climate_type
        ]

    def climate_types(self) -> list[str]:
        """Return sorted unique climate types across all regions."""
        return sorted({r.climate_type for r in self._regions.values()})

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> RegionRegistry:
        """Load regions from a YAML config file.

        The YAML file must contain a top-level ``regions`` key whose value
        is a list of region definitions.  Each definition must include at
        minimum ``name``, ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``,
        and ``climate_type``.
        """
        resolved = resolve_project_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Benchmark config not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        if not isinstance(raw, dict):
            raise ValueError(f"Benchmark config must be a YAML mapping: {resolved}")

        region_dicts: list[dict[str, Any]] = raw.get("regions", [])
        if not region_dicts:
            raise ValueError(f"No 'regions' key found in benchmark config: {resolved}")

        regions: list[Region] = []
        for entry in region_dicts:
            regions.append(
                Region(
                    name=str(entry["name"]),
                    lat_min=float(entry["lat_min"]),
                    lat_max=float(entry["lat_max"]),
                    lon_min=float(entry["lon_min"]),
                    lon_max=float(entry["lon_max"]),
                    climate_type=str(entry["climate_type"]),
                    description=str(entry.get("description", "")),
                )
            )
        return cls(regions)

    def to_dict(self) -> dict[str, Any]:
        """Export all regions as a dictionary suitable for YAML serialization."""
        return {
            "regions": [
                {
                    "name": r.name,
                    "lat_min": r.lat_min,
                    "lat_max": r.lat_max,
                    "lon_min": r.lon_min,
                    "lon_max": r.lon_max,
                    "climate_type": r.climate_type,
                    "description": r.description,
                }
                for r in self.list_all()
            ]
        }

    def validate_region_bounds(self, region: Region) -> list[str]:
        """Validate a Region against the registered definition.

        Returns a list of validation warning messages (empty if all checks pass).

        Checks performed:
        - Region name is registered
        - Bounds match the registered definition exactly
        """
        warnings: list[str] = []
        try:
            registered = self.get(region.name)
        except KeyError:
            return [f"Region '{region.name}' is not in the benchmark registry."]

        if registered.lat_min != region.lat_min:
            warnings.append(
                f"lat_min mismatch: got {region.lat_min}, "
                f"expected {registered.lat_min} for '{region.name}'"
            )
        if registered.lat_max != region.lat_max:
            warnings.append(
                f"lat_max mismatch: got {region.lat_max}, "
                f"expected {registered.lat_max} for '{region.name}'"
            )
        if registered.lon_min != region.lon_min:
            warnings.append(
                f"lon_min mismatch: got {region.lon_min}, "
                f"expected {registered.lon_min} for '{region.name}'"
            )
        if registered.lon_max != region.lon_max:
            warnings.append(
                f"lon_max mismatch: got {region.lon_max}, "
                f"expected {registered.lon_max} for '{region.name}'"
            )
        if registered.climate_type != region.climate_type:
            warnings.append(
                f"climate_type mismatch: got '{region.climate_type}', "
                f"expected '{registered.climate_type}' for '{region.name}'"
            )
        return warnings

    def __len__(self) -> int:
        return len(self._regions)

    def __contains__(self, name: str) -> bool:
        return name in self._regions

    def __repr__(self) -> str:
        return f"RegionRegistry([{', '.join(sorted(self._regions.keys()))}])"


# ---------------------------------------------------------------------------
# singleton-like default registry
# ---------------------------------------------------------------------------

_default_registry: RegionRegistry | None = None


def get_default_registry() -> RegionRegistry:
    """Return the default benchmark region registry (lazy-loaded).

    Loads from ``configs/benchmark/evap_anomaly_v1.yaml`` on first call.
    Subsequent calls return the cached instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = RegionRegistry.from_yaml("configs/benchmark/evap_anomaly_v1.yaml")
    return _default_registry


# ---------------------------------------------------------------------------
# convenience top-level functions
# ---------------------------------------------------------------------------


def get_region(name: str) -> Region:
    """Convenience: get a single region from the default registry."""
    return get_default_registry().get(name)


def list_regions() -> list[str]:
    """Convenience: list all region names in the default registry."""
    return get_default_registry().list_regions()


def filter_by_climate_type(climate_type: str) -> list[Region]:
    """Convenience: filter default registry by climate type."""
    return get_default_registry().filter_by_climate_type(climate_type)


def validate_region_bounds(region: Region) -> list[str]:
    """Convenience: validate a Region against the default registry."""
    return get_default_registry().validate_region_bounds(region)


def load_regions_from_yaml(path: str | Path) -> RegionRegistry:
    """Convenience: load a RegionRegistry from a YAML config file."""
    return RegionRegistry.from_yaml(path)
