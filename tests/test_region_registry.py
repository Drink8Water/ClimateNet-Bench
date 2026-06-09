"""Tests for the benchmark region registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from climatenet.benchmark.region_registry import (
    Region,
    RegionRegistry,
    filter_by_climate_type,
    get_default_registry,
    get_region,
    list_regions,
    load_regions_from_yaml,
    validate_region_bounds,
)


# ---------------------------------------------------------------------------
# Region dataclass tests
# ---------------------------------------------------------------------------


class TestRegion:
    """Unit tests for the Region dataclass."""

    def test_valid_region_construction(self) -> None:
        r = Region("Test", 10.0, 20.0, -5.0, 5.0, "temperate")
        assert r.name == "Test"
        assert r.lat_span == 10.0
        assert r.lon_span == 10.0
        assert r.center == (15.0, 0.0)

    def test_contains_point_inside(self) -> None:
        r = Region("Test", 10.0, 20.0, -5.0, 5.0, "temperate")
        assert r.contains(15.0, 0.0) is True

    def test_contains_point_outside_lat(self) -> None:
        r = Region("Test", 10.0, 20.0, -5.0, 5.0, "temperate")
        assert r.contains(25.0, 0.0) is False

    def test_contains_point_outside_lon(self) -> None:
        r = Region("Test", 10.0, 20.0, -5.0, 5.0, "temperate")
        assert r.contains(15.0, 10.0) is False

    def test_to_cds_area(self) -> None:
        """CDS format is [North, West, South, East]."""
        r = Region("Test", 10.0, 20.0, -5.0, 5.0, "temperate")
        assert r.to_cds_area() == [20.0, -5.0, 10.0, 5.0]

    def test_sahara_to_cds_area(self) -> None:
        sahara = Region("Sahara", 15.0, 30.0, -20.0, 30.0, "arid")
        assert sahara.to_cds_area() == [30.0, -20.0, 15.0, 30.0]

    def test_frozen_after_creation(self) -> None:
        r = Region("Test", 10.0, 20.0, -5.0, 5.0, "temperate")
        with pytest.raises(Exception):
            r.lat_min = 12.0  # type: ignore[misc]

    # --- bounds validation ---

    @pytest.mark.parametrize(
        "lat_min,lat_max,lon_min,lon_max,expected_error",
        [
            # Use regex-safe substrings (error messages contain brackets
            # and parentheses that break re.search patterns).
            (-100, 20, 0, 10, r"lat_min must be in"),
            (10, 100, 0, 10, r"lat_max must be in"),
            (20, 10, 0, 10, r"lat_min .* must be .* lat_max"),
            (10, 20, -200, 10, r"lon_min must be in"),
            (10, 20, 0, 200, r"lon_max must be in"),
            (10, 20, 10, 0, r"lon_min .* must be .* lon_max"),
        ],
    )
    def test_invalid_bounds_raise_error(
        self, lat_min, lat_max, lon_min, lon_max, expected_error
    ) -> None:
        with pytest.raises(ValueError, match=expected_error):
            Region("Bad", lat_min, lat_max, lon_min, lon_max, "arid")


# ---------------------------------------------------------------------------
# RegionRegistry tests — from YAML
# ---------------------------------------------------------------------------

SAMPLE_YAML = """
benchmark_name: test
regions:
  - name: Sahara
    lat_min: 15.0
    lat_max: 30.0
    lon_min: -20.0
    lon_max: 30.0
    climate_type: arid
    description: North African desert
  - name: East China
    lat_min: 20.0
    lat_max: 35.0
    lon_min: 105.0
    lon_max: 122.0
    climate_type: monsoon
  - name: Amazon
    lat_min: -15.0
    lat_max: 5.0
    lon_min: -75.0
    lon_max: -50.0
    climate_type: tropical_humid
  - name: Central Europe
    lat_min: 45.0
    lat_max: 55.0
    lon_min: 0.0
    lon_max: 20.0
    climate_type: temperate
  - name: Western US
    lat_min: 30.0
    lat_max: 45.0
    lon_min: -125.0
    lon_max: -105.0
    climate_type: semi_arid
"""


@pytest.fixture
def sample_registry() -> RegionRegistry:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(SAMPLE_YAML)
        tmp_path = f.name
    registry = RegionRegistry.from_yaml(tmp_path)
    Path(tmp_path).unlink()
    return registry


class TestRegionRegistry:
    """Unit tests for RegionRegistry."""

    def test_all_five_regions_exist(self, sample_registry: RegionRegistry) -> None:
        names = sample_registry.list_regions()
        assert len(names) == 5
        expected = ["Amazon", "Central Europe", "East China", "Sahara", "Western US"]
        assert names == expected

    def test_get_existing_region(self, sample_registry: RegionRegistry) -> None:
        r = sample_registry.get("Sahara")
        assert r.name == "Sahara"
        assert r.lat_min == 15.0
        assert r.lat_max == 30.0
        assert r.lon_min == -20.0
        assert r.lon_max == 30.0
        assert r.climate_type == "arid"

    def test_get_missing_region_raises(self, sample_registry: RegionRegistry) -> None:
        with pytest.raises(KeyError, match="Antarctica"):
            sample_registry.get("Antarctica")

    def test_filter_by_climate_type_arid(self, sample_registry: RegionRegistry) -> None:
        arid = sample_registry.filter_by_climate_type("arid")
        assert len(arid) == 1
        assert arid[0].name == "Sahara"

    def test_filter_by_climate_type_no_match(self, sample_registry: RegionRegistry) -> None:
        result = sample_registry.filter_by_climate_type("polar")
        assert result == []

    def test_climate_types(self, sample_registry: RegionRegistry) -> None:
        types = sample_registry.climate_types()
        assert "arid" in types
        assert "monsoon" in types
        assert "tropical_humid" in types
        assert "temperate" in types
        assert "semi_arid" in types

    def test_contains(self, sample_registry: RegionRegistry) -> None:
        assert "Sahara" in sample_registry
        assert "Antarctica" not in sample_registry

    def test_len(self, sample_registry: RegionRegistry) -> None:
        assert len(sample_registry) == 5

    def test_duplicate_region_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate region name"):
            RegionRegistry(
                [
                    Region("X", 0, 10, 0, 10, "arid"),
                    Region("X", 10, 20, 10, 20, "arid"),
                ]
            )

    # --- validate_region_bounds ---

    def test_validate_matching_region(self, sample_registry: RegionRegistry) -> None:
        r = Region("Sahara", 15.0, 30.0, -20.0, 30.0, "arid")
        warnings = sample_registry.validate_region_bounds(r)
        assert warnings == []

    def test_validate_unregistered_region(self, sample_registry: RegionRegistry) -> None:
        r = Region("Unknown", 0, 10, 0, 10, "arid")
        warnings = sample_registry.validate_region_bounds(r)
        assert len(warnings) == 1
        assert "not in the benchmark registry" in warnings[0]

    def test_validate_mismatched_bounds(self, sample_registry: RegionRegistry) -> None:
        r = Region("Sahara", 10.0, 30.0, -20.0, 30.0, "arid")
        warnings = sample_registry.validate_region_bounds(r)
        assert any("lat_min mismatch" in w for w in warnings)

    def test_validate_mismatched_climate_type(self, sample_registry: RegionRegistry) -> None:
        r = Region("Sahara", 15.0, 30.0, -20.0, 30.0, "monsoon")
        warnings = sample_registry.validate_region_bounds(r)
        assert any("climate_type mismatch" in w for w in warnings)

    # --- round-trip ---

    def test_to_dict_and_reload(self, sample_registry: RegionRegistry) -> None:
        d = sample_registry.to_dict()
        reloaded = RegionRegistry(
            [
                Region(
                    name=e["name"],
                    lat_min=e["lat_min"],
                    lat_max=e["lat_max"],
                    lon_min=e["lon_min"],
                    lon_max=e["lon_max"],
                    climate_type=e["climate_type"],
                    description=e.get("description", ""),
                )
                for e in d["regions"]
            ]
        )
        assert reloaded.list_regions() == sample_registry.list_regions()


# ---------------------------------------------------------------------------
# YAML loading tests (using real config files)
# ---------------------------------------------------------------------------


class TestYamlLoading:
    """Integration-style tests that load the actual benchmark YAML files."""

    def test_load_evap_anomaly_v1_yaml(self) -> None:
        registry = RegionRegistry.from_yaml("configs/benchmark/evap_anomaly_v1.yaml")
        assert len(registry) == 5
        # Verify all required regions
        for name in ["Sahara", "East China", "Amazon", "Central Europe", "Western US"]:
            r = registry.get(name)
            assert r.lat_min < r.lat_max
            assert r.lon_min < r.lon_max
            assert r.climate_type != ""

    def test_load_smoke_test_yaml(self) -> None:
        registry = RegionRegistry.from_yaml("configs/benchmark/smoke_test.yaml")
        assert len(registry) == 2
        assert "Sahara" in registry
        assert "East China" in registry

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            RegionRegistry.from_yaml("configs/benchmark/nonexistent.yaml")


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for the top-level convenience functions."""

    def test_get_region(self) -> None:
        r = get_region("Amazon")
        assert r.climate_type == "tropical_humid"

    def test_list_regions(self) -> None:
        names = list_regions()
        assert len(names) == 5
        assert "Western US" in names

    def test_filter_by_climate_type(self) -> None:
        semi = filter_by_climate_type("semi_arid")
        assert len(semi) == 1
        assert semi[0].name == "Western US"

    def test_validate_region_bounds_pass(self) -> None:
        r = Region("East China", 20.0, 35.0, 105.0, 122.0, "monsoon")
        assert validate_region_bounds(r) == []

    def test_validate_region_bounds_fail(self) -> None:
        r = Region("East China", 0.0, 35.0, 105.0, 122.0, "monsoon")
        warnings = validate_region_bounds(r)
        assert len(warnings) > 0

    def test_load_regions_from_yaml(self) -> None:
        registry = load_regions_from_yaml("configs/benchmark/evap_anomaly_v1.yaml")
        assert len(registry) == 5
