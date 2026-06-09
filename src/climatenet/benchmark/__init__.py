"""ClimateNet-Bench benchmark definitions, region registry, and split protocols.

This subpackage provides:

- ``Region`` dataclass with validation and conversion helpers
- ``RegionRegistry`` for managing fixed benchmark regions
- ``split_protocols`` — 6 train/val/test split protocols of increasing difficulty
- YAML-based benchmark configuration support

Longitude convention
--------------------
All longitude values in this module use the **degrees east** convention
(negative = west of Greenwich, positive = east of Greenwich, range -180 to +180).
This is the standard convention for ERA5-Land and most climate libraries.

When converting to CDS API format, bounds are reordered to
``[North, West, South, East]`` (``[lat_max, lon_min, lat_min, lon_max]``)
because that is the format the Copernicus Climate Data Store expects.

Split protocols
---------------
1. **random** — optimistic baseline (sample-level shuffle)
2. **spatial_block** — hold out entire spatial blocks
3. **temporal** — train on earlier years, test on future years
4. **region_transfer** — train on some regions, test on disjoint regions
5. **climate_zone_transfer** — train on some climate zones, test on held-out zone
6. **spatiotemporal** — joint spatial-block + temporal holdout
"""

from climatenet.benchmark.leaderboard import build_leaderboard
from climatenet.benchmark.region_registry import Region, RegionRegistry, get_default_registry
from climatenet.benchmark.split_protocols import (
    SplitResult,
    generate_all_splits,
    load_split_result,
    make_random_split,
    make_spatial_block_split,
    make_temporal_split,
    make_region_transfer_split,
    make_climate_zone_transfer_split,
    make_spatiotemporal_split,
    save_split_result,
    validate_split,
)
