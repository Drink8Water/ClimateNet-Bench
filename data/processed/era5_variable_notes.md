# ERA5-Land Variable Notes

These notes describe how `src/preprocess_era5.py` maps ERA5-Land NetCDF variables into the tabular schema used by the ML pipeline.

## Dataset

- CDS dataset: `reanalysis-era5-land-monthly-means`
- Product type: `monthly_averaged_reanalysis`
- Output format requested by this project: NetCDF
- Spatial grid in CDS: regular latitude-longitude, approximately 0.1 degrees

## Variable Mapping

| CDS variable | Typical NetCDF short name | Output column | ERA5 unit | Project unit |
|---|---:|---|---|---|
| `2m_temperature` | `t2m` | `temperature` | K | degrees Celsius |
| `total_precipitation` | `tp` | `precipitation` | m/day water equivalent in monthly means | monthly total mm water equivalent |
| `surface_solar_radiation_downwards` | `ssrd` | `radiation` | J m-2/day in monthly means | monthly total MJ m-2 |
| `volumetric_soil_water_layer_1` | `swvl1` | `soil_moisture` | m3 m-3 | m3 m-3 |
| `10m_u_component_of_wind` | `u10` | `u_wind` | m s-1 | m s-1 |
| `10m_v_component_of_wind` | `v10` | `v_wind` | m s-1 | m s-1 |
| `total_evaporation` | `e` | `evaporation` | m/day water equivalent in monthly means | positive monthly total mm water equivalent |

## Sign Conventions

ERA5 evaporation-like fluxes may be stored as negative values when water leaves the surface. For this project, `total_evaporation` is multiplied by `-1000` so that positive values mean positive land evaporation in millimetres.

For the ERA5-Land `monthly_averaged_reanalysis` product, accumulated variables are monthly means of daily accumulations. The preprocessing script multiplies precipitation, radiation, and evaporation by the number of days in the month so the output table contains monthly totals.

## Caution

ERA5-Land monthly accumulated variables require careful interpretation. For Phase 2, this project keeps the processing simple and reproducible, but any scientific write-up should explicitly describe the CDS product, unit conversions, and the evaporation sign convention.
