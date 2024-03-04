"""ECMWF Data Access."""

from __future__ import annotations

from pycontrails.datalib.ecmwf.arco_era5 import ARCOERA5
from pycontrails.datalib.ecmwf.era5 import ERA5
from pycontrails.datalib.ecmwf.hres import HRES
from pycontrails.datalib.ecmwf.ifs import IFS
from pycontrails.datalib.ecmwf.variables import (
    ECMWF_VARIABLES,
    PRESSURE_LEVEL_VARIABLES,
    SURFACE_VARIABLES,
    CloudAreaFraction,
    CloudAreaFractionInLayer,
    Divergence,
    PotentialVorticity,
    RelativeHumidity,
    RelativeVorticity,
    SpecificCloudIceWaterContent,
    SpecificCloudLiquidWaterContent,
    SurfaceSolarDownwardRadiation,
    TOAIncidentSolarRadiation,
    TopNetSolarRadiation,
    TopNetThermalRadiation,
)

__all__ = [
    "ARCOERA5",
    "ERA5",
    "HRES",
    "IFS",
    "CloudAreaFraction",
    "CloudAreaFractionInLayer",
    "Divergence",
    "PotentialVorticity",
    "RelativeHumidity",
    "RelativeVorticity",
    "SpecificCloudIceWaterContent",
    "SpecificCloudLiquidWaterContent",
    "SurfaceSolarDownwardRadiation",
    "TOAIncidentSolarRadiation",
    "TopNetSolarRadiation",
    "TopNetThermalRadiation",
    "ECMWF_VARIABLES",
    "PRESSURE_LEVEL_VARIABLES",
    "SURFACE_VARIABLES",
]
