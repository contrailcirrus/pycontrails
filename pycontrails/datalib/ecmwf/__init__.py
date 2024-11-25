"""ECMWF Data Access."""

from __future__ import annotations

from pycontrails.datalib.ecmwf.arco_era5 import (
    ERA5ARCO,
    open_arco_era5_model_level_data,
    open_arco_era5_single_level,
)
from pycontrails.datalib.ecmwf.common import CDSCredentialsNotFound
from pycontrails.datalib.ecmwf.era5 import ERA5
from pycontrails.datalib.ecmwf.era5_model_level import ERA5ModelLevel
from pycontrails.datalib.ecmwf.hres import HRES
from pycontrails.datalib.ecmwf.hres_model_level import HRESModelLevel
from pycontrails.datalib.ecmwf.ifs import IFS
from pycontrails.datalib.ecmwf.model_levels import (
    MODEL_LEVELS_PATH,
    ml_to_pl,
    model_level_pressure,
    model_level_reference_pressure,
)
from pycontrails.datalib.ecmwf.variables import (
    ECMWF_VARIABLES,
    MODEL_LEVEL_VARIABLES,
    PRESSURE_LEVEL_VARIABLES,
    SURFACE_VARIABLES,
    CloudAreaFraction,
    CloudAreaFractionInLayer,
    Divergence,
    OzoneMassMixingRatio,
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
    "ECMWF_VARIABLES",
    "ERA5",
    "ERA5ARCO",
    "HRES",
    "IFS",
    "MODEL_LEVELS_PATH",
    "MODEL_LEVEL_VARIABLES",
    "PRESSURE_LEVEL_VARIABLES",
    "SURFACE_VARIABLES",
    "CDSCredentialsNotFound",
    "CloudAreaFraction",
    "CloudAreaFractionInLayer",
    "Divergence",
    "ERA5ModelLevel",
    "HRESModelLevel",
    "OzoneMassMixingRatio",
    "PotentialVorticity",
    "RelativeHumidity",
    "RelativeVorticity",
    "SpecificCloudIceWaterContent",
    "SpecificCloudLiquidWaterContent",
    "SurfaceSolarDownwardRadiation",
    "TOAIncidentSolarRadiation",
    "TopNetSolarRadiation",
    "TopNetThermalRadiation",
    "ml_to_pl",
    "model_level_pressure",
    "model_level_reference_pressure",
    "open_arco_era5_model_level_data",
    "open_arco_era5_single_level",
]
