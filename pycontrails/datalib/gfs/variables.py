"""GFS Parameter Support.

Parameter definitions:

- `Reanalysis <https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.anl.shtml>`_
- `Surface <https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f000.shtml>`_
- `Pressure Levels <https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f003.shtml>`_
"""

from __future__ import annotations

from pycontrails.core import met_var
from pycontrails.core.met import MetVariable

TotalCloudCoverIsobaric = MetVariable(
    short_name="tcc",
    standard_name="total_cloud_cover_isobaric",
    long_name="Total cloud cover at isobaric surface",
    level_type="isobaricInhPa",
    ecmwf_id=228164,
    grib2_id=(0, 6, 1),
    units="%",
    description=(
        "This parameter is the percentage of a grid box covered by cloud (liquid or ice) at a"
        " specific pressure level."
    ),
)

CloudIceWaterMixingRatio = MetVariable(
    short_name="icmr",
    standard_name="ice_water_mixing_ratio",
    long_name="Cloud ice water mixing ratio",
    units="kg kg**-1",
    level_type="isobaricInhPa",
    ecmwf_id=260019,
    grib2_id=(0, 1, 23),
    description=(
        "This parameter is the mass of cloud ice particles per kilogram of the total mass of dry"
        " air. "
    ),
)


TOAUpwardShortwaveRadiation = MetVariable(
    short_name="uswrf",
    standard_name="toa_upward_shortwave_flux",
    long_name="Top of atmosphere upward shortwave radiation",
    units="W m**-2",
    level_type="nominalTop",
    grib2_id=(0, 4, 193),
    description=(
        "This parameter is the outgoing shortwave (solar) radiation at the nominal top of the"
        " atmosphere."
    ),
)

TOAUpwardLongwaveRadiation = MetVariable(
    short_name="ulwrf",
    standard_name="toa_upward_longwave_flux",
    long_name="Top of atmosphere upward longwave radiation",
    units="W m**-2",
    level_type="nominalTop",
    grib2_id=(0, 5, 193),
    description=(
        "This parameter is the outgoing longwave (thermal) radiation at the nominal top of the"
        " atmosphere."
    ),
)

Visibility = MetVariable(
    short_name="vis",
    standard_name="visibility",
    long_name="Visibility at ground or water surface",
    units="m",
    level_type="surface",
    grib2_id=(0, 19, 0),
    description="This parameter is the visibility at the ground or water surface, in meters.",
)


PRESSURE_LEVEL_VARIABLES = [
    met_var.AirTemperature,
    met_var.SpecificHumidity,
    met_var.RelativeHumidity,
    met_var.GeopotentialHeight,
    met_var.EastwardWind,
    met_var.NorthwardWind,
    met_var.VerticalVelocity,
    CloudIceWaterMixingRatio,
    TotalCloudCoverIsobaric,
]
SURFACE_VARIABLES = [
    met_var.SurfacePressure,
    Visibility,
    TOAUpwardShortwaveRadiation,
    TOAUpwardLongwaveRadiation,
]

GFS_VARIABLES = PRESSURE_LEVEL_VARIABLES + SURFACE_VARIABLES
