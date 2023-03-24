"""ECMWF Parameter Support.

Sourced from the ECMWF Parameter DB:
    https://apps.ecmwf.int/codes/grib/param-db
"""

from __future__ import annotations

from pycontrails.core import met_var
from pycontrails.core.met import MetVariable

RelativeVorticity = MetVariable(
    short_name="vo",
    standard_name="atmosphere_upward_relative_vorticity",
    long_name="Vorticity (relative)",
    units="s**-1",
    level_type="isobaricInhPa",
    grib1_id=43,
    ecmwf_id=138,
    grib2_id=(0, 2, 12),
    description=(
        "Atmosphere upward relative vorticity is the vertical component of the 3D air vorticity"
        " vector. The vertical component arises from horizontal velocity only. 'Relative' in this"
        " context means the vorticity of the air relative to the rotating solid earth reference"
        " frame, i.e. excluding the Earth's own rotation. In contrast, the quantity with standard"
        " name atmosphere_upward_absolute_vorticity includes the Earth's rotation. 'Upward'"
        " indicates a vector component which is positive when directed upward (negative downward)."
        " A positive value of atmosphere_upward_relative_vorticity indicates anticlockwise rotation"
        " when viewed from above."
    ),
)

PotentialVorticity = MetVariable(
    short_name="pv",
    standard_name="potential_vorticity",
    long_name="Potential vorticity (K m^2 / kg s)",
    units="K m**2 kg**-1 s**-1",
    level_type="isobaricInhPa",
    grib1_id=128,
    ecmwf_id=60,
    grib2_id=(0, 2, 14),
    amip="pvu",
    description=(
        "Potential vorticity is a measure of the capacity for air to rotate in the atmosphere.If we"
        " ignore the effects of heating and friction, potential vorticity is conserved following an"
        " air parcel.It is used to look for places where large wind storms are likely to originate"
        " and develop.Potential vorticity increases strongly above the tropopause and therefore, it"
        " can also be used in studiesrelated to the stratosphere and stratosphere-troposphere"
        " exchanges. Large wind storms develop when a columnof air in the atmosphere starts to"
        " rotate. Potential vorticity is calculated from the wind, temperature andpressure across a"
        " column of air in the atmosphere."
    ),
)

CloudAreaFractionInLayer = MetVariable(
    short_name="cc",
    standard_name="fraction_of_cloud_cover",
    long_name="Cloud area fraction in atmosphere layer",
    ecmwf_id=248,
    level_type="isobaricInhPa",
    grib2_id=(0, 6, 32),
    units="[0 - 1]",
    amip="cl",
    description=(
        "This parameter is the proportion of a grid box covered by cloud (liquid or ice) at a"
        " specific pressure level."
    ),
)

SpecificCloudLiquidWaterContent = MetVariable(
    short_name="clwc",
    standard_name="specific_cloud_liquid_water_content",
    long_name="Specific cloud liquid water content",
    units="kg kg**-1",
    level_type="isobaricInhPa",
    ecmwf_id=246,
    grib2_id=(0, 1, 83),
    description=(
        "This parameter is the mass of cloud liquid water droplets per kilogram of the total mass"
        " of moist air. The 'total mass of moist air' is the sum of the dry air, water vapour,"
        " cloud liquid, cloud ice, rain and falling snow. This parameter represents the average"
        " value for a grid box."
    ),
)


SpecificCloudIceWaterContent = MetVariable(
    short_name="ciwc",
    standard_name="specific_cloud_ice_water_content",
    long_name="Specific cloud ice water content",
    units="kg kg**-1",
    level_type="isobaricInhPa",
    ecmwf_id=247,
    grib2_id=(0, 1, 84),
    description=(
        "This parameter is the mass of cloud ice particles per kilogram of the total mass of moist"
        " air. The 'total mass of moist air' is the sum of the dry air, water vapour, cloud liquid,"
        " cloud ice, rain and falling snow. This parameter represents the average value for a grid"
        " box."
    ),
)

# Override units and description on Relative humidity
RelativeHumidity = MetVariable(
    short_name=met_var.RelativeHumidity.short_name,
    standard_name=met_var.RelativeHumidity.standard_name,
    long_name=met_var.RelativeHumidity.long_name,
    units="%",
    level_type=met_var.RelativeHumidity.level_type,
    ecmwf_id=met_var.RelativeHumidity.ecmwf_id,
    grib2_id=met_var.RelativeHumidity.grib2_id,
    description=(
        "This parameter is the water vapour pressure as a percentage of the value at which the air"
        " becomes saturated "
        "(the point at which water vapour begins to condense into liquid water or deposition into"
        " ice)."
        "For temperatures over 0°C (273.15 K) it is calculated for saturation over water. "
        "At temperatures below -23°C it is calculated for saturation over ice. "
        "Between -23°C and 0°C this parameter is calculated by interpolating between the ice and"
        " water values using a quadratic function."
        "See https://www.ecmwf.int/sites/default/files/elibrary/2016/17117-part-iv-physical-processes.pdf#subsection.7.4.2"  # noqa: E501
    ),
)

TOAIncidentSolarRadiation = MetVariable(
    short_name="tisr",
    standard_name="toa_incident_solar_radiation",
    long_name="Top of atmosphere incident shortwave radiation",
    units="J m**-2",
    level_type="nominalTop",
    ecmwf_id=212,
    grib2_id=(192, 128, 212),  # reference ECMWF
    description="Top of atmosphere incident solar radiation. Accumulated field.",
)

TopNetSolarRadiation = MetVariable(
    short_name="tsr",
    standard_name="top_net_solar_radiation",
    long_name="Top of atmosphere net solar (shortwave) radiation",
    units="J m**-2",
    level_type="nominalTop",
    ecmwf_id=178,
    grib2_id=(0, 4, 1),
    description=(
        "This parameter is the incoming solar radiation (also known as shortwave radiation) "
        "minus the outgoing solar radiation at the top of the atmosphere. "
        "It is the amount of radiation passing through a horizontal plane. "
        "The incoming solar radiation is the amount received from the Sun. "
        "The outgoing solar radiation is the amount reflected and scattered by the Earth's"
        " atmosphere and surface"
        "See https://www.ecmwf.int/sites/default/files/elibrary/2015/18490-radiation-quantities-ecmwf-model-and-mars.pdf"  # noqa: E501
    ),
)

TopNetThermalRadiation = MetVariable(
    short_name="ttr",
    standard_name="top_net_thermal_radiation",
    long_name="Top of atmosphere net thermal (longwave) radiation",
    units="J m**-2",
    level_type="nominalTop",
    ecmwf_id=179,
    grib2_id=(0, 5, 5),
    description=(
        "The thermal (also known as terrestrial or longwave) "
        "radiation emitted to space at the top of the atmosphere is commonly known as the Outgoing"
        " Longwave Radiation (OLR). "
        "The top net thermal radiation (this parameter) is equal to the negative of OLR."
        "See https://www.ecmwf.int/sites/default/files/elibrary/2015/18490-radiation-quantities-ecmwf-model-and-mars.pdf"  # noqa: E501
    ),
)

SurfaceSolarDownwardRadiation = MetVariable(
    short_name="ssrd",
    standard_name="surface_solar_downward_radiation",
    long_name="Surface Solar Downward Radiation",
    units="J m**-2",
    level_type="surface",
    ecmwf_id=169,
    grib2_id=(0, 4, 7),
    description=(
        "This parameter is the amount of solar radiation (also known as shortwave radiation) that"
        " reaches a horizontal plane at the surface of the Earth. This parameter comprises both"
        " direct and diffuse solar radiation."
    ),
)

CloudAreaFraction = MetVariable(
    short_name="tcc",
    standard_name="total_cloud_cover",
    long_name="Cloud area fraction (total)",
    level_type="surface",
    grib1_id=71,
    ecmwf_id=164,
    grib2_id=(192, 128, 164),  # reference ECMWF
    units="[0 - 1]",
    amip="clt",
    description=(
        "This parameter is the proportion of a grid box covered by cloud (liquid or ice) for a"
        " whole atmosphere column."
    ),
)

PRESSURE_LEVEL_VARIABLES = [
    met_var.AirTemperature,
    met_var.SpecificHumidity,
    met_var.Geopotential,
    met_var.EastwardWind,
    met_var.NorthwardWind,
    met_var.VerticalVelocity,
    RelativeHumidity,
    RelativeVorticity,
    CloudAreaFractionInLayer,
    SpecificCloudIceWaterContent,
    SpecificCloudLiquidWaterContent,
    PotentialVorticity,
]
SURFACE_VARIABLES = [
    met_var.SurfacePressure,
    TOAIncidentSolarRadiation,
    TopNetSolarRadiation,
    TopNetThermalRadiation,
    CloudAreaFraction,
    SurfaceSolarDownwardRadiation,
]

ECMWF_VARIABLES = PRESSURE_LEVEL_VARIABLES + SURFACE_VARIABLES
