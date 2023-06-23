"""Module containing core met variables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetVariable:
    """Met variable defined using CF, ECMWF, and WMO conventions.

    When there is a conflict between CF, ECMWF, and WMO conventions,
    CF takes precendence, then WMO, then ECMWF.

    References
    ----------
    - `CF Standard Names, version 77
      <https://cfconventions.org/Data/cf-standard-names/77/build/cf-standard-name-table.html>`_
    - `ECMWF Parameter Database <https://apps.ecmwf.int/codes/grib/param-db>`_
    - `NCEP Grib v1 Code Table <https://www.nco.ncep.noaa.gov/pmb/docs/on388/table2.html>`
    - `WMO Codes Registry, Grib Edition 2 <https://codes.wmo.int/_grib2>`_
    - `NCEP Grib v2 Code Table <https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2.shtml>`

    Used for defining support parameters in a grib-like fashion.
    """  # noqa: E501

    #: Short variable name.
    #: Chosen for greatest consistency between data sources.
    short_name: str

    #: CF standard name, if defined.
    #: Otherwise a standard name is chosen for consistency.
    standard_name: str

    #: Long variable name.
    long_name: str | None = None

    #: Level type
    #: One of "surface", "isobaricInhPa", "nominalTop"
    level_type: str | None = None

    #: ECMWF Grib variable id, if defined.
    #: See `ECMWF Parameter Database <https://apps.ecmwf.int/codes/grib/param-db>`_
    ecmwf_id: int | None = None

    #: WMO Grib v1 variable id, if defined.
    #: See `WMO Codes Registry, Grib Edition 1 <https://codes.wmo.int/_grib1>`_
    #: and `CF Standard Names, version 77 <https://cfconventions.org/Data/cf-standard-names/77/build/cf-standard-name-table.html>`_  # noqa: E501
    grib1_id: int | None = None

    #: WMO Grib 2 variable id, if defined.
    #: See `WMO Codes Registry, Grib Edition 2 <https://codes.wmo.int/_grib2>`_
    #: Tuple represents (disciple, category, number)
    grib2_id: tuple[int, int, int] | None = None

    #: Canonical CF units, if defined.
    units: str | None = None

    #: AMIP identifier, if defined.
    amip: str | None = None

    #: Description
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate inputs.

        Raises
        ------
        ValueError
            If any of the inputs have an unknown :attr:`level_type`.
        """
        level_types = ["surface", "isobaricInPa", "isobaricInhPa", "nominalTop"]
        if self.level_type is not None and self.level_type not in level_types:
            raise ValueError(f"`level_type` must be one of {level_types}")

    @property
    def ecmwf_link(self) -> str | None:
        """Database link in the ECMWF Paramter Database if :attr:`ecmwf_id` is defined.

        Returns
        -------
        str
            Database link in the ECMWF Paramter Database
        """
        return (
            f"https://apps.ecmwf.int/codes/grib/param-db?id={self.ecmwf_id}"
            if self.ecmwf_id
            else None
        )

    @property
    def attrs(self) -> dict[str, str]:
        """Return a dictionary of met variable attributes.

        Compatible with xr.Dataset or xr.DataArray attrs.

        Returns
        -------
        dict[str, str]
            Dictionary with MetVariable attributes.
        """

        # return only these keys if they are not None
        keys = ["short_name", "standard_name", "long_name", "units"]
        return {k: getattr(self, k) for k in keys if getattr(self, k, None) is not None}


# ----
# Dimensions
# ----


AirPressure = MetVariable(
    short_name="p",
    standard_name="air_pressure",
    long_name="Air pressure",
    grib1_id=1,
    ecmwf_id=54,
    units="Pa",
    amip="plev",
    description=(
        "Air pressure is the force per unit area which would be "
        "exerted when the moving gas molecules of which the air is "
        "composed strike a theoretical surface of any orientation."
    ),
)

Altitude = MetVariable(
    short_name="alt",
    standard_name="altitude",
    long_name="Altitude",
    grib1_id=8,
    units="m",
    amip="ta",
    description=(
        "Altitude is the (geometric) height above the geoid, which is the "
        "reference  geopotential surface. The geoid is similar to mean sea level."
    ),
)


# ----
# Pressure level variables
# ----


AirTemperature = MetVariable(
    short_name="t",
    standard_name="air_temperature",
    long_name="Air Temperature",
    units="K",
    level_type="isobaricInhPa",
    grib1_id=11,
    ecmwf_id=130,
    grib2_id=(0, 0, 0),
    amip="ta",
    description=(
        "Air temperature is the bulk temperature of the air, not the surface (skin) temperature."
    ),
)

SpecificHumidity = MetVariable(
    short_name="q",
    standard_name="specific_humidity",
    long_name="Specific Humidity",
    units="kg kg**-1",
    level_type="isobaricInhPa",
    grib1_id=51,
    ecmwf_id=133,
    grib2_id=(0, 1, 0),
    amip="hus",
    description=(
        "Specific means per unit mass. Specific humidity is the mass "
        "fraction of water vapor in (moist) air."
    ),
)

RelativeHumidity = MetVariable(
    short_name="r",
    standard_name="relative_humidity",
    long_name="Relative Humidity",
    units="1",
    level_type="isobaricInhPa",
    grib1_id=52,
    ecmwf_id=157,
    grib2_id=(0, 1, 1),
    amip="hur",
    description=(
        "This parameter is the water vapour pressure as a percentage of "
        "the value at which the air becomes saturated liquid."
    ),
)

Geopotential = MetVariable(
    short_name="z",
    standard_name="geopotential",
    long_name="Geopotential",
    units="m**2 s**-2",
    level_type="isobaricInhPa",
    grib1_id=6,
    ecmwf_id=129,
    grib2_id=(0, 3, 4),
    description=(
        "Geopotential is the sum of the specific gravitational potential energy "
        "relative to the geoid and the specific centripetal potential energy."
    ),
)

GeopotentialHeight = MetVariable(
    short_name="gh",
    standard_name="geopotential_height",
    long_name="Geopotential Height",
    units="m",
    level_type="isobaricInhPa",
    grib1_id=7,
    ecmwf_id=156,
    grib2_id=(0, 3, 5),
    amip="zg",
    description=(
        "Geopotential is the sum of the specific gravitational potential energy "
        "relative to the geoid and the specific centripetal potential energy. "
        "Geopotential height is the geopotential divided by the standard "
        "acceleration due to gravity. It is numerically similar to the altitude "
        "(or geometric height) and not to the quantity with standard name height, "
        "which is relative to the surface."
    ),
)

EastwardWind = MetVariable(
    short_name="u",
    long_name="Eastward Wind",
    standard_name="eastward_wind",
    level_type="isobaricInhPa",
    units="m s**-1",
    grib1_id=33,
    ecmwf_id=131,
    grib2_id=(0, 2, 2),
    amip="ua",
    description=(
        '"Eastward" indicates a vector component which is positive '
        "when directed eastward (negative westward). Wind is defined "
        "as a two-dimensional (horizontal) air velocity vector, with no vertical component."
    ),
)

NorthwardWind = MetVariable(
    short_name="v",
    standard_name="northward_wind",
    long_name="Northward Wind",
    units="m s**-1",
    level_type="isobaricInhPa",
    grib1_id=34,
    ecmwf_id=132,
    grib2_id=(0, 2, 3),
    amip="va",
    description=(
        '"Northward" indicates a vector component which is positive when '
        "directed northward (negative southward). Wind is defined as a "
        "two-dimensional (horizontal) air velocity vector, with no vertical component."
    ),
)

VerticalVelocity = MetVariable(
    short_name="w",
    standard_name="lagrangian_tendency_of_air_pressure",
    long_name="Vertical Velocity (omega)",
    units="Pa s**-1",
    level_type="isobaricInhPa",
    grib1_id=39,
    ecmwf_id=135,
    grib2_id=(0, 2, 8),
    amip="wap",
    description=(
        'The Lagrangian tendency of air pressure, often called "omega", plays '
        "the role of the upward component of air velocity when air pressure "
        "is being used as the vertical coordinate. If the vertical air velocity "
        "is upwards, it is negative when expressed as a tendency of air pressure; "
        "downwards is positive. Air pressure is the force per unit area which "
        "would be exerted when the moving gas molecules of which the air is "
        "composed strike a theoretical surface of any orientation."
    ),
)


# ----
# Single level variables
# ----


SurfacePressure = MetVariable(
    short_name="sp",
    standard_name="surface_air_pressure",
    long_name="Surface air pressure",
    level_type="surface",
    grib1_id=1,
    ecmwf_id=134,
    grib2_id=(0, 3, 0),
    units="Pa",
    amip="ps",
    description=(
        "This parameter is the pressure (force per unit area) of the atmosphere "
        "on the surface of land, sea and in-land water. It is a measure of the "
        "weight of all the air in a column vertically above the area of the "
        "Earth's surface represented at a fixed point."
    ),
)
