"""IAGOS utilties."""

import os
import pathlib
import re
import warnings

import numpy as np
import xarray as xr

from pycontrails.core import Flight, MetVariable, cache, met_var
from pycontrails.datalib._met_utils import metsource
from pycontrails.utils.types import DatetimeLike

Altitude = MetVariable(
    short_name="z",
    standard_name="gps_altitude",
    units="m",
    description="Altitude above the geoid, as measured by GPS.",
)

WaterVaporMoleFraction = MetVariable(
    short_name="etav",
    standard_name="mole_fraction_of_water_vapor_in_air",
    units="ppm",
    description="The number of molecules of water per molecule of air.",
)


#: Variables measured by aircraft directly.
#: These variables are measured on all IAGOS flights and are accompanied
#: by validity flags.
AIRCRAFT_VARIABLES = [
    met_var.AirPressure,
    Altitude,
    met_var.EastwardWind,
    met_var.NorthwardWind,
    met_var.AirTemperature,
]


#: Variables measured by IAGOS instruments.
#: These variables are not measured on all IAGOS flights.
#: When present, they are accompanied by validity and processing flags.
IAGOS_VARIABLES = [WaterVaporMoleFraction]


def extract_flight_id(filename: str) -> str:
    """Extract IAGOS flight id."""
    match = re.fullmatch(r"IAGOS_timeseries_(\d{8})(\d{8})_L2_(\d.\d.\d).nc4", filename)
    if match is None:
        msg = f"Name of IAGOS file {filename} does not match expected format."
        raise ValueError(msg)
    return match.group(1) + match.group(2)


class IAGOS:
    """EXPERIMENTAL: class for processing raw IAGOS L2 data.

    Parameters
    ----------
    time: metsource.TimeInput | None = None, optional
        Time range for data retrieval. If provided, input must be a tuple of two
        datetime-likes. If not provided, data will be retrieved for all available times.
    variables: metsource.VariableInput, optional
        Names of required variables. If not provided, data will be retrieved without
        filtering by included variables.
    paths : list[str | pathlib.Path] | None, optional
        Paths to IAGOS NetCDF files to load manually.
        Can include glob patterns to load specific files.
        Defaults to None, which looks for files in the :attr:`cachestore` or IAGOS database.
    cachestore: cache.CacheStore | None, optional
        Cache data store for retrieved IAGOS data files. Defaults of :class:`cache.DiskCacheStore`.
        If set to None, retrieved files are loaded directly into memory without caching.

    Notes
    -----
    IAGOS files are expected to be named as follows::

        IAGOS_timeseries_{YYYYmmdd}{XXXXXXXX}_L2_{V.V.V}.nc4

    - {YYYYmmdd} is the flight departure date
    - {XXXXXXXX} is an 8-digit flight identifier
    - {V.V.V} is the file version

    {YYYYmmmdd}{XXXXXXXX} is used as a unique identifier for IAGOS flights.

    IAGOS files are expected to contain a single datetime-like dimension coordinate
    called "UTC_time". In addition, they are expected to contain the following
    non-dimension coordinates:

    - Longitude (variable: "lon", units: "degree_east")
    - Latitude (variable: "lat", units: "degree_north")
    - Barometric altitude (variable: "baro_alt_AC", units: "m")

    and the following one-dimensional variables:

    - Geometric altitude (standard_name: "gps_altitude", units: "m")
    - Air pressure (standard_name: "air_pressure", units: "Pa")
    - Air temperature (standard_name: "air_temperature", units: "K")
    - Zonal wind (standard_name: "eastward_wind", units: "m s-1")
    - Meridional wind (standard_name: "northward_wind", unit: "m s-1")

    IAGOS files *may* (but are not guaranteed to) contain the following
    one-dimensional variables, along with associated standard errors (variable suffix "_error")
    and processing flags (variable suffix "_process_flag").

    - Water vapor mole fraction (standard_name: "mole_fraction_of_water_vapor", units: "ppm")

    All variables and non-dimension coordinates are expected to have "standard_name" and
    "units" attributes, and all variables and non-dimension coordinates except latitude and
    longitude are expected to have associated validity flags (variable suffix "_validity_flag).

    """

    __marker = object()
    __slots__ = "cachestore", "paths", "timespan", "variables"

    def __init__(
        self,
        time: tuple[str | DatetimeLike, str | DatetimeLike] | None = None,
        variables: metsource.VariableInput | None = None,
        paths: list[str | pathlib.Path] | None = None,
        cachestore: cache.CacheStore | None = __marker,  # type:ignore
    ) -> None:
        if cachestore is self.__marker:
            cachestore = cache.DiskCacheStore()
        self.cachestore = cachestore

        if time and len(time) != 2:
            msg = "If provided, time must be a tuple of length 2."
            raise ValueError(msg)
        self.timespan = metsource.parse_timesteps(time, freq=None)
        self.variables = metsource.parse_variables(variables or [], self.supported_variables)

        self.paths = [str(p) for p in paths] if paths else None

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__}"
        if self.timespan:
            _repr += f"\n\tTimespan: {[t.isoformat() for t in self.timespan]}"
        if self.variables:
            _repr += f"\n\tVariables: {self.variable_shortnames}"
        if self.paths:
            _repr += f"\n\tPaths: {len(self.paths)} files"
        return _repr

    @property
    def variable_shortnames(self) -> list[str]:
        """Return a list of variable short names.

        Returns
        -------
        list[str]
            List of variable short names.
        """
        return [v.short_name for v in self.variables]

    @property
    def supported_variables(self) -> list[MetVariable]:
        """Parameters available from IAGOS measurements.

        Returns
        -------
        list[MetVariable]:
            List of parameters available in data source.
            Note that not all IAGOS flights are guaranteed
            to record all available parameters.
        """
        return AIRCRAFT_VARIABLES + IAGOS_VARIABLES

    def list_files(self) -> list[str]:
        """List available files.

        Returns
        -------
        list[str]
            List of available IAGOS files. Will return the files in :attr:`paths` if defined,
            or a list of filenames available in the IAGOS database otherwise.

        Raises
        ------
        ValueError
            If :attr:`paths` is defined and includes multiple filenames with the same
            flight id.

        IAGOSProcessingError
            If :attr:`paths` is defined and includes a filename that does not match
            the expected format.
        """
        if self.paths:
            flight_ids = [extract_flight_id(os.path.basename(p)) for p in self.paths]
            duplicates = [f for f in set(flight_ids) if flight_ids.count(f) > 1]
            if duplicates:
                msg = (
                    f"IAGOS flight ids {duplicates} appear more than once in `paths`. "
                    f"Deduplicate files included in `paths` before using."
                )
                raise ValueError(msg)
            return sorted([os.path.basename(p) for p in self.paths])

        msg = (
            "File retrieval from the IAGOS database has not been implemented. "
            "To use this datalib, provide paths to local IAGOS files using the `paths` parameter."
        )
        raise NotImplementedError(msg)

    def get(self, filename: str) -> xr.Dataset:
        """Open a single IAGOS file.

        Parameters
        ----------
        filename : str
            IAGOS filename to open. Will look for a matching file in :attr:`paths` if defined,
            and the :attr:`cachestore` or IAGOS database otherwise.
        """
        if self.paths:
            try:
                return xr.open_dataset(
                    next(p for p in self.paths if os.path.basename(p) == filename)
                )
            except StopIteration as e:
                msg = f"IAGOS filename {filename} not found in `paths`."
                raise ValueError(msg) from e

        msg = (
            "File retrieval from the IAGOS database has not been implemented. "
            "To use this datalib, provide paths to local IAGOS files using the `paths` parameter."
        )
        raise NotImplementedError(msg)

    def load_flights(self) -> list[Flight]:
        """Process IAGOS data and return as a list of flights.

        Returns
        -------
        list[Flight]
            Processed IAGOS data.
        """
        flights = []

        for filename in self.list_files():
            ds = self.get(filename)
            if not self._includes_time_and_variables(ds):
                continue

            flight = self._create_flight(ds)
            flight_id = extract_flight_id(filename)
            flight.attrs["flight_id"] = flight_id
            flights.append(flight)

        return flights

    def _includes_time_and_variables(self, ds: xr.Dataset) -> bool:
        """Check if IAGOS file includes requested time and variables.

        This method will issue a warning and return ``False`` if problems
        with the contents of the IAGOS file are detected.
        """
        # check for presence of time coordinate
        if "UTC_time" not in ds.coords:
            msg = "IAGOS data file does not contain time coordinate"
            warnings.warn(msg)
            return False
        if ds["UTC_time"].dtype.kind != "M":
            msg = "IAGOS time coordinate is not datetime-like"
            warnings.warn(msg)
            return False

        # check if flight overlaps target timepan
        if self.timespan:
            start, end = self.timespan
            t = ds["UTC_time"]
            if not ((t >= np.datetime64(start)) & (t <= np.datetime64(end))).any():
                return False

        present = {
            v.attrs["standard_name"]: k
            for k, v in ds.variables.items()
            if "standard_name" in v.attrs
        }
        for variable in self.variables:
            # check if dataset includes variable
            if variable.standard_name not in present:
                return False

            # once presence of variable is established, check units and
            # presence of flags and errors.
            # raise an error if expected variables are not found.
            key = present[variable.standard_name]
            # IAGOS units don't include "**"
            if variable.units and ds[key].attrs["units"] != variable.units.replace("*", ""):
                msg = f"IAGOS variable {key} does not have expected units"
                warnings.warn(msg)
                return False
            flag = f"{key}_validity_flag"
            if flag not in ds:
                msg = f"IAGOS variable {key} does not have a validity flag"
                warnings.warn(msg)
                return False

            if variable not in IAGOS_VARIABLES:
                continue

            error = f"{key}_error"
            if error not in ds:
                msg = f"IAGOS variable {key} does not have a standard error"
                warnings.warn(msg)
                return False
            # IAGOS units don't include "**"
            if variable.units and ds[error].attrs["units"] != variable.units.replace("*", ""):
                msg = f"IAGOS variable {key} standard error does not have expected units"
                warnings.warn(msg)
                return False
            flag = f"{key}_process_flag"
            if flag not in ds:
                msg = f"IAGOS variable {key} does not have a processing flag"
                warnings.warn(msg)
                return False

        # check for presence of other coordinates
        for coord, units in (("lon", "degree_east"), ("lat", "degree_north"), ("baro_alt_AC", "m")):
            if coord not in ds.coords:
                msg = f"IAGOS data file does not contain coordinate {coord}"
                warnings.warn(msg)
                return False
            if ds[coord].units != units:
                msg = f"IAGOS coordinate {coord} does not have expected units"
                warnings.warn(msg)
                return False
        flag = "baro_alt_AC_validity_flag"
        if flag not in ds:
            msg = "IAGOS variable baro_alt_AC does not have a validity flag"
            warnings.warn(msg)
            return False

        return True

    def _create_flight(self, ds: xr.Dataset) -> Flight:
        """Create a flight contents of IAGOS data file.

        Should be called after first using _includes_time_and_variables
        to check that required variables are present.
        """
        # set coordinates
        flight = Flight(
            longitude=ds["lon"], latitude=ds["lat"], altitude=ds["baro_alt_AC"], time=ds["UTC_time"]
        )
        flight["altitude_validity_flag"] = ds["baro_alt_AC_validity_flag"]

        # add variables
        present = {
            v.attrs["standard_name"]: k
            for k, v in ds.variables.items()
            if "standard_name" in v.attrs
        }
        for variable in self.variables:
            name = variable.standard_name
            key = present[name]
            flight[name] = ds[key]
            flight[f"{name}_validity_flag"] = ds[f"{key}_validity_flag"]
            if variable not in IAGOS_VARIABLES:
                continue
            flight[f"{name}_standard_error"] = ds[f"{key}_error"]
            flight[f"{name}_process_flag"] = ds[f"{key}_process_flag"]

        return flight
