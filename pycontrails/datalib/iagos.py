"""Support for accessing `IAGOS <https://iagos.aeris-data.fr>`_ data."""

import functools
import os
import pathlib
import re
import tempfile
import warnings
from datetime import datetime

import numpy as np
import xarray as xr

from pycontrails.core import Flight, MetVariable, cache, met_var
from pycontrails.datalib._met_utils import metsource
from pycontrails.utils import dependencies
from pycontrails.utils.types import DatetimeLike

try:
    import keycloak
except ModuleNotFoundError as e:
    dependencies.raise_module_not_found_error(
        name="datalib.iagos module",
        package_name="python-keycloak",
        module_not_found_error=e,
        pycontrails_optional_package="iagos",
    )

try:
    import requests
except ModuleNotFoundError as e:
    dependencies.raise_module_not_found_error(
        name="datalib.iagos module",
        package_name="requests",
        module_not_found_error=e,
        pycontrails_optional_package="iagos",
    )


#: Variables measured by aircraft directly.
#: These variables are measured on all IAGOS flights and are accompanied
#: by validity flags.
AIRCRAFT_VARIABLES = [
    met_var.AirPressure,
    met_var.Altitude,
    met_var.EastwardWind,
    met_var.NorthwardWind,
    met_var.AirTemperature,
]


#: Variables measured by IAGOS instruments.
#: These variables are not measured on all IAGOS flights.
#: When present, they are accompanied by validity and processing flags
#: and standard errors.
IAGOS_VARIABLES = [
    met_var.MoleFractionOfWaterVaporInAir,
]


# Mapping from met variable to standard_name in IAGOS data files
_met_var_to_iagos_standard_name_mapping = {
    met_var.AirPressure: "air_pressure",
    met_var.Altitude: "gps_altitude",
    met_var.EastwardWind: "eastward_wind",
    met_var.NorthwardWind: "northward_wind",
    met_var.AirTemperature: "air_temperature",
    met_var.MoleFractionOfWaterVaporInAir: "mole_fraction_of_water_vapor_in_air",
}


# Mapping from met variable to name in IAGOS data files
# and scale factor for converting to met variable's units.
_met_var_to_iagos_units_mapping = {
    met_var.AirPressure: ("Pa", 1.0),
    met_var.Altitude: ("m", 1.0),
    met_var.EastwardWind: ("m s-1", 1.0),
    met_var.NorthwardWind: ("m s-1", 1.0),
    met_var.AirPressure: ("K", 1.0),
    met_var.MoleFractionOfWaterVaporInAir: ("ppm", 1e-6),
}

# Mapping from met variable to parameter name in IAGOS API.
_met_var_to_iagos_parameter_mapping = {met_var.MoleFractionOfWaterVaporInAir: "H2O"}


def match_flight_id(filename: str) -> str | None:
    """Get IAGOS flight id matches."""
    match = re.fullmatch(r"IAGOS_timeseries_(\d{8})(\d{8})_L2_(\d.\d.\d).nc4", filename)
    if match is None:
        return None
    return match.group(1) + match.group(2)


def extract_flight_id(filename: str) -> str:
    """Extract IAGOS flight id."""
    flight_id = match_flight_id(filename)
    if flight_id is None:
        msg = f"Could not extract IAGOS flight ID from {filename}"
        raise ValueError(msg)
    return flight_id


def validate_paths(paths: list[str | pathlib.Path] | None) -> dict[str, str] | None:
    """Validate provided IAGOS paths.

    Parameters
    ----------
    paths : list[str | pathlib.Path]
        List of paths to local IAGOS data files.

    Returns
    -------
    dict[str, str] | None
        Mapping from IAGOS flight ids to validated paths, or None if :param:`paths` is None.
        Input paths that include filenames that cannot be parsed into IAGOS flight ids are ignored,
        and a warning is raised if any are encountered. An error is raised if the same IAGOS flight
        id is parsed from more than one file.

    Raises
    ------
    ValueError
        If multiple paths contain the same IAGOS flight id.

    """
    if paths is None:
        return None

    flight_ids = []
    validated = []
    for path in paths:
        try:
            flight_id = extract_flight_id(os.path.basename(path))
        except ValueError:
            msg = f"Could not parse IAGOS flight id from {path}. This file will be ignored."
            warnings.warn(msg)
            continue
        flight_ids.append(flight_id)
        validated.append(str(path))

    unique_ids = set(flight_ids)
    if len(unique_ids) != len(flight_ids):
        duplicates = sorted([f for f in unique_ids if flight_ids.count(f) > 1])
        msg = (
            f"IAGOS flight ids {duplicates} appear more than once in `paths`. "
            f"Deduplicate files included in `paths` before using."
        )
        raise ValueError(msg)

    return dict(zip(flight_ids, validated, strict=True))


class IAGOS:
    """Class for downloading and processing L2 `IAGOS <https://iagos.aeris-data.fr>`_ data.

    IAGOS data access is free but requires
    `registering <https://iagos.aeris-data.fr/registration>_` with the IAGOS database
    and complying with the `IAGOS data policy <https://iagos.aeris-data.fr/data-policy>`_.

    We recommend authenticating using AERIS rather than ORCID or eduGAIN during registration,
    as an AERIS username and password are required to retrieve files from the IAGOS database
    using this datalib.

    Parameters
    ----------
    time: metsource.TimeInput | None = None, optional
        Time range for data retrieval. If provided, input must be a tuple of two
        datetime-likes. If not provided, data will be retrieved for all available times.
    variables: metsource.VariableInput, optional
        Names of required variables. If provided, this datalib will return data for flights
        that measure any (not necessarily all) of the specified variables. If not provided,
        data will be returned for flights that include any of the variables supported by this
        datalib.
    paths : list[str | pathlib.Path] | None, optional
        Paths to IAGOS NetCDF files to load manually.
        Can include glob patterns to load specific files.
        Defaults to None, which looks for files in the :attr:`cachestore` or IAGOS database.
    url : str, optional
        Override the default
        `IAGOS API <https://services.iagos-data.fr/prod/swagger-ui/index.html>`_ url.
    user : str | None, optional
        AERIS username for IAGOS database authentication. Required to retrieve files from
        the IAGOS database.
    password : str | None, optional
        AERIS password for IAGOS database authentication. Required to retrieve files from
        the IAGOS database.
    cachestore: cache.CacheStore | None, optional
        Cache data store for retrieved IAGOS data files. Defaults of :class:`cache.DiskCacheStore`.
        If set to None, retrieved files are loaded directly into memory without caching.

    Notes
    -----
    To inspect raw IAGOS data files without additional processing:

        1. Instantiate ``IAGOS(time, variables)`` or ``IAGOS(time, variables, paths)``
        2. Call ``list_flight_ids()`` to get a list of available flights
        3. Call ``get(flight_id)`` to open a file as an ``xarray.Dataset``

    If a set of local ``paths`` is provided during instantiation, ``list_files()``
    will flight ids based on ``paths`` without additional filtering. If local ``paths``
    are not provided, ``list_files()`` will filter for files that overlap with the
    requested ``time`` and include the requested ``variables`` when calling the IAGOS
    API.

    To open processed IAGOS files as a list of ``Flight``:

        1. Instantiate ``IAGOS(time, variables)`` or ``IAGOS(time, variables, paths)``
        2. Call ``load_flights()``

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

    __slots__ = (
        "__dict__",
        "cachestore",
        "password",
        "paths",
        "timespan",
        "token",
        "url",
        "user",
        "variables",
    )

    #: IAGOS data timespan as a length-2 list
    timespan: list[datetime]

    #: IAGOS variables
    variables: list[MetVariable]

    #: Paths to local IAGOS data files
    paths: dict[str, str] | None

    #: Cachestore for retrieved IAGOS files
    cachestore: cache.CacheStore | None

    #: IAGOS API url
    url: str

    #: AERIS username
    user: str | None

    #: AERIS password
    password: str | None

    #: AERIS authentication token
    token: str | None

    def __init__(
        self,
        time: tuple[str | DatetimeLike, str | DatetimeLike] | None = None,
        variables: metsource.VariableInput | None = None,
        paths: list[str | pathlib.Path] | None = None,
        url: str = "https://services.iagos-data.fr/prod/v2.0",
        user: str | None = None,
        password: str | None = None,
        cachestore: cache.CacheStore | None = __marker,  # type:ignore
    ) -> None:
        if cachestore is self.__marker:
            cache_root = cache._get_user_cache_dir()
            cache_dir = f"{cache_root}/iagos"
            cachestore = cache.DiskCacheStore(cache_dir=cache_dir)
        self.cachestore = cachestore

        if time and (isinstance(time, str) or len(time) != 2):
            msg = "If provided, time must be a tuple of length 2."
            raise ValueError(msg)

        self.timespan = metsource.parse_timesteps(time, freq=None)
        self.variables = metsource.parse_variables(variables or [], self.supported_variables)
        self.paths = validate_paths(paths)

        self.url = url
        self.user = user
        self.password = password
        self.token = None

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
    def aircraft_variables(self) -> list[MetVariable]:
        """Return a list of variables measured directly by aircraft.

        Returns
        -------
        list[MetVariable]
            Subset of :attr:`variables` measured by aircraft.
        """
        return [v for v in self.variables if v in AIRCRAFT_VARIABLES]

    @property
    def iagos_variables(self) -> list[MetVariable]:
        """Return a list of variables measured by IAGOS instruments.

        Returns
        -------
        list[MetVariable]
            Subset of :attr:`variables` measured by IAGOS instruments.
        """
        return [v for v in self.variables if v in IAGOS_VARIABLES]

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

    def list_flight_ids(self) -> list[str]:
        """List available flight_ids.

        Returns
        -------
        list[str]
            List of available IAGOS flight ids. Will return flight ids extracted from
            the files in :attr:`paths` if defined, or a list of flights available in
            the IAGOS database otherwise.
        """
        if self.paths:
            return list(self.paths.keys())

        url, params, headers = self._build_list_request()
        response = self._retry_once(url, params, headers)

        return [flight["name"] for flight in response.json()]

    def get(self, flight_id: str) -> xr.Dataset:
        """Open a single IAGOS file.

        Parameters
        ----------
        flight_id : str
            IAGOS flight to open. Will look for a matching file in :attr:`paths` if defined,
            and the :attr:`cachestore` or IAGOS database otherwise.
        """
        if self.paths:
            if flight_id not in self.paths:
                msg = f"IAGOS flight id {flight_id} not found in `paths`."
                raise ValueError(msg)
            return xr.open_dataset(self.paths[flight_id])

        if self.cachestore is None:
            return self._get_no_cache(flight_id)
        return self._get_with_cache(flight_id)

    def load_flights(self) -> list[Flight]:
        """Process IAGOS data and return as a list of flights.

        Returns
        -------
        list[Flight]
            Processed IAGOS data.
        """
        flights = []

        for flight_id in self.list_flight_ids():
            ds = self.get(flight_id)
            if not self._includes_time_and_variables(ds):
                continue

            flight = self._create_flight(ds)
            flight.attrs["flight_id"] = flight_id
            flights.append(flight)

        return flights

    def _get_with_cache(self, flight_id: str) -> xr.Dataset:
        """Get data using cachestore."""
        if not self.cachestore:
            raise ValueError("Cachestore not configured")

        lpath = self.cachestore.path(f"{flight_id}.nc")
        if self.cachestore.exists(lpath):
            return xr.open_dataset(lpath)

        url, params, headers = self._build_download_request(flight_id)
        response = self._retry_once(url, params, headers)

        with open(lpath, "wb") as f:
            f.write(response.content)
        return xr.open_dataset(lpath)

    def _get_no_cache(self, flight_id: str) -> xr.Dataset:
        """Get data without using cachestore."""
        url, params, headers = self._build_download_request(flight_id)
        response = self._retry_once(url, params, headers)

        try:
            # On windows, NamedTemporaryFile cannot be reopened while still open.
            # After python 3.11 support is dropped, we can use delete_on_close=False
            # in NamedTemporaryFile to streamline this.
            with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
                tmp.write(response.content)
            return xr.load_dataset(tmp.name)
        finally:
            os.remove(tmp.name)

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

        # get mapping from dataset standard_names to variable names.
        # this is used repeatedly to find dataset keys corresponding
        # to requested variables.
        present = {
            v.attrs["standard_name"]: k
            for k, v in ds.variables.items()
            if "standard_name" in v.attrs
        }

        for variable in self.variables:
            # check if dataset includes variable
            standard_name = _met_var_to_iagos_standard_name_mapping[variable]
            if standard_name not in present:
                return False
            key = present[standard_name]

            # check units and presence of validity flag.
            # raise an error if expected variables are not found.
            units, _ = _met_var_to_iagos_units_mapping[variable]
            if ds[key].attrs.get("units", None) != units:
                msg = f"IAGOS variable {key} does not have expected units {units}"
                warnings.warn(msg)
                return False
            flag = f"{key}_validity_flag"
            if flag not in ds:
                msg = f"IAGOS variable {key} does not have a validity flag"
                warnings.warn(msg)
                return False

            # only variables measured by the iagos instrument (as opposed
            # to by the aircraft directly) are required to have standard
            # errors and processing flags.
            if variable not in IAGOS_VARIABLES:
                continue

            # check presence of and units on standard error
            error = f"{key}_error"
            if error not in ds:
                msg = f"IAGOS variable {key} does not have a standard error"
                warnings.warn(msg)
                return False
            # IAGOS units don't include "**"
            if ds[error].attrs.get("units", None) != units:
                msg = f"IAGOS variable {key} standard error does not have expected units"
                warnings.warn(msg)
                return False

            # check presence of processing flag
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
            standard_name = _met_var_to_iagos_standard_name_mapping[variable]
            key = present[standard_name]
            _, scale = _met_var_to_iagos_units_mapping[variable]

            flight[name] = scale * ds[key]
            flight[f"{name}_validity_flag"] = ds[f"{key}_validity_flag"]

            # should only look for standard errors and processing flags
            # for variables measured by the IAGOS instrument.
            # we already have everything for variables measured by the
            # aircraft directly.
            if variable not in IAGOS_VARIABLES:
                continue

            flight[f"{name}_standard_error"] = scale * ds[f"{key}_error"]
            flight[f"{name}_process_flag"] = ds[f"{key}_process_flag"]

        return flight

    @functools.cached_property
    def _auth_client(self) -> keycloak.KeycloakOpenID:
        """Get auth token generation client."""

        return keycloak.KeycloakOpenID(
            server_url="https://sso.aeris-data.fr/auth/",
            client_id="aeris-public",
            realm_name="aeris",
            verify=True,
        )

    def _refresh_token(self) -> None:
        """Refresh authentication token."""
        if not self.user or not self.password:
            msg = "Must provide 'user' and 'password' to download data from IAGOS API."
            raise ValueError(msg)

        try:
            token = self._auth_client.token(self.user, self.password)
            self.token = token["access_token"]
        except Exception as e:
            msg = "Failed to refresh authentication token. Check username and password."
            raise RuntimeError(msg) from e

    def _build_list_request(self) -> tuple[str, dict[str, str], dict[str, str]]:
        """Return components of request to list available files."""
        url = f"{self.url}/flights"

        t0, t1 = self.timespan
        variables = self.variables or self.iagos_variables
        parameters = [_met_var_to_iagos_parameter_mapping[v] for v in variables]

        params = {
            "from": t0.strftime("%Y-%m-%d"),
            "to": t1.strftime("%Y-%m-%d"),
            "bbox": "-180,-90,180,90",
            "mission": "IAGOS-CORE,IAGOS-MOZAIC,IAGOS-CARIBIC",
            "level": "2",
            "parameters": ",".join(parameters),
        }

        if not self.token:
            self._refresh_token()

        headers = {"accept": "application/json", "authorization": f"bearer {self.token}"}

        return url, params, headers

    def _build_download_request(self, flight_id: str) -> tuple[str, dict[str, str], dict[str, str]]:
        """Return components of request to download a files."""
        url = f"{self.url}/downloads/{flight_id}"

        params = {"level": "2", "format": "netcdf", "type": "timeseries"}

        if not self.token:
            self._refresh_token()

        headers = {"accept": "application/octet-stream", "authorization": f"bearer {self.token}"}

        return url, params, headers

    def _retry_once(
        self, url: str, params: dict[str, str], headers: dict[str, str]
    ) -> requests.Response:
        """Submit request, re-trying once if authentication fails."""
        with requests.Session() as session:
            session.headers.update(headers)
            response = session.get(url, params=params)

            if response.status_code == 401:  # could be using expired token
                self._refresh_token()
                session.headers.update({f"authorizationbearer {self.token}"})
                response = session.get(url, params=params)

            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                msg = "Error listing available IAGOS flight ids."
                raise RuntimeError(msg) from e

            return response
