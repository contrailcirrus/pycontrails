"""IAGOS utilties."""

import logging
import os
import pathlib
import traceback
import warnings

import numpy as np
import xarray as xr

from pycontrails.core import MetVariable, cache, met_var
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

SUPPORTED_VARIABLES = [
    met_var.AirPressure,
    Altitude,
    met_var.EastwardWind,
    met_var.NorthwardWind,
    met_var.AirTemperature,
    WaterVaporMoleFraction,
]


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

        if paths:
            filenames = [os.path.basename(p) for p in paths]
            duplicates = [f for f in set(filenames) if filenames.count(f) > 1]
            if duplicates:
                msg = (
                    f"IAGOS filenames {duplicates} appear more than once in `paths`. "
                    f"Deduplicate files included in `paths` before using."
                )
                raise ValueError(msg)
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
        return SUPPORTED_VARIABLES

    def _filter(self, path: str) -> bool:
        """Filter local IAGOS files."""
        try:
            ds = xr.open_dataset(path, decode_cf=False)

            # check if flight includes time coordinate
            try:
                # decode_cf requires a Dataset
                t = xr.decode_cf(ds[["UTC_time"]])["UTC_time"]
                assert t.dtype.kind == "M"
            except Exception:
                logging.debug(f"Could not decode 'UTC_time' coordinate from {path}")
                return False

            # check if flight overlaps target timepan
            if self.timespan:
                start, end = self.timespan
                if not ((t >= np.datetime64(start)) & (t <= np.datetime64(end))).any():
                    logging.debug(f"No overlap between target time range and {path}")
                    return False

            # get list of available variables
            present = {
                (v.attrs["standard_name"], v.dims, v.attrs["units"]) for v in ds.variables.values()
            }

            # check for required coordinate variables
            required = {
                ("longitude", ("UTC_time",), "degree_east"),
                ("latitude", ("UTC_time",), "degree_north"),
                ("barometric_altitude", ("UTC_time",), "m"),
            }
            missing = required.difference(present)
            if missing:
                logging.debug(f"Coordinates {[m[0] for m in missing]} missing from {path}")
                return False

            # check for required data variables
            required = {
                # IAGOS files don't include ** in unit exponents
                (v.standard_name, ("UTC_time",), (v.units or "1").replace("*", ""))
                for v in self.variables
            }
            missing = required.difference(present)
            if missing:
                logging.debug(f"Variables {[m[0] for m in missing]} missing from {path}")
                return False

            return True

        except Exception:
            msg = (
                f"Unexpected error while processing IAGOS data file at {path}. "
                f"Traceback: {traceback.format_exc()}"
            )
            warnings.warn(msg)
            return False

    def list_files(self) -> list[str]:
        """List available files.

        Returns
        -------
        list[str]
            List of available IAGOS files. Will be a subset of files in :attr:`paths` if defined,
            or a list of filenames available in the IAGOS database otherwise.
        """
        if self.paths:
            return sorted([os.path.basename(p) for p in self.paths if self._filter(p)])

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
