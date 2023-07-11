"""ECWMF IFS forecast data access."""

from __future__ import annotations

import logging
import pathlib
import warnings
from datetime import datetime
from typing import Any

LOG = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import xarray as xr
from overrides import overrides

from pycontrails.core import datalib, met
from pycontrails.datalib.ecmwf.variables import ECMWF_VARIABLES
from pycontrails.physics import constants
from pycontrails.utils.types import DatetimeLike


class IFS(datalib.MetDataSource):
    """
    ECMWF Integrated Forecasting System (IFS) data source.

    .. warning::

        This data source is fully implemented.

    Parameters
    ----------
    time : datalib.TimeInput | None
        The time range for data retrieval, either a single datetime or (start, end) datetime range.
        Input must be a single datetime-like or tuple of datetime-like
        (datetime, :class:`pandas.Timestamp`, :class:`numpy.datetime64`)
        specifying the (start, end) of the date range, inclusive.
        If None, all time coordinates will be loaded.
    variables : datalib.VariableInput
        Variable name (i.e. "air_temperature", ["air_temperature, relative_humidity"])
        See :attr:`pressure_level_variables` for the list of available variables.
    pressure_levels : datalib.PressureLevelInput, optional
        Pressure level bounds for data (min, max), in hPa (mbar)
        Set to -1 for to download surface level parameters.
        Defaults to -1.
    paths : str | list[str] | pathlib.Path | list[pathlib.Path] | None, optional
        UNSUPPORTED FOR IFS
    forecast_path: str | pathlib.Path | None, optional
        Path to local forecast files.
        Defaults to None
    forecast_date: DatetimeLike, optional
        Forecast date to load specific netcdf files.
        Defaults to None

    Notes
    -----
    This takes an average pressure of the model level to create
    pressure level dimensions.
    """

    __slots__ = ("forecast_date", "forecast_path")

    #: Root path of IFS data
    forecast_path: pathlib.Path

    #: Forecast datetime of IFS forecast
    forecast_date: pd.Timestamp

    def __init__(
        self,
        time: datalib.TimeInput | None,
        variables: datalib.VariableInput,
        pressure_levels: datalib.PressureLevelInput = -1,
        paths: str | list[str] | pathlib.Path | list[pathlib.Path] | None = None,
        grid: float | None = None,
        forecast_path: str | pathlib.Path | None = None,
        forecast_date: DatetimeLike | None = None,
    ) -> None:
        self.paths = paths  # TODO: this is currently unused
        self.grid = grid  # TODO: this is currently unused

        # path to forecast files
        if forecast_path is None:
            raise ValueError("Forecast path input is required for IFS")
        self.forecast_path = pathlib.Path(forecast_path)

        # TODO: automatically select a forecast_date from input time range?
        self.forecast_date = pd.to_datetime(forecast_date).to_pydatetime()

        # parse inputs
        self.timesteps = datalib.parse_timesteps(time, freq="3H")
        self.pressure_levels = datalib.parse_pressure_levels(pressure_levels, None)
        self.variables = datalib.parse_variables(variables, self.supported_variables)

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}\n\tForecast date: {self.forecast_date}"

    @property
    def supported_variables(self) -> list[met.MetVariable]:
        """IFS parameters available.

        Returns
        -------
        list[MetVariable] | None
            List of MetVariable available in datasource
        """
        return ECMWF_VARIABLES

    @property
    def supported_pressure_levels(self) -> None:
        """IFS does not provide constant pressure levels and instead uses model levels.

        Returns
        -------
        list[int]
        """
        return None

    @overrides
    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> met.MetDataset:
        xr_kwargs = xr_kwargs or {}

        #  short-circuit dataset or file paths if provided
        if self.paths is not None or dataset is not None:
            raise NotImplementedError("IFS input paths or input dataset is not supported")

        # load / merge datasets
        ds = self._open_ifs_dataset(**xr_kwargs)

        # drop ancillary vars
        ds = ds.drop_vars(names=["hyai", "hybi"])

        # downselect dataset if only a subset of times, pressure levels, or variables are requested
        if self.timesteps:
            ds = ds.sel(time=self.timesteps)
        else:
            # set timesteps from dataset "time" coordinates
            # np.datetime64 doesn't covert to list[datetime] unless its unit is us
            self.timesteps = ds["time"].values.astype("datetime64[us]").tolist()

        # downselect hyam/hybm coefficients by the "lev" coordinate
        # (this is a 1-indexed verison of nhym)
        ds["hyam"] = ds["hyam"][dict(nhym=(ds["lev"] - 1).astype(int))]
        ds["hybm"] = ds["hybm"][dict(nhym=(ds["lev"] - 1).astype(int))]

        # calculate air_pressure (Pa) by hybrid sigma pressure
        ds["air_pressure"] = ds["hyam"] + (ds["hybm"] * ds["surface_pressure"])
        ds["air_pressure"].attrs["units"] = "Pa"
        ds["air_pressure"].attrs["long_name"] = "Air pressure"

        # calculate virtual temperature (t_virtual)
        # the temperature at which dry air would have the same density
        # as the moist air at a given pressure
        ds["t_virtual"] = ds["t"] * (1 + ds["q"] * ((constants.R_v / constants.R_d) - 1))
        ds["t_virtual"].attrs["units"] = "K"
        ds["t_virtual"].attrs["long_name"] = "Virtual Temperature"

        # calculate geopotential
        if "z" in self.variable_shortnames:
            ds["z"] = self._calc_geopotential(ds)

        # take the mean of the air pressure to create quasi-gridded level coordinate
        ds = ds.assign_coords(
            {"level": ("lev", (ds["air_pressure"].mean(dim=["time", "lat", "lon"]) / 100).values)}
        )
        ds = ds.swap_dims({"lev": "level"})
        ds = ds.drop_vars(names=["lev"])

        # rename dimensions
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        # downselect variables
        ds = ds[self.variable_shortnames]

        # TODO: fix this correctly
        if "level" not in ds.dims:
            ds = ds.expand_dims({"level": [-1]})

        # harmonize variable names
        ds = met.standardize_variables(ds, self.variables)

        ds.attrs["met_source"] = type(self).__name__
        return met.MetDataset(ds, **kwargs)

    @overrides
    def download_dataset(self, times: list[datetime]) -> None:
        raise NotImplementedError("IFS download is not supported")

    @overrides
    def cache_dataset(self, dataset: xr.Dataset) -> None:
        raise NotImplementedError("IFS dataset caching not supported")

    @overrides
    def create_cachepath(self, t: datetime) -> str:
        raise NotImplementedError("IFS download is not supported")

    def _open_ifs_dataset(self, **xr_kwargs: Any) -> xr.Dataset:
        # get the path to each IFS file for each forecast date
        date_str = self.forecast_date.strftime("%Y%m%d")
        path_full = f"{self.forecast_path}/FC_{date_str}_00_144.nc"
        path_fl = f"{self.forecast_path}/FC_{date_str}_00_144_fl.nc"
        path_surface = f"{self.forecast_path}/FC_{date_str}_00_144_sur.nc"
        path_rad = f"{self.forecast_path}/FC_{date_str}_00_144_rad.nc"

        # load each dataset
        LOG.debug(f"Loading IFS forecast date {date_str}")

        # load each dataset
        xr_kwargs.setdefault("chunks", datalib.DEFAULT_CHUNKS)
        xr_kwargs.setdefault("engine", datalib.NETCDF_ENGINE)
        ds_full = xr.open_dataset(path_full, **xr_kwargs)
        ds_fl = xr.open_dataset(path_fl, **xr_kwargs)
        ds_surface = xr.open_dataset(path_surface, **xr_kwargs)
        ds_rad = xr.open_dataset(path_rad, **xr_kwargs)

        # calculate surface pressure from ln(surface pressure) var, squeeze out "lev" dim
        ds_full["surface_pressure"] = np.exp(ds_full["lnsp"]).squeeze()
        ds_full["surface_pressure"].attrs["units"] = "Pa"
        ds_full["surface_pressure"].attrs["long_name"] = "Surface air pressure"

        # swap dim names for consistency
        ds_full = ds_full.drop_vars(names=["lnsp", "lev"])
        ds_full = ds_full.rename({"lev_2": "lev"})

        # drop vars so all datasets can merge
        ds_fl = ds_fl.drop_vars(names=["hyai", "hybi", "hyam", "hybm"])

        # merge all datasets using the "ds_fl" dimensions as the join keys
        ds = xr.merge([ds_fl, ds_full, ds_surface, ds_rad], join="left")  # order matters!

        return ds

    def _calc_geopotential(self, ds: xr.Dataset) -> xr.DataArray:
        warnings.warn(
            "The geopotential calculation implementation may assume the underlying grid "
            "starts at ground level. This may not be the case for IFS data. It may be "
            "better to use geometric height (altitude) instead of geopotential for downstream "
            "applications (tau cirrus, etc.).",
            UserWarning,
        )

        # TODO: this could be done via a mapping on the "lev" dimension
        # groupby("lev")

        z_level = ds["z"].copy()
        p_level = ds["surface_pressure"].copy()
        geopotential = xr.zeros_like(ds["t"])

        geopotential.attrs["standard_name"] = "geopotential"
        geopotential.attrs["units"] = "m**2 s**-2"
        geopotential.attrs["long_name"] = "Geopotential"

        # iterate through level layers from the bottom up
        for k in ds["lev"][::-1]:
            d_log_p = np.log(p_level / ds["air_pressure"].loc[dict(lev=k)])

            denom = p_level - ds["air_pressure"].loc[dict(lev=k)]
            alpha = 1 - d_log_p * ds["air_pressure"].loc[dict(lev=k)] / denom

            geopotential.loc[dict(lev=k)] = (
                z_level + ds["t_virtual"].loc[dict(lev=k)] * alpha * constants.R_d
            )

            # Update values for next loop
            z_level = geopotential.loc[dict(lev=k)].copy()
            p_level = ds["air_pressure"].loc[dict(lev=k)].copy()

        return geopotential
