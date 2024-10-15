"""Support for `ARCO ERA5 <https://cloud.google.com/storage/docs/public-datasets/era5>`_.

This module supports:

- Downloading ARCO ERA5 model level data for specific times and pressure level variables.
- Downloading ARCO ERA5 single level data for specific times and single level variables.
- Interpolating model level data to a target lat-lon grid and pressure levels.
- Local caching of the downloaded and interpolated data as netCDF files.
- Opening cached data as a :class:`pycontrails.MetDataset` object.

This module requires the following additional dependencies:

- `gcsfs <https://gcsfs.readthedocs.io/en/latest/>`_
- `zarr <https://zarr.readthedocs.io/en/stable/>`_

"""

from __future__ import annotations

import datetime
import hashlib
import sys
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy.typing as npt
import xarray as xr

from pycontrails.core import cache, met_var
from pycontrails.core.met import MetDataset
from pycontrails.datalib._met_utils import metsource
from pycontrails.datalib.ecmwf import common as ecmwf_common
from pycontrails.datalib.ecmwf import model_levels as mlmod
from pycontrails.datalib.ecmwf import variables as ecmwf_variables

MODEL_LEVEL_STORE = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
# This combined store holds both pressure level and surface data
# It contains 273 variables (as of Sept 2024)
COMBINED_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


PRESSURE_LEVEL_VARIABLES = [
    ecmwf_variables.Divergence,
    ecmwf_variables.CloudAreaFractionInLayer,
    met_var.Geopotential,
    ecmwf_variables.OzoneMassMixingRatio,
    ecmwf_variables.SpecificCloudIceWaterContent,
    ecmwf_variables.SpecificCloudLiquidWaterContent,
    met_var.SpecificHumidity,
    # "specific_rain_water_content",
    # "specific_snow_water_content",
    met_var.AirTemperature,
    met_var.EastwardWind,
    met_var.NorthwardWind,
    met_var.VerticalVelocity,
    ecmwf_variables.RelativeVorticity,
]


_met_vars_to_arco_model_level_mapping = {
    ecmwf_variables.Divergence: "divergence",
    ecmwf_variables.CloudAreaFractionInLayer: "fraction_of_cloud_cover",
    met_var.Geopotential: "geopotential",
    ecmwf_variables.OzoneMassMixingRatio: "ozone_mass_mixing_ratio",
    ecmwf_variables.SpecificCloudIceWaterContent: "specific_cloud_ice_water_content",
    ecmwf_variables.SpecificCloudLiquidWaterContent: "specific_cloud_liquid_water_content",
    met_var.SpecificHumidity: "specific_humidity",
    met_var.AirTemperature: "temperature",
    met_var.EastwardWind: "u_component_of_wind",
    met_var.NorthwardWind: "v_component_of_wind",
    met_var.VerticalVelocity: "vertical_velocity",
    ecmwf_variables.RelativeVorticity: "vorticity",
}

_met_vars_to_arco_surface_level_mapping = {
    met_var.SurfacePressure: "surface_pressure",
    ecmwf_variables.TOAIncidentSolarRadiation: "toa_incident_solar_radiation",
    ecmwf_variables.TopNetSolarRadiation: "top_net_solar_radiation",
    ecmwf_variables.TopNetThermalRadiation: "top_net_thermal_radiation",
    ecmwf_variables.CloudAreaFraction: "total_cloud_cover",
    ecmwf_variables.SurfaceSolarDownwardRadiation: "surface_solar_radiation_downwards",
}


def _open_arco_model_level_stores(
    times: list[datetime.datetime],
    variables: list[met_var.MetVariable],
) -> tuple[xr.Dataset, xr.DataArray]:
    """Open slices of the ARCO ERA5 model level Zarr stores."""
    kw: dict[str, Any] = {"chunks": None, "consolidated": True}  # keep type hint for mypy

    # This is too slow to open with chunks={} or chunks="auto"
    ds = xr.open_zarr(MODEL_LEVEL_STORE, **kw)
    names = {
        name: var.short_name
        for var in variables
        if (name := _met_vars_to_arco_model_level_mapping.get(var))
    }
    if not names:
        msg = "No valid variables provided"
        raise ValueError(msg)

    ds = ds[list(names)].sel(time=times).rename(hybrid="model_level").rename_vars(names)
    sp = xr.open_zarr(COMBINED_STORE, **kw)["surface_pressure"].sel(time=times)

    # Chunk here in a way that is harmonious with the zarr store itself
    # https://github.com/google-research/arco-era5?tab=readme-ov-file#025-model-level-data
    ds = ds.chunk(time=1)
    sp = sp.chunk(time=1)

    return ds, sp


def open_arco_era5_model_level_data(
    times: list[datetime.datetime],
    variables: list[met_var.MetVariable],
    pressure_levels: npt.ArrayLike,
) -> xr.Dataset:
    r"""Open ARCO ERA5 model level data for a specific time and variables.

    Data is not loaded into memory, and the data is not cached.

    Parameters
    ----------
    times : list[datetime.datetime]
        Time of the data to open.
    variables : list[met_var.MetVariable]
        List of variables to open. Unsupported variables are ignored.
    pressure_levels : npt.ArrayLike
        Target pressure levels, [:math:`hPa`].

    Returns
    -------
    xr.Dataset
        Dataset with the requested variables on the target grid and pressure levels.
        Data is reformatted for :class:`MetDataset` conventions.

    References
    ----------
    - :cite:`carverARCOERA5AnalysisReadyCloudOptimized2023`
    - `ARCO ERA5 moisture workflow <https://github.com/google-research/arco-era5/blob/main/docs/moisture_dataset.py>`_
    - `Model Level Walkthrough <https://github.com/google-research/arco-era5/blob/main/docs/1-Model-Levels-Walkthrough.ipynb>`_
    - `Surface Reanalysis Walkthrough <https://github.com/google-research/arco-era5/blob/main/docs/0-Surface-Reanalysis-Walkthrough.ipynb>`_
    """
    ds, sp = _open_arco_model_level_stores(times, variables)
    out = mlmod.ml_to_pl(ds, pressure_levels, sp=sp)
    return MetDataset(out).data


def open_arco_era5_single_level(
    times: list[datetime.datetime],
    variables: list[met_var.MetVariable],
) -> xr.Dataset:
    """Open ARCO ERA5 single level data for a specific date and variables.

    Data is not loaded into memory, and the data is not cached.

    Parameters
    ----------
    times : list[datetime.date]
        Time of the data to open.
    variables : list[met_var.MetVariable]
        List of variables to open.

    Returns
    -------
    xr.Dataset
        Dataset with the requested variables.
        Data is reformatted for :class:`MetDataset` conventions.

    Raises
    ------
    FileNotFoundError
        If the variable is not found at the requested date. This could
        indicate that the variable is not available in the ARCO ERA5 dataset,
        or that the time requested is outside the available range.
    """
    # This is too slow to open with chunks={} or chunks="auto"
    ds = xr.open_zarr(COMBINED_STORE, consolidated=True, chunks=None)
    names = {
        name: var.short_name
        for var in variables
        if (name := _met_vars_to_arco_surface_level_mapping.get(var))
    }
    if not names:
        msg = "No valid variables provided"
        raise ValueError(msg)

    ds = ds[list(names)].sel(time=times).rename_vars(names)

    # But we need to chunk it here for lazy loading (the call expand_dims below
    # would materialize the data if chunks=None). So we chunk in a way that is
    # harmonious with the zarr store itself.
    # https://github.com/google-research/arco-era5?tab=readme-ov-file#025-pressure-and-surface-level-data
    ds = ds.chunk(time=1)

    ds = ds.expand_dims(level=[-1])
    return MetDataset(ds).data


class ERA5ARCO(ecmwf_common.ECMWFAPI):
    r"""ARCO ERA5 data accessed remotely through Google Cloud Storage.

    This is a high-level interface to access and cache
    `ARCO ERA5 <https://cloud.google.com/storage/docs/public-datasets/era5>`_
    for a predefined set of times, variables, and pressure levels.

        .. versionadded:: 0.50.0

    Parameters
    ----------
    time : TimeInput
        Time of the data to open.
    variables : VariableInput
        List of variables to open.
    pressure_levels : PressureLevelInput, optional
        Target pressure levels, [:math:`hPa`]. For pressure level data, this should be
        a sorted (increasing or decreasing) list of integers. For single level data,
        this should be ``-1``. By default, the pressure levels are set to the
        pressure levels at each model level between 20,000 and 50,000 ft assuming a
        constant surface pressure.
    cachestore : CacheStore, optional
        Cache store to use. By default, a new disk cache store is used. If None, no caching is done.
        In this case, the data returned by :meth:`open_metdataset` is not loaded into memory.

    References
    ----------
    :cite:`carverARCOERA5AnalysisReadyCloudOptimized2023`

    See Also
    --------
    :func:`open_arco_era5_model_level_data`
    :func:`open_arco_era5_single_level`
    """

    __marker = object()

    def __init__(
        self,
        time: metsource.TimeInput,
        variables: metsource.VariableInput,
        pressure_levels: metsource.PressureLevelInput | None = None,
        cachestore: cache.CacheStore | None = __marker,  # type: ignore[assignment]
    ) -> None:
        self.timesteps = metsource.parse_timesteps(time)

        if pressure_levels is None:
            self.pressure_levels = mlmod.model_level_reference_pressure(20_000.0, 50_000.0)
        else:
            self.pressure_levels = metsource.parse_pressure_levels(pressure_levels)

        self.paths = None
        self.variables = metsource.parse_variables(variables, self.supported_variables)
        self.cachestore = cache.DiskCacheStore() if cachestore is self.__marker else cachestore

    @property
    def pressure_level_variables(self) -> list[met_var.MetVariable]:
        """Variables available in the ARCO ERA5 model level data.

        Returns
        -------
        list[MetVariable] | None
            List of MetVariable available in datasource
        """
        return PRESSURE_LEVEL_VARIABLES

    @property
    def single_level_variables(self) -> list[met_var.MetVariable]:
        """Variables available in the ARCO ERA5 single level data.

        Returns
        -------
        list[MetVariable] | None
            List of MetVariable available in datasource
        """
        return ecmwf_variables.SURFACE_VARIABLES

    @override
    def download_dataset(self, times: list[datetime.datetime]) -> None:
        if not times:
            return

        if self.is_single_level:
            ds = open_arco_era5_single_level(times, self.variables)
        else:
            ds = open_arco_era5_model_level_data(times, self.variables, self.pressure_levels)

        self.cache_dataset(ds)

    @override
    def create_cachepath(self, t: datetime.datetime) -> str:
        if self.cachestore is None:
            msg = "Attribute self.cachestore must be defined to create cache path"
            raise ValueError(msg)

        string = (
            f"{t:%Y%m%d%H}-"
            f"{'.'.join(str(p) for p in self.pressure_levels)}-"
            f"{'.'.join(sorted(self.variable_shortnames))}-"
        )
        name = hashlib.md5(string.encode()).hexdigest()
        cache_path = f"arcoera5-{name}.nc"

        return self.cachestore.path(cache_path)

    @override
    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MetDataset:
        if dataset:
            msg = "Parameter 'dataset' is not supported for ARCO ERA5"
            raise ValueError(msg)

        if self.cachestore is None:
            if self.is_single_level:
                ds = open_arco_era5_single_level(self.timesteps, self.variables)
            else:
                ds = open_arco_era5_model_level_data(
                    self.timesteps, self.variables, self.pressure_levels
                )
        else:
            xr_kwargs = xr_kwargs or {}
            self.download(**xr_kwargs)

            disk_cachepaths = [self.cachestore.get(f) for f in self._cachepaths]
            ds = self.open_dataset(disk_cachepaths, **xr_kwargs)

        mds = self._process_dataset(ds, **kwargs)
        self.set_metadata(mds)
        return mds

    @override
    def set_metadata(self, ds: xr.Dataset | MetDataset) -> None:
        ds.attrs.update(
            provider="ECMWF",
            dataset="ERA5",
            product="reanalysis",
        )
