"""Support for `ARCO ERA5 <https://cloud.google.com/storage/docs/public-datasets/era5>`_.

This module supports:

- Downloading ARCO ERA5 model level data for specific times and pressure level variables.
- Downloading ARCO ERA5 single level data for specific times and single level variables.
- Interpolating model level data to a target lat-lon grid and pressure levels.
- Local caching of the downloaded and interpolated data as netCDF files.
- Opening cached data as a :class:`pycontrails.MetDataset` object.

This module requires the following additional dependencies:

- `metview (binaries and python bindings) <https://metview.readthedocs.io/en/latest/python.html>`_
- `gcsfs <https://gcsfs.readthedocs.io/en/latest/>`_
- `zarr <https://zarr.readthedocs.io/en/stable/>`_

"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import multiprocessing
import pathlib
import tempfile
import warnings
from collections.abc import Iterable
from typing import Any

import xarray as xr
from overrides import overrides

from pycontrails.core import cache, met_var
from pycontrails.core.met import MetDataset
from pycontrails.datalib._met_utils import metsource
from pycontrails.datalib.ecmwf import common as ecmwf_common
from pycontrails.datalib.ecmwf import variables as ecmwf_variables
from pycontrails.datalib.ecmwf.model_levels import pressure_levels_at_model_levels
from pycontrails.utils import dependencies

try:
    import gcsfs
except ModuleNotFoundError as e:
    dependencies.raise_module_not_found_error(
        "arco_era5 module",
        package_name="gcsfs",
        module_not_found_error=e,
        pycontrails_optional_package="zarr",
    )

MOISTURE_STORE = "gs://gcp-public-data-arco-era5/co/model-level-moisture.zarr"
WIND_STORE = "gs://gcp-public-data-arco-era5/co/model-level-wind.zarr"
SURFACE_STORE = "gs://gcp-public-data-arco-era5/co/single-level-surface.zarr"
SINGLE_LEVEL_PREFIX = "gs://gcp-public-data-arco-era5/raw/date-variable-single_level"

WIND_STORE_VARIABLES = [
    met_var.AirTemperature,
    met_var.VerticalVelocity,
    met_var.EastwardWind,
    met_var.NorthwardWind,
    ecmwf_variables.RelativeVorticity,
    ecmwf_variables.Divergence,
]

MOISTURE_STORE_VARIABLES = [
    met_var.SpecificHumidity,
    ecmwf_variables.CloudAreaFractionInLayer,
    ecmwf_variables.SpecificCloudIceWaterContent,
    ecmwf_variables.SpecificCloudLiquidWaterContent,
]

PRESSURE_LEVEL_VARIABLES = [*WIND_STORE_VARIABLES, *MOISTURE_STORE_VARIABLES, met_var.Geopotential]


def _attribute_fix(ds: xr.Dataset | None) -> None:
    """Fix GRIB attributes.

    See:
    https://github.com/google-research/arco-era5/blob/90f4c3dfc31692be73006e0ee841b620ecf81e7c/docs/moisture_dataset.py#L12
    """

    if ds is None:
        return

    for da in ds.values():
        da.attrs.pop("GRIB_cfName", None)


@dataclasses.dataclass
class _ARCOERA5Datasets:
    wind: xr.Dataset | None
    moisture: xr.Dataset | None
    surface: xr.Dataset | None


def _required_wind_short_names(variables: list[met_var.MetVariable]) -> list[str]:
    """Get the required wind variable short names needed to compute the requested variables."""
    out = set()
    for var in variables:
        if var in (met_var.AirTemperature, met_var.Geopotential):
            out.add("t")
        elif var in (met_var.EastwardWind, met_var.NorthwardWind):
            out.add("d")
            out.add("vo")
        elif var == met_var.VerticalVelocity:
            out.add("w")
        elif var == ecmwf_variables.RelativeVorticity:
            out.add("vo")
        elif var == ecmwf_variables.Divergence:
            out.add("d")

    return sorted(out)


def _required_moisture_short_names(variables: list[met_var.MetVariable]) -> list[str]:
    """Get the required moisture variable short names needed to compute the requested variables."""
    moisture_vars = set(MOISTURE_STORE_VARIABLES)

    out = set()
    for var in variables:
        if var in moisture_vars:
            out.add(var.short_name)
        elif var == met_var.Geopotential:
            out.add("q")
    return sorted(out)


def _required_surface_short_names(variables: list[met_var.MetVariable]) -> list[str]:
    """Get the required surface variable short names needed to compute the requested variables."""
    if met_var.Geopotential in variables:
        return ["lnsp", "z"]
    return ["lnsp"] if variables else []


def _download_data(
    t: datetime.datetime,
    variables: list[met_var.MetVariable],
) -> _ARCOERA5Datasets:
    """Download slices of the ARCO ERA5 model level Zarr stores."""

    wind_vars = _required_wind_short_names(variables)
    moisture_vars = _required_moisture_short_names(variables)
    surface_vars = _required_surface_short_names(variables)

    kw: dict[str, Any] = {"chunks": None, "consolidated": True}
    wind_ds = xr.open_zarr(WIND_STORE, **kw)[wind_vars].sel(time=t) if wind_vars else None
    moisture_ds = (
        xr.open_zarr(MOISTURE_STORE, **kw)[moisture_vars].sel(time=t) if moisture_vars else None
    )
    surface_ds = (
        xr.open_zarr(SURFACE_STORE, **kw)[surface_vars].sel(time=t) if surface_vars else None
    )
    return _ARCOERA5Datasets(wind=wind_ds, moisture=moisture_ds, surface=surface_ds)


def _handle_metview(
    data: _ARCOERA5Datasets,
    variables: list[met_var.MetVariable],
    pressure_levels: list[int],
    grid: float,
) -> xr.Dataset:
    try:
        import metview as mv
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            "arco_era5 module",
            package_name="metview",
            module_not_found_error=exc,
            extra="See https://metview.readthedocs.io/en/latest/install.html for instructions.",
        )
    except ImportError as exc:
        msg = "Failed to import metview"
        raise ImportError(msg) from exc

    # Extract any moisture data (defined on a Gaussian grid)
    gg_ml = mv.Fieldset()  # Gaussian grid on model levels
    if data.moisture:
        moisture_gg = mv.dataset_to_fieldset(data.moisture, no_warn=True)
        gg_ml = mv.merge(gg_ml, moisture_gg)

    # Convert any wind data (defined on a spherical harmonic grid) to the Gaussian grid
    if data.wind:
        wind_sh = mv.dataset_to_fieldset(data.wind, no_warn=True)
        if met_var.EastwardWind in variables or met_var.NorthwardWind in variables:
            uv_wind_sh = mv.uvwind(data=wind_sh, truncation=639)
            wind_sh = mv.merge(wind_sh, uv_wind_sh)
        wind_gg = mv.read(data=wind_sh, grid="N320")
        gg_ml = mv.merge(gg_ml, wind_gg)

    # Convert any surface data (defined on a spherical harmonic grid) to the Gaussian grid
    surface_sh = mv.dataset_to_fieldset(data.surface, no_warn=True)
    surface_gg = mv.read(data=surface_sh, grid="N320")
    lnsp = surface_gg.select(shortName="lnsp")

    # Compute Geopotential if requested
    if met_var.Geopotential in variables:
        t = gg_ml.select(shortName="t")
        q = gg_ml.select(shortName="q")
        zs = surface_gg.select(shortName="z")
        zp = mv.mvl_geopotential_on_ml(t, q, lnsp, zs)
        gg_ml = mv.merge(gg_ml, zp)

    # Convert the Gaussian grid to a lat-lon grid
    gg_pl = mv.Fieldset()  # Gaussian grid on pressure levels
    for var in variables:
        var_gg_ml = gg_ml.select(shortName=var.short_name)
        var_gg_pl = mv.mvl_ml2hPa(lnsp, var_gg_ml, pressure_levels)
        gg_pl = mv.merge(gg_pl, var_gg_pl)

    # Regrid the Gaussian grid pressure level data to a lat-lon grid
    ll_pl = mv.read(data=gg_pl, grid=[grid, grid])

    ds = ll_pl.to_dataset()
    return MetDataset(ds.rename(isobaricInhPa="level").expand_dims("time")).data


def open_arco_era5_model_level_data(
    t: datetime.datetime,
    variables: list[met_var.MetVariable],
    pressure_levels: list[int],
    grid: float,
) -> xr.Dataset:
    r"""Open ARCO ERA5 model level data for a specific time and variables.

    This function downloads moisture, wind, and surface data from the
    `ARCO ERA5 <https://cloud.google.com/storage/docs/public-datasets/era5>`_
    Zarr stores and interpolates the data to a target grid and pressure levels.

    This function requires the `metview <https://metview.readthedocs.io/en/latest/python.html>`_
    package to be installed. It is not available as an optional pycontrails dependency,
    and instead must be installed manually.

    Parameters
    ----------
    t : datetime.datetime
        Time of the data to open.
    variables : list[met_var.MetVariable]
        List of variables to open. Unsupported variables are ignored.
    pressure_levels : list[int]
        Target pressure levels, [:math:`hPa`]. For ``metview`` compatibility, this should be
        a sorted (increasing or decreasing) list of integers. Floating point values
        are treated as integers in ``metview``.
    grid : float
        Target grid resolution, [:math:`\deg`]. A value of 0.25 is recommended.

    Returns
    -------
    xr.Dataset
        Dataset with the requested variables on the target grid and pressure levels.
        Data is reformatted for :class:`MetDataset` conventions.
        Data **is not** cached.

    References
    ----------
    - :cite:`carverARCOERA5AnalysisReadyCloudOptimized2023`
    - `ARCO ERA5 moisture workflow <https://github.com/google-research/arco-era5/blob/main/docs/moisture_dataset.py>`_
    - `Model Level Walkthrough <https://github.com/google-research/arco-era5/blob/main/docs/1-Model-Levels-Walkthrough.ipynb>`_
    - `Surface Reanalysis Walkthrough <https://github.com/google-research/arco-era5/blob/main/docs/0-Surface-Reanalysis-Walkthrough.ipynb>`_
    """
    data = _download_data(t, variables)

    if not data.surface:
        msg = "No variables provided"
        raise ValueError(msg)

    _attribute_fix(data.wind)
    _attribute_fix(data.moisture)
    _attribute_fix(data.surface)

    # Ignore all the metview warnings from deprecated pandas usage
    # This could be removed after metview updates their python API
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A value is trying to be set on a copy of a DataFrame",
            category=FutureWarning,
        )
        return _handle_metview(data, variables, pressure_levels, grid)


def open_arco_era5_single_level(
    t: datetime.date,
    variables: list[met_var.MetVariable],
) -> xr.Dataset:
    """Open ARCO ERA5 single level data for a specific date and variables.

    Parameters
    ----------
    t : datetime.date
        Date of the data to open.
    variables : list[met_var.MetVariable]
        List of variables to open.

    Returns
    -------
    xr.Dataset
        Dataset with the requested variables.
        Data is reformatted for :class:`MetDataset` conventions.
        Data **is not** cached.

    Raises
    ------
    FileNotFoundError
        If the variable is not found at the requested date. This could
        indicate that the variable is not available in the ARCO ERA5 dataset,
        or that the time requested is outside the available range.
    """
    gfs = gcsfs.GCSFileSystem()

    prefix = f"{SINGLE_LEVEL_PREFIX}/{t.year}/{t.month:02}/{t.day:02}"

    ds_list = []
    for var in variables:
        uri = f"{prefix}/{var.standard_name}/surface.nc"

        try:
            data = gfs.cat(uri)
        except FileNotFoundError as exc:
            msg = f"Variable {var.standard_name} at date {t} not found"
            raise FileNotFoundError(msg) from exc

        ds = xr.open_dataset(data)
        ds_list.append(ds)

    ds = xr.merge(ds_list)
    return MetDataset(ds.expand_dims(level=[-1])).data


class ARCOERA5(ecmwf_common.ECMWFAPI):
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
    grid : float, optional
        Target grid resolution, [:math:`\deg`]. Default is 0.25.
    cachestore : CacheStore, optional
        Cache store to use. By default, a new disk cache store is used. If None, no caching is done.
    n_jobs : int, optional
        EXPERIMENTAL: Number of parallel jobs to use for downloading data. By default, 1.
    cleanup_metview_tempfiles : bool, optional
        If True, cleanup all ``TEMP_DIRECTORY/tmp*.grib`` files. Implementation is brittle and may
        not work on all systems. By default, True.

    References
    ----------
    :cite:`carverARCOERA5AnalysisReadyCloudOptimized2023`

    See Also
    --------
    :func:`open_arco_era5_model_level_data`
    :func:`open_arco_era5_single_level`
    """

    grid: float

    __marker = object()

    def __init__(
        self,
        time: metsource.TimeInput,
        variables: metsource.VariableInput,
        pressure_levels: metsource.PressureLevelInput | None = None,
        grid: float = 0.25,
        cachestore: cache.CacheStore | None = __marker,  # type: ignore[assignment]
        n_jobs: int = 1,
        cleanup_metview_tempfiles: bool = True,
    ) -> None:
        self.timesteps = metsource.parse_timesteps(time)

        if pressure_levels is None:
            self.pressure_levels = pressure_levels_at_model_levels(20_000.0, 50_000.0)
        else:
            self.pressure_levels = metsource.parse_pressure_levels(pressure_levels)

        self.paths = None
        self.variables = metsource.parse_variables(variables, self.supported_variables)
        self.grid = grid
        self.cachestore = cache.DiskCacheStore() if cachestore is self.__marker else cachestore
        self.n_jobs = max(1, n_jobs)
        self.cleanup_metview_tempfiles = cleanup_metview_tempfiles

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

    @overrides
    def download_dataset(self, times: list[datetime.datetime]) -> None:
        if not times:
            return

        # Download single level data sequentially
        if self.is_single_level:
            unique_dates = sorted({t.date() for t in times})
            for t in unique_dates:
                ds = open_arco_era5_single_level(t, self.variables)
                self.cache_dataset(ds)
            return

        stack = contextlib.ExitStack()
        if self.cleanup_metview_tempfiles:
            stack.enter_context(_MetviewTempfileHandler())

        n_jobs = min(self.n_jobs, len(times))

        # Download sequentially if n_jobs == 1
        if n_jobs == 1:
            for t in times:
                with stack:  # cleanup after each iteration
                    _download_convert_cache_handler(self, t)
            return

        # Download in parallel
        args = [(self, t) for t in times]
        mp = multiprocessing.get_context("spawn")
        with mp.Pool(n_jobs) as pool, stack:  # cleanup after pool finishes work
            pool.starmap(_download_convert_cache_handler, args, chunksize=1)

    @overrides
    def create_cachepath(self, t: datetime.datetime) -> str:
        if self.cachestore is None:
            msg = "Attribute self.cachestore must be defined to create cache path"
            raise ValueError(msg)

        string = (
            f"{t:%Y%m%d%H}-"
            f"{'.'.join(str(p) for p in self.pressure_levels)}-"
            f"{'.'.join(sorted(self.variable_shortnames))}-"
            f"{self.grid}"
        )
        name = hashlib.md5(string.encode()).hexdigest()
        cache_path = f"arcoera5-{name}.nc"

        return self.cachestore.path(cache_path)

    @overrides
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
            msg = "Cachestore is required to download data"
            raise ValueError(msg)

        xr_kwargs = xr_kwargs or {}
        self.download(**xr_kwargs)

        disk_cachepaths = [self.cachestore.get(f) for f in self._cachepaths]
        ds = self.open_dataset(disk_cachepaths, **xr_kwargs)

        mds = self._process_dataset(ds, **kwargs)

        self.set_metadata(mds)
        return mds

    @overrides
    def set_metadata(self, ds: xr.Dataset | MetDataset) -> None:
        ds.attrs.update(
            provider="ECMWF",
            dataset="ERA5",
            product="reanalysis",
        )


def _download_convert_cache_handler(arco: ARCOERA5, t: datetime.datetime) -> None:
    """Download, convert, and cache ARCO ERA5 model level data."""
    ds = open_arco_era5_model_level_data(t, arco.variables, arco.pressure_levels, arco.grid)
    arco.cache_dataset(ds)


def _get_grib_files() -> Iterable[pathlib.Path]:
    """Get all temporary GRIB files."""
    tmp = pathlib.Path(tempfile.gettempdir())
    return tmp.glob("tmp*.grib")


class _MetviewTempfileHandler:
    def __enter__(self) -> None:
        self.existing_grib_files = set(_get_grib_files())

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        new_grib_files = _get_grib_files()
        for f in new_grib_files:
            if f not in self.existing_grib_files:
                f.unlink(missing_ok=True)
