"""Test Fixtures.

Create test data for pycontrails. The data itself is committed to the repository,
so this script only needs to be run once. Because ECMWF updates their reanalysis
data from time to time, re-running this script will alter the data.

See output in ``pycontrails/tests/unit/static``.
"""

# import logging
# logger = logging.getLogger("pycontrails")
# logging.basicConfig(level=logging.DEBUG)

import pathlib
from datetime import datetime

import xarray as xr

from pycontrails import DiskCacheStore
from pycontrails.datalib.ecmwf import ERA5, ERA5ModelLevel

# use separate disk cache
disk_cache = DiskCacheStore(cache_dir="cache")

# paths
parent = pathlib.Path(__file__).parents[1].absolute()
static_dir = parent / "unit" / "static"

# domain
times = (datetime(2019, 5, 31, 5, 0, 0), datetime(2019, 5, 31, 6, 0, 0))
pl_variables = ["air_temperature", "specific_humidity", "specific_cloud_ice_water_content"]
sl_variables = ["surface_air_pressure"]
pressure_levels = [300, 250, 225]
step = 100  # used for downselecting the domain

### ERA5 PL data

era5pl = ERA5(
    times,
    variables=pl_variables,
    pressure_levels=pressure_levels,
    cachestore=disk_cache,
)
era5pl.download()

# NOTE we are saving the cached CDS *source* files here.
# These are not the same as purely raw downloaded files from CDS.
ds_pl = xr.open_mfdataset(era5pl._cachepaths)
ds = ds_pl.isel(latitude=slice(0, None, step), longitude=slice(0, None, step))

target = static_dir / "met-ecmwf-pl.nc"
if not target.is_file():
    ds.to_netcdf(target)


### ERA5 Single Level
era5sl = ERA5(times, variables=sl_variables, cachestore=disk_cache)
era5sl.download()

# NOTE we are saving the raw CDS *source* files here
ds_sl = xr.open_mfdataset(era5sl._cachepaths)
ds = ds_sl.isel(latitude=slice(0, None, step), longitude=slice(0, None, step))

target = static_dir / "met-ecmwf-sl.nc"
if not target.is_file():
    ds.to_netcdf(target)


### Model level data
era5ml = ERA5ModelLevel(
    times[0],  # download a single time to save space
    variables=pl_variables,
    model_levels=list(range(74, 85)),
    pressure_levels=pressure_levels,
    cachestore=disk_cache,
    cache_download=True,
)
era5ml.download()

lnsp_name = "era5ml-bdb942b20d6a170e4c0240715101e6c7-raw.nc"
ds = xr.open_mfdataset(disk_cache.path(lnsp_name))
ds = ds.isel(latitude=slice(0, None, step), longitude=slice(0, None, step))

target = static_dir / "met-ecmwf-lnsp.nc"
if not target.is_file():
    ds.to_netcdf(target)


ml_name = "era5ml-2ad0995e9616aa91c702bf6c3abe752f-raw.nc"
ds = xr.open_mfdataset(disk_cache.path(ml_name))
ds = ds.isel(latitude=slice(0, None, step), longitude=slice(0, None, step))

target = static_dir / "met-ecmwf-ml.nc"
if not target.is_file():
    ds.to_netcdf(target)
