"""Test Fixtures.

Create test data for pycontrails.

See output in `pycontrails/tests/static`
"""

# import logging
# logger = logging.getLogger("pycontrails")
# logging.basicConfig(level=logging.DEBUG)

import pathlib
import sys
from datetime import datetime

import xarray as xr

from pycontrails import DiskCacheStore
from pycontrails.datalib.ecmwf import ERA5

# use separate disk cache
disk_cache = DiskCacheStore(cache_dir="cache")

# paths
parent = pathlib.Path(sys.argv[0]).parents[1].absolute()
static_dir = parent / "unit" / "static"

# domain
times = (datetime(2019, 5, 31, 5, 0, 0), datetime(2019, 5, 31, 6, 0, 0))
pl_variables = ["air_temperature", "specific_humidity", "specific_cloud_ice_water_content"]
sl_variables = ["surface_air_pressure"]
pressure_levels = [300, 250, 225]

### ERA5 PL data

era5pl = ERA5(
    times,
    variables=pl_variables,
    pressure_levels=pressure_levels,
    cachestore=disk_cache,
)
_ = era5pl.open_metdataset(xr_kwargs={"parallel": False})

# NOTE we are saving the cached CDS *source* files here.
# These are not the same as purely raw downloaded files from CDS.
ds_pl = xr.open_mfdataset(era5pl._cachepaths)
step = 100

ds = ds_pl.reindex(
    latitude=ds_pl["latitude"].values[0::step], longitude=ds_pl["longitude"].values[0::step]
)
ds.to_netcdf(static_dir / "met-ecmwf-pl.nc")


### ERA5 Single Level
era5sl = ERA5(times, variables=sl_variables, cachestore=disk_cache)
_ = era5sl.open_metdataset(xr_kwargs={"parallel": False})

# NOTE we are saving the raw CDS *source* files here
ds_sl = xr.open_mfdataset(era5sl._cachepaths)
step = 100
ds = ds_sl.reindex(
    latitude=ds_sl["latitude"].values[0::step], longitude=ds_sl["longitude"].values[0::step]
)
ds.to_netcdf(static_dir / "met-ecmwf-sl.nc")
