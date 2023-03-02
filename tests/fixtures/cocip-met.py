"""
Build netcdf assets for cocip unit tests.

- Requires account with [Copernicus Data Portal](https://cds.climate.copernicus.eu/cdsapp#!/home)
and  local `~/.cdsapirc` file with credentials.
"""

import pathlib
from datetime import datetime

from pycontrails import DiskCacheStore
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models import Cocip

# use separate disk cache
disk_cache = DiskCacheStore(cache_dir="cache")

# paths
parent = pathlib.Path(__file__).parents[1].absolute()
static_dir = parent / "unit" / "static"

## Set Domain
time = (datetime(2019, 1, 1, 0), datetime(2019, 1, 1, 12))
pressure_levels = [300, 250, 225, 200]

## Get Data
variables = (*Cocip.met_variables, "fraction_of_cloud_cover")
era5pl = ERA5(time, variables, pressure_levels, cachestore=disk_cache)
era5sl = ERA5(time, Cocip.rad_variables, cachestore=disk_cache)

# download data from ERA5 (or open from cache)
met = era5pl.open_metdataset(chunks={"time": 1, "level": 2})
rad = era5sl.open_metdataset(chunks={"time": 1})

## Write out for tests

met.data = met.data[
    dict(
        latitude=((met.data["latitude"] < 60) & (met.data["latitude"] > 50)),
        longitude=((met.data["longitude"] < -20) & (met.data["longitude"] > -40)),
    )
]
rad.data = rad.data[
    dict(
        latitude=((rad.data["latitude"] < 60) & (rad.data["latitude"] > 50)),
        longitude=((rad.data["longitude"] < -20) & (rad.data["longitude"] > -40)),
    )
]


step = 5
met.data = met.data.reindex(
    latitude=met.data["latitude"].values[0::step], longitude=met.data["longitude"].values[0::step]
)
rad.data = rad.data.reindex(
    latitude=rad.data["latitude"].values[0::step], longitude=rad.data["longitude"].values[0::step]
)

# outputs
met.data.to_netcdf(static_dir / "met-era5-cocip1.nc")
rad.data.to_netcdf(static_dir / "rad-era5-cocip1.nc")


# ## Test import

# met = MetDataset(xr.open_dataset("met-era5-cocip1.nc"))
# rad = MetDataset(xr.open_dataset("rad-era5-cocip1.nc"))
