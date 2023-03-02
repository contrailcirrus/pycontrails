"""Generate GFS test fixtures."""

import pathlib

from pycontrails import DiskCacheStore
from pycontrails.datalib.gfs import GFSForecast
from pycontrails.models import Cocip

# use separate disk cache
disk_cache = DiskCacheStore(cache_dir="cache")

# paths
parent = pathlib.Path(__file__).parents[1].absolute()
static_dir = parent / "unit" / "static"

# TODO: add test fixture for original GFS grib file
#


# see cocip_persistent test fixture
# this has to be > 2021 since older data is not available
times = ("2022-01-01 00:00:00", "2022-01-01 06:00:00")
met = GFSForecast(
    times,
    variables=Cocip.met_variables,
    pressure_levels=[200, 250, 300],
    cachestore=disk_cache,
    show_progress=True,
)
rad = GFSForecast(times, variables=Cocip.rad_variables, show_progress=True)

mds = met.open_metdataset(xr_kwargs=dict(parallel=False))
rds = rad.open_metdataset(xr_kwargs=dict(parallel=False))

for datasource, name in [(met, "met"), (rad, "rad")]:
    mds = datasource.open_metdataset(xr_kwargs=dict(parallel=False))

    # downselect region
    mds = mds.downselect([-40, 40, -20, 60])

    # reindex
    step = 5
    mds.data = mds.data.reindex(
        latitude=mds.data["latitude"].values[0::step],
        longitude=mds.data["longitude"].values[0::step],
    )
    mds.data.to_netcdf(static_dir / f"gfs-{name}.nc")
