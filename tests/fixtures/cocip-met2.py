"""
Build assets for second set of cocip unit tests.

Built from static data in `/tests/regression/cocip-fortran/inputs`.
See `/tests/regression/cocip-fortran/README.md` for instructions to download inputs.

Run this file from the terminal to generate assets:

```python
$ python cocip-met2.py
```

"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from pycontrails import DiskCacheStore, Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models import Cocip
from pycontrails.physics import units

# use separate disk cache
disk_cache = DiskCacheStore(cache_dir="cache")


# from pycontrails.models.humidity_scaling import ConstantHumidityScaling


# setup
flight_input_filename = "flight1.csv"

input_path = pathlib.Path("../regression/cocip-fortran/inputs")
output_path = pathlib.Path("../unit/static")


## ----
# Load Flight
# ----

# load flight waypoints
df_flight = pd.read_csv(input_path / flight_input_filename)

# constant properties along the length of the flight
attrs = {
    "flight_id": flight_input_filename,
    "aircraft_type": df_flight["ICAO Aircraft Type"].values[0],
    "wingspan": df_flight["Wingspan (m)"].values[0],
}

# convert UTC timestamp to np.datetime64
df_flight["time"] = pd.to_datetime(df_flight["UTC time"], origin="unix", unit="s")

# get altitude in m
df_flight["altitude"] = units.ft_to_m(df_flight["Altitude (feet)"])

# rename a few columns for compatibility with `Flight` requirements
df_flight = df_flight.rename(
    columns={
        "Longitude (degrees)": "longitude",
        "Latitude (degrees)": "latitude",
        "True airspeed (m s-1)": "true_airspeed",
        "Mach Number": "mach_number",
        "Aircraft mass (kg)": "aircraft_mass",
        "Fuel mass flow rate (kg s-1)": "fuel_flow",
        "Overall propulsion efficiency": "engine_efficiency",
        "nvPM number emissions index (kg-1)": "nvpm_ei_n",
    }
)

# clean up a few columns before building Flight class
df_flight = df_flight.drop(
    columns=["ICAO Aircraft Type", "Wingspan (m)", "UTC time", "Altitude (feet)"]
)

fl = Flight(data=df_flight, attrs=attrs)


# get met dims from Flight
time = (
    pd.to_datetime(fl["time"][0]).floor("H"),
    pd.to_datetime(fl["time"][-1]).ceil("H") + pd.Timedelta("5H"),
)
pressure_levels = [
    400,
    350,
    300,
    250,
    225,
    200,
    175,
    150,
]  # select pressure levels

# load remote met - only necessary when not using regression test inputs
# era5pl = ERA5(
#     time=time,
#     variables=Cocip.met_variables,
#     pressure_levels=pressure_levels,
#     cachestore=disk_cache,
# )
# era5sl = ERA5(time=time, variables=Cocip.rad_variables, cachestore=disk_cache)

# load met locally from `input_path`
filenames = [t.strftime("%Y-%m-%dT%H") for t in pd.date_range(time[0], time[1], freq="1H")]

met_filepaths = [str(input_path / "met" / f"{f}.nc") for f in filenames]
rad_filepaths = [str(input_path / "rad" / f"{f}.nc") for f in filenames]


era5pl = ERA5(
    time=time, variables=Cocip.met_variables, pressure_levels=pressure_levels, paths=met_filepaths
)
era5sl = ERA5(time=time, variables=Cocip.rad_variables, paths=rad_filepaths)

# create `MetDataset` from sources
met = era5pl.open_metdataset(xr_kwargs=dict(parallel=False))
rad = era5sl.open_metdataset(xr_kwargs=dict(parallel=False))


params = {
    "downselect_met": True,
    "process_emissions": False,
    "max_age": np.timedelta64(4, "h"),
    # "interpolation_fill_value": 0.0,
    "verbose_outputs": True,
    # "humidity_scaling": ConstantHumidityScaling(rhi_adj=1)
}

cocip = Cocip(met=met, rad=rad, params=params)
fl_out = cocip.eval(source=fl)


## Unit test output
df_flight["flight_id"] = "test2"
df_flight["wingspan"] = 65
df_flight.to_csv(output_path / "flight-cocip2.csv", index=False)


met_test = cocip.met.data
rad_test = cocip.rad.data

step = 8
met_test = met_test.reindex(
    latitude=met_test["latitude"].values[0::step],
    longitude=met_test["longitude"].values[0::step],
    level=[200, 225, 300],
)
rad_test = rad_test.reindex(
    latitude=rad_test["latitude"].values[0::step], longitude=rad_test["longitude"].values[0::step]
)

# outputs
met_test.to_netcdf(output_path / "met-era5-cocip2.nc")
rad_test.to_netcdf(output_path / "rad-era5-cocip2.nc")


print(f"Static test assets written to {output_path}")
