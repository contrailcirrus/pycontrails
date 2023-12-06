import numpy as np
import xarray as xr
import pandas as pd
import datetime
from pycontrails.datalib.ecmwf import ERA5
from init_chem import CHEM
from chem import ChemDataset
import boxm_for
from pycontrails.physics import units


# This file needs to generate a common input to feed to both original BOXM and new f2py implementation. The aim is to generate two sets of outputs (species concs over time), and to diff these outputs to see how different they are to each other.

# Hard coded inputs
time = ("2022-01-01 00:00:00", "2022-01-01 12:00:00")
lon_bounds = (0, 5) 
lat_bounds = (0, 5)
alt_bounds = (8000, 8500)
horiz_res = 5
vert_res = 500
ts_met = "1H"
ts_disp = "1min"
ts_chem = "1H"

met_pressure_levels = np.array([400, 300, 200, 100])

lons = np.arange(lon_bounds[0], lon_bounds[1], horiz_res)
lats = np.arange(lat_bounds[0], lat_bounds[1], horiz_res)
alts = np.arange(alt_bounds[0], alt_bounds[1], vert_res)

# Import met data from ERA5
era5 = ERA5(
        time=time,
        timestep_freq=ts_met,
        variables=[
                "t",
                "q",
                "u",
                "v",
                "w",
                "z",
                "relative_humidity"
        ],
        pressure_levels=met_pressure_levels
)

# download data from ERA5 (or open from cache)
met = era5.open_metdataset()
met.data = met.data.transpose("latitude", "longitude", "level", "time", ...)

# Initialise low-res chem xarray dataset, and import species files
init_chem = CHEM(
        time=time,
        timestep_freq=ts_chem,
)

# Populate chem dataset with species data
chem = init_chem.open_chemdataset()

# Downselect and interpolate both met and chem datasets to high-res grid
met.data = met.data.interp(longitude=lons, latitude=lats, level=units.m_to_pl(alts), method="linear")
chem.data = chem.data.interp(longitude=lons, latitude=lats, level=units.m_to_pl(alts), method="linear")

print(met["air_temperature"].data.shape)
print(met["air_pressure"].data.shape)

temp = met["air_temperature"].data.values[0,0,0,0]
pressure = met["air_pressure"].data.values[0]

# set arrays of temp and pressure to constant scalar value
met["air_temperature"].data.values = temp
met["air_pressure"].data.values = pressure

# # Run original boxm
# run_boxm(met, chem)

# # Run f2py implementation
# run_f2py(met, chem)





