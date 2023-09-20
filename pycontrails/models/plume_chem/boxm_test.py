import numpy as np
import pandas as pd
import xarray as xr
import datetime
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.emissions import Emissions
from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.core.met_var import (
    AirTemperature,
    SpecificHumidity,
)
from init_chem import ChemDataset
from boxm import BoxModel

# Initialise coord arrays
lon_bounds = (-180, 180) #np.arange(-180, 180, 5)
lat_bounds = (-90, 90) #np.arange(-90, 90, 5)

grid_met = 0.5
grid_chem = 5.0
met_pressure_levels = np.array([400, 300, 200, 100])

time = ("2022-03-01 00:00:00", "2022-03-01 08:00:00")
ts_met = "1H"
ts_disp = "5min"
ts_chem = "5min"

runtime = "24H"

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

print(met)

# initialise example chem MetDataset
chem = ChemDataset(
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        met=met,
        time=time,
)

chem.open_chemdataset()

print(chem.data["Y"])

# # Initialise box model with met and chem data
# boxm = BoxModel(met, chem)

# # Import emissions data from dry advection

# # Convert dry advection to emissions dataset
# emi = xr.Dataset(
#     {
#         "EM": (["latitude", "longitude", "level", "time", "species"],
#                 np.zeros((len(latitude), len(longitude), len(level), len(time), len(species)))),
#     },
#     coords={
#         "latitude": latitude,
#         "longitude": longitude, 
#         "level": level,
#         "time": time,
#         "species": species,
#     }
# )

# # Run box model
# boxm.eval(source=emi)

