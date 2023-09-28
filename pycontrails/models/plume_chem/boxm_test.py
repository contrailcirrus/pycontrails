import numpy as np
import pandas as pd
import xarray as xr
import datetime
import matplotlib.pyplot as plt
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.emissions import Emissions
from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.physics import units
from pycontrails.core.met_var import (
    AirTemperature,
    SpecificHumidity,
)
from init_chem import CHEM
from chem import ChemDataset
from boxm import BoxModel
import cartopy.crs as ccrs


# Initialise coord arrays
lon_bounds = (-180, 180)
lat_bounds = (-90, 90)
alt_bounds = (8000, 12000)
horiz_res = 0.1
vert_res = 500
met_pressure_levels = np.array([400, 300, 200, 100])

time = ("2022-01-01 00:00:00", "2022-01-01 08:00:00")
ts_met = "1H"
ts_disp = "1min"
ts_chem = "5min"

runtime = "24H"

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


# Initialise low-res chem xarray dataset, and import species files
init_chem = CHEM(
        time=time,
        timestep_freq=ts_chem,
)

# Populate chem dataset with species data
chem = init_chem.open_chemdataset()


# Preprocess function that takes met, chem and disp data and downselects and interpolates it. Also preprocess chem as much as poss.
met.data = met.data.transpose("latitude", "longitude", "level", "time", ...)

met.data = met.data.interp(longitude=lons, latitude=lats, level=units.m_to_pl(alts), method="linear")

print(met.data)

# Plot data

p = met.data["air_temperature"].isel(level=0, time=0).plot(
subplot_kws=dict(projection=ccrs.Orthographic(20, 10), facecolor="gray"),
transform=ccrs.PlateCarree(),
)

#plt.colorbar(p, ax=p.axes)

p.axes.set_global()

p.axes.coastlines()

# save figure
p.figure.savefig(str(ilat) + ".png")
plt.close(p.figure)




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

