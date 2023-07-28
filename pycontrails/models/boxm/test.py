# This is a script to test all current functions to be in boxm.py

import pandas as pd
import xarray as xr
import numpy as np
from pycontrails import MetDataset, MetDataArray, MetVariable
from pysolar.solar import *
import datetime
from IPython.display import display
from zenith import zenith
from photol import photol
from pycontrails.physics.geo import cosine_solar_zenith_angle, orbital_position
from pycontrails.datalib.ecmwf import ERA5

# Set up date
date = datetime.datetime(2000, 6, 1, 0, 0, tzinfo=datetime.timezone.utc)

# PHOTOLYSIS PARAMETERS IN FORMAT J = L*COSX**M*EXP(-N*SECX)
consts = pd.read_pickle('J.pkl')
    
# Extract the constants
photol_idx, L, M, N = np.array(consts).T

# Initialise coord arrays
longitude = np.arange(-180, 180, 10)
latitude = np.arange(-90, 90, 10)
level = np.array([1013, 912, 810, 709, 607, 505, 404, 302, 201, 100])
gm_time = [datetime.datetime(2000, 6, 1, h, 0) for h in range(0, 10)] # always GMT!
photol_params = photol_idx
photol_coeffs = np.arange(1, 96 + 1) # from Fortran indexing (1 to 96)
therm_coeffs = np.arange(1, 510 + 1) # from Fortran indexing (1 to 510)
species = np.arange(1, 220 + 1) # from Fortran indexing (1 to 220)

# Grab bg atmospheric species data from 1998 UKMO Unified Model dataset
#Y_bg = 

# Import ERA5 pressure level data
era5 = ERA5(
        time=(gm_time[0], gm_time[-1]),
        variables=[
                "air_temperature",
                "q",
                "u",
                "v",
                "w",
                "z"
        ],
        pressure_levels=[1000, 925, 800, 700, 600, 500, 400, 300, 200, 100] 
)

# Initialise MetDataset
met = era5.open_metdataset()

# chem MetDataset
chem = xr.Dataset(
    {
        
        "local_time": (["latitude", "longitude", "gm_time"],
                np.zeros((len(latitude), len(longitude), len(gm_time)))),
        "sza": (["latitude", "longitude", "gm_time"], 
                np.zeros((len(latitude), len(longitude), len(gm_time)))),
        "J": (["latitude", "longitude", "gm_time", "photol_params"], 
                np.zeros((len(latitude), len(longitude), len(gm_time), len(photol_params)))),
        "DJ": (["latitude", "longitude", "level", "gm_time", "photol_coeffs"], 
                np.zeros((len(latitude), len(longitude), len(level), len(gm_time), len(photol_coeffs)))),
        "RC": (["latitude", "longitude", "level", "gm_time", "therm_coeffs"],
                np.zeros((len(latitude), len(longitude), len(level), len(gm_time), len(therm_coeffs)))),
        "Y": (["latitude", "longitude", "level", "gm_time", "species"], 
                np.zeros((len(latitude), len(longitude), len(level), len(gm_time), len(species)))),
        "EM": (["latitude", "longitude", "level", "gm_time", "species"],
                np.zeros((len(latitude), len(longitude), len(level), len(gm_time), len(species)))),
        "FL": (["latitude", "longitude", "level", "gm_time", "species"],
                np.zeros((len(latitude), len(longitude), len(level), len(gm_time), len(species)))),

    },
    coords={
        "latitude": latitude,
        "longitude": longitude, 
        "level": level,
        "gm_time": gm_time,
        "photol_params": photol_params,
        "photol_coeffs": photol_coeffs,
        "therm_coeffs": therm_coeffs,
        "species": species,
    }
)

# Interpolate to model grid
met.data = met.data.interp_like(chem.DJ).transpose("latitude", "longitude", "level", "time")

print(type(met.data))
print(type(met))

t_df = pd.DataFrame(met["air_temperature"].data[:, :, 9, 1])


# test zenith function
def test_zenith(chem):
        sza = zenith(chem) * 180 / np.pi
        chem["sza"] = sza

        pys = np.zeros((18, 36, 12))
        pyc = np.zeros((18, 36, 12))

        for t, date in enumerate(chem.gm_time.values):
                date_dt = date.astype(datetime.datetime) / 1e9
                date_dt = datetime.datetime.utcfromtimestamp(date_dt).replace(tzinfo=datetime.timezone.utc)
                
                for i, lat in enumerate(range(-90, 90, 10)):
                        for j, lon in enumerate(range(-180, 180, 10)):
                                pys[i, j, t] = (90 - get_altitude(lat, lon, date_dt))
                                theta_rad = orbital_position(date)
                                pyc[i, j, t] = np.degrees(np.arccos(cosine_solar_zenith_angle(lon, lat, date, theta_rad)))
        
        
        return chem, pys, pyc

# test chemco function
def test_chemco(chem):
        return chem

# test photol function
def test_photol(chem):
        
        J = photol(chem)
        print(J)
        print(chem["J"])
        #chem["J"] = J

        J_df = pd.DataFrame(J[:, :, 0, 1])
        return chem, J_df


# test deriv function
def test_deriv(chem):
        return chem

# test boxm.py

# Carry out tests
chem, pys, pyc = test_zenith(chem)

test_chemco(chem)

chem, J_df = test_photol(chem)

test_deriv(chem)

# Create dfs for better visual display
pys_df = pd.DataFrame(pys[:, :, 5])
pyc_df = pd.DataFrame(pyc[:, :, 5])
sza_df = pd.DataFrame(sza[:, :, 5])

diff = pys_df - sza_df 
diff2 = pyc_df - pys_df
perc_diff = diff / pys_df * 100

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
     display(t_df)
     display(sza_df)
#     display(pys_df)
#     display(pyc_df)
#     display(diff)
#     display(diff2)
#     #display(perc_diff)
     display(J_df)