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
from chemco import chemco
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
species = ['O1D', 'O', 'OH', 'NO2', 'NO3', 'O3', 'N2O5', 'NO', 'HO2', 'H2', 'CO', 
           'H2O2', 'HONO', 'HNO3', 'HO2NO2', 'SO2', 'SO3', 'HSO3', 'NA', 'SA', 
           'CH4', 'CH3O2', 'C2H6', 'C2H5O2', 'C3H8', 'IC3H7O2', 'RN10O2', 'NC4H10', 
           'RN13O2', 'C2H4', 'HOCH2CH2O2', 'C3H6', 'RN9O2', 'TBUT2ENE', 'RN12O2', 
           'NRN6O2', 'NRN9O2', 'NRN12O2', 'HCHO', 'HCOOH', 'CH3CO2H', 'CH3CHO', 
           'C5H8', 'RU14O2', 'NRU14O2', 'UCARB10', 'APINENE', 'RTN28O2', 'NRTN28O2', 
           'RTN26O2', 'TNCARB26', 'RCOOH25', 'BPINENE', 'RTX28O2', 'NRTX28O2', 
           'RTX24O2', 'TXCARB24', 'TXCARB22', 'C2H2', 'CARB3', 'BENZENE', 'RA13O2', 
           'AROH14', 'TOLUENE', 'RA16O2', 'AROH17', 'OXYL', 'RA19AO2', 'RA19CO2', 
           'CH3CO3', 'C2H5CHO', 'C2H5CO3', 'CH3COCH3', 'RN8O2', 'RN11O2', 'CH3OH', 
           'C2H5OH', 'NPROPOL', 'IPROPOL', 'CH3CL', 'CH2CL2', 'CHCL3', 'CH3CCL3', 
           'TCE', 'TRICLETH', 'CDICLETH', 'TDICLETH', 'CARB11A', 'RN16O2', 'RN15AO2', 
           'RN19O2', 'RN18AO2', 'RN13AO2', 'RN16AO2', 'RN15O2', 'UDCARB8', 'UDCARB11', 
           'CARB6', 'UDCARB14', 'CARB9', 'MEK', 'HOCH2CHO', 'RN18O2', 'CARB13', 
           'CARB16', 'HOCH2CO3', 'RN14O2', 'RN17O2', 'UCARB12', 'RU12O2', 'CARB7', 
           'RU10O2', 'NUCARB12', 'NRU12O2', 'NOA', 'RTN25O2', 'RTN24O2', 'RTN23O2', 
           'RTN14O2', 'TNCARB10', 'RTN10O2', 'RTX22O2', 'CH3NO3', 'C2H5NO3', 'RN10NO3', 
           'IC3H7NO3', 'RN13NO3', 'RN16NO3', 'RN19NO3', 'HOC2H4NO3', 'RN9NO3', 'RN12NO3', 
           'RN15NO3', 'RN18NO3', 'RU14NO3', 'RA13NO3', 'RA16NO3', 'RA19NO3', 'RTN28NO3', 
           'RTN25NO3', 'RTX28NO3', 'RTX24NO3', 'RTX22NO3', 'CH3OOH', 'C2H5OOH', 'RN10OOH', 
           'IC3H7OOH', 'RN13OOH', 'RN16OOH', 'RN19OOH', 'RA13OOH', 'RA16OOH', 'RA19OOH', 
           'HOC2H4OOH', 'RN9OOH', 'RN12OOH', 'RN15OOH', 'RN18OOH', 'CH3CO3H', 'C2H5CO3H', 
           'HOCH2CO3H', 'RN8OOH', 'RN11OOH', 'RN14OOH', 'RN17OOH', 'RU14OOH', 'RU12OOH', 
           'RU10OOH', 'NRN6OOH', 'NRN9OOH', 'NRN12OOH', 'NRU14OOH', 'NRU12OOH', 'RTN28OOH', 
           'NRTN28OOH', 'RTN26OOH', 'RTN25OOH', 'RTN24OOH', 'RTN23OOH', 'RTN14OOH', 
           'RTN10OOH', 'RTX28OOH', 'RTX24OOH', 'RTX22OOH', 'NRTX28OOH', 'CARB14', 'CARB17', 
           'CARB10', 'CARB12', 'CARB15', 'CCARB12', 'ANHY', 'TNCARB15', 'RAROH14', 'ARNOH14', 
           'RAROH17', 'ARNOH17', 'PAN', 'PPN', 'PHAN', 'RU12PAN', 'MPAN', 'RTN26PAN', 'P2604', 
           'P4608', 'P2631', 'P2635', 'P4610', 'P2605', 'P2630', 'P2629', 'P2632', 'P2637', 
           'P3612', 'P3613', 'P3442', 'CH3O2NO2', 'EMPOA', 'P2007']



# Grab bg atmospheric species data from 1998 UKMO Unified Model dataset
#Y_bg = 

# Import ERA5 pressure level data
era5 = ERA5(
        time=(gm_time[0], gm_time[-1]),
        variables=[
                "t",
                "q",
                "u",
                "v",
                "w",
                "z"
        ],
        pressure_levels=[1000, 925, 800, 700, 600, 500, 400, 300, 200, 100] 
)

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

print(chem.Y.isel(level=1, gm_time=1).sel(species='O').shape)

# Initialise MetDataset
met = era5.open_metdataset()

# Interpolate to model grid
met.data = met.data.interp_like(chem.DJ).transpose("latitude", "longitude", "level", "time")

t_df = pd.DataFrame(met["air_temperature"].data[:, :, 1, 1])
met["air_pressure"] = met.broadcast_coords("air_pressure")

p_df = pd.DataFrame(met["air_pressure"].data[:, :, 1, 1])


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
def test_chemco(met, chem):
        chem, rho_d, M, H2O, O2, N2 = chemco(met, chem)
        return chem, rho_d, M, H2O, O2, N2

# test photol function
def test_photol(chem):
        
        J = photol(chem)
        
        J_df = pd.DataFrame(J[:, :, 1, 1])
        return chem, J_df

# test deriv function
def test_deriv(chem):
        return chem

# test boxm.py

# Carry out tests
chem, pys, pyc = test_zenith(chem)

chem, rho_d, M, H2O, O2, N2 = test_chemco(met, chem)
# print(met)
# print(chem)
print(rho_d)
print(M)
print(H2O)
print(O2)
print(N2)

#rho_df = pd.DataFrame(rho_d[:, :, 1, 1])
#chem, J_df = test_photol(chem)

#test_deriv(chem)

# Create dfs for better visual display
pys_df = pd.DataFrame(pys[:, :, 5])
pyc_df = pd.DataFrame(pyc[:, :, 5])
sza_df = pd.DataFrame(chem.sza[:, :, 5])

diff = pys_df - sza_df 
diff2 = pyc_df - pys_df
perc_diff = diff / pys_df * 100

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#         display(t_df)
#         display(p_df)
#         display(sza_df)
#         display(rho_df)
#         print(rho_df.mean())
#       display(pys_df)
#       display(pyc_df)
#       display(diff)
#       display(diff2)
#       display(perc_diff)
#       display(J_df)