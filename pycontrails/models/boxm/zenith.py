# This script calculates zenith angle. It is the most fundamental of all the functions, 
# as it purely relies on geographical position, time of day/year and year

# See Jacobson et al. (2005) Fundamentals of Atmospheric Modelling
# for more details on the calculation of zenith angle

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pycontrails import MetDataset, MetDataArray, MetVariable
from pycontrails.physics import geo, thermo, units

def zenith(chem: MetDataset):    

    # # For each timestep, calculate zenith for each grid cell in a vectorised way
    _, lon_mesh = xr.broadcast(chem.local_time, chem.longitude)
    _, lat_mesh = xr.broadcast(chem.local_time, chem.latitude)
    _, gm_time_mesh = xr.broadcast(chem.local_time, chem.gm_time)

   
    # Calculate the offset from UTC based on the longitude
    offsets = (lon_mesh / 15)  # Each hour corresponds to 15 degrees of longitude
    offsets = offsets * 3600  # Convert to seconds
    delta = offsets.astype('timedelta64[s]')

    chem["local_time"] = gm_time_mesh + delta
    #local_time, lat_mesh = xr.broadcast(local_time, chem.latitude)
    

    # Calculate number of leap days since or before the year 2000
    if chem["local_time.year"].values.all() >= 2001:
        DL = (chem["local_time.year"].values - 2001) / 4
    else:
        DL = (chem["local_time.year"].values - 2000) / 4 - 1
    
    # Calculate Julian day of year (number of days since 1st Jan)
    DJ = chem["local_time.dayofyear"].values
   
    # # Calculate number of days since 1st Jan 2000
    NJD = 364.5 + (chem["local_time.year"].values - 2001) * 365.25 + DL + DJ     
   
    # Calculate obliquity of the ecliptic
    eps_ob = np.radians(23.439 - 0.0000004 * NJD)
    
    # Calculate mean longitude of the Sun  
    LM = np.radians(280.460 + 0.9856474 * NJD)
    
    # Calculate mean anomaly of the Sun
    GM = np.radians(357.528 + 0.9856003 * NJD)
    
    # Calculate ecliptic longitude of the Sun
    LE = LM + np.radians(1.915 * np.sin(GM) + 0.020 * np.sin(2 * GM))
    
    # Calculate solar declination angle
    d = np.arcsin(np.sin(eps_ob) * np.sin(LE))
    
    # Calculate local hour angle        
    secday = chem["local_time.hour"].values * 3600 \
        + chem["local_time.minute"].values * 60 \
        + chem["local_time.second"].values
    
    lha = (2*np.pi * (secday/86400 - 0.5))

    sza = np.arccos(np.cos(np.radians(lat_mesh))*np.cos(d)*np.cos(lha) + np.sin(np.radians(lat_mesh))*np.sin(d))
    
    return sza


