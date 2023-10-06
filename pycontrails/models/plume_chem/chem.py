import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

from pycontrails.core import datalib
from pycontrails.core.met import MetDataset

class ChemDataset(MetDataset):
    """Instantiated chemistry dataset to store a range of chemical parameters, that will feed into the box model (i.e. pre-integration so species, zenith and photol)."""

    name = "chem"
    long_name = "Semi-populated photochemical xarray dataset"

    def __init__(
            self,
            data: xr.Dataset,
            L: float = None,
            M: float = None, 
            N: float = None,
            lon_bounds: tuple[float, float] | None = (-180, 180),
            lat_bounds: tuple[float, float] | None = (-90, 90),
            time: datalib.TimeInput | None = None,
            ts_chem: str = "1H",
    ):
        super().__init__(data)
        self.L = L
        self.M = M
        self.N = N

    def zenith(self):
        """Calculate zenith angle for each grid cell and timestep"""

        chem = self.data

        # For each timestep, calculate zenith for each grid cell in a vectorised way
        _, lon_mesh = xr.broadcast(chem.local_time, chem.longitude)
        _, lat_mesh = xr.broadcast(chem.local_time, chem.latitude)
        _, time_mesh = xr.broadcast(chem.local_time, chem.time)

        # Calculate the offset from UTC based on the longitude
        offsets = (lon_mesh / 15)  # Each hour corresponds to 15 degrees of longitude
        offsets = offsets * 3600  # Convert to seconds
        delta = offsets.astype('timedelta64[s]')

        chem["local_time"] = time_mesh + delta
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

        self.data["sza"] = np.arccos(np.cos(np.radians(lat_mesh))*np.cos(d)*np.cos(lha) + np.sin(np.radians(lat_mesh))*np.sin(d))

    def get_photol_params(self):
                
        sza_J = self.data["sza"].expand_dims(dim={'photol_params': self.data["photol_params"]}, axis=3)

        print(sza_J)
        print(sza_J.shape)
        # If sza is greater than 90 degrees, set J to zero
        condition = (np.radians(sza_J) < np.pi / 2)
        self.data["J"] = (("latitude", "longitude", "time", "photol_params"), np.where(
                condition, 
                self.L * (np.cos(np.radians(sza_J))**self.M) * np.exp(-self.N / np.cos(np.radians(sza_J))), 0)
        )


        