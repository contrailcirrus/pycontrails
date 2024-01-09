import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

from pycontrails.core import datalib
from pycontrails.core.met import MetDataset
from pycontrails.physics import geo, thermo, units, constants

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

        # # If sza is greater than 90 degrees, set J to zero
        # condition = (np.radians(sza_J) < np.pi / 2)
        # self.data["J"] = (("latitude", "longitude", "time", "photol_params"), np.where(
        #         condition, 
        #         self.L * (np.cos(np.radians(sza_J))**self.M) * np.exp(-self.N / np.cos(np.radians(sza_J))), 0)
        #         )
        
        condition = (sza_J < np.pi / 2)
        self.data["J"] = (("latitude", "longitude", "time", "photol_params"), np.where(
                condition, 
                self.L * (np.cos(sza_J)**self.M) * np.exp(-self.N / np.cos(sza_J)), 0)
                )
        
        self.data["J"] = self.data["J"].expand_dims(dim={'level': self.data["level"]}, axis=2)

    def calc_M_H2O(self, met: MetDataset):

        """Calculate number density of air molecules at each pressure level M"""
        N_A = 6.022e23 # Avogadro's number
        
        # Get air density from pycontrails physics.thermo script
        rho_d = met["air_pressure"].data / (constants.R_d * met["air_temperature"].data)

        # Calculate number density of air (M) to feed into box model calcs
        self.data["M"] = (N_A / constants.M_d) * rho_d * 1e-6  # [molecules / cm^3]
        self.data["M"] = self.data["M"].transpose("latitude", "longitude", "level", "time")
                
        # Use expand_dims to add the new "species" dimension
        #self.data["M"] = self.data["M"].expand_dims(dim={'species': self.data["species"]}, axis=4)


        # Calculate H2O number concentration to feed into box model calcs
        self.data["H2O"] = (met["specific_humidity"].data / constants.M_v) * N_A * rho_d * 1e-6 
        # [molecules / cm^3]

        # Calculate O2 and N2 number concs based on M
        self.data["O2"] = 2.079E-01 * self.data["M"]
        self.data["N2"] = 7.809E-01 * self.data["M"]  

        print(self.data["M"])  
        print(self.data["H2O"])
        print(self.data["O2"])
        print(self.data["N2"])  
        
        


        