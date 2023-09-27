import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

from pycontrails.core import datalib
from pycontrails.core.met import MetDataset
from chem import ChemDataset


class CHEM():
        """Class to support getting bg chem data from STOCHEM outputs Khan (2019).
        Adheres to UKMO meteorological data resolution (5deg x 5deg at 9 pressure levels)."""

        name = "init_chem"
        long_name = "Unpopulated photochemical xarray dataset"
        
        def __init__(
                self,
                lon_bounds: tuple[float, float] | None = (-180, 180),
                lat_bounds: tuple[float, float] | None = (-90, 90),
                time: datalib.TimeInput | None = None,
                grid_chem: float = 5.0,
                ts_chem: str = "1H",
                
        ):
                """Initialise chem dataset with default parameters."""
                self.latitude = np.arange(lat_bounds[0], lat_bounds[1], grid_chem)
                self.longitude = np.arange(lon_bounds[0], lon_bounds[1], grid_chem)
                self.chem_pressure_levels = np.array([962, 861, 759, 658, 556, 454, 353, 251, 150.5])
                # np.array([1013, 912, 810, 709, 607, 505, 404, 302, 201, 100])
                self.timesteps = datalib.parse_timesteps(time, freq=ts_chem)
                                
                # PHOTOLYSIS PARAMETERS IN FORMAT J = L*COSX**M*EXP(-N*SECX)
                consts = pd.read_pickle('J.pkl')
                        
                # Extract the constants
                photol_idx, L, M, N = np.array(consts).T
                # self.L = L
                # self.M = M
                # self.N = N
                self.photol_params = photol_idx
                self.photol_coeffs = np.arange(1, 96 + 1) # from Fortran indexing (1 to 96)
                self.therm_coeffs = np.arange(1, 510 + 1) # from Fortran indexing (1 to 510)
                self.species = np.loadtxt('species.txt', dtype=str)

        def open_chemdataset(self) -> MetDataset: 
                """Instantiate chemdataset with zeros and apply species data to it. Interpolation (and zenith calcs etc.), will be done later."""
                
                # Initialise 5 x 5 chem dataset with all variables, ready for species import.
                self._init_chem()     

                # Get species data for timestep 0, pre-interpolated.
                self._get_species()

                self.data = ChemDataset(self.data)

        
        def _init_chem(self) -> xr.Dataset:
                """Initialise chem dataset with zeros."""
                self.data = xr.Dataset(
                        {
                        
                        "local_time": (["latitude", "longitude", "time"],
                                np.zeros((len(self.latitude), len(self.longitude), len(self.timesteps)))),
                        "sza": (["latitude", "longitude", "time"], 
                                np.zeros((len(self.latitude), len(self.longitude), len(self.timesteps)))),
                        "J": (["latitude", "longitude", "time", "photol_params"], 
                                np.zeros((len(self.latitude), len(self.longitude), len(self.timesteps), len(self.photol_params)))),
                        "DJ": (["latitude", "longitude", "level", "time", "photol_coeffs"], 
                                np.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps), len(self.photol_coeffs)))),
                        "RC": (["latitude", "longitude", "level", "time", "therm_coeffs"],
                                np.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps), len(self.therm_coeffs)))),
                        "soa": (["latitude", "longitude", "level", "time"],
                                np.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps)))),
                        "mom": (["latitude", "longitude", "level", "time"],
                                np.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps)))),
                        "Y": (["latitude", "longitude", "level", "time", "species"], 
                                np.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps), len(self.species)))),
                        "EM": (["latitude", "longitude", "level", "time", "species"],
                                np.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps), len(self.species)))),
                        "FL": (["latitude", "longitude", "level", "time", "species"],
                                np.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps), len(self.species)))),

                        },
                        coords={
                        "latitude": self.latitude,
                        "longitude": self.longitude, 
                        "level": self.chem_pressure_levels,
                        "time": self.timesteps, 
                        "photol_params": self.photol_params,
                        "photol_coeffs": self.photol_coeffs,
                        "therm_coeffs": self.therm_coeffs,
                        "species": self.species,
                        }
                ).chunk({"time": 1})

        def _get_species(self):
                """Get species concentrations for initial timestep from species data files. Then interpolate them to chem grid."""
                chem = self.data
                bg_chem = xr.DataArray(
                        {
                                "Y": (["latitude", "longitude", "level", "species"],
                                      np.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.species))))

                        },
                        coords={
                                "latitude": self.latitude,
                                "longitude": self.longitude, 
                                "level": self.chem_pressure_levels,
                                "species": self.species,
                        }
                )
                for s in chem.species.values:
                        # Find month from first timestep
                        month = self.timesteps[0].month
                        for level_idx, l in enumerate(bg_chem.level.values):
                                bg_chem["Y"].loc[:, :, l, s] = np.loadtxt("species/" + s + "_MONTH_" + str(month) + "_LEVEL_" + str(level_idx + 1) + ".csv", delimiter=",")

                chem["Y"] = bg_chem["Y"].isel(time=0).interp(latitude=chem.latitude, longitude=chem.longitude, level=chem.level)
        
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
                
                sza_J = self.data["sza"].expand_dims(dim={'photol_params': self.photol_params}, axis=3)

                print(sza_J)
                print(sza_J.shape)
                # If sza is greater than 90 degrees, set J to zero
                condition = (np.radians(sza_J) < np.pi / 2)
                self.data["J"] = (("latitude", "longitude", "time", "photol_params"), np.where(
                        condition, 
                        self.L * (np.cos(np.radians(sza_J))**self.M) * np.exp(-self.N / np.cos(np.radians(sza_J))), 0)
                        )

        