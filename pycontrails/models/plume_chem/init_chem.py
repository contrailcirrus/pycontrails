import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from overrides import overrides

import pycontrails
from pycontrails.core import datalib
from pycontrails.core import interpolation
from pycontrails.core.met import MetBase, MetDataArray, MetDataset


class ChemDataset(MetBase):
        """Instantiate background chemistry dataset based on user input of position and time.
        Adheres to UKMO chemical composition data resolution (5deg x 5deg at 9 pressure levels)."""
        name = "chem"
        long_name = "Photochemical xarray dataset"
        # Set default parameters as BoxModelParams
        #default_params = ChemParams

        def __init__(
                self,
                met: MetDataset,
                lon_bounds: tuple[float, float] | None = None,
                lat_bounds: tuple[float, float] | None = None,
                alt_bounds: tuple[float, float] | None = None,
                grid_chem: float = 5.0,
                pressure_levels: datalib.PressureLevelInput = -1,
                time: datalib.TimeInput | None = None,
                ts_chem: str = "1H",
                
        ) -> xr.Dataset:
                self.latitude = np.arange(lat_bounds[0], lat_bounds[1], grid_chem)
                self.longitude = np.arange(lon_bounds[0], lon_bounds[1], grid_chem)
                self.chem_pressure_levels = np.array([962, 861, 759, 658, 556, 454, 353, 251, 150.5])# np.array([1013, 912, 810, 709, 607, 505, 404, 302, 201, 100])
                self.timesteps = datalib.parse_timesteps(time, freq=ts_chem)
                                
                # PHOTOLYSIS PARAMETERS IN FORMAT J = L*COSX**M*EXP(-N*SECX)
                consts = pd.read_pickle('J.pkl')
                        
                # Extract the constants
                photol_idx, L, M, N = np.array(consts).T

                self.photol_params = photol_idx
                self.photol_coeffs = np.arange(1, 96 + 1) # from Fortran indexing (1 to 96)
                self.therm_coeffs = np.arange(1, 510 + 1) # from Fortran indexing (1 to 510)
                self.species = np.loadtxt('species.txt', dtype=str)

        def open_chemdataset(self):
                """Populate chemdataset with all values pre integration."""
                self._init_chem()     

                # Calculate local time and zenith angle
                self._zenith()

                # # Calculate photolysis rate constants
                # self._photolysis()

                # # Calculate photolysis frequency
                # self._photol_freq()

                # # Calculate photolysis frequency
                # self._therm_freq()

                # # Calculate photolysis frequency
                # self._therm_rate()

                # # Calculate photolysis frequency
                # self._soa()

                # # Calculate photolysis frequency
                # self._mom()

                # # Calculate photolysis frequency
                # self._Y()

                # # Calculate photolysis frequency
                # self._EM()

                # # Calculate photolysis frequency
                # self._FL()    


        # initialise example chem MetDataset
        
        def _init_chem(self) -> xr.Dataset:
                
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


        def _zenith(self):
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

        def _get_species(self):
                """Get species from species data files"""
                chem = self.data
                for s in chem.species:
                        for t in chem.local_time:
                                for l in chem.level:
                                        for lat in chem.latitude:
                                                for lon in chem.longitude:
                                                        chem["Y"].loc[lat, lon, l, t, s] = np.loadtxt(f"Y_{s}.txt")
                        chem["Y"]
                return species
        
        
        @property
        @overrides
        def shape(self) -> tuple[int, int, int, int]:
                sizes = self.data.sizes
                return sizes["longitude"], sizes["latitude"], sizes["level"], sizes["time"]

        @property
        @overrides
        def size(self) -> int:
                return np.prod(self.shape).item()
        
        @overrides
        def broadcast_coords(self, name: str) -> xr.DataArray:
                da = xr.ones_like(self.data[list(self.data.keys())[0]]) * self.data[name]
                da.name = name

                return da

        @overrides
        def downselect(self, bbox: tuple[float, ...]) -> MetDataset:
                data = downselect(self.data, bbox)
                return MetDataset(data, cachestore=self.cachestore, copy=False)
                
        # def zenith(self):
        #         """Calculate zenith angle for each grid cell and timestep"""

        # chem = self.chem

        # # For each timestep, calculate zenith for each grid cell in a vectorised way
        # _, lon_mesh = xr.broadcast(chem.data.local_time, chem.data.longitude)
        # _, lat_mesh = xr.broadcast(chem.data.local_time, chem.data.latitude)
        # _, time_mesh = xr.broadcast(chem.data.local_time, chem.data.time)

        # # Calculate the offset from UTC based on the longitude
        # offsets = (lon_mesh / 15)  # Each hour corresponds to 15 degrees of longitude
        # offsets = offsets * 3600  # Convert to seconds
        # delta = offsets.astype('timedelta64[s]')

        # chem["local_time"] = time_mesh + delta
        # #local_time, lat_mesh = xr.broadcast(local_time, chem.latitude)
        

        # # Calculate number of leap days since or before the year 2000
        # if chem["local_time.year"].values.all() >= 2001:
        #     DL = (chem["local_time.year"].values - 2001) / 4
        # else:
        #     DL = (chem["local_time.year"].values - 2000) / 4 - 1
        
        # # Calculate Julian day of year (number of days since 1st Jan)
        # DJ = chem["local_time.dayofyear"].values
    
        # # # Calculate number of days since 1st Jan 2000
        # NJD = 364.5 + (chem["local_time.year"].values - 2001) * 365.25 + DL + DJ     
    
        # # Calculate obliquity of the ecliptic
        # eps_ob = np.radians(23.439 - 0.0000004 * NJD)
        
        # # Calculate mean longitude of the Sun  
        # LM = np.radians(280.460 + 0.9856474 * NJD)
        
        # # Calculate mean anomaly of the Sun
        # GM = np.radians(357.528 + 0.9856003 * NJD)
        
        # # Calculate ecliptic longitude of the Sun
        # LE = LM + np.radians(1.915 * np.sin(GM) + 0.020 * np.sin(2 * GM))
        
        # # Calculate solar declination angle
        # d = np.arcsin(np.sin(eps_ob) * np.sin(LE))
        
        # # Calculate local hour angle        
        # secday = chem["local_time.hour"].values * 3600 \
        #     + chem["local_time.minute"].values * 60 \
        #     + chem["local_time.second"].values
        
        # lha = (2*np.pi * (secday/86400 - 0.5))

        # self.chem["sza"] = np.arccos(np.cos(np.radians(lat_mesh))*np.cos(d)*np.cos(lha) + np.sin(np.radians(lat_mesh))*np.sin(d))
