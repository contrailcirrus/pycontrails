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
                timestep_freq: str = "1H",
                
        ):
                """Initialise chem dataset with default parameters."""
                self.latitude = np.arange(lat_bounds[0], lat_bounds[1], grid_chem)
                self.longitude = np.arange(lon_bounds[0], lon_bounds[1], grid_chem)
                self.chem_pressure_levels = np.array([962, 861, 759, 658, 556, 454, 353, 251, 150.5])
                # np.array([1013, 912, 810, 709, 607, 505, 404, 302, 201, 100])
                self.timesteps = datalib.parse_timesteps(time, freq=timestep_freq)
                                
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

        def open_chemdataset(self) -> ChemDataset: 
                """Instantiate chemdataset with zeros and apply species data to it. Interpolation (and zenith calcs etc.), will be done later."""
                
                # Initialise 5 x 5 chem dataset with all variables, ready for species import.
                ds = self._init_chem()     
                print(ds)
                # Get species data for timestep 0, pre-interpolated.
                self._get_species(ds)

                return ChemDataset(ds)

        
        def _init_chem(self) -> xr.Dataset:
                """Initialise chem dataset with zeros."""
                return xr.Dataset(
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

        def _get_species(self, ds: xr.Dataset):
                """Get species concentrations for initial timestep from species data files. Then interpolate them to chem grid."""
                chem = ds
                # bg_chem = xr.DataArray(
                #         {
                #                 "Y": (["latitude", "longitude", "level", "species"],
                #                       np.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.species))))

                #         },
                #         coords={
                #         "latitude": self.latitude,
                #         "longitude": self.longitude, 
                #         "level": self.chem_pressure_levels,
                #         "species": self.species,
                #         }
                # )
                for s in chem.species.values:
                        # Find month from first timestep
                        month = self.timesteps[0].month
                        
                        for level_idx, l in enumerate(chem.level.values):
                                chem["Y"].loc[:, :, l, self.timesteps[0], s] = np.loadtxt("species/" + s + "_MONTH_" + str(month) + "_LEVEL_" + str(level_idx + 1) + ".csv", delimiter=",")

                
        