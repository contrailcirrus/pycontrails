import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import glob
import re
from datetime import datetime

from pycontrails.physics import geo, thermo, units, constants
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
                # met: MetDataset = None
                
        ):
                """Initialise chem dataset with default parameters."""
                self.latitude = np.arange(lat_bounds[0] + 2.5, lat_bounds[1], grid_chem)
                self.longitude = np.arange(lon_bounds[0] + 2.5, lon_bounds[1], grid_chem)
                self.chem_pressure_levels = np.array([962, 861, 759, 658, 556, 454, 353, 251, 150.5])
                # np.array([1013, 912, 810, 709, 607, 505, 404, 302, 201, 100])
                self.timesteps = datalib.parse_timesteps(time, freq=timestep_freq)
                                
                # PHOTOLYSIS PARAMETERS IN FORMAT J = L*COSX**M*EXP(-N*SECX)
                consts = pd.read_pickle('J.pkl')
                        
                # Extract the constants
                photol_idx, L, M, N = np.array(consts).T
                self.L = L
                self.M = M
                self.N = N
                self.photol_params = photol_idx
                self.photol_coeffs = np.arange(1, 96 + 1) # from Fortran indexing (1 to 96)
                self.therm_coeffs = np.arange(1, 510 + 1) # from Fortran indexing (1 to 510)
                self.species = np.loadtxt('species_num.txt', dtype=str)

        def open_chemdataset(self, met: MetDataset) -> ChemDataset: 
                """Instantiate chemdataset with zeros and apply species data to it. Interpolation (and zenith calcs etc.), will be done later."""
                
                # Initialise 5 x 5 chem dataset with all variables, ready for species import.
                ds = self._init_chem()     

                # Get species data for timestep 0, pre-interpolated.
                self._get_species(ds)

                return ChemDataset(ds, self.L, self.M, self.N)

        def _init_chem(self) -> xr.Dataset:
                """Initialise chem dataset with zeros."""
                return xr.Dataset(
                        {
                        "local_time": (["latitude", "longitude", "time"],
                                da.zeros((len(self.latitude), len(self.longitude), len(self.timesteps)))),
                        "sza": (["latitude", "longitude", "time"], 
                                da.zeros((len(self.latitude), len(self.longitude), len(self.timesteps)))),
                        "M": (["latitude", "longitude", "level", "time"],
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps)))),
                        "H2O": (["latitude", "longitude", "level", "time"],
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps)))),
                        "O2": (["latitude", "longitude", "level", "time"],
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps)))),
                        "N2": (["latitude", "longitude", "level", "time"],
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps)))),
                        "J": (["latitude", "longitude", "time", "photol_params"], 
                                da.zeros((len(self.latitude), len(self.longitude), len(self.timesteps), len(self.photol_params)))),
                        "DJ": (["latitude", "longitude", "level", "time", "photol_coeffs"], 
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps), len(self.photol_coeffs)))),
                        "RC": (["latitude", "longitude", "level", "time", "therm_coeffs"],
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps), len(self.therm_coeffs)))),
                        "soa": (["latitude", "longitude", "level", "time"],
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps)))),
                        "mom": (["latitude", "longitude", "level", "time"],
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps)))),
                        "BR01": (["latitude", "longitude", "level", "time"],
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps)))),
                        "RO2": (["latitude", "longitude", "level", "time"],
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps)))),
                        "Y": (["latitude", "longitude", "level", "time", "species"], 
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps), len(self.species)))),
                        "EM": (["latitude", "longitude", "level", "time", "species"],
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps), len(self.species)))),
                        "FL": (["latitude", "longitude", "level", "time", "species"],
                                da.zeros((len(self.latitude), len(self.longitude), len(self.chem_pressure_levels), len(self.timesteps), len(self.species)))),

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
                bg_chem = xr.open_dataarray("species.nc")

                month = self.timesteps[0].month
                print(month)

                # get list of all files in species dir
                species_txt = open("species.txt", "r").read()
                species_tolist = species_txt.replace('\n', ' ').split(".")
                species_string = ''.join(species_tolist)
                print(species_string)
                
                for species in self.species:
                        if re.search(species + " ", species_string):
                                
                                chem["Y"].loc[:, :, :, self.timesteps[0], species] = bg_chem.sel(month=month-1, species=species) * 1E+09

                        else:
                                print("No files found for " + species)

                
        