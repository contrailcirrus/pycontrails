"""Photochemical trajectory model for the Earth's atmosphere."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload
import numpy as np
import datetime
import boxm_for

import xarray as xr

import pycontrails
from pycontrails.core import datalib
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataArray, MetDataset, standardize_variables
from chem import ChemDataset
from pycontrails.core.met_var import (
    AirTemperature,
    RelativeHumidity,
    SpecificHumidity,
    AirPressure
)
from pycontrails.core.models import Model, ModelParams
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.datalib import ecmwf
from pycontrails.physics import geo, thermo, units, constants
   
@dataclass
class ChemParams(ModelParams):
    """Default trajectory model parameters."""
    lat_bounds: tuple[float, float] | None = None
    lon_bounds: tuple[float, float] | None = None
    alt_bounds: tuple[float, float] | None = None
    start_date: str = "2021-01-01"
    time: tuple[str, str] = ("2021-01-01 00:00:00", "2021-01-02 00:00:00")
    ts_chem: int = 60 # seconds between chemistry calculations
    disp_ts: int = 300 # seconds between dispersion calculations
    runtime: int = 24 # hours model runtime
    horiz_res: float = 0.25 # degrees
    bgoam: float = 0.7 # background organic aerosol mass
    microgna: float = 0.0 # microgram of nitrate aerosol
    microgsa: float = 0.0 # microgram of sulfate aerosol


class BoxModel(Model):
    """Compute chemical concentrations along a trajectory."""
    name = "boxm"
    long_name = "Photochemical Trajectory Model"
    met_variables = (
        AirTemperature,
        SpecificHumidity,
        RelativeHumidity,
        AirPressure
    )
    default_params = ChemParams

    # Met, chem data is not optional
    met: MetDataset
    chem: ChemDataset
    met_required = True

    timesteps: npt.NDArray[np.datetime64]


    def __init__(
        self,
        met: MetDataset,
        chem: MetDataset,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ):
        super().__init__(met, params=params, **params_kwargs)
        
        self.met = met


        self.chem = chem


        #self.met = self.require_source_type(MetDataset)
        #self.chem = self.require_source_type(ChemDataset)

        self.timesteps = self.chem.data["time"].values

    # ----------
    # Public API
    # ----------

    def eval(
        self, source: ChemDataset | None = None, **params: Any
    ) -> ChemDataset:
        """Evaluate chem on meteorology grid, subject to flight emissions.
        
        Parameters
        ----------
        source : MetDataset | None, optional
            Input MetDataset
            If None, evaluates at the :attr:`met` grid points.
        **params : Any
            Overwrite model parameters before eval

        Returns
        -------
        MetDataset
            Returns `np.nan` if interpolating outside meteorology grid.

        Raises
        ------
        NotImplementedError
            Raises if input ``source`` is not supported.       
        """

        self.update_params(params)
        self.source = source
       
        # Interpolate emi to chem grid
        # self.source.data = self.source.data.interp_like(self.chem.data)

        # Assign emissions dataarray to chem dataset
        # self.chem["EM"].data = self.source["EM"].data

        # Export all variables to fortran
        self._f2py_export()
        
        # # Calculate chemical concentrations and flux rates for each timestep (requires iteration)
        for time_idx, ts in enumerate(self.timesteps):
           
            print(time_idx, ts)
        
            # Calculate mass of organic particulate material and mass of organic aerosol
            self.calc_aerosol(time_idx)

            self._chemco(time_idx)

            self._photol(time_idx)

            if time_idx != 0:
                self._deriv(time_idx)

        self._f2py_import()

    # -------------
    # Model Methods
    # -------------

    def _f2py_export(self):
        met = self.met
        chem = self.chem

        # Clear all variables prior to allocation
        boxm_for.boxm.temp = None
        boxm_for.boxm.pressure = None
        boxm_for.boxm.spec_hum = None
        boxm_for.boxm.m = None
        boxm_for.boxm.h2o = None
        boxm_for.boxm.o2 = None
        boxm_for.boxm.n2 = None

        boxm_for.boxm.y = None
        boxm_for.boxm.rc = None
        boxm_for.boxm.dj = None
        boxm_for.boxm.em = None
        boxm_for.boxm.fl = None

        boxm_for.boxm.j = None
        boxm_for.boxm.soa = None
        boxm_for.boxm.mom = None
        boxm_for.boxm.br01 = None
        boxm_for.boxm.ro2 = None

        # Position and time
        boxm_for.boxm.lat = chem.data.sizes["latitude"]
        boxm_for.boxm.lon = chem.data.sizes["longitude"]
        boxm_for.boxm.alt = chem.data.sizes["level"]
        boxm_for.boxm.dts = 60 # FIX THIS
        # Met variables

        print(met["air_temperature"].data.values)
        boxm_for.boxm.temp = met["air_temperature"].data.values
        boxm_for.boxm.pressure = met["air_pressure"].data.values
        #boxm_for.boxm.spec_hum = met["specific_humidity"].data.values
        print(chem["M"].data.values)
        boxm_for.boxm.m = chem["M"].data.values
        boxm_for.boxm.h2o = chem["H2O"].data.values
        print(chem["O2"].data.values)
        boxm_for.boxm.o2 = chem["O2"].data.values
        print(chem["N2"].data.values)
        boxm_for.boxm.n2 = chem["N2"].data.values
        print("met done")

        # Chem variables
        boxm_for.boxm.y = chem["Y"].data.values * chem["M"].data.expand_dims(dim={'species': chem.data["Y"].species}, axis=4).values / 1E+09
        print("chem y done")
        boxm_for.boxm.rc = chem["RC"].data.values
        print("chem rc done")
        boxm_for.boxm.dj = chem["DJ"].data.values
        print("chem dj done")
        boxm_for.boxm.em = chem["EM"].data.values
        print("chem em done")
        boxm_for.boxm.fl = chem["FL"].data.values
        print("chem fl done")
        boxm_for.boxm.j = chem["J"].data.values
        print("chem j done")
        boxm_for.boxm.soa = chem["soa"].data.values
        boxm_for.boxm.mom = chem["mom"].data.values
        boxm_for.boxm.br01 = chem["BR01"].data.values
        boxm_for.boxm.ro2 = chem["RO2"].data.values

        #print(boxm_for.boxm.__doc__)

    def _deriv(self, time_idx):
        """Calculate the derivatives of species concentrations and flux rates at each timestep."""
        boxm_for.boxm.deriv(int(time_idx+1))
   
    def _chemco(self, time_idx):
        """Calculate the thermal rate coefficients and reaction rates for each timestep."""
        boxm_for.boxm.chemco(int(time_idx+1))

    def _photol(self, time_idx):
        """Calculate photolysis rates for each grid cell and timestep"""
        boxm_for.boxm.photol(int(time_idx+1))
      
    def calc_aerosol(self, time_idx):
        """Calculate aerosol masses for each grid cell and timestep"""
        boxm_for.boxm.calc_aerosol(int(time_idx+1))
        
    def _f2py_import(self):
        chem = self.chem

        # Chem variables
        chem["Y"].data.values = boxm_for.boxm.y * 1E+09 / chem["M"].data.expand_dims(dim={'species': chem.data["Y"].species}, axis=4).values
        chem["RC"].data.values = boxm_for.boxm.rc
        chem["DJ"].data.values = boxm_for.boxm.dj
        chem["EM"].data.values = boxm_for.boxm.em
        #chem["FL"].data.values = boxm_for.boxm.fl
        print("chem 5d done")
        chem["J"].data.values = boxm_for.boxm.j
        chem["soa"].data.values = boxm_for.boxm.soa
        chem["mom"].data.values = boxm_for.boxm.mom
        chem["BR01"].data.values = boxm_for.boxm.br01
        chem["RO2"].data.values = boxm_for.boxm.ro2

        # # Deallocate all variables once done to clear memory in Fortran
        # boxm_for.boxm.temp = None
        # boxm_for.boxm.pressure = None
        # boxm_for.boxm.spec_hum = None
        # boxm_for.boxm.m = None
        # boxm_for.boxm.h2o = None
        # boxm_for.boxm.o2 = None
        # boxm_for.boxm.n2 = None

        # boxm_for.boxm.y = None
        # #boxm_for.boxm.yp = None
        # boxm_for.boxm.rc = None
        # boxm_for.boxm.dj = None
        # boxm_for.boxm.em = None
        # boxm_for.boxm.fl = None

        # boxm_for.boxm.j = None
        # boxm_for.boxm.soa = None
        # boxm_for.boxm.mom = None
        # boxm_for.boxm.br01 = None
        # boxm_for.boxm.ro2 = None


