"""Photochemical trajectory model for the Earth's atmosphere."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload
import numpy as np
import pysolar.solar
import datetime

import xarray as xr

import pycontrails
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataArray, MetDataset, standardize_variables
from pycontrails.core.met_var import (
    AirTemperature,
    RelativeHumidity,
    SpecificHumidity,
    AirPressure
)
from pycontrails.core.models import Model, ModelParams
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.datalib import ecmwf

@dataclass
class BoxModelParams(ModelParams):
    """Default trajectory model parameters."""
    lat_bound: tuple[float, float] | None = None
    lon_bound: tuple[float, float] | None = None
    alt_bound: tuple[float, float] | None = None
    start_date: str = "2021-01-01"
    start_time: str = "00:00:00"
    chem_ts: int = 60 # seconds between chemistry calculations
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
        AirPressure,
        ecmwf.CloudAreaFractionInLayer
    )
    
    # Set default parameters as BoxModelParams
    default_params = BoxModelParams

    # Met, chem data is not optional
    met: MetDataset
    chem: MetDataset
    met_required = True

    timesteps: npt.NDArray[np.datetime64]


    def __init__(
        self,
        met: MetDataset,
        chem: MetDataset,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ):
        
        # Normalize ECMWF variables
        met = standardize_variables(met, self.met_variables)

        super().__init__(met, params=params, **params_kwargs)
        
        self.met = met
        self.chem = chem


    # ----------
    # Public API
    # ----------

    def eval(
        self, source: MetDataset | None = None, **params: Any
    ) -> MetDataset:
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
        self.set_source(source)
        self.source = self.require_source_type(MetDataset)

        self.calc_timesteps()


        
    def deriv(self):
        return
    # This function calculate concentrations and flux rates for each of the chemical species

    # INPUTS: Y(#), RC(#), DJ(#), EM(#), FL(#), chem_ts, 
    # OUTPUTS: Y(#) and FL(#)

    # P/L equations in the format 
    # P = EM(#) + RC(#) * Y(#) + DJ(#) * Y(#)
    # L = EM(#) + RC(#) * Y(#) + DJ(#) * Y(#)

    # Chemical concs for the two boxes
    # Y(#) = P/L for steady state approximation
    # Y(#) = (YP(#) + chem_ts*P) / (1 + chem_ts*L) for non-steady state approximation
    # FL(#) = FL(#) + RC(#)*Y(#)*chem_ts -> flux rates of thermal reactions

    def chemco():
        return
    # This function calculates the thermal rate coeffs for deriv

    # INPUTS: Temp, M, H2O
    # OUTPUTS: RC(#)

    # Simple rate coeffs e.g: KRO2NO  = 2.70D-12*EXP(360/TEMP)
    # Complex rate coeffs e.g: K170 = 5.00D-30*((TEMP/298)**(-1.5))*M
    # List of all reactions e.g: RC(#) = 5.60D-34*O2*N2*((TEMP/300)**(-2.6))

    def photol():
        return
    # Calculate photolysis rate coeffs for deriv
    
    # INPUTS: J(#), BR01
    # OUTPUTS: DJ(#)
    
    # e.g: DJ(46) = J(53)*(1-BR01)

    def zenith(self):
        return
    # Calculate solar zenith angle for photolysis rate coeffs, based on time of year and day
    # INPUTS: COSZEN, TTIME, FYEAR, SECYEAR, ZENNOW, LONGRAD, LATRAD, XYEAR
    # OUTPUTS: J(#), COSZEN

