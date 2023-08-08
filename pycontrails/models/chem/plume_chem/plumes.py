"""Create plume dataset from input flight and meteorology data."""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

import pycontrails
from pycontrails.core import models
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataset
from pycontrails.core.vector import GeoVectorDataset, VectorDataset
from pycontrails.models import humidity_scaling, sac
from pycontrails.models.cocip import cocip, contrail_properties, wake_vortex, wind_shear
from pycontrails.models.emissions import black_carbon, emissions
from pycontrails.physics import geo, thermo, units
from pycontrails.core.met_var import (
    AirTemperature,
    EastwardWind,
    Geopotential,
    NorthwardWind,
    VerticalVelocity,
    RelativeHumidity,
    SpecificHumidity,
    AirPressure
)
from pycontrails.core.models import Model, ModelParams
from pycontrails.datalib import ecmwf

class Plumes(MetDataset):
    """Compute emissions concentrations in eulerian grid format, based on input flight emissions data (emissions.py)."""
    name = "plumes"
    long_name = "Plume dispersion model"
    met_variables = (
        AirTemperature,
        EastwardWind,
        Geopotential,
        NorthwardWind,
        VerticalVelocity,
        SpecificHumidity,
        RelativeHumidity,
        AirPressure,
        ecmwf.CloudAreaFractionInLayer,
    )   

    
