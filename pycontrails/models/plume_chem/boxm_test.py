import numpy as np
import pandas as pd
import xarray as xr
import datetime
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.emissions import Emissions
from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.core.met_var import (
    AirTemperature,
    SpecificHumidity,
)
from boxm import BoxModel

# Initialise