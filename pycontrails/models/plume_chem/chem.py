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
            lon_bounds: tuple[float, float] | None = (-180, 180),
            lat_bounds: tuple[float, float] | None = (-90, 90),
            time: datalib.TimeInput | None = None,
            ts_chem: str = "1H",
    ):
        super().__init__(data)
        