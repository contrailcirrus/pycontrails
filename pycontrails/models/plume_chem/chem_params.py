"""Default parameters for chem models.

Used by :class:`BoxModel` and :class:`ChemDataset`.
"""
import dataclasses

import numpy as np
import numpy.typing as npt

from pycontrails.core.models import ModelParams


@dataclass
class ChemParams(ModelParams):
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