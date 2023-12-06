import pandas as pd
import numpy as np
import xarray as xr


def run_boxm(met, chem):
    """Function to generate input files corresponding to individual grid cells, run instances of original box model on each grid cell, and then merg outputs into single .nc file."""

    