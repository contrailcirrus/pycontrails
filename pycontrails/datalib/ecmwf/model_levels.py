"""Utilities for working with ECMWF model-level data.

This module requires the following additional dependency:

- `lxml <https://lxml.de/>`_
"""

import pathlib

import pandas as pd

from pycontrails.physics import units
from pycontrails.utils import dependencies

_path_to_static = pathlib.Path(__file__).parent / "static"
MODEL_LEVELS_PATH = _path_to_static / "model_level_dataframe_v20240418.csv"


def pressure_levels_at_model_levels(alt_ft_min: float, alt_ft_max: float) -> list[int]:
    """Return the pressure levels at each model level assuming a constant surface pressure.

    The pressure levels are rounded to the nearest hPa.

    Parameters
    ----------
    alt_ft_min : float
        Minimum altitude, [:math:`ft`].
    alt_ft_max : float
        Maximum altitude, [:math:`ft`].

    Returns
    -------
    list[int]
        List of pressure levels, [:math:`hPa`].
    """
    df = pd.read_csv(MODEL_LEVELS_PATH)
    alt_m_min = units.ft_to_m(alt_ft_min)
    alt_m_max = units.ft_to_m(alt_ft_max)
    filt = df["Geometric Altitude [m]"].between(alt_m_min, alt_m_max)
    return df.loc[filt, "pf [hPa]"].round().astype(int).tolist()


def _cache_model_level_dataframe() -> pd.DataFrame:
    """Regenerate static model level data file.

    Read the ERA5 model level definitions published by ECMWF
    and cache it in a static file for use by this module.
    This should only be used by model developers, and only if ECMWF model
    level definitions change. ``MODEL_LEVEL_PATH`` must be manually
    updated to use newly-cached files.

    Requires the lxml package to be installed.
    """
    import os
    from datetime import datetime

    url = "https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions"
    try:
        df = pd.read_html(url, na_values="-", index_col="n")[0]
        today = datetime.now()
        new_file_path = _path_to_static / f"model_level_dataframe_v{today.strftime('%Y%m%d')}.csv"
        if os.path.exists(new_file_path):
            msg = f"Static file already exists at {new_file_path}"
            raise ValueError(msg)
        df.to_csv(new_file_path)

    except ImportError as exc:
        if "lxml" in exc.msg:
            dependencies.raise_module_not_found_error(
                "model_level_utils._read_model_level_dataframe function",
                package_name="lxml",
                module_not_found_error=exc,
                extra=(
                    "Alternatively, if instantiating a model-level ECMWF datalib, you can provide "
                    "the 'pressure_levels' parameter directly to avoid the need to read the "
                    "ECMWF model level definitions."
                ),
            )
        raise
