"""Utilities for working with ECMWF model-level data.

This module requires the following additional dependency:

- `lxml <https://lxml.de/>`_
"""

import functools

import pandas as pd

from pycontrails.physics import units
from pycontrails.utils import dependencies


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
    df = _read_model_level_dataframe()
    alt_m_min = units.ft_to_m(alt_ft_min)
    alt_m_max = units.ft_to_m(alt_ft_max)
    filt = df["Geometric Altitude [m]"].between(alt_m_min, alt_m_max)
    return df.loc[filt, "pf [hPa]"].round().astype(int).tolist()


@functools.cache
def _read_model_level_dataframe() -> pd.DataFrame:
    """Read the ERA5 model level definitions published by ECMWF.

    This requires the lxml package to be installed.
    """
    url = "https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions"
    try:
        return pd.read_html(url, na_values="-", index_col="n")[0]
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
