"""Utilities for working with ECMWF model-level data."""

import datetime
import pathlib

import dask.array
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pycontrails.physics import units

_path_to_static = pathlib.Path(__file__).parent / "static"
MODEL_LEVELS_PATH = _path_to_static / "model_level_dataframe_v20240418.csv"


def pressure_levels_at_model_levels_constant_surface_pressure(
    alt_ft_min: float | None = None,
    alt_ft_max: float | None = None,
) -> list[int]:
    """Return the pressure levels at each model level assuming a constant surface pressure.

    This function assumes
    `137 model levels <https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions>`_
    and the constant ICAO ISA surface pressure of 1013.25 hPa.

    The returned pressure levels are rounded to the nearest hPa.

    Parameters
    ----------
    alt_ft_min : float | None
        Minimum altitude, [:math:`ft`]. If None, there is no minimum altitude
        used in filtering the ``MODEL_LEVELS_PATH`` table.
    alt_ft_max : float | None
        Maximum altitude, [:math:`ft`]. If None, there is no maximum altitude
        used in filtering the ``MODEL_LEVELS_PATH`` table.

    Returns
    -------
    list[int]
        List of pressure levels, [:math:`hPa`] between the minimum and maximum altitudes.

    See Also
    --------
    pressure_level_at_model_levels
    """
    usecols = ["n", "Geometric Altitude [m]", "pf [hPa]"]
    df = pd.read_csv(MODEL_LEVELS_PATH, usecols=usecols, index_col="n")

    filt = df.index >= 1  # exclude degenerate model level 0
    if alt_ft_min is not None:
        alt_m_min = units.ft_to_m(alt_ft_min)
        filt &= df["Geometric Altitude [m]"] >= alt_m_min
    if alt_ft_max is not None:
        alt_m_max = units.ft_to_m(alt_ft_max)
        filt &= df["Geometric Altitude [m]"] <= alt_m_max

    return df.loc[filt, "pf [hPa]"].round().astype(int).tolist()


def _cache_model_level_dataframe() -> None:
    """Regenerate static model level data file.

    Read the ERA5 L137 model level definitions published by ECMWF
    and cache it in a static file for use by this module.
    This should only be used by model developers, and only if ECMWF model
    level definitions change. ``MODEL_LEVEL_PATH`` must be manually
    updated to use newly-cached files.

    Requires the `lxml <https://lxml.de/>`_ package to be installed.
    """

    url = "https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions"
    df = pd.read_html(url, na_values="-", index_col="n")[0]

    today = datetime.datetime.now()
    new_file_path = _path_to_static / f"model_level_dataframe_v{today.strftime('%Y%m%d')}.csv"
    if new_file_path.is_file():
        msg = f"Static file already exists at {new_file_path}"
        raise ValueError(msg)

    df.to_csv(new_file_path)


def pressure_level_at_model_levels(
    lnsp: xr.DataArray, model_levels: npt.NDArray[np.integer]
) -> xr.DataArray:
    r"""Return the pressure levels at each model level given the surface pressure.

    This function assumes
    `137 model levels <https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions>`_.
    Unlike :func:`pressure_levels_at_model_levels_constant_surface_pressure`, this function
    does not assume constant pressure. Instead, it uses the
    `half-level pressure formula <https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height#heading-Pressureonmodellevels>`_
    :math:`p = a + b \cdot \exp(\ln(\text{sp}))` where :math:`a` and :math:`b` are constants
    for each model level.

    Parameters
    ----------
    lnsp : xr.DataArray
        Natural logarithm of surface pressure, [:math:`\ln(\text{Pa})`].
    model_levels : npt.ArrayLike
        Target model levels. Expected to be a one-dimensional array of integers between 1 and 137.

    Returns
    -------
    xr.DataArray
        Pressure levels at each model level, [:math:`hPa`]. The shape of the output is
        the product of the shape of the input and the length of ``model_levels``. In
        other words, the output will have dimensions of the input plus a new dimension
        for ``model_levels``.

        If ``lnsp`` is not dask-backed, the output will be computed eagerly. In particular,
        if ``lnsp`` has a large size and ``model_levels`` is a large range, this function
        may consume a large amount of memory.

        The ``dtype`` of the output is the same as the ``dtype`` of the ``lnsp`` parameter.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr

    >>> sp = np.linspace(101325.0, 90000.0, 16).reshape(4, 4)
    >>> longitude = np.linspace(-180, 180, 4)
    >>> latitude = np.linspace(-90, 90, 4)
    >>> lnsp = xr.DataArray(np.log(sp), coords={"longitude": longitude, "latitude": latitude})

    >>> model_levels = [80, 100]
    >>> pressure_level_at_model_levels(lnsp, model_levels)
    <xarray.DataArray (model_level: 2, longitude: 4, latitude: 4)> Size: 256B
    array([[[259.75493944, 259.27107504, 258.78721064, 258.30334624],
            [257.81948184, 257.33561744, 256.85175304, 256.36788864],
            [255.88402424, 255.40015984, 254.91629544, 254.43243104],
            [253.94856664, 253.46470224, 252.98083784, 252.49697344]],
           [[589.67975444, 586.47283154, 583.26590864, 580.05898574],
            [576.85206284, 573.64513994, 570.43821704, 567.23129414],
            [564.02437124, 560.81744834, 557.61052544, 554.40360254],
            [551.19667964, 547.98975674, 544.78283384, 541.57591094]]])
    Coordinates:
      * longitude    (longitude) float64 32B -180.0 -60.0 60.0 180.0
      * latitude     (latitude) float64 32B -90.0 -30.0 30.0 90.0
      * model_level  (model_level) int64 16B 80 100

    See Also
    --------
    pressure_levels_at_model_levels_constant_surface_pressure
    """
    model_levels = np.asarray(model_levels, dtype=int)
    if not np.all((model_levels >= 1) & (model_levels <= 137)):
        msg = "model_levels must be integers between 1 and 137"
        raise ValueError(msg)

    usecols = ["n", "a [Pa]", "b"]
    df = (
        pd.read_csv(MODEL_LEVELS_PATH, usecols=usecols)
        .rename(columns={"n": "model_level", "a [Pa]": "a"})
        .set_index("model_level")
    )

    a = df["a"].to_xarray()
    b = df["b"].to_xarray()

    if "model_level" in lnsp.dims:
        lnsp_model_levels = lnsp["model_level"]
        if len(lnsp_model_levels) != 1:
            msg = "Found multiple model levels in lnsp, expected at most one"
            raise ValueError(msg)
        if lnsp_model_levels.item() != 1:
            msg = f"lnsp must be at model level 1, found model level {lnsp_model_levels.item()}"
            raise ValueError(msg)
        # Remove the model level dimension to allow automatic broadcasting below
        lnsp = lnsp.squeeze("model_level")

    sp = dask.array.exp(lnsp)
    dtype = sp.dtype
    a = a.astype(dtype, copy=False)
    b = b.astype(dtype, copy=False)

    indexer = {"model_level": model_levels}
    p_half_below = a.sel(indexer) + b.sel(indexer) * sp

    indexer = {"model_level": model_levels - 1}
    p_half_above = (a.sel(indexer) + b.sel(indexer) * sp).assign_coords(model_level=model_levels)

    p_full = (p_half_above + p_half_below) / 2.0
    return p_full / 100.0
