"""Cocip Grid Outputs."""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.models.cocip import contrail_properties
from pycontrails.physics import units
from pycontrails.physics.geo import spatial_bounding_box

np.random.seed(1)





# ---
# natural_cirrus.py
# ---


def get_max_cirrus_cover_and_tau(
    cloud_cover: xr.DataArray,
    tau_cirrus: xr.DataArray,
    *,
    boost_res: bool = False,
    grid_res: float = 0.05,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate the maximum natural cirrus coverage and optical depth.

    Returns these properties in the xr.DataArray format.
    """
    cc_max = _get_2d_cirrus_cover(cloud_cover)
    tau_cirrus_max = tau_cirrus.sel(
        level=tau_cirrus["level"][-1]
    )  # Get maximum values at lowest layer

    if boost_res:
        cc_max, tau_cirrus_max = _boost_cirrus_cover_and_tau_resolution(
            cc_max, tau_cirrus_max, grid_res=grid_res
        )

    return cc_max, tau_cirrus_max


def _get_2d_cirrus_cover(cloud_cover: xr.DataArray) -> xr.DataArray:
    """Calculate 2D cirrus cover effective for observers from above.

    cc_max(x,y,t) = max[cc(x,y,z,t)].
    """
    cc_max_t = [
        _get_2d_cirrus_cover_time_slice(cloud_cover, tt) for tt in range(len(cloud_cover["time"]))
    ]
    cc_max = xr.concat(cc_max_t, dim="time")
    dim_order: list = ["longitude", "latitude", "time"]
    return cc_max.transpose(*dim_order)


def _get_2d_cirrus_cover_time_slice(cloud_cover: xr.DataArray, time_int: int) -> xr.DataArray:
    cirrus_cover_t = cloud_cover.sel(time=cloud_cover["time"][time_int])
    return cirrus_cover_t.max(dim="level")


def _boost_cirrus_cover_and_tau_resolution(
    cc_max: xr.DataArray, tau_cirrus_max: xr.DataArray, *, grid_res: float = 0.05
) -> tuple[xr.DataArray, xr.DataArray]:
    """Increase resolution of cirrus cover and optical depth via approximation method by Schumann.

    This is necessary because the existing spatial resolution is too coarse to resolve the
    contributions from relatively narrow contrails.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    time = cc_max.time.values
    lon_hi_res, lat_hi_res = _boost_spatial_resolution(
        cc_max.longitude.values, cc_max.latitude.values, grid_res=grid_res
    )
    n_rep = int(len(lon_hi_res) / len(cc_max.longitude.values))
    res = [
        _boost_cc_and_tau_res_t(
            cc_max.sel(time=t).values, tau_cirrus_max.sel(time=t).values, n_rep=n_rep
        )
        for t in time
    ]

    # Convert results to xr.DataArray
    dim_order: list = ["longitude", "latitude", "time"]
    cc_max_hi_res = np.array([res[i][0] for i in range(len(res))])
    cc_max_hi_res_xr = xr.DataArray(
        cc_max_hi_res,
        dims=["time", "longitude", "latitude"],
        coords={"time": time, "longitude": lon_hi_res, "latitude": lat_hi_res},
    )
    cc_max_hi_res_xr = cc_max_hi_res_xr.transpose(*dim_order)

    tauc_max_hi_res = np.array([res[i][1] for i in range(len(res))])
    tauc_max_hi_res_xr = xr.DataArray(
        tauc_max_hi_res,
        dims=["time", "longitude", "latitude"],
        coords={"time": time, "longitude": lon_hi_res, "latitude": lat_hi_res},
    )
    tauc_max_hi_res_xr = tauc_max_hi_res_xr.transpose(*dim_order)
    return cc_max_hi_res_xr, tauc_max_hi_res_xr


def _boost_spatial_resolution(
    lon_deg: np.ndarray, lat_deg: np.ndarray, *, grid_res: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Increase the longitude and latitude grid resolution."""
    d_lon = np.abs(lon_deg[1] - lon_deg[0])
    d_lat = np.abs(lat_deg[1] - lat_deg[0])
    is_whole_number = ((d_lon / grid_res) - int(d_lon / grid_res)) == 0

    if d_lon != d_lat:
        raise RuntimeWarning(
            "Note: Resolution of existing longitude and latitude inputs is not equal. "
        )

    if (d_lon <= grid_res) & (d_lat <= grid_res):
        raise ArithmeticError(
            "Error: Resolution of existing longitude or latitude is already higher than"
            ' "grid_res". '
        )

    if ~is_whole_number:
        raise ArithmeticError(
            "Select a grid resolution that provides a whole number to the new number of pixels. "
        )

    n_pixel_per_box = int(d_lon / grid_res)
    edges = grid_res * 0.5 * (n_pixel_per_box - 1)
    lon_boosted = np.linspace(
        min(lon_deg) - edges, max(lon_deg) + edges, (len(lon_deg) * n_pixel_per_box)
    )
    lat_boosted = np.linspace(
        min(lat_deg) - edges, max(lat_deg) + edges, (len(lat_deg) * n_pixel_per_box)
    )

    is_float_residual = np.abs(lon_boosted) < 1e-8
    lon_boosted[is_float_residual] = 0.0

    is_float_residual = np.abs(lat_boosted) < 1e-8
    lat_boosted[is_float_residual] = 0.0
    return lon_boosted, lat_boosted


def _boost_cc_and_tau_res_t(
    cc_max_t: np.ndarray, tau_cirrus_max_t: np.ndarray, *, n_rep: int, tau_threshold: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Scale the mean value of the boosted resolution to the original pixel mean.

    The natural cirrus coverage and optical depth for the boosted resolution is
    distributed randomly in each pixel. This ensures that the mean value in the
    boosted resolution is equal to the value of the original pixel.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    cc_hi_res_t = _repeat_elements_along_rows_columns(cc_max_t, n_rep=n_rep)
    tau_cirrus_hi_res_t = _repeat_elements_along_rows_columns(tau_cirrus_max_t, n_rep=n_rep)

    cc_hi_res_adj_t = np.zeros_like(cc_hi_res_t)
    tauc_hi_res_adj_t = np.zeros_like(tau_cirrus_hi_res_t)

    # Adjusted cirrus optical depth
    rand_num = np.random.uniform(0, 1, np.shape(cc_hi_res_t))
    dx = 0.03  # Prevent division of small values: calibrated to match the original cirrus cover
    has_cover = rand_num > (1 + dx - cc_hi_res_t)
    tauc_hi_res_adj_t[has_cover] = tau_cirrus_hi_res_t[has_cover] / cc_hi_res_t[has_cover]

    # Adjusted cirrus coverage
    is_above_threshold = tauc_hi_res_adj_t >= tau_threshold
    cc_hi_res_adj_t[is_above_threshold] = 1
    return cc_hi_res_adj_t, tauc_hi_res_adj_t


def _repeat_elements_along_rows_columns(array_2d: np.ndarray, *, n_rep: int) -> np.ndarray:
    """Repeat the elements of a 2D numpy array n_rep times along each row and column."""
    dimension = np.shape(array_2d)

    # Repeating elements along axis=1
    array_2d_rep = [np.repeat(array_2d[i, :], n_rep) for i in np.arange(dimension[0])]
    stacked = np.vstack(array_2d_rep)

    # Repeating elements along axis=0
    return np.repeat(stacked, n_rep, axis=0)
