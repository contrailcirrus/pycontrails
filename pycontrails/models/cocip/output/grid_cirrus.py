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


def contrails_to_hi_res_grid(
        time: pd.Timestamp | np.datetime64,
        contrails_t: GeoVectorDataset, *,
        var_name: str,
        spatial_bbox: list[float] = [-180, -90, 180, 90],
        spatial_grid_res: float = 0.05,
) -> xr.DataArray:
    """
    Aggregate contrail segments to a high-resolution longitude-latitude grid.

    Parameters
    ----------
    time : pd.Timestamp | np.datetime64
        UTC time of interest.
    contrails_t : GeoVectorDataset
        All contrail waypoint outputs at `time`.
    var_name : str
        Contrail property for aggregation, where `var_name` must be included in `contrail_segment`.
        For example, `tau_contrail`, `rf_sw`, `rf_lw`, and `rf_net`
    spatial_bbox : list[float]
        Spatial bounding box, [lon_min, lat_min, lon_max, lat_max], [:math:`\deg`]
    spatial_grid_res : float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    xr.DataArray
        Contrail segments and their properties aggregated to a longitude-latitude grid.
    """
    # Ensure the required columns are included in `contrails_t`
    cols_req = [
        "flight_id", "waypoint", "longitude", "latitude",
        "altitude", "time", "sin_a", "cos_a", "width", var_name
    ]
    contrails_t.ensure_vars(cols_req)

    # Ensure that the times in `contrails_t` are the same.
    is_in_time = contrails_t["time"] == time
    if ~np.all(is_in_time):
        warnings.warn(
            f"Contrails have inconsistent times. Waypoints that are not in {time} are removed."
        )
        contrails_t = contrails_t.filter(is_in_time)

    main_grid = _initialise_longitude_latitude_grid(spatial_bbox, spatial_grid_res)

    # Contrail head and tails: continuous segments only
    heads_t = contrails_t.dataframe
    heads_t.sort_values(["flight_id", "waypoint"], inplace=True)
    tails_t = heads_t.shift(periods=-1)

    is_continuous = heads_t["continuous"]
    heads_t = heads_t[is_continuous].copy()
    tails_t = tails_t[is_continuous].copy()
    tails_t["waypoint"] = tails_t['waypoint'].astype('int')

    heads_t.set_index(["flight_id", "waypoint"], inplace=True, drop=False)
    tails_t.index = heads_t.index

    # Aggregate contrail segments to a high resolution longitude-latitude grid
    for i in tqdm(heads_t.index[:2000]):
        contrail_segment = GeoVectorDataset(
            pd.concat([heads_t[cols_req].loc[i], tails_t[cols_req].loc[i]], axis=1).T,
            copy=True
        )

        segment_grid = segment_property_to_hi_res_grid(
            contrail_segment, var_name=var_name, spatial_grid_res=spatial_grid_res
        )
        main_grid = _add_segment_to_main_grid(main_grid, segment_grid)

    return main_grid


def _initialise_longitude_latitude_grid(
        spatial_bbox: list[float] = [-180, -90, 180, 90],
        spatial_grid_res: float = 0.05,
) -> xr.DataArray:
    """
    Create longitude-latitude grid of specified coordinates and spatial resolution.

    Parameters
    ----------
    spatial_bbox : list[float]
        Spatial bounding box, [lon_min, lat_min, lon_max, lat_max], [:math:`\deg`]
    spatial_grid_res : float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    xr.DataArray
        Longitude-latitude grid of specified coordinates and spatial resolution, filled with zeros.

    Notes
    -----
    This empty grid is used to store the aggregated contrail properties of the individual
    contrail segments, such as the gridded contrail optical depth and radiative forcing.
    """
    lon_coords = np.arange(
        spatial_bbox[0], spatial_bbox[2] + spatial_grid_res, spatial_grid_res
    )
    lat_coords = np.arange(
        spatial_bbox[1], spatial_bbox[3] + spatial_grid_res, spatial_grid_res
    )
    return xr.DataArray(
        np.zeros((len(lon_coords), len(lat_coords))),
        dims=["longitude", "latitude"],
        coords={"longitude": lon_coords, "latitude": lat_coords},
    )


def segment_property_to_hi_res_grid(
        contrail_segment: GeoVectorDataset, *,
        var_name: str,
        spatial_grid_res: float = 0.05,
) -> xr.DataArray:
    """
    Convert the contrail segment property to a high-resolution longitude-latitude grid.

    Parameters
    ----------
    contrail_segment : GeoVectorDataset
        Contrail segment waypoints (head and tail).
    var_name : str
        Contrail property of interest, where `var_name` must be included in `contrail_segment`.
        For example, `tau_contrail`, `rf_sw`, `rf_lw`, and `rf_net`
    spatial_grid_res : float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    xr.DataArray
        Contrail segment dimension and property projected to a longitude-latitude grid.

    Notes
    -----
    - See Appendix A11 and A12 of :cite:`schumannContrailCirrusPrediction2012`.
    """
    # Ensure that `contrail_segment` contains the required variables
    contrail_segment.ensure_vars(("sin_a", "cos_a", "width", var_name))

    # Ensure that `contrail_segment` only contains two waypoints and have the same time.
    assert len(contrail_segment) == 2
    assert contrail_segment["time"][0] == contrail_segment["time"][1]

    # Calculate contrail edges
    (
        contrail_segment["lon_edge_l"],
        contrail_segment["lat_edge_l"],
        contrail_segment["lon_edge_r"],
        contrail_segment["lat_edge_r"],
    ) = contrail_properties.contrail_edges(
        contrail_segment["longitude"],
        contrail_segment["latitude"],
        contrail_segment["sin_a"],
        contrail_segment["cos_a"],
        contrail_segment["width"],
    )

    # Initialise contrail segment grid with spatial domain that covers the contrail area.
    lon_edges = np.concatenate(
        [contrail_segment["lon_edge_l"], contrail_segment["lon_edge_r"]], axis=0
    )
    lat_edges = np.concatenate(
        [contrail_segment["lat_edge_l"], contrail_segment["lat_edge_r"]], axis=0
    )
    spatial_bbox = spatial_bounding_box(lon_edges, lat_edges, buffer=0.5)
    segment_grid = _initialise_longitude_latitude_grid(spatial_bbox, spatial_grid_res)

    # Calculate gridded contrail segment properties
    weights = _pixel_weights(contrail_segment, segment_grid)
    dist_perpendicular = _segment_perpendicular_distance_to_pixels(contrail_segment, weights)
    plume_concentration = _gaussian_plume_concentration(
        contrail_segment, weights, dist_perpendicular
    )

    # Distribute selected contrail property to grid
    return plume_concentration * (
            weights * xr.ones_like(weights) * contrail_segment[var_name][1]
            + (1 - weights) * xr.ones_like(weights) * contrail_segment[var_name][0]
    )


def _pixel_weights(
        contrail_segment: GeoVectorDataset,
        segment_grid: xr.DataArray
) -> xr.DataArray:
    """
    Calculate the pixel weights for `segment_grid`.

    Parameters
    ----------
    contrail_segment : GeoVectorDataset
        Contrail segment waypoints (head and tail).
    segment_grid : xr.DataArray
        Contrail segment grid with spatial domain that covers the contrail area.

    Returns
    -------
    xr.DataArray
        Pixel weights for `segment_grid`

    Notes
    -----
    - See Appendix A12 of :cite:`schumannContrailCirrusPrediction2012`.
    - This is the weights (from the beginning of the contrail segment) to the nearest longitude and
        latitude pixel in the `segment_grid`.
    - The contrail segment do not contribute to the pixel if weight < 0 or > 1.
    """
    head = contrail_segment.dataframe.iloc[0]
    tail = contrail_segment.dataframe.iloc[1]

    # Calculate determinant
    dx = units.longitude_distance_to_m(
        (tail["longitude"] - head["longitude"]),
        0.5 * (head["latitude"] + tail["latitude"]),
    )
    dy = units.latitude_distance_to_m(tail["latitude"] - head["latitude"])
    det = dx**2 + dy**2

    # Calculate pixel weights
    lon_grid, lat_grid = np.meshgrid(
        segment_grid["longitude"].values, segment_grid["latitude"].values
    )
    dx_grid = units.longitude_distance_to_m(
        (lon_grid - head["longitude"]),
        0.5 * (head["latitude"] + lat_grid),
    )
    dy_grid = units.latitude_distance_to_m((lat_grid - head["latitude"]))
    weights = (dx * dx_grid + dy * dy_grid) / det
    return xr.DataArray(
        data=weights.T,
        dims=["longitude", "latitude"],
        coords={"longitude": segment_grid["longitude"], "latitude": segment_grid["latitude"]},
    )


def _segment_perpendicular_distance_to_pixels(
        contrail_segment: GeoVectorDataset,
        weights: xr.DataArray
) -> xr.DataArray:
    """
    Calculate perpendicular distance from contrail segment to each segment grid pixel.

    Parameters
    ----------
    contrail_segment : GeoVectorDataset
        Contrail segment waypoints (head and tail).
    weights : xr.DataArray
        Pixel weights for `segment_grid`.
        See `_pixel_weights` function.

    Returns
    -------
    xr.DataArray
        Perpendicular distance from contrail segment to each segment grid pixel, [:math:`m`]

    Notes
    -----
    - See Figure A7 of :cite:`schumannContrailCirrusPrediction2012`.
    """
    head = contrail_segment.dataframe.iloc[0]
    tail = contrail_segment.dataframe.iloc[1]

    # Longitude and latitude along contrail segment
    lon_grid, lat_grid = np.meshgrid(
        weights["longitude"].values, weights["latitude"].values
    )

    lon_s = head["longitude"] + weights.T.values * (tail["longitude"] - head["longitude"])
    lat_s = head["latitude"] + weights.T.values * (tail["latitude"] - head["latitude"])

    lon_dist = units.longitude_distance_to_m(
        np.abs(lon_grid - lon_s),
        0.5 * (lat_s + lat_grid)
    )

    lat_dist = units.latitude_distance_to_m(np.abs(lat_grid - lat_s))
    dist_perp = (lon_dist**2 + lat_dist**2) ** 0.5
    return xr.DataArray(dist_perp.T, coords=weights.coords)


def _gaussian_plume_concentration(
        contrail_segment: GeoVectorDataset,
        weights: xr.DataArray,
        dist_perpendicular: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate relative gaussian plume concentration along the contrail width.

    Parameters
    ----------
    contrail_segment : GeoVectorDataset
        Contrail segment waypoints (head and tail).
    weights : xr.DataArray
        Pixel weights for `segment_grid`.
        See `_pixel_weights` function.
    dist_perpendicular : xr.DataArray
        Perpendicular distance from contrail segment to each segment grid pixel, [:math:`m`]
        See `_segment_perpendicular_distance_to_pixels` function.

    Returns
    -------
    xr.DataArray
        Relative gaussian plume concentration along the contrail width

    Notes
    -----
    - Assume a one-dimensional Gaussian plume.
    - See Appendix A11 of :cite:`schumannContrailCirrusPrediction2012`.
    """
    head = contrail_segment.dataframe.iloc[0]
    tail = contrail_segment.dataframe.iloc[1]

    width = weights.values * tail["width"] + (1 - weights.values) * head["width"]
    sigma_yy = 0.125 * width**2

    concentration = np.where(
        (weights.values < 0) | (weights.values > 1),
        0,
        (4 / np.pi)**0.5 * np.exp(-0.5 * dist_perpendicular.values**2 / sigma_yy)
    )
    return xr.DataArray(concentration, coords=weights.coords)


def _add_segment_to_main_grid(
        main_grid: xr.DataArray,
        segment_grid: xr.DataArray
) -> xr.DataArray:
    """
    Add the gridded contrail segment to the main grid.

    Parameters
    ----------
    main_grid : xr.DataArray
        Aggregated contrail segment properties in a longitude-latitude grid.
    segment_grid : xr.DataArray
        Contrail segment dimension and property projected to a longitude-latitude grid.

    Returns
    -------
    xr.DataArray
        Aggregated contrail segment properties, including `segment_grid`.

    Notes
    -----
    - The spatial domain of `segment_grid` only covers the contrail segment, which is added to
        the `main_grid` which is expected to have a larger spatial domain than the `segment_grid`.
    - This architecture is used to reduce the computational resources.
    """
    lon_main = main_grid["longitude"].values
    lat_main = main_grid["latitude"].values

    lon_segment_grid = np.round(segment_grid["longitude"].values, decimals=2)
    lat_segment_grid = np.round(segment_grid["latitude"].values, decimals=2)

    main_grid_arr = main_grid.values
    subgrid_arr = segment_grid.values

    try:
        ix_ = np.searchsorted(lon_main, lon_segment_grid[0])
        ix = np.searchsorted(lon_main, lon_segment_grid[-1]) + 1
        iy_ = np.searchsorted(lat_main, lat_segment_grid[0])
        iy = np.searchsorted(lat_main, lat_segment_grid[-1]) + 1
    except IndexError:
        warnings.warn(
            "Contrail segment ignored as it is outside spatial bounding box of the main grid. "
        )
    else:
        main_grid_arr[ix_:ix, iy_:iy] = main_grid_arr[ix_:ix, iy_:iy] + subgrid_arr

    return xr.DataArray(main_grid_arr, coords=main_grid.coords)


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
