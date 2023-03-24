"""Cocip Grid Outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from pycontrails.models.cocip import contrail_properties
from pycontrails.physics import constants, units

np.random.seed(1)


def cirrus_summary_statistics(
    df_contrails: pd.DataFrame,
    cloud_cover: xr.DataArray,
    tau_cirrus: xr.DataArray,
    tau_threshold: float = 0.1,
) -> pd.Series:
    """Calculate cirrus summary statistics.

    Calculates the (horizontal) gridded optical depth and fraction of cloud
    coverage arising from natural and
    contrail cirrus, as observed from above (i.e. satellites).


    Calculates the summary statistics, such as the percentage of:

    (1) total cirrus cover
    (2) natural cirrus cover
    (3) contrail cirrus cover; and
    (4) contrail cirrus cover under clear sky conditions for each time slice.

    The contrail cirrus cover, (3), is defined as the total cirrus cover
    (contrails + natural cirrus) minus the natural cirrus cover.

    Parameters
    ----------
    df_contrails : pd.DataFrame
        Aggregated contrail ouput.
        Note this must have a `flight_id` column with the `flight_id` of the causing flight.
    cloud_cover : xr.DataArray
        Description
    tau_cirrus : xr.DataArray
        Description
    tau_threshold : float, optional
        Description


    No Longer Returned
    ------------------
    pd.Series
        Description
    """
    cc_natural, tau_natural = _get_gridded_natural_cirrus_cover_and_tau(cloud_cover, tau_cirrus)
    tau_contrails_clear = get_gridded_tau_contrail(df_contrails, cc_natural)
    cc_contrails_clear = get_cirrus_coverage_from_tau(tau_contrails_clear, tau_threshold)

    tau_total = tau_contrails_clear + tau_natural
    cc_total = get_cirrus_coverage_from_tau(tau_total, tau_threshold)
    cc_contrails = cc_total - cc_natural

    # Get percentage cirrus cover
    lon_coords_met = tau_natural["longitude"].values
    lat_coords_met = tau_natural["latitude"].values
    pixel_area = _get_pixel_area(lon_coords_met, lat_coords_met)

    # This is specific to NATS
    # is_in_airspace = get_2d_airspace_domain_for_met_data(lon_coords_met, lat_coords_met)
    is_in_airspace = None

    cc_pct_total = _get_pct_cirrus_cover(cc_total, pixel_area, is_in_airspace)
    cc_pct_natural = _get_pct_cirrus_cover(cc_natural, pixel_area, is_in_airspace)
    cc_pct_contrails = _get_pct_cirrus_cover(cc_contrails, pixel_area, is_in_airspace)
    cc_pct_contrails_clear_sky = _get_pct_cirrus_cover(
        cc_contrails_clear, pixel_area, is_in_airspace
    )

    # Concatenate statistics
    df_cc_pct = {
        "Total Cirrus Cover Pct": cc_pct_total[0],
        "Natural Cirrus Cover Pct": cc_pct_natural[0],
        "Contrail Cirrus Cover Pct": cc_pct_contrails[0],
        "Contrail Cirrus Cover Pct - Clear Sky": cc_pct_contrails_clear_sky[0],
    }

    return pd.DataFrame(df_cc_pct)


def _get_gridded_natural_cirrus_cover_and_tau(
    cloud_cover: xr.DataArray, tau_cirrus: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    # Load DataArray to memory
    cc_3d = xr.DataArray.load(cloud_cover).copy()
    tau_cirrus_3d = xr.DataArray.load(tau_cirrus).copy()

    # Get gridded natural cirrus cover and optical depth (2D)
    cc_natural, tau_natural = get_max_cirrus_cover_and_tau(
        cc_3d, tau_cirrus_3d, boost_res=True, grid_res=0.05
    )

    return cc_natural, tau_natural


def get_cirrus_coverage_from_tau(tau_cirrus: xr.DataArray, tau_threshold: float) -> xr.DataArray:
    """Calculate the 2D cirrus coverage for the given `tau_cirrus` as observed by satellites.

    Note that the sensitivity of detecting cirrus is instrument dependent, and a tau_threshold
    is currently set at 0.1 to match the capability of satellites.
    """
    cirrus_cover = np.zeros_like(tau_cirrus)
    is_covered_by_cirrus = tau_cirrus.values > tau_threshold
    cirrus_cover[is_covered_by_cirrus] = 1
    da_cirrus_cover = xr.DataArray(data=cirrus_cover, coords=tau_cirrus.coords)
    return da_cirrus_cover


def _get_pct_cirrus_cover(
    cirrus_cover: xr.DataArray,
    pixel_area: xr.DataArray,
    is_in_airspace: xr.DataArray | None = None,
) -> pd.DataFrame:
    """Calculate the percentage cirrus coverage from the 2D cirrus coverage.

    Note that the global/regional cloud cover is calculated by summing
    the area of pixel cells that is covered by cirrus and dividing it
    by the total horizontal area.
    """
    if is_in_airspace is None:
        is_in_airspace = xr.ones_like(pixel_area, dtype=bool)

    cc_pct = np.array(
        [
            _get_pct_cirrus_cover_t(cirrus_cover.sel(time=t), pixel_area, is_in_airspace)
            for t in cirrus_cover["time"]
        ]
    )
    df_cc_pct = pd.DataFrame(cc_pct)
    df_cc_pct.index = cirrus_cover["time"].values
    return df_cc_pct


def _get_pct_cirrus_cover_t(
    cirrus_cover_2d: xr.DataArray, pixel_area: xr.DataArray, is_in_airspace: xr.DataArray
) -> float:
    """Calculate the percentage cirrus coverage for one time step.

    This is a helper function for `get_pct_cirrus_cover`.
    """
    cirrus_cover_2d_np = cirrus_cover_2d.values
    pixel_area_np = pixel_area.values
    is_in_airspace_np = is_in_airspace.values

    numer = np.sum(pixel_area_np[is_in_airspace_np] * cirrus_cover_2d_np[is_in_airspace_np])
    denom = np.sum(pixel_area_np[is_in_airspace_np])
    return numer / denom * 100


# ---
# helpers.py
# ---


def _get_pixel_area(lon_deg: np.ndarray, lat_deg: np.ndarray) -> xr.DataArray:
    """Calculate the area that is covered by the pixel of each grid cell.

    Source: https://www.pmel.noaa.gov/maillists/tmap/ferret_users/fu_2004/msg00023.html
    """
    lon_2d, lat_2d = np.meshgrid(lon_deg, lat_deg)
    d_lon = np.abs(lon_deg[1] - lon_deg[0])
    d_lat = np.abs(lat_deg[1] - lat_deg[0])

    area_lat_btm = _get_area_between_latitude_and_north_pole(lat_2d - d_lat)
    area_lat_top = _get_area_between_latitude_and_north_pole(lat_2d)
    grid_surface_area = (d_lon / 360) * (area_lat_btm - area_lat_top)
    return xr.DataArray(
        grid_surface_area.T,
        dims=["longitude", "latitude"],
        coords={"longitude": lon_deg, "latitude": lat_deg},
    )


def _get_area_between_latitude_and_north_pole(latitude: np.ndarray) -> np.ndarray:
    lat_radians = units.degrees_to_radians(latitude)
    return 2 * np.pi * constants.radius_earth**2 * (1 - np.sin(lat_radians))


# ---
# contrail_cirrus.py
# ---


def get_gridded_tau_contrail(
    df_contrails: pd.DataFrame,
    cc_natural_2d: xr.DataArray,
    *,
    time_slices: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Convert the contrail waypoints to a gridded format.

    Notes
    -----
    The exact methodology of this function is outlined in Appendix A12 in Schumann (2012).

    References
    ----------
    Schumann, U., 2012. A contrail cirrus prediction model.
        Geoscientific Model Development, 5(3), pp.543-580.
    """
    # Get coordinates
    lon_coords_met = cc_natural_2d["longitude"].values
    lat_coords_met = cc_natural_2d["latitude"].values

    if time_slices is None:
        time_slices_met = cc_natural_2d["time"].values
    else:
        time_slices_met = np.copy(time_slices)

    # calculate edges for use in gridded calculations
    (
        df_contrails["lon_edge_l"],
        df_contrails["lat_edge_l"],
        df_contrails["lon_edge_r"],
        df_contrails["lat_edge_r"],
    ) = contrail_properties.contrail_edges(
        df_contrails["longitude"],
        df_contrails["latitude"],
        df_contrails["sin_a"],
        df_contrails["cos_a"],
        df_contrails["width"],
    )

    # Filter required contrail properties
    df_contrails = df_contrails[
        [
            "flight_id",  # TODO: do we really need this for the sort below?
            "waypoint",
            "continuous",
            "longitude",
            "latitude",
            "lon_edge_l",
            "lon_edge_r",
            "lat_edge_l",
            "lat_edge_r",
            "time",
            "width",
            "tau_contrail",
            "rf_net",
        ]
    ].copy()
    df_contrails["time"] = pd.to_datetime(df_contrails["time"])

    da_tau_contrail: list[xr.DataArray] = []
    for t in time_slices_met:
        da_tau_contrail_t = _get_gridded_outputs_time_slice(
            df_contrails, t, lon_coords_met, lat_coords_met
        )
        da_tau_contrail.append(da_tau_contrail_t)

    da_tau_contrail_xr = xr.concat(da_tau_contrail, dim="time")
    dim_order: list[str] = ["longitude", "latitude", "time"]
    return da_tau_contrail_xr.transpose(*dim_order)


def _get_gridded_outputs_time_slice(
    df_contrails: pd.DataFrame,
    time_slice: np.datetime64,
    lon_coords_met: np.ndarray,
    lat_coords_met: np.ndarray,
) -> xr.DataArray:
    """Convert the contrail waypoints for a given hour to a gridded format."""
    da_main_tau_contrail = _initialize_main_grid_xr(lon_coords_met, lat_coords_met)

    if len(df_contrails) == 0:
        return da_main_tau_contrail.assign_coords({"time": time_slice})

    contrail_heads_t = df_contrails[df_contrails["time"] == time_slice].copy()
    contrail_heads_t.sort_values(["flight_id", "waypoint"], inplace=True)
    contrail_tails_t = contrail_heads_t.shift(periods=-1)

    is_continuous = contrail_heads_t["continuous"].values
    contrail_heads_t = contrail_heads_t[is_continuous].copy()
    contrail_tails_t = contrail_tails_t[is_continuous].copy()

    for i in range(len(contrail_heads_t)):
        tau_contrail = get_contrail_segment_subgrid(
            contrail_heads_t.iloc[i], contrail_tails_t.iloc[i], da_main_tau_contrail
        )
        da_main_tau_contrail = _concatenate_segment_to_main_grid(da_main_tau_contrail, tau_contrail)

    return da_main_tau_contrail.expand_dims().assign_coords({"time": time_slice})


def _initialize_main_grid_xr(
    lon_coords_met: np.ndarray, lat_coords_met: np.ndarray
) -> xr.DataArray:
    """Initialize DataArray with specified coordinates.

    Initialize an empty xr.DataArray with longitude and latitude coordinates
    that is provided by the maximum cirrus coverage met variable. This
    "main grid" will be used to store the aggregated properties of the individual
    contrail segments, such as the gridded contrail optical depth and radiative forcing.
    """
    lon_grid, lat_grid = np.meshgrid(lon_coords_met, lat_coords_met)
    return xr.DataArray(
        np.zeros_like(lon_grid.T),
        dims=["longitude", "latitude"],
        coords={"longitude": lon_coords_met, "latitude": lat_coords_met},
    )


def _concatenate_segment_to_main_grid(
    da_vals_main_grid: xr.DataArray, da_vals_subgrid: xr.DataArray
) -> xr.DataArray:
    """Add values of the subgrid (contrail segment contribution to each pixel) to the main grid."""
    vals_main_grid = da_vals_main_grid.values
    vals_subgrid = da_vals_subgrid.values

    lon_main = da_vals_main_grid["longitude"].values
    lat_main = da_vals_main_grid["latitude"].values

    lon_subgrid = da_vals_subgrid["longitude"].values
    lat_subgrid = da_vals_subgrid["latitude"].values

    try:
        # NOTE: mypy is concerned that np.argmax may return something that is not be coercible
        # to a python int. So we cast explicitly here, and if something goes wrong, an error will
        # be raised
        ix_ = int(np.argmax(lon_main == lon_subgrid[0]))
        ix = int(np.argmax(lon_main == lon_subgrid[-1]) + 1)
        iy_ = int(np.argmax(lat_main == lat_subgrid[0]))
        iy = int(np.argmax(lat_main == lat_subgrid[-1]) + 1)
    except IndexError:
        # FIXME: log instead of print
        print(
            "NOTE: Contrail segment ignored as it is located beyond the meteorological boundaries. "
        )
    else:
        vals_main_grid[ix_:ix, iy_:iy] = vals_main_grid[ix_:ix, iy_:iy] + vals_subgrid

    return xr.DataArray(
        vals_main_grid,
        dims=["longitude", "latitude"],
        coords={"longitude": lon_main, "latitude": lat_main},
    )


# ---
# contrail_subgrid.py
# ---


def get_contrail_segment_subgrid(
    contrail_head: pd.Series, contrail_tail: pd.Series, da_main_grid: xr.DataArray
) -> xr.DataArray:
    """Convert the properties of an individual contrail segment to a grid format.

    Parameters
    ----------
    contrail_head : pd.Series
        Description
    contrail_tail : pd.Series
        Description
    da_main_grid : xr.DataArray
        Description
    """
    grid = _initialize_contrail_segment_subgrid(contrail_head, contrail_tail, da_main_grid)
    weights = _get_weights_to_subgrid_pixels(contrail_head, contrail_tail, grid)
    dist_perp = _get_perpendicular_dist_to_pixels(contrail_head, contrail_tail, weights)
    concentration = _get_gaussian_plume_concentration(
        contrail_head, contrail_tail, weights, dist_perp
    )
    tau_contrail = _get_subgrid_contrail_property(
        contrail_head, contrail_tail, weights, concentration
    )

    return tau_contrail


def _initialize_contrail_segment_subgrid(
    contrail_head: pd.Series, contrail_tail: pd.Series, da_main_grid: xr.DataArray
) -> xr.DataArray:
    """Initialize a contrail segment subgrid.

    Function initializes an empty xr.DataArray with a subset of
    longitude and latitude coordinates from the main grid. The spatial
    domain of the subgrid is defined by the area that is covered by the
    contrail. Note that the subgrid architecture is used to reduce the
    computational requirements.
    """
    # Longitude coordinates that are covered by the contrail segment
    lon_edges = np.array(
        [
            contrail_head["lon_edge_l"],
            contrail_head["lon_edge_r"],
            contrail_tail["lon_edge_l"],
            contrail_tail["lon_edge_r"],
        ]
    )
    lon_coords_main = da_main_grid["longitude"].values
    lon_tolerance = np.round(0.5 * (lon_coords_main[1] - lon_coords_main[0]), 3)
    is_near_contrail = (lon_coords_main >= (min(lon_edges) - lon_tolerance)) & (
        lon_coords_main <= (max(lon_edges) + lon_tolerance)
    )
    lon_coords_subgrid = lon_coords_main[is_near_contrail]

    # Latitude coordinates that are covered by the contrail segment
    lat_edges = np.array(
        [
            contrail_head["lat_edge_l"],
            contrail_head["lat_edge_r"],
            contrail_tail["lat_edge_l"],
            contrail_tail["lat_edge_r"],
        ]
    )
    lat_coords_main = da_main_grid["latitude"].values
    lat_tolerance = np.round(0.5 * (lat_coords_main[1] - lat_coords_main[0]), 3)
    is_near_contrail = (lat_coords_main >= (min(lat_edges) - lat_tolerance)) & (
        lat_coords_main <= (max(lat_edges) + lat_tolerance)
    )
    lat_coords_subgrid = lat_coords_main[is_near_contrail]

    # Initialize subgrid
    lon_subgrid, lat_subgrid = np.meshgrid(lon_coords_subgrid, lat_coords_subgrid)
    da_subgrid = xr.DataArray(
        np.zeros_like(lon_subgrid.T),
        dims=["longitude", "latitude"],
        coords={"longitude": lon_coords_subgrid, "latitude": lat_coords_subgrid},
    )
    return da_subgrid


def _get_weights_to_subgrid_pixels(
    contrail_head: pd.Series, contrail_tail: pd.Series, grid: xr.DataArray
) -> xr.DataArray:
    """Calculate weights for subgrid pixels.

    Function calculates the weights (from the beginning of the contrail segment)
    to the nearest longitude and latitude pixel in the subgrid. Note that weights
    with values that is out of range (w < 0 and w > 1) imply that the contrail
    segment do not contribute to the pixel. The methodology is in Appendix A12
    of Schumann (2012).
    """
    dx = units.longitude_distance_to_m(
        np.abs(contrail_head["longitude"] - contrail_tail["longitude"]),
        0.5 * (contrail_head["latitude"] + contrail_tail["latitude"]),
    )
    dy = units.latitude_distance_to_m(contrail_head["latitude"] - contrail_tail["latitude"])
    det = dx**2 + dy**2

    lon_subgrid, lat_subgrid = np.meshgrid(grid["longitude"].values, grid["latitude"].values)
    dx_grid = units.longitude_distance_to_m(
        np.abs(contrail_head["longitude"] - lon_subgrid),
        0.5 * (contrail_head["latitude"] + lat_subgrid),
    )
    dy_grid = units.latitude_distance_to_m(np.abs(contrail_head["latitude"] - lat_subgrid))
    weights = (dx * dx_grid + dy * dy_grid) / det
    return xr.DataArray(data=weights.T, coords=grid.coords)


def _get_perpendicular_dist_to_pixels(
    contrail_head: pd.Series, contrail_tail: pd.Series, weights: xr.DataArray
) -> xr.DataArray:
    """Calculate the perpendicular distance from the contrail segment to each pixel in the subgrid.

    Refer to Figure A7 in Schumann (2012).
    """
    lon_subgrid, lat_subgrid = np.meshgrid(weights["longitude"].values, weights["latitude"].values)

    # Longitude and latitude along contrail segment
    lon_s = contrail_head["longitude"] + weights.T.values * (
        contrail_tail["longitude"] - contrail_head["longitude"]
    )
    lat_s = contrail_head["latitude"] + weights.T.values * (
        contrail_tail["latitude"] - contrail_head["latitude"]
    )

    lon_dist = units.longitude_distance_to_m(
        np.abs(lon_s - lon_subgrid), 0.5 * (lat_s + lat_subgrid)
    )
    lat_dist = units.latitude_distance_to_m(np.abs(lat_s - lat_subgrid))
    dist_perp = (lon_dist**2 + lat_dist**2) ** 0.5
    da_dist_perp = xr.DataArray(dist_perp.T, coords=weights.coords)

    return da_dist_perp


def _get_gaussian_plume_concentration(
    contrail_head: pd.Series,
    contrail_tail: pd.Series,
    weights: xr.DataArray,
    dist_perp: xr.DataArray,
) -> xr.DataArray:
    """Calculate the relative concentration distribution along the axis of the contrail width.

    Assumes a one-dimensional Gaussian plume. The methodology for this function can be found in
    Appendix A11 of Schumann (2012).
    """

    width = weights.values * contrail_tail["width"] + (1 - weights.values) * contrail_head["width"]
    sigma_yy = 0.125 * width**2

    concentration = (4 / np.pi) ** 0.5 * np.exp(-0.5 * dist_perp.values**2 / sigma_yy)
    is_out_of_range = (weights.values < 0) | (weights.values > 1)
    concentration[is_out_of_range] = 0
    return xr.DataArray(concentration, coords=weights.coords)


def _get_subgrid_contrail_property(
    contrail_head: pd.Series,
    contrail_tail: pd.Series,
    weights: xr.DataArray,
    concentration: xr.DataArray,
) -> xr.DataArray:
    """Calculate the contrail segment contribution to each pixel in the subgrid."""
    w_1 = weights.values
    w_2 = 1 - w_1
    conc = concentration.values
    da_property_head = xr.ones_like(weights) * contrail_head["tau_contrail"]
    da_property_tail = xr.ones_like(weights) * contrail_tail["tau_contrail"]
    return (w_1 * da_property_tail + w_2 * da_property_head) * conc


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
