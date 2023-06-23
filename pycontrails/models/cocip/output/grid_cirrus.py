from __future__ import annotations

import numpy as np
import numpy.typing as npt
import xarray as xr
from pycontrails.core.met import MetDataset
from pycontrails.models.tau_cirrus import tau_cirrus


def natural_cirrus_properties_to_hi_res_grid(
        met: MetDataset, *,
        spatial_grid_res: float = 0.05,
        optical_depth_threshold: float = 0.1,
        seed: int = 1
) -> MetDataset:
    """
    Increase the longitude-latitude resolution of natural cirrus cover and optical depth.

    Parameters
    ----------
    met : MetDataset
        Pressure level dataset for one time step containing 'air_temperature', 'specific_humidity',
        'specific_cloud_ice_water_content', 'geopotential',and `fraction_of_cloud_cover`
    spatial_grid_res : float
        Spatial grid resolution for the output, [:math:`\deg`]
    optical_depth_threshold : float
        Sensitivity of cirrus detection, set at 0.1 to match the capability of satellites.
    seed : int
        A number used to initialize a pseudorandom number generator.

    Returns
    -------
    MetDataset
        Single-level dataset containing the high resolution natural cirrus properties.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`

    Notes
    -----
    - The high-resolution natural cirrus coverage and optical depth is distributed randomly,
        ensuring that the mean value is equal to the value of the original grid.
    - Enhancing the spatial resolution is necessary because the existing spatial resolution of
        numerical weather prediction (NWP) models are too coarse to resolve the coverage area of
        relatively narrow contrails.
    """
    # Ensure the required columns are included in `met`
    met.ensure_vars(
        (
            'air_temperature',
            'specific_humidity',
            'specific_cloud_ice_water_content',
            'geopotential',
            'fraction_of_cloud_cover',
        )
    )

    # Ensure `met` only contains one time step, constraint can be relaxed in the future.
    if len(met["time"].data) > 1:
        raise AssertionError(
            "`met` contains more than one time step, but function only accepts one time step. "
        )

    # Calculate tau_cirrus as observed by satellites
    met["tau_cirrus"] = tau_cirrus(met)
    tau_cirrus_max = met["tau_cirrus"].data.sel(level=met["level"].data[-1])

    # Calculate cirrus coverage as observed by satellites, cc_max(x,y,t) = max[cc(x,y,z,t)]
    cirrus_cover_max = met["fraction_of_cloud_cover"].data.max(dim="level")

    # Increase resolution of longitude and latitude dimensions
    lon_coords_hi_res, lat_coords_hi_res = _hi_res_grid_coordinates(
        met["longitude"].values, met["latitude"].values, spatial_grid_res=spatial_grid_res
    )

    # Increase spatial resolution by repeating existing values (temporarily)
    n_reps = int(len(lon_coords_hi_res) / len(met["longitude"].values))
    cc_rep = _repeat_rows_and_columns(cirrus_cover_max.values, n_reps=n_reps)
    tau_cirrus_rep = _repeat_rows_and_columns(tau_cirrus_max.values, n_reps=n_reps)

    # Enhance resolution of `tau_cirrus`
    np.random.seed(seed)
    rand_number = np.random.uniform(0, 1, np.shape(tau_cirrus_rep))
    dx = 0.03  # Prevent division of small values: calibrated to match the original cirrus cover
    has_cirrus_cover = rand_number > (1 + dx - cc_rep)
    tau_cirrus_hi_res = np.where(
        has_cirrus_cover,
        tau_cirrus_rep / cc_rep,
        0
    )

    # Enhance resolution of `cirrus coverage`
    cirrus_cover_hi_res = np.where(
        tau_cirrus_hi_res > optical_depth_threshold,
        1,
        0
    )

    # Package outputs
    ds_hi_res = xr.Dataset(
        data_vars=dict(
            tau_cirrus=(["longitude", "latitude"], cirrus_cover_hi_res),
            cc_natural_cirrus=(["longitude", "latitude"], cirrus_cover_hi_res),
        ),
        coords=dict(longitude=lon_coords_hi_res, latitude=lat_coords_hi_res)
    )
    ds_hi_res = ds_hi_res.expand_dims({"level": np.array([-1])})
    ds_hi_res = ds_hi_res.expand_dims({"time": met["time"].values})
    return MetDataset(ds_hi_res)


def _hi_res_grid_coordinates(
        lon_coords: npt.NDArray[np.float_],
        lat_coords: npt.NDArray[np.float_], *,
        spatial_grid_res: float = 0.05
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Calculate longitude and latitude coordinates for the high resolution grid.

    Parameters
    ----------
    lon_coords : npt.NDArray[np.float_]
        Longitude coordinates provided by the original `MetDataset`.
    lat_coords : npt.NDArray[np.float_]
        Latitude coordinates provided by the original `MetDataset`.
    spatial_grid_res : float
        Spatial grid resolution for the output, [:math:`\deg`]

    Returns
    -------
    tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]
        Longitude and latitude coordinates for the high resolution grid.
    """
    d_lon = np.abs(np.diff(lon_coords)[0])
    d_lat = np.abs(np.diff(lat_coords)[0])
    is_whole_number = (d_lon / spatial_grid_res) - int(d_lon / spatial_grid_res) == 0

    if (d_lon <= spatial_grid_res) | (d_lat <= spatial_grid_res):
        raise ArithmeticError(
            "Spatial resolution of `met` is already higher than `spatial_grid_res`"
        )

    if ~is_whole_number:
        raise ArithmeticError(
            "Select a spatial grid resolution where `spatial_grid_res / existing_grid_res` is "
            "a whole number. "
        )

    lon_coords_hi_res = np.arange(
        lon_coords[0], lon_coords[-1] + spatial_grid_res, spatial_grid_res
    )

    lat_coords_hi_res = np.arange(
        lat_coords[0], lat_coords[-1] + spatial_grid_res, spatial_grid_res
    )

    return (
        np.round(lon_coords_hi_res, decimals=3),
        np.round(lat_coords_hi_res, decimals=3)
    )


def _repeat_rows_and_columns(
        array_2d: npt.NDArray[np.float_, np.float_], *,
        n_reps: int
) -> npt.NDArray[np.float_, np.float_]:
    """
    Repeat the elements in `array_2d` along each row and column.

    Parameters
    ----------
    array_2d : npt.NDArray[np.float_, np.float_]
        2D array containing `tau_cirrus` or `cirrus_coverage` across longitude and latitude.
    n_reps : int
        Number of repetitions.

    Returns
    -------
    npt.NDArray[np.float_, np.float_]
        2D array containing `tau_cirrus` or `cirrus_coverage` at a higher spatial resolution.
        See :func:`_hi_res_grid_coordinates`.
    """
    dimension = np.shape(array_2d)

    # Repeating elements along axis=1
    array_2d_rep = [np.repeat(array_2d[i, :], n_reps) for i in np.arange(dimension[0])]
    stacked = np.vstack(array_2d_rep)

    # Repeating elements along axis=0
    return np.repeat(stacked, n_reps, axis=0)
