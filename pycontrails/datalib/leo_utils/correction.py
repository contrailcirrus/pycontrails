"""Support for overlaying flight and contrail data on Landsat & Sentinel images."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import shapely
import xarray as xr

from pycontrails import Flight
from pycontrails.datalib.landsat import Landsat
from pycontrails.datalib.sentinel import Sentinel


def _ephemeris_ecef_to_utm(ephemeris_df: pd.DataFrame, utm_crs: pyproj.CRS) -> pd.DataFrame:
    # Define the source CRS: ECEF (Earth-Centered, Earth-Fixed) with WGS84 datum
    source_crs = pyproj.CRS(proj="geocent", datum="WGS84")

    # Create a transformer object to convert from source CRS to target CRS
    # The default order for ECEF is (X, Y, Z) and for UTM is (Easting, Northing, Height)
    transformer = pyproj.Transformer.from_crs(source_crs, utm_crs)

    ecef_x = ephemeris_df["EPHEMERIS_ECEF_X"].to_numpy()
    ecef_y = ephemeris_df["EPHEMERIS_ECEF_Y"].to_numpy()
    ecef_z = ephemeris_df["EPHEMERIS_ECEF_Z"].to_numpy()
    ecef_t = ephemeris_df["EPHEMERIS_TIME"].to_numpy()

    x, y, h = transformer.transform(ecef_x, ecef_y, ecef_z)
    return pd.DataFrame({"x": x, "y": y, "z": h, "t": ecef_t})


def scan_angle_correction(
    ds: xr.Dataset,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    z: npt.NDArray[np.floating],
    n_iter: int = 5,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Apply the scan angle correction to the given x, y, z coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the viewing azimuth angle (VAA)
        and viewing zenith angle (VZA) arrays. The units for both are degrees.
    x : npt.NDArray[np.floating]
        The x coordinates of the points to correct. Should be in the
        correct UTM coordinate system
    y : npt.NDArray[np.floating]
        The y coordinates of the points to correct. Should be in the
        correct UTM coordinate system.
    z : npt.NDArray[np.floating]
        The z coordinates (altitude in meters) of the points to correct.

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        The corrected x and y coordinates as numpy arrays in the UTM
        coordinate system. Points that are not contained in the non-nan
        region of the image will contain nan values in the output arrays.
    """
    # Confirm that x is monotonically increasing and y is decreasing
    # (This is assumed in the filtering logic below)
    if not np.all(np.diff(ds["x"]) > 0.0):
        msg = "ds['x'] must be monotonically increasing"
        raise ValueError(msg)
    if not np.all(np.diff(ds["y"]) < 0.0):
        msg = "ds['y'] must be monotonically decreasing"
        raise ValueError(msg)

    try:
        ds = ds[["VZA", "VAA"]].load()  # nice to load these once here instead of repeatedly below
    except KeyError as e:
        raise KeyError("ds must contain the variables 'VZA' and 'VAA'") from e

    x = np.atleast_1d(x).astype(np.float64, copy=False)
    y = np.atleast_1d(y).astype(np.float64, copy=False)
    z = np.atleast_1d(z).astype(np.float64, copy=False)

    x_proj = xr.DataArray(x.copy(), dims="points")  # need to copy because we modify below
    y_proj = xr.DataArray(y.copy(), dims="points")  # need to copy because we modify below

    # Mask inputs outside the dataset extent
    x_min, x_max = ds["x"].min().item(), ds["x"].max().item()
    y_min, y_max = ds["y"].min().item(), ds["y"].max().item()
    out_of_bounds = (x_proj < x_min) | (x_proj > x_max) | (y_proj < y_min) | (y_proj > y_max)

    x_proj[out_of_bounds] = np.nan
    y_proj[out_of_bounds] = np.nan

    for _ in range(n_iter):
        # Use only valid points
        valid = np.isfinite(x_proj) & np.isfinite(y_proj)
        if not np.any(valid):
            break

        # Interpolate angles only for valid points
        # Note that we may get nan values back after interpolation
        # It's arguably better to propagate nans than to keep the original values
        # because the original values may be in the nan region of the image
        vza, vaa = _interpolate_angles(ds, x_proj[valid], y_proj[valid])

        # Convert to radians
        vza_rad = np.deg2rad(vza)
        vaa_rad = np.deg2rad(vaa)

        # Apply spherical projection offset
        offset = z[valid] * np.tan(vza_rad)
        dx_offset = offset * np.sin(vaa_rad)
        dy_offset = offset * np.cos(vaa_rad)

        # Update the newly predicted x and y locations
        x_proj[valid] = x[valid] - dx_offset
        y_proj[valid] = y[valid] - dy_offset

    return x_proj.values, y_proj.values


def _interpolate_angles(
    ds: xr.Dataset,
    xi: xr.DataArray,
    yi: xr.DataArray,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Interpolate view zenith angle (VZA) and view azimuth angle (VAA).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing at least the variables "VZA" and "VAA",
        with coordinates ``x`` and ``y`` that define the spatial grid.
    xi : xr.DataArray
        X-coordinates of the target points for interpolation.
        Must be the same length as ``yi``.
    yi : xr.DataArray
        Y-coordinates of the target points for interpolation.
        Must be the same length as ``xi``.

    Returns
    -------
    vza : np.ndarray
        Interpolated view zenith angles at the given (xi, yi) points.
    vaa : np.ndarray
        Interpolated view azimuth angles at the given (xi, yi) points.
    """
    interped = ds[["VZA", "VAA"]].interp(x=xi, y=yi)
    return interped["VZA"].values, interped["VAA"].values


def estimate_scan_time(
    ephemeris_df: pd.DataFrame,
    utm_crs: pyproj.CRS,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
) -> npt.NDArray[np.datetime64]:
    """Estimate the scan time for the given x, y pixels.

    Project the x, y coordinates (in UTM coordinate system) onto the
    ephemeris track and interpolate the time.

    Parameters
    ----------
    ephemeris_df : pd.DataFrame
        DataFrame containing the ephemeris data with columns:
        - 'EPHEMERIS_ECEF_X': ECEF X coordinates (meters)
        - 'EPHEMERIS_ECEF_Y': ECEF Y coordinates (meters)
        - 'EPHEMERIS_ECEF_Z': ECEF Z coordinates (meters)
        - 'EPHEMERIS_TIME': Timestamps (as datetime64[ns])
    utm_crs : pyproj.CRS
        The UTM coordinate reference system used for projection.
    x : npt.NDArray[np.floating]
        The x coordinates of the points to estimate the scan time for. Should be in the
        correct UTM coordinate system.
    y : npt.NDArray[np.floating]
        The y coordinates of the points to estimate the scan time for. Should be in the
        correct UTM coordinate system.

    Returns
    -------
    npt.NDArray[np.datetime64]
        The estimated scan times as numpy datetime64[ns] array. Points for which
        ``x`` or ``y`` are nan will have ``NaT`` as the corresponding output value.
    """
    ephemeris_utm = _ephemeris_ecef_to_utm(ephemeris_df, utm_crs)

    valid = np.isfinite(x) & np.isfinite(y)
    points = shapely.points(x[valid], y[valid])

    line = shapely.LineString(ephemeris_utm[["x", "y"]])

    distance = line.project(points)
    projected = line.interpolate(distance)
    projected_x = shapely.get_coordinates(projected)[:, 0]

    if ephemeris_utm["t"].dtype != "datetime64[ns]":
        # This could be relaxed if needed, but datetime64[ns] is what we expect
        raise ValueError("ephemeris_utm['t'] must have dtype 'datetime64[ns]'")
    if not ephemeris_utm["x"].diff().iloc[1:].lt(0).all():
        # I think this would always be the case for polar-orbiting satellites for both
        # ascending and descending passes
        raise ValueError("ephemeris_utm['x'] must be strictly decreasing for np.interp")

    out = np.full(x.shape, np.datetime64("NaT", "ns"))
    out[valid] = np.interp(
        projected_x,
        ephemeris_utm["x"].iloc[::-1],
        ephemeris_utm["t"].iloc[::-1].astype(int),
    ).astype("datetime64[ns]")

    return out


def _geodetic_to_utm(
    lon: npt.NDArray[np.floating], lat: npt.NDArray[np.floating], crs: pyproj.CRS
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Convert geographic coordinates (longitude, latitude) to UTM coordinates."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return transformer.transform(lon, lat)


def colocate_flights(
    flight: Flight,
    handler: Sentinel | Landsat,
    utm_crs: pyproj.CRS,
    n_iter: int = 3,
    search_window: int = 5,
) -> tuple[list[float], pd.Timestamp]:
    """
    Colocate IAGOS flight track points with satellite image pixels.

    This function projects IAGOS flight positions into the UTM coordinate system,
    then uses the provided satellite handler (Sentinel or Landsat) to iteratively
    match the satellite pixels with flight points, refining the match to improve alignment.

    Parameters
    ----------
        flight (Flight): The flight object.
        handler (Sentinel | Landsat): The satellite image handler.
        utm_crs (pyproj.CRS): UTM coordinate reference system used for projection.
        n_iter (int, optional): Number of iterations to refine the sensing_time correction.
            Default is 3.
        search_window (int, optional): Time window. Default is 5 minutes.

    Returns
    -------
        correct_aircraft_location and correct_sensing_time
    """
    # get the average sensing_time for the granule to speed up processing
    initial_sensing_time = handler.get_sensing_time()
    initial_sensing_time = initial_sensing_time.tz_localize(None)

    # turn Flight object to dataframe
    flight_df = flight.dataframe.copy()

    # keep only 'search_window' minutes before and after sensing_time to speed up function
    flight_df["time"] = pd.to_datetime(flight_df["time"])
    flight_df = flight_df[
        flight_df["time"].between(
            initial_sensing_time - pd.Timedelta(minutes=search_window),
            initial_sensing_time + pd.Timedelta(minutes=search_window),
        )
    ]
    # add the x and y coordinates in the UTM coordinate system
    x, y = _geodetic_to_utm(flight_df["longitude"], flight_df["latitude"], utm_crs)
    flight_df.loc[:, "x"] = x
    flight_df.loc[:, "y"] = y

    # project the x and y location to the image level
    ds_viewing_angles = handler.get_viewing_angle_metadata()
    x_proj, y_proj = scan_angle_correction(
        ds_viewing_angles, flight_df["x"], flight_df["y"], flight_df["altitude"], n_iter=3
    )
    flight_df.loc[:, "x_proj"] = x_proj
    flight_df.loc[:, "y_proj"] = y_proj

    # get the satellite ephemeris data prepared
    satellite_ephemeris = handler.get_ephemeris()

    # create the initial guess for the location
    x_proj, y_proj = interpolate_columns(
        flight_df, initial_sensing_time, columns=["x_proj", "y_proj"]
    )
    if np.isnan(x_proj) or np.isnan(y_proj):
        raise ValueError("Aircraft is outside of the image bounds")

    # Iteratively correct flight location based on satellite position
    for _ in range(n_iter):
        corrected_sensing_time = estimate_scan_time(satellite_ephemeris, utm_crs, x_proj, y_proj)
        if corrected_sensing_time is None or len(corrected_sensing_time) == 0:
            raise ValueError(
                "No valid scan time could be estimated from the UTM coordinates during iteration."
            )

        corrected_sensing_time = corrected_sensing_time[0]

        x_proj, y_proj = interpolate_columns(
            flight_df, corrected_sensing_time, columns=["x_proj", "y_proj"]
        )
        if x_proj is None or y_proj is None or np.isnan(x_proj) or np.isnan(y_proj):
            raise ValueError("Interpolation failed after scan time correction iteration.")

    # Optional: correct sensing_time based on the detector it is in
    try:
        detector_id = handler.get_detector_id(x_proj, y_proj)
    except ValueError:
        detector_id = None
    if detector_id is not None and detector_id != 0:
        detector_time_offset = handler.get_time_delay_detector(str(detector_id), "B03")
        corrected_sensing_time += detector_time_offset

        x_proj, y_proj = interpolate_columns(
            flight_df, corrected_sensing_time, columns=["x_proj", "y_proj"]
        )

    return [x_proj, y_proj], corrected_sensing_time


def interpolate_columns(
    df: pd.DataFrame, timestamp: pd.Timestamp | str, columns: Sequence[str], time_col: str = "time"
) -> tuple:
    """
    Interpolate multiple columns in a DataFrame at a given timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing timestamped data.
    timestamp : pd.Timestamp or str
        The timestamp at which to interpolate.
    columns : list of str
        List of column names to interpolate.
    time_col : str, default 'timestamp'
        Name of the timestamp column in df.

    Returns
    -------
    tuple
        Tuple of interpolated values in the same order as `columns`.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    timestamp = pd.to_datetime(timestamp)

    if timestamp <= df[time_col].iloc[0]:
        before, after = df.iloc[0], df.iloc[1]
    elif timestamp >= df[time_col].iloc[-1]:
        before, after = df.iloc[-2], df.iloc[-1]
    else:
        before = df[df[time_col] <= timestamp].iloc[-1]
        after = df[df[time_col] >= timestamp].iloc[0]

    t0 = before[time_col].value
    t1 = after[time_col].value
    t = timestamp.value

    ratio = (t - t0) / (t1 - t0) if t1 != t0 else 0

    return tuple(before[col] + ratio * (after[col] - before[col]) for col in columns)
