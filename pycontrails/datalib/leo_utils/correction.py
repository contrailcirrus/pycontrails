"""Support for overlaying flight and contrail data on Landsat & Sentinel images."""

from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import shapely
import xarray as xr

from pycontrails.core import GeoVectorDataset


def ephemeris_ecef_to_utm(ephemeris_df: pd.DataFrame, utm_crs: pyproj.CRS) -> pd.DataFrame:
    """Convert ephemeris data from ECEF to UTM coordinates.

    Parameters
    ----------
    ephemeris_df : pd.DataFrame
        DataFrame containing the ephemeris data with columns:
        - 'EPHEMERIS_ECEF_X': ECEF X coordinates (meters)
        - 'EPHEMERIS_ECEF_Y': ECEF Y coordinates (meters)
        - 'EPHEMERIS_ECEF_Z': ECEF Z coordinates (meters)
        - 'EPHEMERIS_TIME': Timestamps (as datetime64[ns])
    utm_crs : pyproj.CRS
        The UTM coordinate reference system to convert to.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'x': UTM easting (meters)
        - 'y': UTM northing (meters)
        - 'z': Altitude (meters)
        - 't': Timestamps (as datetime64[ns])
    """
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


@overload
def scan_angle_correction(
    ds: xr.Dataset,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    z: npt.NDArray[np.floating],
    *,
    maxiter: int = ...,
    tol: float = ...,
    full_output: Literal[False] = ...,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...


@overload
def scan_angle_correction(
    ds: xr.Dataset,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    z: npt.NDArray[np.floating],
    *,
    maxiter: int = ...,
    tol: float = ...,
    full_output: Literal[True],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.bool_]]: ...


def scan_angle_correction(
    ds: xr.Dataset,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    z: npt.NDArray[np.floating],
    *,
    maxiter: int = 5,
    tol: float = 10.0,
    full_output: bool = False,
) -> (
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.bool_]]
):
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
    maxiter : int, optional
        Maximum number of iterations to perform. Default is 5.
    tol : float, optional
        Tolerance for convergence in meters. Default is 10.0.
    full_output : bool, optional
        If True, return an additional boolean array indicating which points
        successfully converged. Default is False.

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

    offset0 = np.zeros_like(x)

    for _ in range(maxiter):
        # Note that we often get nan values back after interpolation
        # It's arguably better to propagate nans than to keep the original values
        # because the original values may be in the nan region of the image
        # (or outside the image entirely)
        vza, vaa = _interpolate_angles(ds, x_proj, y_proj)

        # Convert to radians
        vza_rad = np.deg2rad(vza)
        vaa_rad = np.deg2rad(vaa)

        # Apply spherical projection offset
        offset = z * np.tan(vza_rad)
        dx_offset = offset * np.sin(vaa_rad)
        dy_offset = offset * np.cos(vaa_rad)

        # Update the newly predicted x and y locations
        x_proj[:] = x - dx_offset
        y_proj[:] = y - dy_offset

        error = np.abs(offset - offset0)
        converged = error < tol
        if np.all(converged | np.isnan(error)):
            break

        offset0 = offset

    if full_output:
        return x_proj.values, y_proj.values, converged
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
    vza : npt.NDArray[np.floating]
        Interpolated view zenith angles at the given (xi, yi) points.
    vaa : npt.NDArray[np.floating]
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
    ephemeris_utm = ephemeris_ecef_to_utm(ephemeris_df, utm_crs)

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
        # This should always be the case for sun-synchronous satellites
        raise ValueError("ephemeris_utm['x'] must be strictly decreasing for np.interp")

    out = np.full(x.shape, np.datetime64("NaT", "ns"))
    out[valid] = np.interp(
        projected_x,
        ephemeris_utm["x"].iloc[::-1],
        ephemeris_utm["t"].iloc[::-1].astype(int),
    ).astype("datetime64[ns]")

    return out


def parallax_correct(
    vector: GeoVectorDataset, crs: pyproj.CRS, ds_angles: xr.Dataset
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return the image coordinates of the waypoints in a vector, corrected for parallax.

    Waypoints are projected into the image CRS and corrected with
    :func:`scan_angle_correction`.

    Parameters
    ----------
    vector : GeoVectorDataset
        Waypoints.
    crs : pyproj.CRS
        Image CRS.
    ds_angles : xr.Dataset
        Viewing zenith (``VZA``) and azimuth (``VAA``) angles in degrees on the image grid.

    Returns
    -------
    x, y : npt.NDArray[np.floating]
        Image coordinates of the waypoints.
    """
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(vector["longitude"], vector["latitude"])
    return scan_angle_correction(ds_angles, x, y, vector.altitude, maxiter=3)


def colocate(
    vector: GeoVectorDataset,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    scan_time: npt.NDArray[np.datetime64],
    group_by: str | list[str] | None = None,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Interpolate each group of waypoints in a vector to the time the satellite imaged it.

    Each group is interpolated on its first segment over which the waypoint time passes
    the scan time.

    Parameters
    ----------
    vector : GeoVectorDataset
        Waypoints of interest. For example, a collection of Flights or advected waypoints.
    x, y : npt.NDArray[np.floating]
        Image coordinates of the waypoints, from :func:`parallax_correct`.
    scan_time : npt.NDArray[np.datetime64]
        Time at which the satellite imaged each waypoint, or ``NaT``.
    group_by : str | list[str] | None, optional
        Columns of ``vector`` that identify groups, for example ``"flight_id"`` for
        flights or ``["flight_id", "waypoint"]`` for advected waypoints. If None,
        ``vector`` is one group.
    variables : list[str] | None, optional
        Variables of ``vector`` to interpolate to the scan time.

    Returns
    -------
    pd.DataFrame
        One row per group imaged by the satellite, with the ``group_by`` columns, the
        ``x`` and ``y`` image coordinates, the scan ``time``, and ``longitude``,
        ``latitude``, ``altitude``, and ``variables`` at that time.
    """
    group_by = [group_by] if isinstance(group_by, str) else group_by or []
    variables = variables or []

    cols = [*group_by, "time", "longitude", "latitude", *variables]
    df = vector.select(cols, copy=False).dataframe
    df = df.assign(altitude=vector.altitude, x=x, y=y, scan_time=scan_time)
    df = df.sort_values([*group_by, "time"]).reset_index(drop=True)

    if group_by:
        group = df.groupby(group_by, sort=False).ngroup().to_numpy()
    else:
        group = np.zeros(len(df), int)

    diff = ((df["time"] - df["scan_time"]) / pd.Timedelta(seconds=1)).to_numpy()

    # Find first segment of each group over which the waypoint time passes the scan time
    i = np.flatnonzero((group[:-1] == group[1:]) & (diff[:-1] <= 0) & (diff[1:] > 0))
    i = i[np.unique(group[i], return_index=True)[1]]
    frac = -diff[i] / (diff[i + 1] - diff[i])

    out = df.loc[i, group_by].reset_index(drop=True)
    for col in ["x", "y", "time", "longitude", "latitude", "altitude", *variables]:
        v = df[col].to_numpy()
        dv = v[i + 1] - v[i]
        if col == "longitude":
            # Wrap so segments crossing the antimeridian interpolate correctly
            dv = (dv + 180.0) % 360.0 - 180.0
            out[col] = (v[i] + frac * dv + 180.0) % 360.0 - 180.0
        else:
            out[col] = v[i] + frac * dv
    return out
