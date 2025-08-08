"""Support for overlaying flight and contrail data on Landsat & Sentinel images."""

import itertools

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import shapely
import xarray as xr
from typing import Sequence, Union

from pycontrails import Flight
from pycontrails.datalib.sentinel import Sentinel
from pycontrails.datalib.landsat import Landsat


def _ephemeris_ecef_to_utm(ephemeris_df: pd.DataFrame, utm_crs: pyproj.CRS) -> pd.DataFrame:
    # Define the source CRS: ECEF (Earth-Centered, Earth-Fixed) with WGS84 datum
    source_crs = pyproj.CRS(proj="geocent", datum="WGS84")

    # Create a transformer object to convert from source CRS to target CRS
    # The default order for ECEF is (X, Y, Z) and for UTM is (Easting, Northing, Height)
    transformer = pyproj.Transformer.from_crs(source_crs, utm_crs, always_xy=False)

    ecef_x = ephemeris_df["EPHEMERIS_ECEF_X"].to_numpy()
    ecef_y = ephemeris_df["EPHEMERIS_ECEF_Y"].to_numpy()
    ecef_z = ephemeris_df["EPHEMERIS_ECEF_Z"].to_numpy()
    ecef_t = ephemeris_df["EPHEMERIS_TIME"].to_numpy()

    x, y, h = transformer.transform(ecef_x, ecef_y, ecef_z)
    return pd.DataFrame({"x": x, "y": y, "z": h, "t": ecef_t})


def scan_angle_correction_iterative(
    ds: xr.Dataset,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    z: npt.NDArray[np.floating],
    iterations: int = 5,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Apply the scan angle correction to the given x, y, z coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the viewing azimuth angle (VAA)
        and viewing zenith angle (VZA) arrays.
    x : npt.NDArray[np.floating]
        The x coordinates of the points to correct. Should be in the 
        correct UTM coorinate system
    y : npt.NDArray[np.floating]
        The y coordinates of the points to correct. Should be in the 
        correct UTM coorinate system.
    z : npt.NDArray[np.floating]
        The z coordinates (altitude in meters) of the points to correct.

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        The corrected x and y coordinates as numpy arrays in the UTM 
        coordinate system.
    """
    # Confirm that x is monotonically increasing and y is decreasing
    # (This is assumed in the filtering logic below)
    if not np.all(np.diff(ds["x"]) > 0.0):
        msg = "ds['x'] must be monotonically increasing"
        raise ValueError(msg)
    if not np.all(np.diff(ds["y"]) < 0.0):
        msg = "ds['y'] must be monotonically decreasing"
        raise ValueError(msg)
    
    x = np.atleast_1d(x).astype(np.float64)
    y = np.atleast_1d(y).astype(np.float64)
    z = np.atleast_1d(z).astype(np.float64)

    x_proj, y_proj = np.copy(x), np.copy(y)

    # Mask inputs outside the dataset extent
    x_min, x_max = ds["x"].min().item(), ds["x"].max().item()
    y_min, y_max = ds["y"].min().item(), ds["y"].max().item()
    out_of_bounds = (x_proj < x_min) | (x_proj > x_max) | (y_proj < y_min) | (y_proj > y_max)
    x_proj[out_of_bounds] = np.nan
    y_proj[out_of_bounds] = np.nan

    for _ in range(iterations):
        # Use only valid points
        valid = ~np.isnan(x_proj) & ~np.isnan(y_proj)
        if not np.any(valid):
            break

        # Interpolate angles only for valid points
        vza, vaa = interpolate_angles(ds, x_proj[valid], y_proj[valid])

        # Flatten and clean up any unexpected shapes
        vza = np.atleast_1d(vza).astype(np.float64).flatten()
        vaa = np.atleast_1d(vaa).astype(np.float64).flatten()

        # Skip any interpolated NaNs
        interp_valid = ~(np.isnan(vza) | np.isnan(vaa))

        # Full index into original arrays
        valid_indices = np.where(valid)[0]
        update_indices = valid_indices[interp_valid]

        # Convert to radians
        vza_rad = np.deg2rad(vza[interp_valid])
        vaa_rad = np.deg2rad(vaa[interp_valid])

        # Apply projection offset
        offset = z[update_indices] * np.tan(vza_rad)
        dx_offset = offset * np.sin(vaa_rad)
        dy_offset = offset * np.cos(vaa_rad)

        # Update the newly predicted x and y locations
        x_proj[update_indices] = x[update_indices] - dx_offset
        y_proj[update_indices] = y[update_indices] - dy_offset

    return x_proj, y_proj


# Interpolation function using xarray
def interpolate_angles(ds, xi, yi):
    """
    Bilinear interpolation for VZA and VAA from ds at points (xi, yi).
        -> method can be switched to linear for better accuracy
    """
    xi = np.atleast_1d(xi).flatten()
    yi = np.atleast_1d(yi).flatten()

    vza = ds["VZA"].interp(x=("x", xi), y=("y", yi), method="nearest").values
    vaa = ds["VAA"].interp(x=("x", xi), y=("y", yi), method="nearest").values
    return vza, vaa


def scan_angle_correction_fl(
    ds: xr.Dataset,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    z: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Apply the scan angle correction to the given x, y, z coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the viewing azimuth angle (VAA)
        and viewing zenith angle (VZA) arrays.
    x : npt.NDArray[np.floating]
        The x coordinates of the points to correct. Must be in the same
        coordinate system as the dataset.
    y : npt.NDArray[np.floating]
        The y coordinates of the points to correct. Must be in the same
        coordinate system as the dataset.
    z : npt.NDArray[np.floating]
        The z coordinates (altitude) of the points to correct.

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        The corrected x and y coordinates as numpy arrays.
    """
    # Confirm that x is monotonically increasing and y is decreasing
    # (This is assumed in the filtering logic below)
    if not np.all(np.diff(ds["x"]) > 0.0):
        msg = "ds['x'] must be monotonically increasing"
        raise ValueError(msg)
    if not np.all(np.diff(ds["y"]) < 0.0):
        msg = "ds['y'] must be monotonically decreasing"
        raise ValueError(msg)

    # Break into overlapping chunks each with core rectangle chunk_size x chunk_size
    # Use a pixel buffer to avoid translating off the edge
    chunk_size = 500
    chunk_buffer = 150  # = 4500m

    x_out = np.full_like(x, fill_value=np.nan)
    y_out = np.full_like(y, fill_value=np.nan)

    for i, j in itertools.product(
        range(0, ds["x"].size, chunk_size),
        range(0, ds["y"].size, chunk_size),
    ):
        i0 = max(0, i - chunk_buffer)
        i1 = min(ds["x"].size, i + chunk_size + chunk_buffer)
        j0 = max(0, j - chunk_buffer)
        j1 = min(ds["y"].size, j + chunk_size + chunk_buffer)

        # The _scan_angle_correction_chunk function fails for dask-backed arrays, so load here
        ds_chunk = ds.isel(x=slice(i0, i1), y=slice(j0, j1))[["VAA", "VZA"]].load()

        if ds_chunk["VAA"].isnull().all():
            continue

        filt = (
            (x >= ds["x"][i].item())
            & (x < ds["x"][min(ds["x"].size - 1, i + chunk_size)].item())
            & (y <= ds["y"][j].item())
            & (y > ds["y"][min(ds["y"].size - 1, j + chunk_size)].item())
        )
        if not np.any(filt):
            continue

        x_chunk = x[filt]
        y_chunk = y[filt]
        z_chunk = z[filt]

        x_chunk_out, y_chunk_out = _scan_angle_correction_chunk(ds_chunk, x_chunk, y_chunk, z_chunk)
        x_out[filt] = x_chunk_out
        y_out[filt] = y_chunk_out

    return x_out, y_out


def _scan_angle_correction_chunk(
    ds: xr.Dataset,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    z: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    zn = ds["VZA"]
    az = ds["VAA"]

    if zn.isnull().all() or az.isnull().all():
        return np.full_like(x, fill_value=np.nan), np.full_like(y, fill_value=np.nan)

    zn_rad = np.deg2rad(zn)
    az_rad = np.deg2rad(az)

    # Lift the whole 2d array to the flight altitude using:
    # z = r * cos(zn)
    # x = r * sin(zn) * sin(az)
    # y = r * sin(zn) * cos(az)

    # Employ lots of xarray broadcasting to do this
    x_da = xr.DataArray(x, dims=["points"], coords={"points": np.arange(len(x))})
    y_da = xr.DataArray(y, dims=["points"], coords={"points": np.arange(len(y))})
    z_da = xr.DataArray(z, dims=["points"], coords={"points": np.arange(len(z))})

    r = z_da / np.cos(zn_rad)
    x_lift = ds["x"] + r * np.sin(zn_rad) * np.sin(az_rad)
    y_lift = ds["y"] + r * np.sin(zn_rad) * np.cos(az_rad)

    # Find the closest point in the lifted array to the original point
    # Do this for each point in the points dimension
    dist_squared = (x_da - x_lift) ** 2 + (y_da - y_lift) ** 2

    # We take an argmin below, so we need to avoid a numpy All-NaN slice error.
    # Simply filling with a large finite value (not np.inf) will work because
    # we check the distance against a threshold later. See comment below for
    # threshold details.
    threshold = 15.0**2 + 15.0**2
    dist_squared = dist_squared.fillna(2.0 * threshold)

    indices = dist_squared.argmin(dim=["x", "y"])

    # After we lift, the distance between the lifted point and the original point
    # should be small. If it is not, we may have lifted off the edge of the image,
    # or nan values in the image may have caused the lifted point to be invalid.
    # So we check that the distance is small enough. Here, ds has 30m pixels, so
    # valid lifts should be within 15^2 + 15^2 of the original point.
    valid = dist_squared.isel(indices) <= threshold

    x_out = ds["x"].isel(x=indices["x"]).where(valid).to_numpy()
    y_out = ds["y"].isel(y=indices["y"]).where(valid).to_numpy()

    return x_out, y_out


def estimate_scan_time(
    ephemeris_df: pd.DataFrame,
    utm_crs: pyproj.CRS,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
) -> npt.NDArray[np.datetime64]:
    """Estimate the scan time for the given x, y pixels.

    Project the x, y coordinates (in UTM coordinate system) onto the 
    ephemeris track and interpolate the time.
    """
    ephemeris_utm = _ephemeris_ecef_to_utm(ephemeris_df, utm_crs)
    points = shapely.points(x, y)

    line = shapely.LineString(ephemeris_utm[["x", "y"]])

    distance = line.project(points)
    projected = line.interpolate(distance)
    projected_x = shapely.get_coordinates(projected)[:, 0]

    assert ephemeris_utm["t"].dtype == "datetime64[ns]"
    assert ephemeris_utm["x"].diff().iloc[1:].lt(0).all()
    return np.interp(
        projected_x,
        ephemeris_utm["x"].iloc[::-1],
        ephemeris_utm["t"].iloc[::-1].astype(int),
    ).astype("datetime64[ns]")


def geodetic_to_utm(lon, lat, crs):
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_utm, y_utm = transformer.transform(lon, lat)
    return x_utm, y_utm


def colocate_flights(flight: Flight, handler: Union[Sentinel, Landsat], utm_crs: pyproj.CRS, iterations=3, search_window:int=1):
    """"""
    # get the average sensing_time for the granule to speed up processing
    initial_sensing_time = handler.get_sensing_time()
    initial_sensing_time = initial_sensing_time.tz_localize(None)

    # turn Flight object to dataframe
    flight_df = flight.dataframe.copy()

    # keep only 'search_window' minutes before and after sensing_time to speed up function
    flight_df["time"] = pd.to_datetime(flight_df["time"])
    flight_df = flight_df[flight_df["time"].between(initial_sensing_time - pd.Timedelta(minutes=search_window),
                                                     initial_sensing_time + pd.Timedelta(minutes=search_window))]
    
    # add the x and y coordinates in the UTM coordinate system
    flight_df[['x', 'y']] = flight_df.apply(
        lambda row: pd.Series(geodetic_to_utm(row['longitude'], row['latitude'], utm_crs)), axis=1
    )
    
    # project the x and y location to the image level
    ds_viewing_angles = handler.get_viewing_angle_metadata()
    flight_df[['x_proj', 'y_proj']] = flight_df.apply(
        lambda row: pd.Series(scan_angle_correction_iterative(ds_viewing_angles, row['x'], row['y'], row["altitude"], iterations=3)), axis=1
    )

    # get the satellite ephemeris data prepared
    satellite_ephemeris = handler.get_ephemeris()

    # create the initial guess for the location 
    x_proj, y_proj = interpolate_columns(flight_df, initial_sensing_time, columns=["x_proj", "y_proj"])

    # Iteratively correct flight location based on satellite position
    for _ in range(iterations):
        corrected_sensing_time = estimate_scan_time(satellite_ephemeris, utm_crs, x_proj, y_proj)
        if corrected_sensing_time is None or len(corrected_sensing_time) == 0:
                raise ValueError("No valid scan time could be estimated from the UTM coordinates during iteration.")
        
        corrected_sensing_time = corrected_sensing_time[0]

        x_proj, y_proj = interpolate_columns(flight_df, corrected_sensing_time, columns=["x_proj", "y_proj"])
        if x_proj is None or y_proj is None or np.isnan(x_proj) or np.isnan(y_proj):
            raise ValueError("Interpolation failed after scan time correction iteration.")
        
    # Optional: correct sensing_time based on the detector it is in
    try:
        detector_id = handler.get_detector_id(x_proj, y_proj)
    except ValueError as e:
        detector_id = None
    if detector_id is not None and detector_id != 0:
        detector_time_offset = handler.get_time_delay_detector(str(detector_id), "B03")
        corrected_sensing_time += detector_time_offset

        x_proj, y_proj = interpolate_columns(flight_df, corrected_sensing_time, columns=["x_proj", "y_proj"])

    return [x_proj, y_proj], corrected_sensing_time


def interpolate_columns(
    df: pd.DataFrame,
    timestamp: pd.Timestamp | str,
    columns: Sequence[str],
    time_col: str = 'time'
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
        before = df[df[time_col] <= timestamp].tail(1).iloc[0]
        after = df[df[time_col] >= timestamp].head(1).iloc[0]

    t0 = before[time_col].value
    t1 = after[time_col].value
    t = timestamp.value

    ratio = (t - t0) / (t1 - t0) if t1 != t0 else 0

    return tuple(
        before[col] + ratio * (after[col] - before[col]) for col in columns
    )
