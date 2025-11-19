"""Tooling and support for GEO satellites."""

import numpy as np
import numpy.typing as npt
import xarray as xr

from pycontrails.utils import dependencies


def parallax_correct(
    longitude: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
    altitude: npt.NDArray[np.floating],
    goes_da: xr.DataArray,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Apply parallax correction to WGS84 geodetic coordinates based on satellite perspective.

    This function considers the ray from the satellite to the points of interest and finds
    the intersection of this ray with the WGS84 ellipsoid. The intersection point is then
    returned as the corrected longitude and latitude coordinates.

    ::

       @ satellite
        \
         \
          \
           \
            \
             * aircraft
              \
               \
                x parallax corrected aircraft
        -------------------------  surface

    If the point of interest is not visible from the satellite (ie, on the opposite side of the
    earth), the function returns nan for the corrected coordinates.

    This function requires the :mod:`pyproj` package to be installed.

    Parameters
    ----------
    longitude : npt.NDArray[np.floating]
        A 1D array of longitudes in degrees.
    latitude : npt.NDArray[np.floating]
        A 1D array of latitudes in degrees.
    altitude : npt.NDArray[np.floating]
        A 1D array of altitudes in meters.
    goes_da : xr.DataArray
        DataArray containing the GOES projection information. Only the ``goes_imager_projection``
        field of the :attr:`xr.DataArray.attrs` is used.

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        A tuple containing the corrected longitude and latitude coordinates.

    """
    goes_imager_projection = goes_da.attrs["goes_imager_projection"]
    sat_lon = goes_imager_projection["longitude_of_projection_origin"]
    sat_lat = goes_imager_projection["latitude_of_projection_origin"]
    sat_alt = goes_imager_projection["perspective_point_height"]

    try:
        import pyproj
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            name="parallax_correct function",
            package_name="pyproj",
            module_not_found_error=exc,
            pycontrails_optional_package="pyproj",
        )

    # Convert from WGS84 to ECEF coordinates
    ecef_crs = pyproj.CRS("EPSG:4978")
    wgs84_crs = pyproj.CRS("EPSG:4326")
    transformer = pyproj.Transformer.from_crs(wgs84_crs, ecef_crs, always_xy=True)

    p0 = np.array(transformer.transform([sat_lon], [sat_lat], [sat_alt]))
    p1 = np.array(transformer.transform(longitude, latitude, altitude))

    # Major and minor axes of the ellipsoid
    a = ecef_crs.ellipsoid.semi_major_metre  # type: ignore[union-attr]
    b = ecef_crs.ellipsoid.semi_minor_metre  # type: ignore[union-attr]
    intersection = _intersection_with_ellipsoid(p0, p1, a, b)

    # Convert back to WGS84 coordinates
    inv_transformer = pyproj.Transformer.from_crs(ecef_crs, wgs84_crs, always_xy=True)
    return inv_transformer.transform(*intersection)[:2]  # final coord is (close to) 0


def _intersection_with_ellipsoid(
    p0: npt.NDArray[np.floating],
    p1: npt.NDArray[np.floating],
    a: float,
    b: float,
) -> npt.NDArray[np.floating]:
    """Find the intersection of a line with the surface of an ellipsoid."""
    # Calculate the direction vector
    px, py, pz = p0
    v = p1 - p0
    vx, vy, vz = v

    # The line between p0 and p1 in parametric form is p(t) = p0 + t * v
    # We need to find t such that p(t) lies on the ellipsoid
    # x^2 / a^2 + y^2 / a^2 + z^2 / b^2 = 1
    # (px + t * vx)^2 / a^2 + (py + t * vy)^2 / a^2 + (pz + t * vz)^2 / b^2 = 1
    # Rearranging gives a quadratic in t

    # Calculate the coefficients of this quadratic equation
    A = vx**2 / a**2 + vy**2 / a**2 + vz**2 / b**2
    B = 2 * (px * vx / a**2 + py * vy / a**2 + pz * vz / b**2)
    C = px**2 / a**2 + py**2 / a**2 + pz**2 / b**2 - 1.0

    # Calculate the discriminant
    D = B**2 - 4 * A * C
    sqrtD = np.sqrt(D, where=D >= 0, out=np.full_like(D, np.nan))

    # Calculate the two possible solutions for t
    t0 = (-B + sqrtD) / (2.0 * A)
    t1 = (-B - sqrtD) / (2.0 * A)

    # Calculate the intersection points
    intersection0 = p0 + t0 * v
    intersection1 = p0 + t1 * v

    # Pick the intersection point that is closer to the aircraft (p1)
    d0 = np.linalg.norm(intersection0 - p1, axis=0)
    d1 = np.linalg.norm(intersection1 - p1, axis=0)
    out = np.where(d0 < d1, intersection0, intersection1)

    # Fill the points in which the aircraft is not visible by the satellite with nan
    # This occurs when the earth is between the satellite and the aircraft
    # In other words, we can check for t0 < 1 (or t1 < 1)
    opposite_side = t0 < 1.0
    out[:, opposite_side] = np.nan

    return out


def to_ash(da: xr.DataArray, convention: str = "SEVIRI") -> npt.NDArray[np.float32]:
    """Compute 3d RGB array for the ASH color scheme.

    Parameters
    ----------
    da : xr.DataArray
        DataArray of GOES data with appropriate bands.
    convention : str, optional
        Convention for color space.

        - SEVIRI convention requires bands C11, C14, C15.
          Used in :cite:`kulikSatellitebasedDetectionContrails2019`.
        - Standard convention requires bands C11, C13, C14, C15

    Returns
    -------
    npt.NDArray[np.float32]
        3d RGB array with ASH color scheme according to convention.

    References
    ----------
    - `Ash RGB quick guide (the color space and color interpretations) <https://rammb.cira.colostate.edu/training/visit/quick_guides/GOES_Ash_RGB.pdf>`_
    - :cite:`SEVIRIRGBCal`
    - :cite:`kulikSatellitebasedDetectionContrails2019`

    Examples
    --------
    >>> from pycontrails.datalib.goes import GOES
    >>> goes = GOES(region="M2", bands=("C11", "C14", "C15"))
    >>> da = goes.get("2022-10-03 04:34:00")
    >>> rgb = to_ash(da)
    >>> rgb.shape
    (500, 500, 3)

    >>> rgb[0, 0, :]
    array([0.0127004 , 0.22793579, 0.3930847 ], dtype=float32)
    """
    if convention == "standard":
        if not np.all(np.isin([11, 13, 14, 15], da["band_id"])):
            msg = "DataArray must contain bands 11, 13, 14, and 15 for standard ash"
            raise ValueError(msg)
        c11 = da.sel(band_id=11).values  # 8.44
        c13 = da.sel(band_id=13).values  # 10.33
        c14 = da.sel(band_id=14).values  # 11.19
        c15 = da.sel(band_id=15).values  # 12.27

        red = c15 - c13
        green = c14 - c11
        blue = c13

    elif convention in ("SEVIRI", "MIT"):  # retain MIT for backwards compatibility
        if not np.all(np.isin([11, 14, 15], da["band_id"])):
            msg = "DataArray must contain bands 11, 14, and 15 for SEVIRI ash"
            raise ValueError(msg)
        c11 = da.sel(band_id=11).values  # 8.44
        c14 = da.sel(band_id=14).values  # 11.19
        c15 = da.sel(band_id=15).values  # 12.27

        red = c15 - c14
        green = c14 - c11
        blue = c14

    else:
        raise ValueError("Convention must be either 'SEVIRI' or 'standard'")

    # See colostate pdf for slightly wider values
    red = _clip_and_scale(red, -4.0, 2.0)
    green = _clip_and_scale(green, -4.0, 5.0)
    blue = _clip_and_scale(blue, 243.0, 303.0)
    return np.dstack([red, green, blue])


def _clip_and_scale(
    arr: npt.NDArray[np.floating], low: float, high: float
) -> npt.NDArray[np.floating]:
    """Clip array and rescale to the interval [0, 1].

    Array is first clipped to the interval [low, high] and then linearly rescaled
    to the interval [0, 1] so that::

        low -> 0
        high -> 1

    Parameters
    ----------
    arr : npt.NDArray[np.floating]
        Array to clip and scale.
    low : float
        Lower clipping bound.
    high : float
        Upper clipping bound.

    Returns
    -------
    npt.NDArray[np.floating]
        Clipped and scaled array.
    """
    return (arr.clip(low, high) - low) / (high - low)


def _coarsen_then_concat(da1: xr.DataArray, da2: xr.DataArray) -> xr.DataArray:
    """Concatenate two DataArrays, averaging da2 to da1's resolution.

    This function is hacky and should not be used publicly. It is used in goes.py
    and himawari.py to combine data from different resolutions.

    The function assumes that da2 has exactly twice the resolution of da1 in both
    the x and y dimensions.
    """
    da2 = da2.coarsen(x=2, y=2, boundary="exact").mean()  # type: ignore[attr-defined]

    # Gut check
    np.testing.assert_allclose(da1["x"], da2["x"], atol=2e-5)
    np.testing.assert_allclose(da1["y"], da2["y"], atol=2e-5)

    # Assign the coarser coords to the coarsened coords to account for any small differences
    da2["x"] = da1["x"]
    da2["y"] = da1["y"]

    # Finally, combine the datasets
    return xr.concat([da1, da2], dim="band_id", coords="different", compat="equals")
