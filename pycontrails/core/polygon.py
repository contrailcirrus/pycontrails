"""Algorithm support for grid to polygon conversion.

See Also
--------
:meth:`pycontrails.MetDataArray.to_polygon_feature`
:meth:`pycontrails.MetDataArray.to_polygon_feature_collection`
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

try:
    import cv2
    import shapely
    import shapely.geometry
    import shapely.validation
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This module requires the 'opencv-python' and 'shapely' packages. "
        "These can be installed with 'pip install pycontrails[vis]'."
    ) from exc


def buffer_and_clean(
    contour: npt.NDArray[np.float_],
    min_area: float,
    convex_hull: bool,
    epsilon: float,
    orient_ccw: bool = True,
) -> shapely.Polygon | None:
    """Buffer and clean a contour.

    Parameters
    ----------
    contour : npt.NDArray[np.float_]
        Contour to buffer and clean. A 2d array of shape (n, 2) where n is the number
        of vertices in the contour.
    min_area : float
        Minimum area of the polygon. If the area of the buffered contour is less than
        this, return None.
    convex_hull : bool
        Whether to take the convex hull of the buffered contour.
    epsilon : float
        Epsilon value for polygon simplification. If 0, no simplification is performed.
    orient_ccw : bool, optional
        Whether to orient the polygon counter-clockwise. If False, orient clockwise.

    Returns
    -------
    shapely.Polygon | None
        Buffered and cleaned polygon. If the area of the buffered contour is less than
        ``min_area``, return None.
    """
    if len(contour) == 1:
        base = shapely.Point(contour)
    elif len(contour) < 4:
        base = shapely.LineString(contour)
    else:
        base = shapely.LinearRing(contour)

    if orient_ccw:
        polygon = base.buffer(0.5, quad_segs=1)
    else:
        # Only buffer the interiors if necessary
        try:
            polygon = shapely.Polygon(base)
        except shapely.errors.TopologicalError:
            return None
        if not polygon.is_valid:
            polygon = polygon.buffer(0.1, quad_segs=1)

    assert isinstance(polygon, shapely.Polygon)
    assert polygon.is_valid

    # Remove all interior rings
    if polygon.interiors:
        polygon = shapely.Polygon(polygon.exterior)

    if polygon.area < min_area:
        return None

    if orient_ccw != polygon.exterior.is_ccw:
        polygon = polygon.reverse()
    if convex_hull:
        polygon = _take_convex_hull(polygon)
    if epsilon:
        polygon = _buffer_simplify_iterate(polygon, epsilon)

    return polygon


def _contours_to_polygons(
    contours: tuple[npt.NDArray[np.float_], ...],
    hierarchy: npt.NDArray[np.int_],
    min_area: float,
    convex_hull: bool,
    epsilon: float,
    i: int = 0,
) -> list[shapely.Polygon]:
    """Parse the outputs of :func:`cv2.findContours` per the GeoJSON spec.

    Parameters
    ----------
    contours : tuple[npt.NDArray[np.float_], ...]
        The contours output from :func:`cv2.findContours`.
    hierarchy : npt.NDArray[np.int_]
        The hierarchy output from :func:`cv2.findContours`.
    min_area : float
        Minimum area of a polygon to be included in the output.
    convex_hull : bool
        Whether to take the convex hull of each polygon.
    epsilon : float
        Epsilon value to use when simplifying the polygons.
    i : int, optional
        The index of the contour to start with. Defaults to 0.

    Returns
    -------
    list[shapely.Polygon]
        A list of polygons. Polygons with a parent-child relationship are merged into
        a single polygon.
    """
    out = []
    while i != -1:
        child_i, parent_i = hierarchy[i, 2:]
        orient_ccw = parent_i == -1

        contour = contours[i][:, 0, ::-1]
        i = hierarchy[i, 0]

        polygon = buffer_and_clean(contour, min_area, convex_hull, epsilon, orient_ccw)
        if polygon is None:
            continue

        assert isinstance(polygon, shapely.Polygon)
        assert polygon.is_valid
        assert polygon.is_simple
        assert not polygon.interiors

        if child_i != -1:
            holes = _contours_to_polygons(
                contours,
                hierarchy,
                min_area=min_area,
                convex_hull=False,
                epsilon=epsilon,
                i=child_i,
            )

            candidate = shapely.Polygon(polygon.exterior, [h.exterior for h in holes])
            # If the candidate isn't valid, ignore all the holes
            # This can happen if there are many holes and the buffer operation
            # causes the holes to overlap
            if candidate.is_valid:
                polygon = candidate

        out.append(polygon)
    return out


def find_multipolygon(
    arr: npt.NDArray[np.float_],
    threshold: float,
    min_area: float,
    epsilon: float,
    interiors: bool = True,
    convex_hull: bool = False,
) -> shapely.MultiPolygon:
    """Compute a multipolygon from a 2d array.

    Parameters
    ----------
    arr : npt.NDArray[np.float_]
        Array to convert to a multipolygon. The array will be converted to a binary
        array by comparing each element to `threshold`. This binary array is then
        passed into :func:`cv2.findContours` to find the contours.
    threshold : float
        Threshold to use when converting `arr` to a binary array.
    min_area : float
        Minimum area of a polygon to be included in the output.
    epsilon : float
        Epsilon value to use when simplifying the polygons. Passed into shapely's
        :meth:`shapely.geometry.Polygon.simplify` method.
    interiors : bool
        Whether to include interior polygons.
    convex_hull : bool
        Experimental. Whether to take the convex hull of each polygon.

    Returns
    -------
    shapely.MultiPolygon
        A multipolygon of the contours.
    """
    arr_bin = np.empty(arr.shape, dtype=np.uint8)
    np.greater_equal(arr, threshold, out=arr_bin)

    mode = cv2.RETR_CCOMP if interiors else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(arr_bin, mode, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return shapely.MultiPolygon()

    assert len(hierarchy) == 1
    hierarchy = hierarchy[0]

    polygons = _contours_to_polygons(contours, hierarchy, min_area, convex_hull, epsilon)
    return _make_valid_multipolygon(polygons, convex_hull)


def _take_convex_hull(polygon: shapely.Polygon) -> shapely.Polygon:
    """Take the convex hull of a linear ring and preserve the orientation.

    Parameters
    ----------
    polygon : shapely.Polygon
        Linear ring to take the convex hull of.

    Returns
    -------
    shapely.Polygon
        Convex hull of the input.
    """
    convex_hull = polygon.convex_hull
    if polygon.exterior.is_ccw == convex_hull.exterior.is_ccw:
        return convex_hull
    return shapely.Polygon(convex_hull.exterior.reverse())


def _buffer_simplify_iterate(polygon: shapely.Polygon, epsilon: float) -> shapely.Polygon:
    """Simplify a linear ring by iterating over a larger buffer.

    This function calls :func:`shapely.buffer` and :func:`shapely.simplify`
    over a range of buffer distances. The buffer allows for the topology
    of the contour to change, which is useful for simplifying contours.
    Applying a buffer does introduce a slight bias towards the exterior
    of the contour.

    .. versionadded:: 0.38.0

    Parameters
    ----------
    polygon : shapely.Polygon
        Linear ring to simplify.
    epsilon : float
        Passed as ``tolerance`` parameter into :func:`shapely.simplify`.

    Returns
    -------
    shapely.LinearRing
        Simplified linear ring.
    """
    # Try to simplify without a buffer first
    # This seems to be computationally faster
    out = polygon.simplify(epsilon, preserve_topology=False)
    if out.is_simple and out.is_valid:
        return out

    # Applying a naive linear_ring.buffer(0) can destroy the polygon completely
    # https://stackoverflow.com/a/20873812

    is_ccw = polygon.exterior.is_ccw

    # Values here are somewhat ad hoc: These seem to allow the algorithm to
    # terminate and are not too computationally expensive
    for i in range(1, 11):
        distance = epsilon * i / 10

        # Taking the buffer can change the orientation of the contour
        out = polygon.buffer(distance, join_style="mitre", quad_segs=2)
        if out.exterior.is_ccw != is_ccw:
            out = shapely.Polygon(out.exterior.coords[::-1])

        out = out.simplify(epsilon, preserve_topology=False)
        if out.is_simple and out.is_valid:
            return out

    warnings.warn(
        f"Could not simplify contour with epsilon {epsilon}. Try passing a smaller epsilon."
    )
    return polygon.simplify(epsilon, preserve_topology=True)


def _make_valid_multipolygon(
    polygons: list[shapely.Polygon], convex_hull: bool
) -> shapely.MultiPolygon:
    """Make a multipolygon valid.

    This function attempts to make a multipolygon valid by iteratively
    applying :func:`shapely.unary_union` to convert non-disjoint polygons
    into disjoint polygons by merging them. If the multipolygon is still
    invalid after 5 attempts, a warning is raised and the last
    multipolygon is returned.

    This function is needed because simplifying a contour can change the
    geometry of it, which can cause non-disjoint polygons to be created.

    .. versionadded:: 0.38.0

    Parameters
    ----------
    polygons : list[shapely.Polygon]
        List of polygons to combine into a multipolygon.
    convex_hull : bool
        If True, take the convex hull of merged polygons.

    Returns
    -------
    shapely.MultiPolygon
        Valid multipolygon.
    """
    mp = shapely.MultiPolygon(polygons)
    if mp.is_empty:
        return mp

    n_attempts = 5
    for _ in range(n_attempts):
        if mp.is_valid:
            return mp

        mp = shapely.unary_union(mp)
        if isinstance(mp, shapely.Polygon):
            mp = shapely.MultiPolygon([mp])

        # Fix the orientation of the polygons
        mp = shapely.MultiPolygon([shapely.geometry.polygon.orient(p, sign=1) for p in mp.geoms])

        if convex_hull:
            mp = shapely.MultiPolygon([_take_convex_hull(p.exterior) for p in mp.geoms])

    warnings.warn(
        f"Could not make multipolygon valid after {n_attempts} attempts. "
        "According to shapely, the multipolygon is invalid because: "
        f"{shapely.validation.explain_validity(mp)}"
    )
    return mp


def polygon_to_lon_lat(
    polygon: shapely.Polygon,
    longitude: npt.NDArray[np.float_],
    latitude: npt.NDArray[np.float_],
    altitude: float | None,
    precision: int | None,
) -> list[list[list[float]]]:
    """Convert polygon longitude-latitude coordinates.

    This function assumes ``polygon`` was created from a padded array of shape
    ``(longitude.size + 2, latitude.size + 2)``.

    .. versionchanged:: 0.25.12

        Previous implementation assumed indexes were integers or half-integers.
        This is the case for binary arrays, but not for continuous arrays.
        The new implementation performs the linear interpolation necessary for
        continuous arrays.

    .. versionchanged:: 0.32.1

        Add ``precision`` parameter. Ensure that the returned contour is not
        degenerate after rounding.

    Parameters
    ----------
    polygon : shapely.Polygon
        Contour to convert to longitude-latitude coordinates.
    longitude : npt.NDArray[np.float_]
        One dimensional array of longitude values.
    latitude : npt.NDArray[np.float_]
        One dimensional array of latitude values.
    altitude : float | None, optional
        Altitude value to use for the output. If not provided, the z-coordinate
        is not included in the output. Default is None.
    precision : int, optional
        Number of decimal places to round the longitude and latitude values to.
        If None, no rounding is performed. If after rounding, the polygon
        becomes degenerate, the rounding is increased by one decimal place.

    Returns
    -------
    list[list[list[float]]]
        Ragged contour array of longitude, latitude values converted from the
        input polygon. The first element of the returned list corresponds to
        the exterior contour, and the remaining elements correspond to the
        interior contours. Each contour is a list of vertices, which is itself
        a list of longitude, latitude, and altitude values.
    """
    rings = polygon.exterior, *polygon.interiors
    args = longitude, latitude, altitude, precision
    return [_ring_to_lon_lat(ring, *args) for ring in rings]


def _ring_to_lon_lat(
    ring: shapely.LinearRing,
    longitude: npt.NDArray[np.float_],
    latitude: npt.NDArray[np.float_],
    altitude: float | None,
    precision: int | None,
) -> list[list[float]]:
    contour = np.asarray(ring.coords)

    # Account for padding
    lon_idx = contour[:, 0] - 1
    lat_idx = contour[:, 1] - 1

    # Calculate interpolated longitude and latitude values
    lon = np.interp(lon_idx, np.arange(longitude.shape[0]), longitude)
    lat = np.interp(lat_idx, np.arange(latitude.shape[0]), latitude)

    # Round to some precision
    if precision is not None:
        while precision < 10:
            rounded_lon = np.round(lon, precision)
            rounded_lat = np.round(lat, precision)
            lr = shapely.LinearRing(np.stack([rounded_lon, rounded_lat], axis=1))
            if lr.is_valid:
                lon = rounded_lon
                lat = rounded_lat
                break
            precision += 1
        else:
            raise RuntimeError("Could not round contour to valid LinearRing.")

    # Include altitude in output if provided
    if altitude is None:
        arrays = [lon, lat]
    else:
        alt = np.full_like(lon, altitude).round(1)
        arrays = [lon, lat, alt]

    stacked = np.stack(arrays, axis=1)
    return stacked.tolist()
