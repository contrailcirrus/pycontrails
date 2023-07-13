"""Algorithm support for grid to polygon conversion.

See Also
--------
:meth:`pycontrails.MetDataArray.to_polygon_feature`
:meth:`pycontrails.MetDataArray.to_polygon_feature_collection`
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

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
    precision: int | None,
    buffer: float,
    is_exterior: bool,
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
    precision : int | None
        Precision of the output polygon. If None, no rounding is performed.
    buffer : float
        Buffer distance.
    is_exterior : bool, optional
        Whether the contour is an exterior contour. If True, the contour is buffered
        with a larger buffer distance. The polygon orientation is CCW iff this is True.

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

    if is_exterior:
        # The contours computed by openCV go directly over array points
        # with value 1. With marching squares, we expect the contours to
        # be the midpoint between the 0-1 boundary. Apply a small buffer
        # to the exterior contours to account for this.
        polygon = base.buffer(buffer, quad_segs=1)
    else:
        # Only buffer the interiors if necessary
        try:
            polygon = shapely.Polygon(base)
        except shapely.errors.TopologicalError:
            return None

    if not polygon.is_valid:
        polygon = polygon.buffer(buffer / 10.0, quad_segs=1)
        assert polygon.is_valid, "Fail to make polygon valid after buffer"

    if isinstance(polygon, shapely.MultiPolygon):
        # In this case, there is often one large polygon and several small polygons
        # Just extract the largest polygon, ignoring the others
        polygon = max(polygon.geoms, key=lambda x: x.area)

    # Remove all interior rings
    if polygon.interiors:
        polygon = shapely.Polygon(polygon.exterior)

    if polygon.area < min_area:
        return None

    # Exterior polygons should have CCW orientation
    if is_exterior != polygon.exterior.is_ccw:
        polygon = polygon.reverse()
    if convex_hull:
        polygon = _take_convex_hull(polygon)
    if epsilon:
        polygon = _buffer_simplify_iterate(polygon, epsilon)

    if precision is not None:
        while precision < 10:
            out = _round_polygon(polygon, precision)
            if out.is_valid:
                return out
            precision += 1

        warnings.warn("Could not round polygon to a valid geometry.")

    return polygon


def _round_polygon(polygon: shapely.Polygon, precision: int) -> shapely.Polygon:
    """Round the coordinates of a polygon.

    Parameters
    ----------
    polygon : shapely.Polygon
        Polygon to round.
    precision : int
        Precision to use when rounding.

    Returns
    -------
    shapely.Polygon
        Polygon with rounded coordinates.
    """
    if polygon.is_empty:
        return polygon

    exterior = np.round(np.asarray(polygon.exterior.coords), precision)
    interiors = [np.round(np.asarray(i.coords), precision) for i in polygon.interiors]
    return shapely.Polygon(exterior, interiors)


def _contours_to_polygons(
    contours: Sequence[npt.NDArray[np.float_]],
    hierarchy: npt.NDArray[np.int_],
    min_area: float,
    convex_hull: bool,
    epsilon: float,
    longitude: npt.NDArray[np.float_] | None,
    latitude: npt.NDArray[np.float_] | None,
    precision: int | None,
    buffer: float,
    i: int = 0,
) -> list[shapely.Polygon]:
    """Convert the outputs of :func:`cv2.findContours` to :class:`shapely.Polygon`.

    Parameters
    ----------
    contours : Sequence[npt.NDArray[np.float_]
        The contours output from :func:`cv2.findContours`.
    hierarchy : npt.NDArray[np.int_]
        The hierarchy output from :func:`cv2.findContours`.
    min_area : float
        Minimum area of a polygon to be included in the output.
    convex_hull : bool
        Whether to take the convex hull of each polygon.
    epsilon : float
        Epsilon value to use when simplifying the polygons.
    longitude : npt.NDArray[np.float_] | None
        Longitude values for the grid.
    latitude : npt.NDArray[np.float_] | None
        Latitude values for the grid.
    precision : int | None
        Precision to use when rounding the coordinates.
    buffer : float
        Buffer to apply to the contours when converting to polygons.
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
        is_exterior = parent_i == -1

        contour = contours[i][:, 0, ::-1]
        i = hierarchy[i, 0]
        if longitude is not None and latitude is not None:
            lon_idx = contour[:, 0]
            lat_idx = contour[:, 1]

            # Calculate interpolated longitude and latitude values and recreate contour
            lon = np.interp(lon_idx, np.arange(longitude.shape[0]), longitude)
            lat = np.interp(lat_idx, np.arange(latitude.shape[0]), latitude)
            contour = np.stack([lon, lat], axis=1)

        polygon = buffer_and_clean(
            contour,
            min_area,
            convex_hull,
            epsilon,
            precision,
            buffer,
            is_exterior,
        )
        if polygon is None:
            continue

        if child_i != -1:
            holes = _contours_to_polygons(
                contours,
                hierarchy,
                min_area=min_area,
                convex_hull=False,
                epsilon=epsilon,
                longitude=longitude,
                latitude=latitude,
                precision=precision,
                buffer=buffer,
                i=child_i,
            )

            candidate = shapely.Polygon(polygon.exterior, [h.exterior for h in holes])
            # Abundance of caution: check if the candidate is valid
            # If the candidate isn't valid, ignore all the holes
            # This can happen if there are many holes and the buffer operation
            # causes the holes to overlap
            if candidate.is_valid:
                polygon = candidate

        out.append(polygon)
    return out


def determine_buffer(longitude: npt.NDArray[np.float_], latitude: npt.NDArray[np.float_]) -> float:
    """Determine the proper buffer size to use when converting to polygons."""

    ndigits = 6

    try:
        d_lon = round(longitude[1] - longitude[0], ndigits)
        d_lat = round(latitude[1] - latitude[0], ndigits)
    except IndexError as e:
        raise ValueError("Longitude and latitude must each have at least 2 elements.") from e

    if d_lon != d_lat:
        warnings.warn(
            "Longitude and latitude are not evenly spaced. Buffer size may be inaccurate."
        )
    if not np.all(np.diff(longitude).round(ndigits) == d_lon):
        warnings.warn("Longitude is not evenly spaced. Buffer size may be inaccurate.")
    if not np.all(np.diff(latitude).round(ndigits) == d_lat):
        warnings.warn("Latitude is not evenly spaced. Buffer size may be inaccurate.")

    return min(d_lon, d_lat) / 2.0


def find_multipolygon(
    arr: npt.NDArray[np.float_],
    threshold: float,
    min_area: float,
    epsilon: float,
    interiors: bool = True,
    convex_hull: bool = False,
    longitude: npt.NDArray[np.float_] | None = None,
    latitude: npt.NDArray[np.float_] | None = None,
    precision: int | None = None,
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
    longitude, latitude : npt.NDArray[np.float_], optional
        If provided, the coordinates values corresponding to the dimensions of `arr`.
        The contour coordinates will be converted to longitude-latitude values by indexing
        into this array. Defaults to None.
    precision : int, optional
        If provided, the precision to use when rounding the coordinates. Defaults to None.

    Returns
    -------
    shapely.MultiPolygon
        A multipolygon of the contours.
    """
    if arr.ndim != 2:
        raise ValueError("Array must be 2d")
    assert (longitude is None) == (latitude is None)
    if longitude is not None:
        assert latitude is not None
        assert arr.shape == (*longitude.shape, *latitude.shape)
        buffer = determine_buffer(longitude, latitude)
    else:
        buffer = 0.5

    arr_bin = np.empty(arr.shape, dtype=np.uint8)
    np.greater_equal(arr, threshold, out=arr_bin)

    mode = cv2.RETR_CCOMP if interiors else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(arr_bin, mode, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return shapely.MultiPolygon()

    assert len(hierarchy) == 1
    hierarchy = hierarchy[0]

    polygons = _contours_to_polygons(
        contours,
        hierarchy,
        min_area,
        convex_hull,
        epsilon,
        longitude,
        latitude,
        precision,
        buffer,
    )
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

    # In some rare situations, calling simplify actually gives a MultiPolygon.
    # In this case, take the polygon with the largest area
    if isinstance(out, shapely.MultiPolygon):
        out = max(out.geoms, key=lambda x: x.area)

    if out.is_simple and out.is_valid:
        return out

    # Applying a naive linear_ring.buffer(0) can destroy the polygon completely
    # https://stackoverflow.com/a/20873812

    is_ccw = polygon.exterior.is_ccw

    # Values here are somewhat ad hoc: These seem to allow the algorithm to
    # terminate and are not too computationally expensive
    for i in range(1, 11):
        distance = epsilon * i / 10.0

        # Taking the buffer can change the orientation of the contour
        out = polygon.buffer(distance, quad_segs=1)
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
            mp = shapely.MultiPolygon([_take_convex_hull(p) for p in mp.geoms])

    warnings.warn(
        f"Could not make multipolygon valid after {n_attempts} attempts. "
        "According to shapely, the multipolygon is invalid because: "
        f"{shapely.validation.explain_validity(mp)}"
    )
    return mp


def multipolygon_to_geojson(
    multipolygon: shapely.MultiPolygon,
    altitude: float | None,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a shapely multipolygon to a GeoJSON feature.

    Parameters
    ----------
    multipolygon : shapely.MultiPolygon
        Multipolygon to convert.
    altitude : float or None
        Altitude of the multipolygon. If provided, the multipolygon coordinates
        will be given a z-coordinate.
    properties : dict[str, Any], optional
        Properties to add to the GeoJSON feature.

    Returns
    -------
    dict[str, Any]
        GeoJSON feature with geometry type "MultiPolygon".
    """
    coordinates = []
    for polygon in multipolygon.geoms:
        poly_coords = []
        rings = polygon.exterior, *polygon.interiors
        for ring in rings:
            if altitude is None:
                coords = np.asarray(ring.coords)
            else:
                shape = len(ring.coords), 3
                coords = np.empty(shape)
                coords[:, :2] = ring.coords
                coords[:, 2] = altitude

            poly_coords.append(coords.tolist())
        coordinates.append(poly_coords)

    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": {"type": "MultiPolygon", "coordinates": coordinates},
    }
