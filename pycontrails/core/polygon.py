"""Algorithm support for grid to polygon conversion.

See Also
--------
:meth:`pycontrails.MetDataArray.to_polygon_feature`
:meth:`pycontrails.MetDataArray.to_polygon_feature_collection`
"""

from __future__ import annotations

import warnings
from typing import Iterator

import numpy as np
import numpy.typing as npt

try:
    import shapely
    import shapely.validation
    from skimage import draw, measure
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This module requires the 'scikit-image' and 'shapely' packages. "
        "These can be installed with 'pip install pycontrails[vis]'."
    ) from exc


class NestedContours:
    """A data structure for storing nested contours.

    This data structure is not intended to be instantiated directly. Rather, the
    :func:`find_contours_to_depth` returns an instance.

    There is no validation to ensure child contours are actually nested. It is up
    to the caller (ie, :func:`find_contours_to_depth`) to provide this.

    Parameters
    ----------
    contour : npt.NDArray[np.float_] | None
        Contour to store at the given node. If None, this is a top level contour.
        None by default.
    """

    def __init__(self, contour: npt.NDArray[np.float_] | None = None):
        if contour is not None:
            self.contour = contour
        self._children: list[NestedContours] = []

    def __iter__(self) -> Iterator[NestedContours]:
        return iter(self._children)

    def add(self, child: NestedContours) -> None:
        """Add a child contour to this instance.

        Parameters
        ----------
        child : NestedContours
            Child contour to add.
        """
        self._children.append(child)

    @property
    def n_children(self) -> int:
        """Get the number of children."""
        return len(self._children)

    @property
    def n_vertices(self) -> int:
        """Get the number of vertices in :attr:`contour`."""
        return len(getattr(self, "contour", []))

    def __repr__(self) -> str:
        if hasattr(self, "contour"):
            out = f"Contour with {self.n_vertices:3} vertices and {self.n_children} children"
        else:
            out = f"Top level NestedContours instance with {self.n_children} children"

        for c in self:
            out += "\n" + "\n".join(["  " + line for line in repr(c).split("\n")])
        return out


def calc_exterior_contours(
    arr: npt.NDArray[np.float_],
    threshold: float,
    min_area: float,
    epsilon: float,
    convex_hull: bool,
    positive_orientation: str,
) -> list[npt.NDArray[np.float_]]:
    """Calculate exterior contours of the padded array ``arr``.

    This function removes degenerate contours (contours with fewer than 3 vertices) and
    contours that are not closed.

    This function proceeds as follows:

    #. Determine all contours at the given ``threshold``.
    #. Convert each contour to a mask and take the union of all masks.
    #. Fill any array values inside the mask lower than the ``threshold`` with some nominal large
       value (such as ``np.inf`` or ``np.max(arr)``). This ensures every value inside the mask
       is above the ``threshold``, and every value outside the mask is below the ``threshold``.
    #. Recalculate all contours of the modified array. This will now contain only contours
       exterior contours of the original array.
    #. Return "clean" contours. See :func:`clean_contours` for details.

    Parameters
    ----------
    arr : npt.NDArray[np.float_]
        Padded array. Assumed to have dtype ``float``, to be padded with some constant value
        below ``threshold``, and not to contain any nan values. These assumptions are *NOT*
        checked.
    threshold : float
        Threshold value for contour creation.
    min_area: float | None
        Minimum area of a contour to be considered. Passed into :func:`clean_contours`.
    epsilon : float
        Passed as ``tolerance`` parameter into :func:`shapely.simplify`.
    convex_hull : bool
        Passed into :func:`clean_contours`.
    positive_orientation: {"high", "low"}
        Passed into :func:`skimage.measure.find_contours`

    Returns
    -------
    list[npt.NDArray[np.float_]]
        List of exterior contours.
    """
    fully_connected = "low" if positive_orientation == "high" else "high"
    kwargs = {
        "level": threshold,
        "positive_orientation": positive_orientation,
        "fully_connected": fully_connected,
    }
    contours = measure.find_contours(arr, **kwargs)

    # The snippet below is a little faster (~1.5x) than using draw.polygon2mask
    # Under the hood, draw.polygon2mask does the exact same thing as here
    mask = np.zeros(arr.shape, dtype=bool)
    for c in contours:
        rr, cc = draw.polygon(*c.T, shape=arr.shape)
        mask[rr, cc] = True

    marr = arr.copy()
    # I've gone back and forth on the "correct" fill value here
    # It is somewhat important for continuous data because measure.find_contours
    # does an interpolation under the hood, and this value *might* play a role there
    # np.max(arr) seems safer than np.inf, and is compatible with integer dtype
    marr[mask & (arr <= threshold)] = np.max(arr)

    # After setting interior points to some high value, when we recalculate
    # contours we are left with just exterior contours
    contours = measure.find_contours(marr, **kwargs)

    return clean_contours(contours, min_area, epsilon, convex_hull)


def find_contours_to_depth(
    arr: npt.NDArray[np.float_],
    threshold: float,
    min_area: float,
    min_area_to_iterate: float,
    epsilon: float,
    depth: int,
    convex_hull: bool = False,
    positive_orientation: str = "high",
    root: NestedContours | None = None,
) -> NestedContours:
    """Find nested contours up to a given depth via DFS.

    At a high level, this function proceeds as follows:

    #. Determine all exterior contours at the given ``threshold``
       (see :func:`calc_exterior_contours`).
    #. For each exterior contour:
        #. Convert the contour to a mask
        #. Copy and modify the ``arr`` array by filling values outside of the mask with
           some nominal large value
        #. Negate both the modified array and the threshold value.
        #. Recurse by calling this function with these new parameters.

    Parameters
    ----------
    arr : npt.NDArray[np.float_]
        Padded array. Assumed to have dtype ``float``, to be padded with some constant value
        below ``threshold``, and not to contain any nan values. These assumptions are *NOT*
        checked.
    threshold : float
        Threshold value for contour creation.
    min_area : float
        Minimum area of a contour to be considered. See :func:`clean_contours` for details.
    min_area_to_iterate : float
        Minimum area of a contour to be considered when recursing.
    epsilon : float
        Passed as ``tolerance`` parameter into :func:`shapely.simplify`.
    depth : int
        Depth to which to recurse. For GeoJSON Polygons, this should be 2 in order to
        generate Polygons with exterior contours and interior contours.
    convex_hull : bool, optional
        Passed into :func:`clean_contours`. Default is False.
    positive_orientation : {"high", "low"}
        Passed into :func:`skimage.measure.find_contours`. By default, "high", meaning
        top level exterior contours always have counter-clockwise orientation. This
        value of this parameter alternates between "high" and "low" in successive recursive
        calls to this function.
    root : NestedContours | None, optional
        Root node to use. If None, a new root node is created. Used for recursion and
        not intended for direct use. Default is None.

    Returns
    -------
    NestedContours
        Root node of the contour tree.
    """
    if depth == 0:
        if root is None:
            raise ValueError("Parameter root must be non-None if depth is zero.")
        return root

    root = root or NestedContours()
    contours = calc_exterior_contours(
        arr, threshold, min_area, epsilon, convex_hull, positive_orientation
    )
    for c in contours:
        child = NestedContours(c)

        # When depth == 1, we are at the bottom of the recursion
        if depth == 1:
            root.add(child)
            continue

        # If the area is too small, don't recurse
        if shapely.Polygon(c).area < min_area_to_iterate:
            root.add(child)
            continue

        # Fill points outside exterior contours with high values
        marr = np.full_like(arr, np.max(arr))
        rr, cc = draw.polygon(*c.T, shape=arr.shape)
        marr[rr, cc] = arr[rr, cc]  # keep the same interior arr values

        # And the important part: recurse on the negative
        child = find_contours_to_depth(
            arr=-marr,
            threshold=-threshold,
            min_area=min_area,
            min_area_to_iterate=min_area_to_iterate,
            epsilon=epsilon,
            depth=depth - 1,
            positive_orientation="low" if positive_orientation == "high" else "high",
            root=child,
        )
        root.add(child)

    return root


def clean_contours(
    contours: list[npt.NDArray[np.float_]],
    min_area: float,
    epsilon: float,
    convex_hull: bool,
) -> list[npt.NDArray[np.float_]]:
    """Remove degenerate contours, contours that are not closed, and contours with negligible area.

    This function also calls :func:`shapely.simplify` to simplify the contours.

    .. versionchanged:: 0.38.0

        Apply a smaller buffer when simplifying contours. This allows for changes
        to the underlying polygon topology. Previously, any contour topology was
        preserved when simplifying.

    Parameters
    ----------
    contours : list[npt.NDArray[np.float_]]
        List of contours to clean.
    min_area : float
        Minimum area for a contour to be kept. If 0, this filter is not applied.
    epsilon : float
        Passed as ``tolerance`` parameter into :func:`shapely.simplify`.
    convex_hull : bool
        If True, use the convex hull of the contour as the simplified contour.

    Returns
    -------
    list[npt.NDArray[np.float_]]
        Cleaned list of contours.
    """
    lr_list = []
    for contour in contours:
        if len(contour) <= 3:
            continue
        lr = shapely.LinearRing(contour)
        if not lr.is_valid:
            continue
        if shapely.Polygon(lr).area < min_area:
            continue
        if convex_hull:
            lr = _take_convex_hull(lr).exterior
        if epsilon:
            lr = _buffer_simplify_iterate(lr, epsilon)

        lr_list.append(lr)

    # After simplifying, the polygons may not longer be disjoint.
    mp = shapely.MultiPolygon([shapely.Polygon(lr) for lr in lr_list])
    mp = _make_multipolygon_valid(mp, convex_hull)

    return [np.asarray(p.exterior.coords) for p in mp.geoms]


def _take_convex_hull(lr: shapely.LinearRing) -> shapely.Polygon:
    """Take the convex hull of a linear ring and preserve the orientation.

    Parameters
    ----------
    lr : shapely.LinearRing
        Linear ring to take the convex hull of.

    Returns
    -------
    shapely.LinearRing
        Convex hull of the input.
    """
    convex_hull = lr.convex_hull
    if lr.is_ccw == convex_hull.exterior.is_ccw:
        return convex_hull
    return shapely.Polygon(convex_hull.exterior.coords[::-1])


def _buffer_simplify_iterate(linear_ring: shapely.LinearRing, epsilon: float) -> shapely.LinearRing:
    """Simplify a linear ring by iterating over a larger buffer.

    This function calls :func:`shapely.buffer` and :func:`shapely.simplify`
    over a range of buffer distances. The buffer allows for the topology
    of the contour to change, which is useful for simplifying contours.
    Applying a buffer does introduce a slight bias towards the exterior
    of the contour.

    .. versionadded:: 0.38.0

    Parameters
    ----------
    linear_ring : shapely.LinearRing
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
    out = linear_ring.simplify(epsilon, preserve_topology=False)
    if out.is_simple and out.is_valid:
        return out

    # Applying a naive linear_ring.buffer(0) can destroy the polygon completely
    # https://stackoverflow.com/a/20873812

    is_ccw = linear_ring.is_ccw

    # Values here are somewhat ad hoc: These seem to allow the algorithm to
    # terminate and are not too computationally expensive
    for i in range(1, 11):
        distance = epsilon * i / 10

        # Taking the buffer can change the orientation of the contour
        out = linear_ring.buffer(distance, join_style="mitre", quad_segs=2).exterior
        if out.is_ccw != is_ccw:
            out = shapely.LineString(out.coords[::-1])

        out = out.simplify(epsilon, preserve_topology=False)
        if out.is_simple and out.is_valid:
            return out

    warnings.warn(
        f"Could not simplify contour with epsilon {epsilon}. Try passing a smaller epsilon."
    )
    return linear_ring.simplify(epsilon, preserve_topology=True)


def _make_multipolygon_valid(mp: shapely.MultiPolygon, convex_hull: bool) -> shapely.MultiPolygon:
    """Make a multipolygon valid.

    This function attempts to make a multipolygon valid by iteratively
    applying :func:`shapely.unary_union` to convert non-disjoint polygons
    into disjoint polygons by merging them. If the multipolygon is still
    invalid after 5 attempts, a warning is raised and the last
    multipolygon is returned.

    .. versionadded:: 0.38.0

    Parameters
    ----------
    mp : shapely.MultiPolygon
        Multipolygon to make valid.
    convex_hull : bool
        If True, take the convex hull of merged polygons.

    Returns
    -------
    shapely.MultiPolygon
        Valid multipolygon.
    """
    if mp.is_empty:
        return mp

    # Get orientation of the first polygon. We assume that all polygons
    # share this orientation.
    is_ccw = mp.geoms[0].exterior.is_ccw

    n_attemps = 5
    for _ in range(n_attemps):
        if mp.is_valid:
            return mp

        mp = shapely.unary_union(mp)
        if isinstance(mp, shapely.Polygon):
            mp = shapely.MultiPolygon([mp])

        # Make sure the orientation of the polygons is consistent
        # There is a shapely.geometry.polygon.orient function, but it doesn't look any better
        if mp.geoms[0].exterior.is_ccw != is_ccw:
            mp = shapely.MultiPolygon([shapely.Polygon(p.exterior.coords[::-1]) for p in mp.geoms])

        if convex_hull:
            mp = shapely.MultiPolygon([_take_convex_hull(p.exterior) for p in mp.geoms])

    warnings.warn(
        f"Could not make multipolygon valid after {n_attemps} attempts. "
        "According to shapely, the multipolygon is invalid because: "
        f"{shapely.validation.explain_validity(mp)}"
    )
    return mp


def contour_to_lon_lat(
    contour: npt.NDArray[np.float_],
    longitude: npt.NDArray[np.float_],
    latitude: npt.NDArray[np.float_],
    altitude: float | None,
    precision: int | None,
) -> list[list[float]]:
    """Convert contour longitude-latitude coordinates.

    This function assumes ``contour`` was created from a padded array of shape
    ``(longitude.size + 2, latitude.size + 2)``.

    .. versionchanged:: 0.25.12

        Previous implementation assumed indexes were integers or half-integers.
        This is the case for binary arrays, but not for continuous arrays.
        The new implementation performs the linear interpolation necessary for
        continuous arrays. See :func:`skimage.measure.find_contours`.

    .. versionchanged:: 0.32.1

        Add ``precision`` parameter. Ensure that the returned contour is not
        degenerate after rounding.

    Parameters
    ----------
    contour : npt.NDArray[np.float_]
        Contour array of shape ``(n, 2)``.
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
    list[list[float]]
        Contour array of longitude, latitude values with shape ``(n, 2)`` converted
        to a list of lists. The vertices of the returned contours are rounded to some
        hard-coded precision to reduce the size of the corresponding JSON output.
        If ``altitude`` is provided, the returned list of lists will have shape
        ``(n, 3)`` (each vertex includes a z-coordinate).
    """
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
