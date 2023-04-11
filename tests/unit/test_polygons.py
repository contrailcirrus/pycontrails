"""Unit tests for the pycontrails.core.polygons.py module."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from skimage import draw

from pycontrails import MetDataArray
from pycontrails.core import polygon


@pytest.fixture
def carr():
    """Fixture with continuous padded data."""
    rng = np.random.default_rng(654)
    arr = rng.uniform(0, 0.8, size=(100, 100))
    arr[25:35, 55:70] = 1  # big island
    arr[5:10, 5:10] = 1  # small island
    return np.pad(arr, 1, constant_values=0)


@pytest.fixture
def barr():
    """Fixture with binary padded data."""
    rng = np.random.default_rng(654)
    arr = rng.integers(0, 2, size=(100, 100)).astype(float)
    return np.pad(arr, 1, constant_values=0)


@pytest.fixture
def hawaiian_earrings():
    """Fixture reminiscent of Hawaiian earrings."""
    img = np.zeros((10, 10))

    start = 1, 1
    extent = 8, 8
    rr, cc = draw.rectangle(start, extent=extent, shape=img.shape)
    img[rr, cc] = 1

    start = 2, 2
    extent = 6, 6
    rr, cc = draw.rectangle(start, extent=extent, shape=img.shape)
    img[rr, cc] = 0

    start = 3, 3
    extent = 4, 4
    rr, cc = draw.rectangle(start, extent=extent, shape=img.shape)
    img[rr, cc] = 1

    start = 4, 4
    extent = 2, 2
    rr, cc = draw.rectangle(start, extent=extent, shape=img.shape)
    img[rr, cc] = 0

    return img


def test_find_contours_to_depth(carr: np.ndarray):
    """Test the `find_contours_to_depth` function.

    Confirm properties of `NestedContours` instance.
    """
    nc = polygon.find_contours_to_depth(
        carr,
        threshold=0.5,
        min_area=0,
        min_area_to_iterate=0,
        epsilon=0.1,
        depth=1,
    )
    assert isinstance(nc, polygon.NestedContours)

    # pin the number of children
    assert sum(1 for _ in nc) == 1168
    assert nc.n_children == 1168
    assert not hasattr(nc, "contour")
    assert nc.n_vertices == 0
    assert repr(nc).split("\n")[0] == "Top level NestedContours instance with 1168 children"

    # only down to depth 1
    for child in nc:
        assert isinstance(child, polygon.NestedContours)
        assert child.n_children == 0
        assert hasattr(child, "contour")
        assert child.n_vertices >= 3
        assert len(repr(child)) == 40


@pytest.mark.parametrize("min_area,n_children ", [(1, 524), (2, 351), (3, 251), (12, 33)])
def test_find_contours_to_depth_min_area(carr, min_area: float, n_children: int):
    """Show that increasing min_area decreases the number of children."""
    nc = polygon.find_contours_to_depth(
        carr,
        threshold=0.5,
        min_area=min_area,
        min_area_to_iterate=0,
        epsilon=0.1,
        depth=1,
    )
    assert nc.n_children == n_children


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_find_contours_to_higher_depth(carr, depth: int):
    """Test find_contours with depth > 1."""
    nc = polygon.find_contours_to_depth(
        carr,
        threshold=0.3,
        min_area=0,
        min_area_to_iterate=0,
        epsilon=0.1,
        depth=depth,
    )
    depth_found = 0
    for c1 in nc:
        depth_found = max(depth_found, 1)
        for c2 in c1:
            depth_found = max(depth_found, 2)
            for c3 in c2:
                # With the random data input, we never actually go past depth 3
                depth_found = max(depth_found, 3)
                assert c3.n_children == 0
    assert depth_found == min(depth, 3)


def test_find_contours_to_depth_2_with_min_area_to_iterate(barr):
    """Test find_contours with depth=2 and min_area_to_iterate > 0."""
    nc1 = polygon.find_contours_to_depth(
        barr,
        threshold=0.5,
        min_area=0,
        min_area_to_iterate=30,
        epsilon=0.2,
        depth=2,
    )

    nc2 = polygon.find_contours_to_depth(
        barr,
        threshold=0.5,
        min_area=0,
        min_area_to_iterate=0,
        epsilon=0.2,
        depth=2,
    )

    # Both have the same number of children at depth 1
    assert nc1.n_children == nc2.n_children

    # But the children at depth 2 are different
    n_grandchildren1 = sum(c.n_children for c in nc1)
    n_grandchildren2 = sum(c.n_children for c in nc2)

    assert n_grandchildren1 == 27
    assert n_grandchildren2 == 31


@pytest.mark.parametrize("altitude", [None, 10001.2])
def test_contour_to_lon_lat(barr: np.ndarray, altitude: float | None):
    """Test the `contour_to_lon_lat` function."""
    contours = polygon.calc_exterior_contours(
        barr,
        threshold=0.5,
        min_area=0,
        epsilon=0.1,
        convex_hull=False,
        positive_orientation="high",
    )
    longitude = np.arange(0, 25, 0.25)
    latitude = np.arange(-75, -25, 0.5)
    assert barr.shape == (longitude.size + 2, latitude.size + 2)

    # Pin the length
    assert len(contours) == 695
    for c in contours:
        out = polygon.contour_to_lon_lat(c, longitude, latitude, altitude, precision=2)
        out = np.array(out)
        assert len(out) == len(c)

        assert np.all(out[:, 0] <= 24.75)
        assert np.all(out[:, 0] >= 0)
        assert np.all(out[:, 1] <= -24.5)
        assert np.all(out[:, 1] >= -75)

        if altitude is not None:
            assert out.shape[1] == 3
            assert np.all(out[:, 2] == altitude)
        else:
            assert out.shape[1] == 2


def test_contour_find_islands(carr: np.ndarray):
    """Confirm both large islands are found.

    See fixture definition.
    """
    nc = polygon.find_contours_to_depth(
        carr,
        threshold=0.9,
        min_area=0,
        min_area_to_iterate=0,
        epsilon=0.05,
        depth=1,
    )
    assert nc.n_children == 2
    child1, child2 = nc
    assert child1.n_children == 0
    assert child2.n_children == 0
    assert child1.n_vertices == 18
    assert child2.n_vertices == 37

    # Confirm each have ccw orientation
    # Shift each island by its center (hack)
    centered = child1.contour - [7.5, 7.5]  # shift by island center
    _check_centered_contour_orientation(centered)
    centered = child2.contour - [30, 62]  # shift by island center
    _check_centered_contour_orientation(centered)

    # with high enough min_area, only one island is found
    nc = polygon.find_contours_to_depth(
        carr,
        threshold=0.9,
        min_area=25,
        min_area_to_iterate=0,
        epsilon=0.1,
        depth=1,
    )
    assert nc.n_children == 1

    # and if min_area is even higher, no islands are found
    nc = polygon.find_contours_to_depth(
        carr,
        threshold=0.9,
        min_area=150,
        min_area_to_iterate=0,
        epsilon=0.1,
        depth=1,
    )
    assert nc.n_children == 0


def _check_centered_contour_orientation(centered: np.ndarray):
    """Check that the centered contour has CCW orientation.

    Somewhat hacky.
    """
    x = centered[:, 0]
    y = centered[:, 1]
    angles = np.unwrap(np.arctan2(y, x))
    assert np.all(angles > 0)
    assert np.all(np.diff(angles) < 1.5)  # 1.5 works for these examples


def test_hawaiian_earrings(hawaiian_earrings: np.ndarray):
    """Show that `find_contours_to_depth` can find deeply nested contours."""
    nc = polygon.find_contours_to_depth(hawaiian_earrings, 0.5, 0, 0, 0.1, 10, "high")
    expected_repr = (
        "Top level NestedContours instance with 1 children\n"
        "  Contour with   9 vertices and 1 children\n"
        "    Contour with   9 vertices and 1 children\n"
        "      Contour with   9 vertices and 1 children\n"
        "        Contour with   9 vertices and 0 children"
    )
    assert repr(nc) == expected_repr


@pytest.mark.parametrize("iso_value", [3e8, 6e8, 1e9])
@pytest.mark.parametrize("min_area", [0.0, 1.0, 2.0])
def test_polygon_roundoff_error(polygon_bug: MetDataArray, iso_value: float, min_area: float):
    """Confirm that a polygon bug is fixed.

    On pycontrails 0.32.0, rounding polygon vertices caused a bug where the
    second and second to last vertices were the same. This test confirms that
    the bug is fixed.
    """
    assert polygon_bug.shape == (720, 261, 1, 1)

    feature = polygon_bug.to_polygon_feature(iso_value=iso_value, min_area=min_area)
    coords = feature["geometry"]["coordinates"]
    for polys in coords:
        for i, ring in enumerate(polys):
            lr = shapely.LinearRing(ring)
            assert lr.is_valid
            assert lr.is_simple
            assert lr.is_ring
            assert lr.is_closed
            assert lr.is_ccw == (i == 0)

            assert np.array_equal(ring[0], ring[-1])
            assert not np.array_equal(ring[1], ring[-2])
            # Check that no other vertices are duplicated
            assert len(np.unique(ring, axis=0)) == len(ring) - 1
