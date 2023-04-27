"""Unit tests for the pycontrails.core.polygons.py module."""

from __future__ import annotations

import numpy as np
import pytest
import shapely

from pycontrails import MetDataArray
from pycontrails.core import polygon


@pytest.fixture
def carr() -> np.ndarray:
    """Fixture with continuous padded data."""
    rng = np.random.default_rng(654)
    arr = rng.uniform(0, 0.8, size=(100, 100))
    arr[25:35, 55:70] = 1  # big island
    arr[5:10, 5:10] = 1  # small island
    return np.pad(arr, 1, constant_values=0)


@pytest.fixture
def barr() -> np.ndarray:
    """Fixture with binary padded data."""
    rng = np.random.default_rng(654)
    arr = rng.integers(0, 2, size=(100, 100)).astype(float)
    return np.pad(arr, 1, constant_values=0)


@pytest.fixture
def hawaiian_earrings() -> np.ndarray:
    """Fixture reminiscent of Hawaiian earrings."""
    img = np.zeros((10, 10))

    sl = slice(1, 9)
    img[sl, sl] = 1

    sl = slice(2, 8)
    img[sl, sl] = 0

    sl = slice(3, 7)
    img[sl, sl] = 1

    sl = slice(4, 6)
    img[sl, sl] = 0

    return img


@pytest.mark.parametrize("arr_name", ["carr", "barr", "hawaiian_earrings"])
def test_find_multipolygon_depth1(arr_name: str, request: pytest.FixtureRequest):
    """Test the `polygon.find_multipolygon` function at depth 1."""

    arr = request.getfixturevalue(arr_name)
    assert isinstance(arr, np.ndarray)

    mp = polygon.find_multipolygon(
        arr,
        threshold=0.5,
        min_area=0,
        epsilon=0.1,
        interiors=False,
    )
    assert isinstance(mp, shapely.MultiPolygon)
    assert mp.is_valid

    # All polygons have correct orientation and no holes
    n_polys = 0
    for p in mp.geoms:
        assert isinstance(p, shapely.Polygon)
        assert p.is_valid
        assert not p.has_z
        assert p.is_simple
        assert p.exterior.is_ccw
        assert not p.interiors
        assert p.area
        assert p.length
        n_polys += 1

    # pin the number of children
    if arr_name == "carr":
        assert n_polys == 227
    elif arr_name == "barr":
        assert n_polys == 23
    elif arr_name == "hawaiian_earrings":
        assert n_polys == 1


@pytest.mark.parametrize(
    "min_area,n_children ",
    [(1, 144), (2, 100), (3, 87), (4, 76), (6, 58), (12, 36), (20, 27)],
)
def test_find_multipolygon_min_area(carr: np.ndarray, min_area: float, n_children: int):
    """Show that increasing min_area decreases the number of children."""
    mp = polygon.find_multipolygon(
        carr,
        threshold=0.5,
        min_area=min_area,
        epsilon=0.1,
        interiors=False,
    )
    n_polys = len(mp.geoms)
    assert n_polys == n_children
    for p in mp.geoms:
        assert p.area >= min_area
        assert not p.interiors


@pytest.mark.parametrize("arr_name", ["barr", "hawaiian_earrings"])
def test_find_multipolygons_with_interiors(arr_name: str, request: pytest.FixtureRequest):
    """Test find_multipolygon with interiors=True."""

    arr = request.getfixturevalue(arr_name)
    assert isinstance(arr, np.ndarray)

    mp = polygon.find_multipolygon(
        arr,
        threshold=0.02,
        min_area=0,
        epsilon=0.1,
        interiors=True,
    )

    interior_found = False
    assert isinstance(mp, shapely.MultiPolygon)
    assert mp.is_valid
    for p in mp.geoms:
        assert p.is_valid
        interior_found = interior_found or p.interiors
    assert interior_found


@pytest.mark.filterwarnings("ignore:Longitude and latitude are not evenly spaced")
@pytest.mark.parametrize("altitude", [None, 10001.2])
def test_multipolygon_to_geojson(barr: np.ndarray, altitude: float | None):
    """Test the `multipolygon_to_geojson` function."""
    sl = slice(1, -1)
    mp = polygon.find_multipolygon(
        barr[sl, sl],
        threshold=0.5,
        min_area=0,
        epsilon=0.1,
        convex_hull=False,
        longitude=np.arange(0, 25, 0.25),
        latitude=np.arange(-75, -25, 0.5),
    )
    assert len(mp.geoms) == 23

    geojson = polygon.multipolygon_to_geojson(mp, altitude=altitude)
    assert isinstance(geojson, dict)
    assert geojson["type"] == "Feature"
    assert geojson["geometry"]["type"] == "MultiPolygon"
    assert len(geojson["geometry"]["coordinates"]) == 23
    for poly in geojson["geometry"]["coordinates"]:
        for ring in poly:
            contour = np.array(ring)
            assert contour.ndim == 2

            assert np.all(contour[:, 0] <= 25.25)
            assert np.all(contour[:, 0] >= -0.5)
            assert np.all(contour[:, 1] <= -25.0)
            assert np.all(contour[:, :1] >= -75.5)

            if altitude is not None:
                assert contour.shape[1] == 3
                assert np.all(contour[:, 2] == altitude)
            else:
                assert contour.shape[1] == 2


@pytest.mark.parametrize("interiors", [True, False])
def test_contour_find_islands(carr: np.ndarray, interiors: bool):
    """Confirm both large islands are found.

    See fixture definition.
    """
    mp = polygon.find_multipolygon(
        carr,
        threshold=0.9,
        min_area=0,
        epsilon=0.05,
        interiors=interiors,
    )
    n_polys = len(mp.geoms)
    assert n_polys == 2
    poly1, poly2 = mp.geoms
    assert not poly1.interiors
    assert not poly2.interiors
    assert len(poly1.exterior.coords) == 9
    assert len(poly2.exterior.coords) == 9

    assert poly1.exterior.is_ccw
    assert poly2.exterior.is_ccw
    assert poly1.area == 149.5
    assert poly2.area == 24.5

    # with high enough min_area, only one island is found
    mp = polygon.find_multipolygon(
        carr,
        threshold=0.9,
        min_area=25,
        epsilon=0.1,
        interiors=interiors,
    )
    assert len(mp.geoms) == 1

    # and if min_area is even higher, no islands are found
    mp = polygon.find_multipolygon(
        carr,
        threshold=0.9,
        min_area=150,
        epsilon=0.1,
        interiors=interiors,
    )
    assert len(mp.geoms) == 0


def test_hawaiian_earrings(hawaiian_earrings: np.ndarray):
    """Show that `find_multipolygon` can find deeply nested contours."""

    mp = polygon.find_multipolygon(hawaiian_earrings, threshold=0.5, min_area=0, epsilon=0.1)
    assert isinstance(mp, shapely.MultiPolygon)
    assert mp.is_valid
    assert len(mp.geoms) == 2

    for poly in mp.geoms:
        assert isinstance(poly, shapely.Polygon)
        assert poly.is_valid
        exterior = poly.exterior
        assert exterior.is_ccw
        assert len(exterior.coords) == 9

        assert len(poly.interiors) == 1
        interior = poly.interiors[0]
        assert isinstance(interior, shapely.LinearRing)
        assert not interior.is_ccw
        assert len(interior.coords) == 9


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
