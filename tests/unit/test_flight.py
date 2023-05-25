"""Test flight module."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod

from pycontrails import Flight, GeoVectorDataset, MetDataArray, MetDataset, VectorDataset
from pycontrails.core import flight
from pycontrails.models.issr import ISSR
from pycontrails.physics import constants, units

from .conftest import get_static_path

##########
# Fixtures
##########


@pytest.fixture(scope="module")
def fl(flight_data: pd.DataFrame, flight_attrs: dict[str, Any]) -> Flight:
    flight_data["constant_numeric_column"] = 1729
    flight_data["constant_poorly_formatted_str_column"] = "   contrails  "
    return Flight(flight_data, attrs=flight_attrs)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(seed=31415)


@pytest.fixture(scope="module")
def random_flight_list(rng: np.random.Generator) -> list[Flight]:
    n = 1000

    def _random_flight() -> Flight:
        return Flight(
            latitude=90 * rng.random(n),
            longitude=180 * rng.random(n),
            altitude=11000 * rng.random(n),
            time=pd.date_range("2019-05-31 00:00:00", "2019-05-31 00:59:00", periods=n),
        )

    return [_random_flight() for _ in range(100)]


##########
# Tests
##########


def test_flight_init(fl: Flight) -> None:
    assert isinstance(fl, GeoVectorDataset)
    assert isinstance(fl.attrs, dict)
    assert isinstance(fl.data, dict)

    assert "longitude" in fl.data
    assert "latitude" in fl.data
    assert "altitude" in fl.data
    assert "time" in fl.data
    assert "constant_numeric_column" in fl.data

    assert "equip" in fl.attrs

    assert isinstance(fl.level, np.ndarray)
    assert isinstance(fl.air_pressure, np.ndarray)


def test_flight_crs(flight_data: pd.DataFrame) -> None:
    fl = Flight(flight_data)

    # crs defaults to "EPSG:4326"
    assert fl.attrs["crs"] == "EPSG:4326"

    # crs maintained as attr
    fl = Flight(flight_data, crs="EPSG:3857")
    assert fl.attrs["crs"] == "EPSG:3857"


def test_flight_creation() -> None:
    """Test Flight.__init__() with unsorted time data."""
    # sorted
    longitude = np.array([0, 1, 2])
    latitude = np.array([0, 1, 2])
    altitude = np.array([11000, 11200, 11300])
    time = np.array(
        [
            np.datetime64("2021-09-01 00:00:00"),
            np.datetime64("2021-08-31 23:00:00"),
            np.datetime64("2021-09-01 01:00:00"),
        ]
    )

    # warn if copy=True and time not sorted
    with pytest.warns(UserWarning, match="Sorting Flight data by time."):
        fl = Flight(longitude=longitude, latitude=latitude, altitude=altitude, time=time)

    assert fl["latitude"][0] == latitude[1]
    assert fl["latitude"][1] == latitude[0]
    assert fl["latitude"][2] == latitude[2]

    assert fl["longitude"][0] == longitude[1]
    assert fl["longitude"][1] == longitude[0]
    assert fl["longitude"][2] == longitude[2]

    assert fl["altitude"][0] == altitude[1]
    assert fl["altitude"][1] == altitude[0]
    assert fl["altitude"][2] == altitude[2]

    assert fl["time"][0] == time[1]
    assert fl["time"][1] == time[0]
    assert fl["time"][2] == time[2]

    # raise if copy=False and time not sorted
    with pytest.raises(ValueError, match="must be sorted"):
        fl = Flight(
            longitude=longitude, latitude=latitude, altitude=altitude, time=time, copy=False
        )


def test_flight_constants(fl: Flight) -> None:
    assert isinstance(fl.constants, dict)
    assert fl.constants["equip"] == "A359"
    assert fl.constants["flight"] == "QR172"
    assert fl.constants["real_to"] == fl.constants["schd_to"]
    assert fl.constants["constant_numeric_column"] == 1729
    assert fl.constants["constant_poorly_formatted_str_column"] == "contrails"


def test_flight_properties(fl: Flight) -> None:
    # test level
    assert isinstance(fl.level, np.ndarray)
    assert np.all(fl.level == units.m_to_pl(fl["altitude"]))

    # test air_pressure
    assert isinstance(fl.air_pressure, np.ndarray)
    assert np.all(fl.air_pressure == 100 * fl.level)

    # constants tested separates in `test_flight_constants`

    # test coords
    assert isinstance(fl.coords, dict)
    assert "longitude" in fl.coords and fl.coords["longitude"] is fl.data["longitude"]
    assert "latitude" in fl.coords and fl.coords["latitude"] is fl.data["latitude"]
    assert "time" in fl.coords and fl.coords["time"] is fl.data["time"]
    assert "level" in fl.coords and np.all(fl.coords["level"] == fl.level)


def test_flight_hashable(fl: Flight) -> None:
    assert isinstance(fl.hash, str)
    assert fl.hash == "f78fcb6235a6ba3a0e0dc418034736b2a3618720"


def test_flight_empty() -> None:
    fl = Flight.create_empty(attrs=dict(equip="A359"))

    assert isinstance(fl.data, dict)
    assert "longitude" in fl.data and len(fl.data["longitude"]) == 0
    assert "latitude" in fl.data and len(fl.data["latitude"]) == 0
    assert "altitude" in fl.data and len(fl.data["altitude"]) == 0
    assert "time" in fl.data and len(fl.data["time"]) == 0
    assert len(fl.air_pressure) == 0
    assert len(fl.level) == 0

    assert fl.attrs["crs"] == "EPSG:4326"
    assert isinstance(fl.constants, dict) and fl.constants == fl.attrs

    assert not fl


def test_flight_validate(flight_data: pd.DataFrame) -> None:
    """Test altitude warning and coordinate errors in Flight __init__."""
    df = flight_data.copy()
    df["altitude"] = 17000
    match = (
        "Flight altitude is high. Expected altitude unit is meters. "
        "Found waypoint with altitude 17000 m."
    )
    with pytest.warns(UserWarning, match=match):
        Flight(df)

    match = (
        "Flight altitude is high for flight xyz. Expected altitude unit is meters. "
        "Found waypoint with altitude 17000 m."
    )
    with pytest.warns(UserWarning, match=match):
        Flight(df, flight_id="xyz")

    df = flight_data.copy()
    df["longitude"] = 200
    with pytest.raises(ValueError, match="EPSG:4326 longitude coordinates"):
        Flight(df)

    df = flight_data.copy()
    df["latitude"] = 101
    with pytest.raises(ValueError, match="EPSG:4326 latitude coordinates"):
        Flight(df)


def test_flight_fitting() -> None:
    df = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    df.rename(columns={"altitude_baro": "altitude_ft", "timestamp": "time"}, inplace=True)
    df["time"] = df["time"].dt.tz_localize(None)
    flight = Flight(df[df["callsign"] == "BAW506"], drop_duplicated_times=True)
    f_copy = flight.copy()
    smoothed_flight = flight.fit_altitude()

    assert abs(smoothed_flight["altitude_ft"][1100] - 37000) < 1e-6
    assert abs(smoothed_flight["altitude_ft"][1200] - 37000) < 1e-6
    assert abs(smoothed_flight["altitude_ft"][1300] - 37000) < 1e-6
    assert f_copy == flight


def test_flight_eq(fl: Flight) -> None:
    # classes should be equal
    fl2 = Flight(fl.data, attrs=fl.attrs)
    fl3 = Flight(fl.data, attrs=fl.attrs)
    assert fl2.data is not fl3.data
    assert fl2.attrs is not fl3.attrs
    assert fl2 == fl3

    # data should be compared elementwise
    fl4 = Flight(fl.data, attrs=fl.attrs)
    fl4.data["altitude"][3] = 10
    assert fl2 != fl4

    # attributes with ndarrays should be compared elementwise
    fl2.attrs["np attr"] = np.array([1, 2, 3])
    fl3.attrs["np attr"] = np.array([1, 2, 3])
    assert fl2 == fl3

    fl5 = Flight(fl.data, attrs=fl.attrs)
    fl5.attrs["np attr"] = np.array([1, 2, 4])
    assert fl2 != fl5

    # nan values okay
    fl2.data["altitude"][3] = np.nan
    fl3.data["altitude"][3] = np.nan
    assert fl2 == fl3


def test_flight_repr(fl: Flight) -> None:
    s = repr(fl)
    assert isinstance(s, str)
    assert "Keys:" in s
    assert "Attributes:" in s
    assert "time" in s
    assert "longitude" in s
    assert "latitude" in s
    assert "altitude" in s
    assert "equip" in s
    assert "flight" in s
    assert "real_to" in s
    assert "constant_numeric_column" not in s
    for key in fl.data:
        assert key in fl._repr_html_()


def test_ensure_vars(fl: Flight) -> None:
    assert fl.ensure_vars(["time", "latitude", "longitude", "equip", "flight"])
    with pytest.raises(KeyError, match="Flight instance does not contain data or attr"):
        fl.ensure_vars(["time2"])

    is_var = fl.ensure_vars(["time2"], raise_error=False)
    assert not is_var


def test_plot(fl: Flight) -> None:
    ax = fl.plot()
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.get_xlabel() == "longitude"
    assert ax.get_ylabel() == "latitude"

    # kwargs
    assert isinstance(fl.plot(figsize=(20, 20)), matplotlib.axes.Axes)
    assert isinstance(fl.plot(alpha=0.5), matplotlib.axes.Axes)
    assert isinstance(fl.plot(marker="o", linestyle=""), matplotlib.axes.Axes)


def test_flight_time_methods(fl: Flight) -> None:
    assert isinstance(fl.duration, pd.Timedelta)
    assert isinstance(fl.max_time_gap, pd.Timedelta)
    assert fl.max_time_gap < fl.duration


def test_flight_length_methods(fl: Flight) -> None:
    assert isinstance(fl.length, float)
    x0, y0 = fl.data["longitude"][0], fl.data["latitude"][0]
    x1, y1 = fl.data["longitude"][-1], fl.data["latitude"][-1]

    # comparing length against geodesic from start to end
    geod = Geod(ellps="WGS84")
    crow_flies_dist = geod.line_length([x0, x1], [y0, y1]) / 1000
    assert crow_flies_dist <= fl.length

    assert isinstance(fl.max_distance_gap, float)
    assert fl.max_distance_gap < fl.length


def test_flight_filtering_methods(flight_data: pd.DataFrame, flight_attrs: dict[str, Any]) -> None:
    data = flight_data.copy()
    data.loc[100] = data.loc[101]
    data.loc[51] = data.loc[52]
    with pytest.warns(UserWarning, match="Flight contains 2 duplicate times"):
        fl = Flight(data=data, attrs=flight_attrs)
    fl2 = fl.filter_by_first()
    assert fl.attrs == fl2.attrs
    assert fl.duration == fl2.duration
    assert fl.max_time_gap <= fl2.max_time_gap
    assert len(fl) - len(fl2) == 2
    for col in fl.data:
        assert set(fl.data[col]) == set(fl2.data[col])

    fl3 = fl2.resample_and_fill("10S")
    assert fl3.max_time_gap == pd.Timedelta("10S")
    assert np.all(np.diff(fl3.data["time"]) == pd.Timedelta("10S"))
    assert set(fl3.data.keys()) == {"time", "latitude", "longitude", "altitude"}


def test_resampling(fl: Flight, flight_meridian: Flight) -> None:
    fl2 = fl.resample_and_fill("10S")
    fl3 = fl.resample_and_fill("10T")
    # flight with more waypoints will waver more, and have longer length
    assert len(fl2) > len(fl3)

    # fl2 has 60 times more data
    assert abs(len(fl3) - len(fl2) / 60) <= 1

    # at large threshold, geodesics are not calculated
    fl4 = fl.resample_and_fill("1T", "linear")
    fl5 = fl.resample_and_fill("1T", "geodesic", 1000e3)
    assert fl4 == fl5

    # test resample over the meridian
    # all resampled waypoints should be < 1 deg of longitude apart on [0, 360] scale
    fl6 = flight_meridian.resample_and_fill("1T")
    assert np.all(np.abs(np.diff(fl6["longitude"] % 360)) < 1)

    fl7 = flight_meridian.resample_and_fill("3T")
    assert np.all(np.abs(np.diff(fl7["longitude"] % 360)) < 1)

    fl8 = flight_meridian.resample_and_fill("5T")
    assert np.all(np.abs(np.diff(fl8["longitude"] % 360)) < 2)


def test_geodesic_interpolation(fl: Flight) -> None:
    fl2 = fl.resample_and_fill("1T", "geodesic", geodesic_threshold=1e3)
    fl3 = fl.resample_and_fill("1T", "geodesic", geodesic_threshold=200e3)
    # fine geodesic interpolation will beat linear interpolation for minimizing spherical distance
    assert fl2.length < fl3.length
    with pytest.raises(ValueError, match="Unknown `fill_method`"):
        fl.resample_and_fill(fill_method="nearest")


def test_altitude_interpolation(fl: Flight) -> None:
    # check default flight
    fl1 = fl.resample_and_fill("10S")
    fl2 = fl.resample_and_fill("10T")

    def _check_rocd(_fl: Flight, nominal_rocd: float = constants.nominal_rocd) -> np.bool_:
        """Check rate of climb/descent."""
        dt = np.diff(_fl["time"], append=np.datetime64("NaT")) / np.timedelta64(1, "s")
        dalt = np.diff(_fl.altitude, append=np.nan)
        rocd = np.abs(dalt / dt)
        return np.all(rocd[:-1] < 2 * nominal_rocd)

    # confirm that all d_altitude values are < 2 * the nominal rate of climb/descent
    assert _check_rocd(fl1)
    assert _check_rocd(fl2)

    # test more aggresive altitude interpolation
    fl_alt = Flight(
        longitude=np.linspace(0, 10, 10),
        latitude=np.linspace(0, 10, 10),
        time=pd.date_range("2000-01-01 00:00:00", "2000-01-01 05:00:00", periods=10),
        altitude=np.array([0, 11000, 5000, 9000, 9000, 11000, 5000, 11000, 11000, 0]),
    )

    # resample to 1 min
    fl10 = fl_alt.resample_and_fill("1T")

    # confirm that rocd is appropriate
    assert _check_rocd(fl10)

    # resample to 5 minutes
    # confirm that all values exist in previous resampling
    fl11 = fl_alt.resample_and_fill("5T")
    assert _check_rocd(fl11)
    assert np.in1d(fl11["altitude"], fl10["altitude"]).all()

    # test altitude interpolation with level
    fl_lev = Flight(
        longitude=np.linspace(0, 10, 10),
        latitude=np.linspace(0, 10, 10),
        time=pd.date_range("2000-01-01 00:00:00", "2000-01-01 05:00:00", periods=10),
        level=np.array([1000, 200, 300, 300, 250, 250, 300, 200, 200, 1000]),
    )

    # resample to 1 min
    fl13 = fl_lev.resample_and_fill("1T")
    assert "level" not in fl13
    assert _check_rocd(fl13)

    # test nominal rocd
    fl14 = fl_alt.resample_and_fill("1T", nominal_rocd=30)
    assert _check_rocd(fl14, nominal_rocd=30)

    # test warning with low nominal rocd
    with pytest.warns(UserWarning, match="Rate of climb/descent values greater than nominal"):
        fl15 = fl_alt.resample_and_fill("1T", nominal_rocd=1)
        assert not _check_rocd(fl15, nominal_rocd=1)

    # test `drop` kwarg
    fl_alt["extrakey"] = np.linspace(0, 10, 10)
    fl_alt["level"] = fl_alt.level
    fl16 = fl_alt.resample_and_fill("1T", drop=False)
    assert "extrakey" in fl16 and np.any(np.isnan(fl16["extrakey"]))
    assert "level" not in fl16

    fl17 = fl_alt.resample_and_fill("1T")
    assert "extrakey" not in fl17


def test_geojson_methods(fl: Flight, rng: np.random.Generator) -> None:
    d1 = fl.to_geojson_points()
    assert d1["type"] == "FeatureCollection"
    for feature in d1["features"]:
        assert feature["geometry"]["type"] == "Point"
        assert len(feature["geometry"]["coordinates"]) == 3
    assert json.dumps(d1)

    d2 = fl.to_geojson_linestring()
    assert d2["type"] == "FeatureCollection"
    assert json.dumps(d2)
    assert len(d1["features"]) == len(d2["features"][0]["geometry"]["coordinates"])
    for feature, coord in zip(d1["features"], d2["features"][0]["geometry"]["coordinates"]):
        assert feature["geometry"]["coordinates"] == coord

    with pytest.raises(KeyError):
        fl.to_geojson_multilinestring("bad key")
    fl["issr"] = rng.integers(0, 2, len(fl))
    d3 = fl.to_geojson_multilinestring("issr")
    assert d3["type"] == "FeatureCollection"
    assert len(d3["features"]) == 2

    coords1 = [f["geometry"]["coordinates"] for f in d3["features"]]
    # flattening, casting to tuple to allow inclusion into a set
    coords1 = [tuple(c) for c2 in coords1 for c1 in c2 for c in c1]
    coords2 = [tuple(c) for c in d2["features"][0]["geometry"]["coordinates"]]
    assert set(coords1) == set(coords2)


# ignoring the warnings that come with traffic
# NOTE: this does not work on docker, so we skip
def _is_docker() -> bool:
    path1 = pathlib.Path("/proc/self/cgroup")
    path2 = pathlib.Path("/.dockerenv")
    return (path1.is_file() and any("docker" in line for line in open(path1))) or path2.is_file()


def _is_traffic_installed() -> bool:
    try:
        import traffic  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


@pytest.mark.skipif(not _is_traffic_installed(), reason="Traffic is no longer optional dependency")
def test_to_traffic(fl: Flight) -> None:
    tr = fl.to_traffic()
    # NOTE: tf and fl have inconsistent column names (time and timestamp), so just ensuring
    # the values were copied correctly on tr instantiation.
    np.testing.assert_array_equal(fl.dataframe.to_numpy(), tr.data.to_numpy())
    assert fl.duration == tr.duration


def test_crs(fl: Flight, met_issr: MetDataset, rng: np.random.Generator) -> None:
    fl2 = fl.to_pseudo_mercator()
    assert fl.attrs["crs"] == "EPSG:4326"
    assert fl2.attrs["crs"] == "EPSG:3857"

    assert np.all(fl.data["altitude"] == fl2.data["altitude"])
    assert np.all(fl.data["time"] == fl2.data["time"])
    assert np.all(fl.data["latitude"] != fl2.data["latitude"])
    assert np.all(fl.data["longitude"] != fl2.data["longitude"])

    with pytest.raises(NotImplementedError):
        fl2.length
    fl2.update(issr=rng.integers(0, 2, len(fl2)))
    with pytest.raises(NotImplementedError, match="Only implemented for EPSG:4326"):
        fl2.length_met("issr")
    with pytest.raises(NotImplementedError, match="Only implemented for EPSG:4326"):
        fl2.proportion_met("issr")
    with pytest.raises(AttributeError, match="has no attribute 'interpolate'"):
        fl2.intersect_met(met_issr)

    with pytest.raises(KeyError, match="Column key does not exist in data"):
        fl.length_met("key")  # key not in fl.data


def test_antimeridian_jump() -> None:
    df = pd.DataFrame(
        {
            "longitude": [-177, -179, 179, 178],
            "latitude": [40, 41, 42, 41],
            "time": pd.date_range("2000-01-01 00:00:00", "2000-01-01 00:30:00", periods=4),
            "altitude": [10000, 10001, 10003, 10006],
            "issr": [0, 1, 1, 1],
        }
    )

    fl = Flight(df)
    d = fl.to_geojson_multilinestring("issr", split_antimeridian=True)
    issr_feature = d["features"][1]
    assert len(issr_feature["geometry"]["coordinates"]) == 2

    d = fl.to_geojson_multilinestring("issr", split_antimeridian=False)
    issr_feature = d["features"][1]
    assert len(issr_feature["geometry"]["coordinates"]) == 1

    fl = fl.to_pseudo_mercator()
    d = fl.to_geojson_multilinestring("issr", split_antimeridian=True)
    issr_feature = d["features"][1]
    assert len(issr_feature["geometry"]["coordinates"]) == 2

    d = fl.to_geojson_multilinestring("issr", split_antimeridian=False)
    issr_feature = d["features"][1]
    assert len(issr_feature["geometry"]["coordinates"]) == 1

    df["longitude"] = [-177, -179, 179, -178]
    fl = Flight(df)
    # jumps antimeridian twice
    with pytest.raises(ValueError):
        fl.to_geojson_multilinestring("issr", split_antimeridian=True)


def test_segment_properties(fl: Flight, rng: np.random.Generator) -> None:
    # segment duration
    segment_duration = fl.segment_duration()
    assert len(segment_duration) == len(fl)
    assert np.isnan(segment_duration[-1])
    assert 1 <= np.nanmax(segment_duration) <= 198

    # segment lengths
    segment_length = fl.segment_length()
    assert len(segment_length) == len(fl)
    assert np.isnan(segment_length[-1])
    assert np.nanmax(segment_length) <= 5.1e4

    # segment angles
    sin_a, cos_a = fl.segment_angle()
    assert len(sin_a) == len(fl)
    assert len(cos_a) == len(fl)
    assert np.all(sin_a[:-1] <= 1)
    assert np.all(sin_a[:-1] >= -1)
    assert np.all(cos_a[:-1] <= 1)
    assert np.all(cos_a[:-1] >= -1)
    assert np.isnan(sin_a[-1])
    assert np.isnan(cos_a[-1])

    # segment ground airspeed
    segment_groundspeed = fl.segment_groundspeed()
    assert len(segment_groundspeed) == len(fl)
    assert np.nanmin(segment_groundspeed) > 0 and np.nanmax(segment_groundspeed) <= 300
    assert np.isnan(segment_groundspeed[-1])

    # segment true airspeed
    segment_true_airspeed = fl.segment_true_airspeed(smooth=False)
    assert len(segment_true_airspeed) == len(fl)
    assert np.isnan(segment_true_airspeed[-1])
    abs_diff = np.abs(segment_true_airspeed - segment_groundspeed)
    # should be similar with no wind speed
    assert np.all(abs_diff[:-1] < 1)

    segment_true_airspeed = fl.segment_true_airspeed()
    assert len(segment_true_airspeed) == len(fl)
    assert np.isnan(segment_true_airspeed[-1])
    abs_diff = np.abs(segment_true_airspeed - segment_groundspeed)
    assert np.all(abs_diff[:-1] <= 50)

    u_wind = rng.random(len(fl)) * 10
    v_wind = rng.random(len(fl)) * 5
    segment_true_airspeed = fl.segment_true_airspeed(u_wind=u_wind, v_wind=v_wind)
    assert len(segment_true_airspeed) == len(fl)
    assert np.isnan(segment_true_airspeed[-1])
    assert np.all(abs_diff[:-1] <= 50)

    # mach number
    air_temperature = rng.random(len(fl)) * 30 + 273
    segment_mach_number = fl.segment_mach_number(segment_true_airspeed, air_temperature)
    assert len(segment_mach_number) == len(fl)
    assert np.all(segment_mach_number[:-1] < 2)


def test_downselect_met(fl: Flight, met_issr: MetDataset) -> None:
    # met_issr is session scoped and we're about to mutate it
    met = met_issr.copy()

    # adjust met level to get more overlap
    np.testing.assert_array_equal(met["level"].values, [225, 250, 300])
    met.update(level=[150, 300, 500])
    np.testing.assert_array_equal(met["level"].values, [150, 300, 500])
    np.testing.assert_array_equal(met["air_temperature"].data["level"], [150, 300, 500])

    # clip flight domain so it is definitely smaller than met
    mask = (fl.level < 500) & (fl.level > 150)
    fl = fl.filter(mask)

    assert tuple(met.data.sizes.values()) == (15, 8, 3, 2)
    met_downselected = fl.downselect_met(met)

    # original not mutated
    assert tuple(met.data.sizes.values()) == (15, 8, 3, 2)
    assert tuple(met_downselected.data.sizes.values()) == (3, 3, 3, 2)
    assert met_downselected is not met
    # name convenience
    ds = met_downselected.data

    # check the downselection
    assert ds["longitude"].max() == 75
    assert ds["longitude"].values[-2] < fl["longitude"].max() < ds["longitude"].values[-1]
    assert ds["latitude"].max() == 65
    assert ds["latitude"].values[-2] < fl["latitude"].max() < ds["latitude"].values[-1]
    assert ds["level"].max() == 500
    assert ds["level"].values[-2] < fl.level.max() < ds["level"].values[-1]
    assert ds["time"].max() == np.datetime64("2019-05-31T06:00:00")

    assert ds["longitude"].min() == 25
    assert ds["longitude"].values[0] < fl["longitude"].min() < ds["longitude"].values[1]
    assert ds["latitude"].min() == 15
    assert ds["latitude"].values[0] < fl["latitude"].min() < ds["latitude"].values[1]
    assert ds["level"].min() == 150
    assert ds["level"].values[0] < fl.level.min() < ds["level"].values[1]
    assert ds["time"].min() == np.datetime64("2019-05-31T05:00:00")
    assert ds["time"].values[0] < fl["time"].min() < ds["time"].values[1]


@pytest.mark.parametrize("freq", ["10S", "1T", "10T"])
@pytest.mark.parametrize("shift", [-180, 0])
def test_interpolation_edge_cases(shift: int, freq: str) -> None:
    """Test interpolation adjustments for antimeridian crossing.

    This test probes both the core logic of `resample_and_fill` as well as
    the logic in `_altitude_interpolation`.

    The parameter `shift` translates the longitude coordinates. In the case of
    `shift = 0` and `shift=6`, the flight crosses the antimeridian.

    The parameter `freq` various widely, allowing for both up and down sampling.
    """
    # Using level with many missing values
    fl = Flight(
        longitude=(np.arange(-10, 10, dtype=float) % 360 + shift) % 360 - 180,
        latitude=np.arange(20),
        level=np.r_[200, [np.nan] * 5, 300, 305, 309, 313, 311, [np.nan] * 7, 189, 188],
        time=pd.date_range("2022-01-01", "2022-01-01T03", 20),
    )
    fl["longitude"][3] = np.nan  # make test more interesting
    fl2 = fl.resample_and_fill(freq)
    jump = np.nonzero(np.diff(fl2["longitude"]) < 0)[0]
    if shift == -180:
        assert jump.size == 0
    elif shift == 0:
        assert jump.size == 1

    # Interpolated altitudes should increase then decrease
    np.diff(np.sign(np.diff(fl2.altitude))).nonzero()[0].size == 1

    assert not fl2.dataframe.isna().any().any()


@pytest.mark.parametrize("direction", ["east", "west"])
@pytest.mark.parametrize("cross_anti", [True, False])
def test_antimeridian_long_cross(direction: str, cross_anti: bool) -> None:
    """Test interpolation adjustments for antimeridian crossing.

    This test uses longitude values spanning at least three quadarants AND
    crossing the antimeridian.
    """
    # Intentionally keep the logic very boilerplate
    # In all cases, d_lon = 170
    n = 100
    if direction == "east" and cross_anti:
        longitude = np.linspace(80, 80 + 170, n)
    elif direction == "east" and not cross_anti:
        longitude = np.linspace(-40, -40 + 170, n)
    elif direction == "west" and cross_anti:
        longitude = np.linspace(190, 190 - 170, n)
    elif direction == "west" and not cross_anti:
        longitude = np.linspace(60, 60 - 170, n)
    else:
        raise ValueError("Unknown param")
    longitude = (longitude + 180) % 360 - 180
    fl = Flight(
        longitude=longitude,
        latitude=np.zeros(n),
        altitude=np.full(n, 10000),
        time=pd.date_range("2022-01-01", "2022-01-01T03", n),
    )
    fl2 = fl.resample_and_fill("5S")
    assert np.all(fl2["longitude"] < 180)
    assert np.all(fl2["longitude"] >= -180)
    assert np.all(np.isfinite(fl2["longitude"]))
    assert np.all(fl2.segment_length()[:-1] < 10000)
    assert np.all(fl2.segment_length()[:-1] > 8000)


def test_intersect_issr_met(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test `intersect_met` and `length_met` methods."""
    issr = ISSR(met_era5_fake).eval()
    fl = flight_fake.copy()
    fl["issr"] = fl.intersect_met(issr)

    # at least one waypoint with ISSR
    assert np.nansum(fl["issr"])

    # somes ISSR values are interpolated between 0 and 1
    assert np.any((fl["issr"] > 0) & (fl["issr"] < 1))
    with pytest.warns(UserWarning, match="Column issr contains real numbers between 0 and 1."):
        fl.length_met("issr")

    # relaxing threshold
    assert fl.length_met("issr", threshold=0.5)

    # waypoints below lowest-altitude level in era5 get filled with NaN
    low_altitude_waypoints = fl["altitude"] < met_era5_fake.data["altitude"].min().item()
    assert np.isnan(fl["issr"][low_altitude_waypoints]).all()
    assert ~np.isnan(fl["issr"][~low_altitude_waypoints]).all()

    # based on interpolation, we end up with an issr value exactly when flight waypoint
    # longitude value is within 1 of a multiple of 5
    lons = fl["longitude"][~low_altitude_waypoints]
    expect_issr = np.abs((lons + 2.5) % 5 - 2.5) < 1
    actual_issr = fl["issr"][~low_altitude_waypoints].astype(bool)
    assert np.all(actual_issr == expect_issr)


def test_intersect_met_method_nearest(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test `intersect_met` with "nearest" interpolation."""
    fl = flight_fake.copy()

    # results agree when calling intersect_met and intersect_model with method='nearest'
    issr = ISSR(met_era5_fake).eval()
    fl["issr"] = fl.intersect_met(issr, method="nearest", fill_value=0)

    # values in issr column are either 0 or 1
    assert np.all(np.isin(fl["issr"], [0, 1]))
    assert fl["issr"][0] == 0

    # extrapolates values when "nearest" and fill_value set to None
    fl.update(issr=fl.intersect_met(issr, method="nearest", fill_value=None))
    assert np.all(np.isin(fl["issr"], [0, 1]))
    assert fl["issr"][0] == 1


def test_intersect_met_method_linear(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test `intersect_met` with "linear" interpolation."""
    fl = flight_fake.copy()

    # results disagree when calling intersect_met and intersect_model with default method='linear'
    issr = ISSR(met_era5_fake).eval()
    fl["issr"] = fl.intersect_met(issr, method="linear")

    # values in fl2 issr column are floating point values
    assert (fl["issr"][~np.isnan(fl["issr"])] < 1).all()


def test_intersect_other_mets(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    fl = flight_fake.copy()

    da = met_era5_fake["specific_humidity"]
    fl["specific_humidity"] = fl.intersect_met(da)
    assert "specific_humidity" in fl.data

    # creating a meaningless DataArray that has the right dimension variables
    da = (
        da.data["longitude"]
        + da.data["altitude"]
        + da.data["latitude"]
        + 0 * da.data["time"].astype(int)
    )

    fl.data["meaningless"] = fl.intersect_met(MetDataArray(da))

    assert "specific_humidity" in fl
    assert "meaningless" in fl
    # 5 geospatial columns, specific_humidity, and meaningless
    assert len(fl.data) == 6

    cruising_altitude_waypoints = fl["altitude"] >= met_era5_fake.data["altitude"].min().item()

    # altitude is the dominate term in the meaningless column
    assert (
        (fl["meaningless"][cruising_altitude_waypoints] > 7500)
        & (fl["meaningless"][cruising_altitude_waypoints] < 13500)
    ).all()


def test_cast_to_vector(flight_fake: Flight) -> None:
    """Ensure casting works as expected."""
    # Casting down
    v1 = VectorDataset(flight_fake)
    assert isinstance(v1, VectorDataset)
    assert not isinstance(v1, Flight)
    assert v1 == flight_fake

    v1["longitude"][5] = np.nan
    assert v1 != flight_fake

    v2 = GeoVectorDataset(flight_fake)
    assert isinstance(v2, GeoVectorDataset)
    assert not isinstance(v2, Flight)
    assert v2 == flight_fake

    # Casting up
    lon = v1.data.pop("longitude")
    with pytest.raises(KeyError, match="Missing required data keys"):
        GeoVectorDataset(v1)

    v1["longitude"] = lon
    v3 = GeoVectorDataset(v1)
    assert isinstance(v3, GeoVectorDataset)
    assert v3 == v1


def test_flight_duplicated_times(flight_fake: Flight) -> None:
    """Ensure that duplicate times are removed."""
    fl = flight_fake.copy()
    fl["time"][3] = fl["time"][2]
    with pytest.warns(UserWarning, match="Flight contains 1 duplicate times"):
        Flight(fl.data)

    fl = Flight(fl.data, drop_duplicated_times=True)
    assert fl.size == flight_fake.size - 1

    # We kept the 2nd row and the 3rd is different
    pd.testing.assert_series_equal(fl.dataframe.iloc[2], flight_fake.dataframe.iloc[2])
    pd.testing.assert_series_equal(
        fl.dataframe.iloc[3], flight_fake.dataframe.iloc[4], check_names=False
    )


@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9, 11, 13, 15, 17])
def test_filter_altitude(kernel_size: int) -> None:
    """Check that noise in cruise altitude is removed."""

    altitude_ft = np.array([40000, 39975, 40000, 40000, 39975, 40000, 40000, 40025, 40025, 40000])
    altitude_cleaned = flight.filter_altitude(altitude_ft, kernel_size=kernel_size)
    assert altitude_cleaned.size == altitude_ft.size

    np.testing.assert_array_equal(altitude_cleaned[:-1], 40000.0)

    # Final waypoint has altitude 40002 +/- 2 ft
    assert altitude_cleaned[-1] == pytest.approx(40002, abs=2.5)


# Compare with OpenAP implementation
try:
    from openap import FlightPhase

    OPENAP_AVAILABLE = True
except ImportError:
    OPENAP_AVAILABLE = False


@pytest.mark.skipif(not OPENAP_AVAILABLE, reason="OpenAP not installed")
def test_compare_segment_phase() -> None:
    """Compare flight phase algorithm with OpenAP.

    See https://github.com/junzis/openap/blob/master/test/test_phase.py
    """
    df = pd.read_csv(
        "https://raw.githubusercontent.com/junzis/openap/master/test/data/flight_phlab.csv"
    )

    ts0 = df["ts"].values
    ts0 = ts0 - ts0[0]
    alt0 = df["alt"].values
    spd0 = df["spd"].values
    roc0 = df["roc"].values

    ts = np.arange(0, ts0[-1], 1)
    alt = np.interp(ts, ts0, alt0)
    spd = np.interp(ts, ts0, spd0)
    roc = np.interp(ts, ts0, roc0)

    fp = FlightPhase()
    fp.set_trajectory(ts, alt, spd, roc)
    labels = fp.phaselabel()

    assert labels

    Flight(data=df.rename({"alt": "altitude", "lat": "latitude", "lon": "longitude"}))
