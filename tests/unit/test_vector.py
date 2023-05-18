"""Test vector module."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pycontrails import GeoVectorDataset, MetDataset, VectorDataset
from pycontrails.core.vector import AttrDict, VectorDataDict


@pytest.fixture(scope="module")
def random_path() -> VectorDataset:
    """Return meaningless `VectorDataset` fixture."""
    rng = np.random.default_rng(123)
    data = {
        "a": rng.random(100),
        "b": rng.random(100),
        "c": np.arange(100),
    }
    attrs = {"test": "attribute"}
    return VectorDataset(data=data, attrs=attrs, copy=False)


@pytest.fixture(scope="module")
def random_geo_path() -> GeoVectorDataset:
    """Demo GeoVectorDataset.

    Returns
    -------
    GeoVectorDataset
    """
    data = {
        "longitude": np.array([-15, -14, -13, -12]),
        "latitude": np.array([10, 11, 12, 13]),
        "time": np.array(
            [
                "2021-10-05 00:00:00",
                "2021-10-05 01:00:00",
                "2021-10-05 02:00:00",
                "2021-10-05 03:00:00",
            ]
        ),
        "altitude": np.array([1000, 5000, 10000, 40000]),
        "a": np.array([1, 2, 3, 4]),
        "b": np.array([5, 6, 7, 8]),
    }
    attrs = {"test": "attribute"}
    with warnings.catch_warnings():
        # Pandas has no trouble doing the conversion
        warnings.filterwarnings("ignore", "Time data is not np.datetime64")
        return GeoVectorDataset(data=data, attrs=attrs, copy=False)


def test_vector_init() -> None:
    """VectorDataset __init__."""

    # test constructor
    data = {
        "a": np.ones(100),
        "b": np.empty(100),
    }
    attrs = {
        "test": "value",
    }
    vds = VectorDataset(data, attrs=attrs)
    assert isinstance(vds.data, VectorDataDict)
    assert isinstance(vds.attrs, AttrDict)

    # copy by default
    assert vds.data["a"] is not data["a"]
    assert np.all(vds.data["a"] == data["a"])

    assert vds.attrs is not attrs
    assert vds.attrs["test"] == attrs["test"]

    # it will acceptVectorDataDict/AttrDict input, but still copy
    data = VectorDataDict(
        {
            "a": np.ones(100),
            "b": np.empty(100),
        }
    )
    attrs = AttrDict(
        {
            "test": "value",
        }
    )
    vds = VectorDataset(data, attrs=attrs)
    assert vds.data is not data
    assert vds.attrs is not attrs

    vds = VectorDataset(data, attrs=attrs, copy=False)
    assert vds.data is data
    assert vds.attrs is attrs


def test_bad_size_pass_data() -> None:
    """Ensure an error is raised when inconsistent sizes are passed in the constructor."""
    data = {
        "a": np.arange(5),
        "b": np.arange(6),
        "c": np.arange(5),
    }
    with pytest.raises(ValueError, match="sizes"):
        VectorDataset(data)

    del data["b"]
    vds1 = VectorDataset(data)
    assert vds1.size == 5

    vds2 = VectorDataset()
    vds2["a"] = data["a"]
    vds2["c"] = data["c"]
    assert vds2.size == 5
    assert vds1 == vds2


def test_vector_empty() -> None:
    """Confirm an empty `VectorDataset` instance behaves as expected."""
    attrs = {"test": "attribute"}
    vds = VectorDataset(attrs=attrs)

    assert vds.data == {}
    assert vds.attrs["test"] == "attribute"
    assert vds.size == 0
    assert "0 keys" in repr(vds)
    assert "VectorDataset" in repr(vds)

    vds["orcas"] = np.ones(10)
    assert vds
    assert vds.size == 10
    assert "orcas" in repr(vds)
    assert "VectorDataset" in repr(vds)

    # deleting that last key sets the size back to 0
    del vds["orcas"]
    assert vds.size == 0


def test_vector_dict_like(random_path: VectorDataset) -> None:
    """Test vector dictlike methods"""
    vds = random_path.copy()

    # test default dict methods
    # get
    assert vds["a"] is vds.data["a"]
    assert vds.get("a", None) is vds.data["a"]
    assert vds.get("d", None) is None

    # set
    d = np.random.random(vds["a"].size)
    vds["d"] = d
    assert vds.data["d"] is d
    with pytest.raises(ValueError, match="sizes"):
        vds["e"] = np.empty(5)

    # del
    del vds["d"]
    assert "d" not in vds.data

    # update
    vds.update({"d": d})
    assert vds.data["d"] is d
    vds.update(e=d)
    assert vds.data["e"] is d
    # (`kwargs`` overwrites `other``)
    vds.update({"d": np.zeros_like(d)}, e=np.ones_like(d), d=np.ones_like(d))
    assert np.all(vds.data["d"] == 1)
    assert np.all(vds.data["e"] == 1)

    with pytest.raises(ValueError, match="sizes"):
        vds.update(d=np.empty(5))
    with pytest.raises(ValueError, match="sizes"):
        vds.data.update(d=np.empty(5))

    # setdefault
    out = vds.setdefault("d", d)
    assert np.all(vds.data["d"] == 1)
    assert vds.data["d"] is out
    out = vds.setdefault("f", d)
    assert np.all(vds.data["f"] == d)
    assert vds.data["f"] is out

    with pytest.raises(ValueError, match="sizes"):
        out = vds.setdefault("g", np.empty(5))

    with pytest.raises(ValueError, match="sizes"):
        out = vds.setdefault("g")

    # setdefault overwrites None values
    # out = vds["g"] = None

    # iter
    assert list(vds) == list(vds.data)


def test_vector_attrs(random_path: VectorDataset) -> None:
    """Vector attr tests."""
    vds = random_path.copy()

    # setdefault has special behavior to overwrite None
    vds.attrs["none"] = None
    default = vds.attrs.setdefault("none", "value")
    assert vds.attrs["none"] == "value" and default == "value"


def test_vector_not_dict_like(random_path: VectorDataset) -> None:
    """Test un-dict-like methods."""

    vds = random_path.copy()

    # len, size, shape
    assert len(vds) == 100
    assert vds.size == 100
    assert vds.shape == (100,)

    # bool
    # when all data is deleted, the size should be reset
    del vds["a"]
    del vds["b"]
    assert len(vds) == 100
    assert vds

    del vds["c"]
    assert not vds
    assert len(vds) == 0
    assert vds.size == 0

    # eq
    vds = random_path.copy()
    vds2 = random_path.copy()
    assert vds == vds2
    vds.update(a=np.ones_like(vds["a"]))
    assert vds != vds2


def test_vector_repr() -> None:
    """Confirm `VectorDataset.__repr__` clips very long attributes."""
    attrs = {"short_key": "short_val", "long_key": "long_val" * 100}
    vds = VectorDataset(attrs=attrs)
    out = repr(vds)

    assert "short_key" in out
    assert "long_key" in out
    assert "short_val" in out

    # The very long value gets truncated
    assert out.endswith("...")
    assert "long_val" * 5 in out
    assert "long_val" * 10 not in out


def test_warning_on_set(random_path: VectorDataset) -> None:
    """Test the warning on __setitem__() and set_attr()"""

    # should not warn on empty and the same value object
    vds = VectorDataset()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        val = np.ones(10)
        vds["orcas"] = val
        vds["orcas"] = val  # won't warn for the same object

        # shouldn't warn on data either
        vds.data["orcas"] = val

        # shouldn't warn when `update` is used
        vds.update(orcas=np.zeros(10))
        vds.update({"orcas": np.empty(10)})
        vds.data.update(orcas=np.zeros(10))

    # WILL warn for the same values, different object
    with pytest.warns(UserWarning, match="Overwriting data"):
        vds["orcas"] = np.ones(10)
    with pytest.warns(UserWarning, match="Overwriting data"):
        vds.data["orcas"] = np.ones(10)

    # WILL warn for different values too
    with pytest.warns(UserWarning, match="Overwriting data"):
        vds["orcas"] = np.zeros(10)
    with pytest.warns(UserWarning, match="Overwriting data"):
        vds.data["orcas"] = np.zeros(10)

    # should not warn on None attr and the same value object
    vds.attrs["test"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        val = "value"
        vds.attrs["test"] = val
        vds.attrs["test"] = val  # won't warn for the same object
        vds.attrs["test"] = "value"  # won't warn for the same object

        # should not warn with `update`
        vds.attrs.update(test="different")

    # should warn on overwrite attr
    with pytest.warns(UserWarning, match="Overwriting attr"):
        vds.attrs["test"] = "new_value"


def test_to_dataframe(random_path: VectorDataset, random_geo_path: GeoVectorDataset) -> None:
    """Test method `to_dataframe`."""
    df = random_path.to_dataframe()
    assert df.shape == (100, 3)
    assert df.columns.tolist() == list(random_path.data)

    # reverse gives same data dictionary
    data = df.to_dict(orient="list")
    assert random_path.data.keys() == data.keys()
    for arr1, arr2 in zip(random_path.data.values(), data.values()):
        np.testing.assert_array_equal(arr1, arr2)

    df = random_geo_path.to_dataframe()
    assert "latitude" in df
    assert "longitude" in df
    assert "time" in df
    assert "altitude" in df


def test_sort() -> None:
    """Test `.sort` method."""
    data = {
        "a": np.array([1, 10, 2, 2]),
        "b": np.array([1, 4, 3, 2]),
        "c": np.array([1, 2, 3, 10]),
    }
    attrs = {"test": "attribute"}
    vds = VectorDataset(data=data, attrs=attrs)
    sorted_vds = vds.sort("a")
    assert sorted_vds is not vds
    assert np.all(sorted_vds["a"] == np.array([1, 2, 2, 10]))
    assert np.all(sorted_vds["b"] == np.array([1, 3, 2, 4]))
    assert np.all(sorted_vds["c"] == np.array([1, 3, 10, 2]))

    sorted_vds = vds.sort(["a", "b"])
    assert sorted_vds is not vds
    assert np.all(sorted_vds["a"] == np.array([1, 2, 2, 10]))
    assert np.all(sorted_vds["b"] == np.array([1, 2, 3, 4]))
    assert np.all(sorted_vds["c"] == np.array([1, 10, 3, 2]))


def test_geovector_init() -> None:
    """Test GeoVectorDataset.__init__()."""
    longitude = np.linspace(0, 50, 100)
    latitude = np.linspace(0, 10, 100)
    time = pd.date_range("2019-01-01 00:00:00", "2019-01-01 02:00:00", periods=100)

    # with altitude
    altitude = np.linspace(11000, 11500, 100)
    gvds = GeoVectorDataset(longitude=longitude, latitude=latitude, altitude=altitude, time=time)
    assert gvds["longitude"] is not longitude
    assert gvds["latitude"] is not latitude
    assert gvds["altitude"] is not altitude
    assert gvds["time"] is not time
    assert np.all(gvds["longitude"] == longitude)
    assert np.all(gvds["latitude"] == latitude)
    assert np.all(gvds["altitude"] == altitude)
    assert np.all(gvds["time"] == time)
    assert "level" not in gvds

    # with level
    level = np.linspace(250, 300, 100)
    gvds = GeoVectorDataset(longitude=longitude, latitude=latitude, level=level, time=time)
    assert gvds["longitude"] is not longitude
    assert gvds["latitude"] is not latitude
    assert gvds["level"] is not level
    assert gvds["time"] is not time
    assert np.all(gvds["longitude"] == longitude)
    assert np.all(gvds["latitude"] == latitude)
    assert np.all(gvds["level"] == level)
    assert np.all(gvds["time"] == time)
    assert "altitude" not in gvds

    # dataframe
    df = pd.DataFrame()
    df["longitude"] = longitude
    df["latitude"] = latitude
    df["altitude"] = 11000
    df["time"] = time
    gvds = GeoVectorDataset(data=df)
    assert gvds["longitude"] is not longitude
    assert gvds["latitude"] is not latitude
    assert gvds["altitude"] is not altitude
    assert gvds["time"] is not time
    assert np.all(gvds["longitude"] == longitude)
    assert np.all(gvds["latitude"] == latitude)
    assert np.all(gvds["altitude"] == 11000)
    assert np.all(gvds["time"] == time)

    # mixed
    df = pd.DataFrame()
    df["longitude"] = longitude
    df["latitude"] = latitude
    gvds = GeoVectorDataset(data=df, altitude=altitude, time=time)
    assert np.all(gvds["longitude"] == longitude)
    assert np.all(gvds["latitude"] == latitude)
    assert np.all(gvds["altitude"] == altitude)
    assert np.all(gvds["time"] == time)

    # required keys
    df = pd.DataFrame()
    df["longitude"] = longitude
    df["latitude"] = latitude
    df["altitude"] = altitude
    df["time"] = time
    for col in ["time", "latitude", "longitude"]:
        data = df.drop(col, axis=1)
        with pytest.raises(KeyError, match="Missing required data keys:"):
            GeoVectorDataset(data=data)

    # fail with different lengths
    longitude = np.array([0, 1, 2])
    latitude = np.array([0, 1])
    altitude = np.array([11000, 11200])
    time = pd.date_range("2019-01-01 00:00:00", "2019-01-01 02:00:00", periods=2)
    df = pd.DataFrame()
    df["longitude"] = longitude
    with pytest.raises(ValueError, match="Incompatible array sizes: 2 and 3"):
        gvds = GeoVectorDataset(
            longitude=longitude, latitude=latitude, altitude=altitude, time=time
        )
    with pytest.raises(ValueError, match="Incompatible array sizes: 2 and 3"):
        gvds = GeoVectorDataset(data=df, latitude=latitude, altitude=altitude, time=time)

    # parse time
    longitude = np.array([0, 1])
    latitude = np.array([0, 1])
    altitude = np.array([11000, 11200])
    time = np.array(["2021-09-01 00:00:00", "2021-09-01 01:00:00"])

    with pytest.warns(UserWarning, match="not np.datetime64. Attempting to coerce"):
        gvds = GeoVectorDataset(
            longitude=longitude,
            latitude=latitude,
            altitude=altitude,
            time=time,
        )
    assert isinstance(gvds["time"][0], np.datetime64)
    assert np.all(gvds["longitude"] == longitude)
    assert np.all(gvds["latitude"] == latitude)
    assert np.all(gvds["altitude"] == altitude)

    # level / altitude keys
    longitude = np.linspace(0, 50, 100)
    latitude = np.linspace(0, 10, 100)
    time = pd.date_range("2019-01-01 00:00:00", "2019-01-01 02:00:00", periods=100)
    altitude = np.linspace(11000, 11500, 100)
    level = np.linspace(250, 300, 100)

    # See TODO in vector regarding level/altitude
    # with pytest.warns(Warning, match="contains both `level` and `altitude` keys"):
    #     gvds = GeoVectorDataset(
    #         longitude=longitude, latitude=latitude, level=level, altitude=altitude, time=time
    #     )

    match = "GeoVectorDataset requires either `level` or `altitude` or `altitude_ft` data key"
    with pytest.raises(KeyError, match=match):
        gvds = GeoVectorDataset(longitude=longitude, latitude=latitude, time=time)


def test_geovector_empty() -> None:
    """Test empty geovector handling."""

    gv = GeoVectorDataset()
    gv2 = GeoVectorDataset.create_empty()

    assert gv == gv2

    # TODO: what do you do with an empty GeoVectorDataset()
    # Do we need a special `update` so you can set the coordinates?
    # these works but could be error prone or clunky
    gv["longitude"] = np.linspace(0, 50, 100)
    assert gv.size == 100

    gv.update(
        latitude=np.linspace(0, 10, 100),
        longitude=np.linspace(0, 50, 100),
        altitude=np.linspace(11000, 11500, 100),
        time=pd.date_range("2019-01-01 00:00:00", "2019-01-01 02:00:00", periods=100),
    )
    assert gv.size == 100


def test_geovector_properties() -> None:
    """Test GeoVectorDataset @property()."""
    longitude = np.linspace(0, 50, 100)
    latitude = np.linspace(0, 10, 100)
    altitude = np.linspace(11000, 11500, 100)
    time = pd.date_range("2019-01-01 00:00:00", "2019-01-01 02:00:00", periods=100)

    gvds = GeoVectorDataset(longitude=longitude, latitude=latitude, altitude=altitude, time=time)
    assert gvds

    level = np.linspace(250, 300, 100)
    gvds = GeoVectorDataset(longitude=longitude, latitude=latitude, level=level, time=time)
    assert gvds["longitude"] is not longitude
    assert gvds["latitude"] is not latitude
    assert gvds["level"] is not level
    assert gvds["time"] is not time
    assert np.all(gvds["longitude"] == longitude)
    assert np.all(gvds["latitude"] == latitude)
    assert np.all(gvds["level"] == level)
    assert np.all(gvds["time"] == time)
    assert "altitude" not in gvds


def test_coord_intersect_met(met_issr: MetDataset) -> None:
    """Test coord_intersect_met."""

    longitude = np.linspace(0, 50, 100)
    latitude = np.linspace(0, 10, 100)
    level = np.linspace(230, 250, 100)
    time = pd.date_range("2019-05-31 05:01:00", "2019-05-31 05:50:00", periods=100)

    gvds = GeoVectorDataset(longitude=longitude, latitude=latitude, level=level, time=time)
    assert gvds.coords_intersect_met(met_issr).all()

    gvds = GeoVectorDataset(
        longitude=longitude, latitude=latitude, level=np.linspace(200, 250, 100), time=time
    )
    assert gvds.coords_intersect_met(met_issr).any()

    gvds = GeoVectorDataset(
        longitude=longitude,
        latitude=latitude,
        level=level,
        time=pd.date_range("2019-05-30 21:00:00", "2019-05-31 05:50:00", periods=100),
    )
    assert gvds.coords_intersect_met(met_issr).any()

    gvds = GeoVectorDataset(
        longitude=longitude, latitude=latitude, level=np.linspace(200, 220, 100), time=time
    )
    assert not gvds.coords_intersect_met(met_issr).any()


def test_broadcast_attrs(random_path: VectorDataset):
    """Test method `broadcast_attrs`."""
    path = random_path.copy()

    with pytest.raises(KeyError, match="does not contain attr `not-in-attr`"):
        path.broadcast_attrs("not-in-attr")

    assert len(path.attrs) == 1
    assert len(path.data.keys()) == 3
    path.broadcast_attrs("test")
    assert len(path.data.keys()) == 4
    assert np.all(path["test"] == "attribute")

    path.attrs.update(test="newattr")
    with pytest.warns(match="Found duplicate key"):
        path.broadcast_attrs("test")

    path.broadcast_attrs("test", overwrite=True)
    assert np.all(path["test"] == "newattr")


def test_broadcast_numeric(random_path: VectorDataset):
    """Test method `broadcast_numeric_attrs`."""
    path = random_path.copy()
    assert path.attrs is not random_path.attrs
    path.attrs["numeric"] = 5
    assert len(path.attrs) == 2
    assert len(random_path.attrs) == 1

    assert len(path.data.keys()) == 3
    path.broadcast_numeric_attrs()
    assert len(path.data.keys()) == 4
    assert "numeric" in path
    assert np.all(path["numeric"] == 5)

    # overwrite and ignore
    path.attrs.update(numeric=1)
    path.attrs["numeric_ignored"] = 6
    path.broadcast_numeric_attrs(ignore_keys="numeric_ignored", overwrite=True)
    assert len(path.data.keys()) == 4
    assert "numeric_ignored" not in path
    assert np.all(path["numeric"] == 1)

    # ignore str
    path = random_path.copy()
    path.attrs["numeric_ignored"] = 6
    path.broadcast_numeric_attrs(ignore_keys="numeric_ignored")
    assert len(path.data.keys()) == 3
    assert "numeric_ignored" not in path

    # ignore list
    path.broadcast_numeric_attrs(ignore_keys=["numeric_ignored"])
    assert len(path.data.keys()) == 3
    assert "numeric_ignored" not in path


def test_vector_copy(random_path: VectorDataset):
    """Confirm various "copy" patterns work as expected."""
    # Without copying, the resulting array is a view of the original array
    df = random_path.to_dataframe(copy=False)
    a = df["a"].to_numpy()
    assert a.base is random_path["a"]
    assert np.may_share_memory(a, random_path["a"])

    # Create a new VectorDataset and show that copying is still respected
    random_path2 = VectorDataset(df, copy=False)
    assert random_path2 == random_path
    assert random_path2["a"].base is random_path["a"]
    assert np.may_share_memory(random_path2["a"], random_path["a"])

    random_path3 = VectorDataset(df, copy=True)
    assert random_path3 == random_path
    assert random_path3["a"].base is None
    assert not np.may_share_memory(random_path3["a"], random_path["a"])

    # With copying, the resulting array is a honest copy of the original array
    df = random_path.to_dataframe(copy=True)
    a = df["a"].to_numpy()
    assert a.base is not random_path["a"]
    assert not np.may_share_memory(random_path3["a"], random_path["a"])

    data = random_path.data
    assert isinstance(data, dict)
    assert len(data) == 3

    # Here, when we grab the original data, the base is None
    random_path4 = VectorDataset(data, copy=False)
    assert random_path4 == random_path
    assert random_path4["a"].base is None
    assert np.may_share_memory(random_path4["a"], random_path["a"])

    random_path5 = VectorDataset(data, copy=True)
    assert random_path5 == random_path
    assert random_path5["a"].base is None
    assert not np.may_share_memory(random_path5["a"], random_path["a"])


def test_bad_time(random_geo_path: VectorDataset):
    """Test that bad time data is handled correctly."""
    data = random_geo_path.copy().data

    # time is successfully coerced to datetime64 (but values are nanoseconds after 1970)
    data.update(time=[1, 2, 3, 4])
    with pytest.warns(UserWarning, match="time"):
        GeoVectorDataset(data)

    # time cannot be converted
    data.update(time=["hello", "world", "con", "trail"])
    with pytest.warns(UserWarning, match="time"):
        with pytest.raises(ValueError, match="time"):
            GeoVectorDataset(data)

    data.update(
        time=[
            "2021-10-05 00:00:00Z",
            "2021-10-05 01:00:00Z",
            "2021-10-05 02:00:00Z",
            "2021-10-05 03:00:00Z",
        ]
    )
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])

    assert df["time"].dt.tz is not None
    with pytest.raises(ValueError, match="The 'time' field must be timezone naive."):
        GeoVectorDataset(df)

    df["time"] = df["time"].dt.tz_localize(None)
    vector = GeoVectorDataset(df)
    assert vector["time"].dtype == np.dtype("datetime64[ns]")


def test_vector_from_array_like():
    """Confirm a Vector instance can be instanted from data castable to array."""
    data = {
        "a": [0, 1 / 3, 2 / 3],
        "b": [0, 0.4, 0.8],
    }
    vector = VectorDataset(data)
    assert vector.size == 3

    for k in vector:
        assert isinstance(vector[k], np.ndarray)


def test_vector_set_item_array_like():
    """Confirm __setitem__ correctly casts lists to arrays."""
    data = {
        "a": np.linspace(0, 1, 3),
        "b": np.arange(0, 1, 0.4),
    }
    vector = VectorDataset(data)
    assert vector.size == 3

    for k in vector:
        assert isinstance(vector[k], np.ndarray)

    # Correctly casts list
    c = [3, 2, 1]
    vector["c"] = c
    assert isinstance(vector["c"], np.ndarray)
    np.testing.assert_array_equal(vector["c"], c)
    assert vector["c"] is not c

    # If working with a numpy array, it's not copied
    d = np.array([3.3, 4.4, 5.5])
    vector["d"] = d
    assert vector["d"] is d
