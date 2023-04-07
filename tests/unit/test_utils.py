"""test utils module."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    from pycontrails.utils.synthetic_flight import SAMPLE_AIRCRAFT_TYPES, SyntheticFlight
except ImportError:
    synthetic_flight_unavailable = True
    SAMPLE_AIRCRAFT_TYPES = []
else:
    synthetic_flight_unavailable = False


from pycontrails import Flight, MetDataset
from pycontrails.utils import types
from pycontrails.utils.iteration import chunk_list
from pycontrails.utils.json import NumpyEncoder
from pycontrails.utils.temp import remove_tempfile, temp_file, temp_filename
from pycontrails.utils.types import type_guard
from tests import BADA3_PATH, BADA4_PATH, BADA_AVAILABLE


def test_temp_files() -> None:
    """Test temp file handling."""
    temp_filename_ = temp_filename()
    assert isinstance(temp_filename_, str)
    assert not pathlib.Path(temp_filename_).is_file()

    with open(temp_filename_, "w") as f:
        f.write("test")

    # file persists after closing
    assert pathlib.Path(temp_filename_).is_file()

    # remove tempfile
    remove_tempfile(temp_filename_)
    assert not pathlib.Path(temp_filename_).is_file()


def test_temp_file_no_exception() -> None:
    """Check that `temp_file` automatically removes the temp file."""
    with temp_file() as tempfile:
        assert not pathlib.Path(tempfile).is_file()
        with open(tempfile, "w") as f:
            f.write("test")
        assert pathlib.Path(tempfile).is_file()

    # automatically removes file
    assert not pathlib.Path(tempfile).is_file()


@pytest.mark.parametrize("exc", [ValueError, RuntimeError, KeyError, IndexError, KeyboardInterrupt])
def test_temp_file_with_exception(exc: Exception) -> None:
    """Check that `temp_file` removes the temp file even when an exception is raised."""
    try:
        with temp_file() as tempfile:
            assert not pathlib.Path(tempfile).is_file()
            with open(tempfile, "w") as f:
                f.write("test")
            assert pathlib.Path(tempfile).is_file()
            raise exc
    except exc:  # type: ignore[misc]
        # automatically removes file
        assert not pathlib.Path(tempfile).is_file()


def test_numpy_encoder() -> None:
    """Test NumpyEncoder JSON utility."""
    int32 = 1
    float64 = 2.1
    array = [4, 5, 6]
    bool_ = True
    date_range = pd.date_range("2019-01-01 00:00:00", "2019-01-01 02:00:00", periods=10)
    _dict = {
        "int32": np.int32(int32),
        "float64": np.float64(float64),
        "array": np.array(array),
        "bool_": np.bool_(bool_),
        "series": pd.Series(array),
        "date_range": date_range,
    }

    jsonstr = json.dumps(_dict, cls=NumpyEncoder)
    loaded_dict = json.loads(jsonstr)

    assert loaded_dict["int32"] == int32
    assert loaded_dict["float64"] == float64
    assert loaded_dict["array"] == array
    assert loaded_dict["bool_"] == bool_
    assert loaded_dict["series"] == array
    assert loaded_dict["date_range"] == date_range.to_numpy().tolist()


@pytest.mark.skipif(synthetic_flight_unavailable, reason="synthetic_flight unavailable")
def test_synthetic_flight(met_cocip1: MetDataset) -> None:
    """Test SyntheticFlight class."""
    # synthetic flights
    bounds = {
        "longitude": np.array([-40, 40]),
        "latitude": np.array([-75, 75]),
        "level": np.array([175, 350]),
        "time": np.array([np.datetime64("2019-01-15"), np.datetime64("2019-02-16")]),
    }

    # no met, bada
    fl_gen = SyntheticFlight(aircraft_type="B737", bounds=bounds, seed=1, speed_m_per_s=240)
    fl = fl_gen()

    assert isinstance(fl, Flight)
    assert fl.attrs["aircraft_type"] == "B737"

    # random aircraft_type
    fl_gen = SyntheticFlight(bounds=bounds, seed=1, speed_m_per_s=240)
    fl = fl_gen()
    assert fl.attrs["aircraft_type"] in SAMPLE_AIRCRAFT_TYPES

    # with met
    fl_gen = SyntheticFlight(
        aircraft_type="B737",
        bounds=met_cocip1.coords,
        seed=1,
        speed_m_per_s=240,
        u_wind=met_cocip1["eastward_wind"],
        v_wind=met_cocip1["northward_wind"],
    )
    fl = fl_gen()

    assert isinstance(fl, Flight)


@pytest.mark.skipif(synthetic_flight_unavailable, reason="synthetic_flight unavailable")
@pytest.mark.skipif(not BADA_AVAILABLE, reason="No BADA or EDB Files")
@pytest.mark.parametrize("aircraft_type", SAMPLE_AIRCRAFT_TYPES)
def test_synthetic_flight_bada(aircraft_type: str) -> None:
    """Test BADA support in SyntheticFlight."""
    # synthetic flights
    bounds = {
        "longitude": np.array([-40, 40]),
        "latitude": np.array([-75, 75]),
        "level": np.array([175, 350]),
        "time": np.array([np.datetime64("2019-01-15"), np.datetime64("2019-02-16")]),
    }

    fl_gen = SyntheticFlight(
        aircraft_type=aircraft_type,
        bada3_path=BADA3_PATH,
        bada4_path=BADA4_PATH,
        bounds=bounds,
        seed=1,
    )
    fl = fl_gen()

    assert isinstance(fl, Flight)
    assert fl.attrs["aircraft_type"] == aircraft_type
    tas = fl.segment_true_airspeed()
    assert np.isnan(tas[-1])
    assert np.all(tas[:-1] > 100)
    assert np.all(tas[:-1] < 300)


def test_nan_mask() -> None:
    """Check that `apply_nan_mask_to_arraylike` returns correct type."""
    np_arr = np.arange(10).astype(float)
    nan_mask = np.zeros_like(np_arr).astype(bool)
    nan_mask[5] = True

    np_out = types.apply_nan_mask_to_arraylike(np_arr, nan_mask)
    assert isinstance(np_out, np.ndarray)
    assert np.isnan(np_out).nonzero()[0].item() == 5

    pd_arr = pd.Series(np_arr)
    pd_out = types.apply_nan_mask_to_arraylike(pd_arr, nan_mask)
    assert isinstance(pd_out, pd.Series)
    assert (pd_out.isna() == nan_mask).all()

    xr_arr = xr.DataArray(np_arr)
    nan_mask = xr.DataArray(nan_mask)
    xr_out = types.apply_nan_mask_to_arraylike(xr_arr, nan_mask)
    assert isinstance(xr_out, xr.DataArray)
    assert np.all(np.isnan(xr_out) == nan_mask)


def test_support_arraylike() -> None:
    """Test `support_arraylike` decorator."""
    np_arr = np.arange(10)

    @types.support_arraylike
    def func(arr):
        """Use a piecewise function here to mimic in internal use-cases."""
        condlist = [arr < 5]
        funclist = [lambda x: x**2, lambda x: 3 * x]
        return np.piecewise(arr, condlist, funclist)

    expected = [0, 1, 4, 9, 16, 15, 18, 21, 24, 27]

    # numpy case
    np_out = func(np_arr)
    assert isinstance(np_out, np.ndarray)
    np.testing.assert_array_equal(np_out, expected)

    # pandas case
    pd_arr = pd.Series(np_arr)
    pd_out = func(pd_arr)
    assert isinstance(pd_out, pd.Series)
    assert (pd_out == expected).all()

    # xarray case
    xr_arr = xr.DataArray(np_arr)
    xr_out = func(xr_arr)
    assert isinstance(xr_out, xr.DataArray)
    assert (xr_out == expected).all()

    # float, int case
    for i, e in enumerate(expected):
        assert func(i) == func(float(i)) == e


def test_chunk_list() -> None:
    """Test chunk list utility."""

    a = list(range(0, 50))
    for chunk in chunk_list(a, 10):
        assert len(chunk) == 10
        assert chunk[0] % 10 == 0


def test_type_guard(met_cocip1: MetDataset) -> None:
    """Test type_guard."""

    type_guard("str", str)
    type_guard(0, int)
    type_guard(met_cocip1, MetDataset)
    type_guard("str", (str, int))

    # returns object
    a = "test"
    b = type_guard(a, (str, int))
    assert a is b and isinstance(b, str)

    with pytest.raises(ValueError, match="must be of type"):
        type_guard("str", int)

    with pytest.raises(ValueError, match="pycontrails.core.met.MetDataset"):
        type_guard("str", (MetDataset, float))

    # custom error message
    with pytest.raises(ValueError, match="custom error"):
        type_guard("str", int, error_message="custom error")
