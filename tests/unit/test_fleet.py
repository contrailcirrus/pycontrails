"""Test Fleet data structure."""

import numpy as np
import pandas as pd
import pytest

from pycontrails import Flight, HydrogenFuel
from pycontrails.core.fleet import Fleet

try:
    from pycontrails.ext.synthetic_flight import SyntheticFlight
except ImportError:
    pytest.skip("SyntheticFlight not available", allow_module_level=True)


@pytest.fixture(scope="module")
def syn():
    """Build `SyntheticFlight` generator."""
    bounds = {
        "time": (np.datetime64("2019-10-05"), np.datetime64("2019-10-05T12")),
        "longitude": (-20, 20),
        "latitude": (30, 60),
        "level": (200, 300),
    }
    return SyntheticFlight(bounds, speed_m_per_s=220, aircraft_type="A320", seed=19)


def test_fleet_io(syn: SyntheticFlight) -> None:
    """Confirm that a fleet can be instantiated from a list of `Flights`.

    Check that the flights coming out of method `to_flight_list` agree with inputs.
    """
    in_fls = [syn() for _ in range(100)]
    for fl in in_fls:
        # In the Fleet constructor, a waypoint and flight_id field are added
        # Do this here to ensure the data is the same
        fl["waypoint"] = np.arange(len(fl))
        fl.broadcast_numeric_attrs()  # add flight_id

    fleet = Fleet.from_seq(in_fls, broadcast_numeric=False)
    assert fleet.n_flights == 100

    out_fls = fleet.to_flight_list()
    for fl1, fl2 in zip(in_fls, out_fls, strict=True):
        assert fl1 == fl2


def test_fleet_init(syn: SyntheticFlight) -> None:
    """Check that `Fleet` can be instantiated directly from a GeoVectorDataset."""
    fl = syn(np.timedelta64(5, "s"))
    flight_id = np.arange(len(fl)) % 10
    fl["flight_id"] = flight_id

    with pytest.raises(ValueError, match="Fleet must have contiguous waypoint blocks"):
        Fleet(fl)

    df = fl.dataframe.sort_values(["flight_id", "time"])

    with pytest.raises(ValueError, match=r"Unexpected flight_id\(s\) \{'hello'\} in fl_attrs."):
        Fleet(df, fl_attrs={"hello": "world"})

    fleet = Fleet(df)
    assert fleet.fl_attrs.keys() == set(range(10))
    assert fleet.n_flights == 10


def test_fleet_calc_final_waypoints(syn: SyntheticFlight) -> None:
    """Confirm the calculation of `final_waypoints`."""

    fls = [syn() for _ in range(25)]
    fleet = Fleet.from_seq(fls)

    assert isinstance(fleet.final_waypoints, np.ndarray)
    assert np.sum(fleet.final_waypoints) == len(fls)
    assert fleet.final_waypoints.dtype == bool
    idx = np.cumsum([fl.size for fl in fls]) - 1
    assert np.all(fleet.final_waypoints[idx])

    fls.append(fls[3])
    with pytest.raises(ValueError, match="Duplicate 'flight_id'"):
        Fleet.from_seq(fls)

    data = fleet.data
    data["flight_id"][3] = data["flight_id"][-3]

    with pytest.raises(ValueError, match="Fleet must have contiguous waypoint blocks"):
        Fleet(data)


def test_fleet_segment_length(syn: SyntheticFlight) -> None:
    """Check compatibility of `segment_length` and `segment_angle` on `Fleet`."""
    fls = [syn() for _ in range(20)]
    fleet = Fleet.from_seq(fls)
    assert fleet.n_flights == 20

    sl1 = fleet.segment_length()
    sl2 = np.concatenate([fl.segment_length() for fl in fls])
    np.testing.assert_array_equal(sl1, sl2)

    _, cos1 = fleet.segment_angle()
    cos2 = np.concatenate([fl.segment_angle()[1] for fl in fls])
    np.testing.assert_array_equal(cos1, cos2)


def test_fleet_run_all_methods(syn: SyntheticFlight) -> None:
    """Ensure a few common methods on `Fleet` instance run without error."""
    fls = [syn() for _ in range(30)]
    fleet = Fleet.from_seq(fls)
    assert fleet.n_flights == 30

    assert isinstance(fleet.coords, dict)
    assert isinstance(fleet.segment_angle(), tuple)
    assert isinstance(fleet.altitude_ft, np.ndarray)
    assert isinstance(fleet.level, np.ndarray)


def test_fleet_init_from_gen(syn: SyntheticFlight) -> None:
    """Check that Fleet can be instantiated from a generator of `Flights`."""
    fls = (syn() for _ in range(200))
    fleet = Fleet.from_seq(fls)
    assert fleet.n_flights == 200


def test_fleet_waypoints(syn: SyntheticFlight) -> None:
    """Check Cocip continuity conventions on `Fleet` instance."""
    fls = (syn() for _ in range(321))
    fleet = Fleet.from_seq(fls)
    assert fleet.n_flights == 321
    assert "waypoint" in fleet
    assert fleet["waypoint"].dtype == int

    # copied from `Cocip._calc_continuous`
    continuous = np.append((np.diff(fleet["waypoint"]) == 1).astype(bool), False)
    assert np.sum(~continuous) == 321  # each flight-change generates a discontinuity

    flight_ids = fleet.filter(~continuous)["flight_id"]
    # expect one of each flight_id
    for id1, id2 in zip(flight_ids, fleet.fl_attrs, strict=True):
        assert id1 == id2


def test_duplicate_flight_ids(syn: SyntheticFlight) -> None:
    """Check that Fleet raises exception if duplicate flight id is found."""
    fls = [syn() for _ in range(222)]
    fls[151].attrs.update(flight_id=333)
    fls[97].attrs.update(flight_id=333)
    with pytest.raises(ValueError, match="Duplicate 'flight_id' 333 found."):
        Fleet.from_seq(fls)


def test_type_of_fleet(syn: SyntheticFlight) -> None:
    """Confirm that a `Fleet` is an instance of `Flight`, but an iterable of `Flight` is not.

    This logic is used in `Cocip.eval` method.
    """
    fls = [syn() for _ in range(100)]
    fleet = Fleet.from_seq(fls)
    fl = fls[0]

    # This logic can be used to distinguish Iterable[Flight] from Flight
    assert isinstance(fl, Flight)
    assert isinstance(fleet, Flight)
    assert not isinstance(fls, Flight)

    # NOTE: All have iter method, so this cannot be used to distinguish
    assert hasattr(fl, "__iter__")
    assert hasattr(fleet, "__iter__")
    assert hasattr(fls, "__iter__")


def test_fleet_numeric_types_to_data(syn: SyntheticFlight) -> None:
    """Check that numeric attributes are attached to underlying `Fleet` data."""
    fls = [syn() for _ in range(40)]
    rng = np.random.default_rng()
    for fl in fls:
        fl.attrs["numeric1"] = rng.random()
        fl.attrs["numeric2"] = rng.integers(0, 5).item()
        fl.attrs["non-numeric"] = "contrail"

    fleet = Fleet.from_seq(fls)
    fleet.ensure_vars(
        [
            "waypoint",
            "longitude",
            "latitude",
            "level",
            "time",
            "flight_id",
            "numeric1",
            "numeric2",
        ]
    )
    assert "non-numeric" not in fleet

    # Nothing in fleet.attrs is numeric
    n_keys = len(fleet.data)
    fleet.broadcast_numeric_attrs()
    assert len(fleet.data) == n_keys


def test_fleet_true_airspeed(syn: SyntheticFlight) -> None:
    """Confirm that method segment_true_airspeed on Fleet agrees with Flight-wise call."""
    fls = [syn() for _ in range(44)]
    fleet = Fleet.from_seq(fls)
    rng = np.random.default_rng(1234)
    u_wind = rng.uniform(-5, 5, fleet.size)
    v_wind = rng.uniform(-5, 5, fleet.size)
    gs = fleet.segment_groundspeed()
    assert gs.size == fleet.size

    tas = fleet.segment_true_airspeed(u_wind, v_wind)
    assert tas.size == fleet.size

    # NaN values occur just before waypoints reset
    expected_nan = np.diff(fleet["waypoint"], append=np.nan) != 1
    np.testing.assert_array_equal(expected_nan, np.isnan(tas))

    i0 = 0
    for fl in fls:
        i1 = i0 + fl.size
        slice_ = slice(i0, i1)
        u_wind_slice = u_wind[slice_]
        v_wind_slice = v_wind[slice_]
        tas_slice = fl.segment_true_airspeed(u_wind_slice, v_wind_slice)
        np.testing.assert_array_equal(tas_slice, tas[slice_])
        i0 = i1


@pytest.mark.parametrize("n_min", [5, 6, 7, 8])
def test_fleet_resample_and_fill(syn: SyntheticFlight, n_min: int) -> None:
    """Test the Fleet.resample_and_fill method."""

    n_flights = 99
    fls = [syn(np.timedelta64(n_min, "m")) for _ in range(n_flights)]
    fleet = Fleet.from_seq(fls)
    fleet.attrs["pycontrails"] = 1234

    resampled = fleet.resample_and_fill()
    assert isinstance(resampled, Fleet)
    assert resampled.n_flights == fleet.n_flights == n_flights

    assert len(resampled) <= n_min * len(fleet)
    assert len(resampled) >= n_min * (len(fleet) - n_flights)

    assert resampled.attrs["pycontrails"] == 1234


def test_fleet_slots(syn: SyntheticFlight) -> None:
    """Ensure slots are respected."""

    fls = (syn() for _ in range(100))
    fleet = Fleet.from_seq(fls)

    assert "__dict__" not in dir(fleet)
    assert fleet.__slots__ == ("final_waypoints", "fl_attrs")
    with pytest.raises(AttributeError, match="'Fleet' object has no attribute 'foo'"):
        fleet.foo = "bar"


@pytest.mark.parametrize("method", ["copy", "filter"])
def test_method_preserves_fuel_fl_attrs(syn: SyntheticFlight, method: str) -> None:
    """Ensure fuel and fl_attrs are preserved for the copy and filter methods."""

    fls = (syn() for _ in range(5))
    fleet = Fleet.from_seq(fls)

    fleet.fuel = HydrogenFuel()

    if method == "copy":
        out = fleet.copy()
    else:
        assert method == "filter"
        mask = np.ones(len(fleet), dtype=bool)
        mask[::3] = False
        out = fleet.filter(mask)

    assert isinstance(out, Fleet)
    assert hasattr(out, "fuel")
    assert out.fuel is fleet.fuel

    assert hasattr(out, "fl_attrs")
    assert out.fl_attrs == fleet.fl_attrs


def test_fleet_filter_removes_flight(syn: SyntheticFlight) -> None:
    """Ensure fl attributes are removed when Fleet.filter removes a entire flight."""

    fls = [syn() for _ in range(5)]
    fleet = Fleet.from_seq(fls)
    assert fleet.n_flights == 5

    id0 = fls[0].attrs["flight_id"]
    mask = fleet["flight_id"] != id0
    out = fleet.filter(mask)
    assert isinstance(out, Fleet)

    assert out.n_flights == 4
    assert id0 not in out.fl_attrs
    assert list(out.fl_attrs) == [fl.attrs["flight_id"] for fl in fls[1:]]


def test_fleet_from_seq_removes_empty_flight() -> None:
    """Ensure empty flights are removed when Fleet.from_seq is called."""
    df = pd.DataFrame()
    df["longitude"] = np.linspace(0, 50, 100)
    df["latitude"] = np.linspace(0, 10, 100)
    df["altitude"] = 11000
    df["time"] = pd.date_range("2022-03-01 00:00:00", "2022-03-01 02:00:00", periods=100)

    flights = [Flight(df, flight_id=1), Flight(flight_id=2)]

    with pytest.warns(UserWarning, match="Empty flight found in sequence."):
        fleet = Fleet.from_seq(flights)

    assert fleet.n_flights == 1
