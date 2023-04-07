"""Test Fleet data structure."""

import numpy as np
import pytest

from pycontrails import Flight
from pycontrails.core.fleet import Fleet

try:
    from pycontrails.utils.synthetic_flight import SyntheticFlight
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


def test_fleet_io(syn: SyntheticFlight):
    """Confirm that a fleet can be instantiated from a list of `Flights`.

    Check that the flights coming out of method `to_flight_list` agree with inputs.
    """
    in_fls = [syn() for _ in range(100)]
    fleet = Fleet.from_seq(in_fls)
    assert fleet.n_flights == 100

    out_fls = fleet.to_flight_list()
    for fl1, fl2 in zip(in_fls, out_fls):
        assert fl1 == fl2


def test_fleet_calc_final_waypoints(syn: SyntheticFlight):
    """Confirm `calc_final_waypoints` method."""
    fls = [syn() for _ in range(25)]
    fleet = Fleet.from_seq(fls)
    assert isinstance(fleet.final_waypoints, np.ndarray)
    assert np.sum(fleet.final_waypoints) == len(fls)
    assert fleet.final_waypoints.dtype == bool
    idx = np.cumsum([fl.size for fl in fls]) - 1
    assert np.all(fleet.final_waypoints[idx])

    fls.append(fls[3])
    with pytest.raises(ValueError, match="Duplicate `flight_id`"):
        Fleet.from_seq(fls)
    data = fleet.data
    data["flight_id"][3] = data["flight_id"][-3]
    match = "Fleet must have contiguous waypoints blocks with constant"
    with pytest.raises(ValueError, match=match):
        Fleet(data, attrs=fleet.attrs)

    with pytest.raises(ValueError, match="Require key 'fl_attrs'"):
        Fleet(data)


def test_fleet_segment_length(syn: SyntheticFlight):
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


def test_fleet_run_all_methods(syn: SyntheticFlight):
    """Ensure a few common methods on `Fleet` instance run without error."""
    fls = [syn() for _ in range(30)]
    fleet = Fleet.from_seq(fls)
    assert fleet.n_flights == 30

    assert isinstance(fleet.coords, dict)
    assert isinstance(fleet.segment_angle(), tuple)
    assert isinstance(fleet.altitude_ft, np.ndarray)
    assert isinstance(fleet.level, np.ndarray)


def test_fleet_init_from_gen(syn: SyntheticFlight):
    """Check that Fleet can be instantiated from a generator of `Flights`."""
    fls = (syn() for _ in range(200))
    fleet = Fleet.from_seq(fls)
    assert fleet.n_flights == 200


def test_fleet_waypoints(syn: SyntheticFlight):
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
    for id1, id2 in zip(flight_ids, fleet.attrs["fl_attrs"]):
        assert id1 == id2


def test_duplicate_flight_ids(syn: SyntheticFlight):
    """Check that Fleet raises exception if duplicate flight id is found."""
    fls = [syn() for _ in range(222)]
    fls[151].attrs.update(flight_id=333)
    fls[97].attrs.update(flight_id=333)
    with pytest.raises(ValueError, match="Duplicate `flight_id` 333 found."):
        Fleet.from_seq(fls)


def test_type_of_fleet(syn: SyntheticFlight):
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


def test_fleet_numeric_types_to_data(syn: SyntheticFlight):
    """Check that numeric attributes are attached to underlying `Fleet` data."""
    fls = [syn() for _ in range(40)]
    for fl in fls:
        fl.attrs["numeric1"] = np.random.random()  # not even seeding it
        fl.attrs["numeric2"] = np.random.randint(5)
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
    n_keys = len(fleet.data.keys())
    fleet.broadcast_numeric_attrs()
    assert len(fleet.data.keys()) == n_keys


def test_fleet_true_airspeed(syn: SyntheticFlight):
    """Confirm that method segment_true_airspeed on Fleet agrees with Flight-wise call."""
    fls = [syn() for _ in range(44)]
    fleet = Fleet.from_seq(fls)
    rng = np.random.default_rng(1234)
    u_wind = rng.uniform(-5, 5, fleet.size)
    v_wind = rng.uniform(-5, 5, fleet.size)
    with pytest.raises(NotImplementedError):
        fleet.segment_groundspeed(u_wind, v_wind)

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
