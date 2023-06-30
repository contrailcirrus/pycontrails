"""Test contrail_grid module."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pytest
import xarray as xr

try:
    from pycontrails.models.cocipgrid import CocipGrid
except ImportError:
    pytest.skip("CocipGrid not available", allow_module_level=True)

from pycontrails import GeoVectorDataset, MetDataset
from pycontrails.models.aircraft_performance import AircraftPerformance
from pycontrails.models.cocip import Cocip
from pycontrails.models.cocipgrid import cocip_grid as cg_module
from pycontrails.models.humidity_scaling import ExponentialBoostHumidityScaling
from pycontrails.utils.synthetic_flight import SyntheticFlight
from tests import BADA3_PATH, BADA4_PATH, BADA_AVAILABLE


@pytest.fixture
def source() -> MetDataset:
    """Return common source for `CocipGrid` model evaluation."""
    return CocipGrid.create_source(
        # levels do NOT need to agree with underlying met_cocip1
        level=[220, 230, 240, 250],
        time=np.datetime64("2019-01-01"),
        # using only a small set of longitude and latitude to avoid evolving out of bounds
        # the latitude usually goes first -- keep the range there tiny
        longitude=np.linspace(-35, -25, 40),
        latitude=np.linspace(51, 57, 20),
    )


@pytest.fixture
def instance_params(met_cocip1: MetDataset, rad_cocip1: MetDataset) -> dict[str, Any]:
    """Return common parameters for `CocipGrid` model instances.

    Note the fixture is scoped "function".
    """
    # met_cocip1 has levels [200, 225, 250, 300]
    np.testing.assert_array_equal(met_cocip1.data["level"], [200, 225, 250, 300])

    return {
        "met": met_cocip1,
        "rad": rad_cocip1,
        "dt_integration": np.timedelta64(5, "m"),
        # need to keep this super small to avoid advecting out of bounds
        "max_age": np.timedelta64(90, "m"),
        # keep low to ensure update_met_slices actually gets called
        "met_slice_dt": np.timedelta64(1, "h"),
        "bada3_path": BADA3_PATH,
        "bada4_path": BADA4_PATH,
        # explicitly raise error if we advect too far
        "interpolation_bounds_error": True,
        "target_split_size": 1000,
        "target_split_size_pre_SAC_boost": 1,
        "humidity_scaling": ExponentialBoostHumidityScaling(rhi_adj=0.9),
    }


def test_init_contrail_grid_minimal_params(met_cocip1: MetDataset, rad_cocip1: MetDataset) -> None:
    """Check that `CocipGrid` can be instantiated with a minimal set of parameters."""
    gc = CocipGrid(
        met=met_cocip1,
        rad=rad_cocip1,
        max_age=np.timedelta64(8, "m"),
        humidity_scaling=ExponentialBoostHumidityScaling(),
    )

    assert isinstance(gc.params, dict)
    assert isinstance(gc.interp_kwargs, dict)

    assert "dt_integration" in gc.params
    assert "max_age" in gc.params
    assert "aircraft_type" in gc.params
    assert "azimuth" in gc.params

    assert gc.params["met_slice_dt"] == np.timedelta64(1, "h")


def test_met_too_short(met_cocip1: MetDataset, rad_cocip1: MetDataset, source: MetDataset) -> None:
    """Check that a warning is issued when met and rad don't adequately overlap the source."""

    gc = CocipGrid(
        met=met_cocip1,
        rad=rad_cocip1,
        max_age=np.timedelta64(10, "h"),
        humidity_scaling=ExponentialBoostHumidityScaling(),
    )

    source = CocipGrid.create_source(level=[1, 2, 3], time=np.datetime64("2019-01-01"))
    gc.set_source(source)
    with pytest.warns(UserWarning, match="does not overlap the grid domain"):
        gc._check_met_source_overlap()

    source = CocipGrid.create_source(level=5, time=np.datetime64("2019-01-01T04"))
    gc.set_source(source)
    with pytest.warns(UserWarning, match="too short in the time dimension"):
        gc.attach_timedict()

    source = CocipGrid.create_source(level=5, time=np.datetime64("2019-01-01"))
    gc.set_source(source)
    with pytest.warns(UserWarning, match="does not overlap the grid domain"):
        gc._check_met_source_overlap()


@pytest.mark.parametrize("met_slice_dt", [np.timedelta64(1, "h"), np.timedelta64(2, "h")])
def test_load_met_slices(
    instance_params: dict[str, Any],
    source: MetDataset,
    met_slice_dt: np.timedelta64,
) -> None:
    """Verify expected behavior in `load_met_slices` method."""
    instance_params["met_slice_dt"] = met_slice_dt
    gc = CocipGrid(**instance_params)
    assert gc.met.shape == (16, 8, 4, 13)
    assert gc.rad.shape == (16, 8, 1, 13)

    (time,) = source.data["time"].values
    met, rad = gc._load_met_slices(time)

    if met_slice_dt == np.timedelta64(1, "h"):
        assert met.shape == (16, 8, 4, 2)
        assert rad.shape == (16, 8, 1, 3)  # rad shifted, so it includes 1 more time slice
    else:
        assert met.shape == (16, 8, 4, 3)
        assert rad.shape == (16, 8, 1, 4)  # rad shifted, so it includes 1 more time slice

    assert met.data["time"].values[0] == time
    assert rad.data["time"].values[0] == time - np.timedelta64(30, "m")

    # Make sure all variables present
    met.ensure_vars(gc.met_variables)


@pytest.mark.parametrize(
    "met_slice_dt",
    [
        np.timedelta64(0, "m"),
        np.timedelta64(55, "m"),
        np.timedelta64(150, "m"),
    ],
)
def test_init_bad_met_slice_dt(
    instance_params: dict[str, Any], met_slice_dt: np.timedelta64
) -> None:
    """Check that an error is raised when `met_slice_dt` is not a positive multiple of an hour."""
    assert instance_params["met_slice_dt"] == np.timedelta64(1, "h")
    instance_params["met_slice_dt"] = met_slice_dt

    if met_slice_dt == 0:
        match = "positive timedelta"
    else:
        match = "must be a multiple of the time difference"
    with pytest.raises(ValueError, match=match):
        CocipGrid(**instance_params)


@pytest.mark.parametrize(
    "dt_integration",
    [
        np.timedelta64(7, "m"),
        np.timedelta64(8, "m"),
        np.timedelta64(40, "m"),
    ],
)
def test_init_bad_dt_integration(
    instance_params: dict[str, Any], dt_integration: np.timedelta64
) -> None:
    """Check that an error is raised when dt_integration is not a positive multiple met_slice_dt."""
    assert instance_params["dt_integration"] == np.timedelta64(5, "m")
    instance_params["dt_integration"] = dt_integration
    with pytest.raises(ValueError, match="must be a multiple of dt_integration"):
        CocipGrid(**instance_params)


def test_create_bad_latitude() -> None:
    """Check that an error is raised when latitude values are too close to the poles."""
    with pytest.raises(ValueError, match="latitude"):
        CocipGrid.create_source(
            level=[220, 230, 240, 250],
            time=np.datetime64("2019-01-01"),
            longitude=np.linspace(-35, -25, 40),
            latitude=np.asarray([83, 85, 86]),
        )


def test_create_lon_lat_unspecified() -> None:
    """Ensure `CocipGrid` uses `lon_step` and `lat_step` of 1."""
    source = CocipGrid.create_source(level=[220, 250], time=np.datetime64("2019-01-01"))
    assert source.data["longitude"].size == 360
    assert source.data["latitude"].size == 161  # -80 to 80
    assert source.shape == (360, 161, 2, 1)


def test_create_lon_lat_step_args() -> None:
    """Ensure `CocipGrid` uses `lon_step` and `lat_step` specified in `__init__` args.."""
    source = CocipGrid.create_source(
        level=[220, 230, 240, 250],
        time=np.datetime64("2019-01-01"),
        lon_step=5,
        lat_step=5,
    )
    assert source.data["longitude"].size == 72
    assert source.data["latitude"].size == 33
    assert source.shape == (72, 33, 4, 1)


def test_init_avoid_double_process(instance_params: dict[str, Any]) -> None:
    """Check that CocipGrid doesn't alter met data twice."""
    # grab original
    met = instance_params["met"].copy()
    assert "tau_cirrus" not in met

    gc1 = CocipGrid(**instance_params)

    # The only change to met is the tau cirrus variable
    for var in met:
        np.testing.assert_array_equal(met[var].values, gc1.met[var].values)
    assert "tau_cirrus" not in met
    assert "tau_cirrus" in instance_params["met"]

    # grab a copy
    met1 = gc1.met.copy()
    rad1 = gc1.rad.copy()

    gc2 = CocipGrid(**instance_params)

    # Nothing has been tweaked the second time around
    assert gc1.met is gc2.met is instance_params["met"]
    assert gc1.rad is gc2.rad is instance_params["rad"]

    # The copied met and rad should not be the same object
    assert met1 is not gc2.met
    assert rad1 is not gc2.rad

    # But should contain the same values
    xr.testing.assert_equal(met1.data, gc2.met.data)
    xr.testing.assert_equal(rad1.data, gc2.rad.data)


@pytest.mark.parametrize("target_split_size", [100, 200, 300, 600, 1000, 10000])
def test_generate_new_grid_vectors(
    instance_params: dict[str, Any], target_split_size: int, source: MetDataset
) -> None:
    """Check that the `generate_new_vectors` method works as expected with MetDataset source."""
    instance_params["target_split_size"] = target_split_size
    gc = CocipGrid(**instance_params)
    gc.set_source(source)
    assert isinstance(gc.source_time, np.ndarray)
    assert gc.source_time.size == 1
    (time,) = gc.source_time
    time = time.astype("datetime64[h]")  # timedict uses this resolution

    gc.attach_timedict()
    assert time in gc.timedict
    filt = gc.timedict[time]

    vectors = gc._generate_new_vectors(filt)
    assert isinstance(vectors, Iterable)
    vectors = list(vectors)

    coords_size = (
        source.data["longitude"].size * source.data["latitude"].size * source.data["level"].size
    )
    vector_size = sum(vector.size for vector in vectors)
    assert coords_size == vector_size
    for vector in vectors:
        assert isinstance(vector, GeoVectorDataset)
        if target_split_size < coords_size:
            assert vector.size == pytest.approx(target_split_size, rel=0.5)
        else:
            assert len(vectors) == 1

    lon_concat = np.concatenate([vector["longitude"] for vector in vectors])
    lon_concat = np.unique(lon_concat)
    lon_concat.sort()
    np.testing.assert_array_almost_equal(lon_concat, source.data["longitude"].values, decimal=5)


##############################################################
# NOTE: No tests below here will run unless BADA is available.
##############################################################


@pytest.mark.skipif(not BADA_AVAILABLE, reason="BADA not available")
@pytest.mark.parametrize("aircraft_type", ["B737", "A320", "A359", "B772"])
@pytest.mark.parametrize("bada_priority", [3, 4])
def test_calc_emissions(
    instance_params: dict[str, Any],
    aircraft_type: str,
    bada_priority: int,
    source: MetDataset,
) -> None:
    """Test `calc_emissions` function."""
    instance_params["aircraft_type"] = aircraft_type
    # must be larger than 3200 so that all meshes are together
    instance_params["target_split_size"] = 5000
    instance_params["bada_priority"] = bada_priority

    gc = CocipGrid(**instance_params)
    gc.set_source(source)
    filt = np.array([True])

    vector = next(gc._generate_new_vectors(filt))

    met = instance_params["met"]
    # air_temperature need for calc_emissions
    vector["air_temperature"] = vector.intersect_met(met["air_temperature"], bounds_error=True)

    cg_module.calc_emissions(vector, gc.params)
    keys = [
        "air_temperature",
        "true_airspeed",
        "engine_efficiency",
        "fuel_flow",
        "nvpm_ei_n",
        "head_tail_dt",
    ]
    vector.ensure_vars(keys)

    assert "wingspan" in gc.params
    assert "aircraft_mass" in gc.params
    if bada_priority == 3:
        assert gc.params["bada_model"] == "BADA3"
    else:
        assert gc.params["bada_model"] == "BADA4"

    # Ensure not seeing crazy fluctuations in true_airspeed and fuel_flow
    # AND check that values are also non-constant
    tas = vector["true_airspeed"]
    ff = vector["fuel_flow"]
    assert len(np.unique(tas)) > 1
    assert len(np.unique(ff)) > 1

    assert 200 < np.mean(tas) < 300
    assert 0.5 < np.mean(ff) < 2.5

    assert np.std(tas) < 2
    assert np.std(ff) < 0.05


@pytest.mark.skipif(not BADA_AVAILABLE, reason="BADA not available")
def test_calc_first_contrail(instance_params: dict[str, Any], source: MetDataset) -> None:
    """Test `calc_first_contrail` method, which is called early in in `eval`."""
    instance_params["target_split_size"] = 5000
    gc = CocipGrid(**instance_params)
    gc.set_source(source)
    filt = np.array([True])

    vector = next(gc._generate_new_vectors(filt))
    assert vector.size == 3200

    met, rad = gc._load_met_slices(gc.source_time[0])

    cg_module.run_interpolators(
        vector,
        met,
        rad,
        dz_m=gc.params["dz_m"],
        humidity_scaling=gc.params["humidity_scaling"],
    )
    cg_module.calc_emissions(vector, gc.params)
    sac_vector = cg_module.find_initial_contrail_regions(vector, gc.params)[0]
    assert "T_crit_sac" in sac_vector

    cg_module.run_interpolators(
        sac_vector,
        met,
        rad,
        dz_m=gc.params["dz_m"],
        humidity_scaling=gc.params["humidity_scaling"],
    )
    contrail = cg_module.simulate_wake_vortex_downwash(sac_vector, gc.params)

    cg_module.run_interpolators(
        contrail,
        met,
        rad,
        dz_m=gc.params["dz_m"],
        humidity_scaling=gc.params["humidity_scaling"],
    )
    contrail = cg_module.find_initial_persistent_contrails(sac_vector, contrail, gc.params)[0]

    assert isinstance(contrail, GeoVectorDataset)

    assert sac_vector.size == 3199  # barely anything cut down in SAC check
    assert contrail.size == 429  # grabbed after letting test fail once

    # and level is slighty higher due to level shifting in downwash
    original_level = vector["level"][contrail["index"]]
    new_level = contrail["level"]
    assert np.all(new_level > original_level + 1)

    # not too much higher
    assert np.all(new_level < original_level + 2)


@pytest.fixture
def grid_results(instance_params: dict[str, Any]) -> MetDataset:
    """Run `CocipGrid` on three distinct times."""
    t_step = np.timedelta64(20, "m")
    start_time = np.datetime64("2019-01-01")
    source = CocipGrid.create_source(
        level=[220, 230, 240, 250],
        time=np.arange(start_time, start_time + np.timedelta64(1, "h"), t_step),
        longitude=np.linspace(-35, -25, 40),
        latitude=np.linspace(51, 57, 20),
    )

    gc = CocipGrid(**instance_params)
    return gc.eval(source=source)


@pytest.mark.skipif(not BADA_AVAILABLE, reason="BADA not available")
def test_grid_results(grid_results: MetDataset) -> None:
    """Test `grid_results` fixture.

    It would be nice to split this test into smaller pieces, but we are limited by the
    scope of grid_results (which we only want to run once).
    """
    # Test basic `MetDataset` type properties of `grid_results`.
    assert isinstance(grid_results, MetDataset)
    assert (np.diff(grid_results.data["time"].values) == np.timedelta64(20, "m")).all()

    for variable in ["contrail_age", "ef_per_m"]:
        assert variable in grid_results.data.data_vars

    assert grid_results["ef_per_m"].data.shape == (40, 20, 4, 3)

    # Ensure values are not NaN, and each level contains some initially persistent contrails
    for variable in ["contrail_age", "ef_per_m"]:
        assert np.all(np.isfinite(grid_results[variable].data.values))

        # nonzero somewhere
        da = grid_results[variable].data
        assert np.any(da.values != 0)

    # Test each `MetDataArray`s in `grid_results` have similar binary structure.
    # convert contrail_age to n_steps_with_persistent
    assert grid_results.attrs["dt_integration"] == "5 minutes"
    assert grid_results.attrs["max_age"] == "90 minutes"
    assert grid_results.attrs["bada_model"] == "BADA4"
    assert grid_results.attrs["aircraft_type"] == "B737"
    assert grid_results.attrs["humidity_scaling_name"] == "exponential_boost"
    assert grid_results.attrs["humidity_scaling_formula"]
    assert grid_results.attrs["humidity_scaling_rhi_adj"] == 0.9
    assert grid_results.attrs["humidity_scaling_rhi_boost_exponent"] == 1.7

    persistent = grid_results["contrail_age"].data > 0
    assert persistent.sum() == 605

    # zero outside persistent
    assert np.all(grid_results["ef_per_m"].data.values[~persistent] == 0)

    # positive inside
    assert np.all(grid_results["ef_per_m"].data.values[persistent] > 0)

    # Fixed in version 0.25.0: max age is adhered to by `contrail_age`
    assert np.all(grid_results["contrail_age"].data.values <= 1.5)

    # Pin the distribution of contrail ages
    ages, counts = np.unique(grid_results["contrail_age"].values, return_counts=True)
    np.testing.assert_allclose(ages, [0, 1 / 12, 1.5])
    np.testing.assert_array_equal(counts, [8995, 2, 603])

    # Ensure a hand picked pinned value is realized.
    # This test is HARD to maintain. Delete if it gets too annoying.
    point = grid_results.data.isel(longitude=14, latitude=19, level=2, time=2)
    assert point["contrail_age"].item() == 1.5
    assert point["ef_per_m"].item() == pytest.approx(46576688, rel=1e-3)


@pytest.fixture
def grid_results_segment_free(
    instance_params: dict[str, Any],
    grid_results: MetDataset,
) -> tuple[MetDataset, MetDataset]:
    """Run `CocipGrid` on skeleton of `grid_results` in a segment-free mode."""
    source = MetDataset.from_coords(**grid_results.coords)
    instance_params["azimuth"] = None
    instance_params["segment_length"] = None
    gc = CocipGrid(**instance_params)
    return grid_results, gc.eval(source)


@pytest.mark.skipif(not BADA_AVAILABLE, reason="BADA not available")
def test_grid_results_segment_free(
    grid_results_segment_free: tuple[MetDataset, MetDataset],
) -> None:
    """Test the `grid_results_segment_free` fixture.

    This is really just a sanity check that the segment-free mode is working.
    """
    out1, out2 = grid_results_segment_free
    assert out1.shape == out2.shape
    assert out1.data.data_vars.keys() == out2.data.data_vars.keys()

    assert np.count_nonzero(out1["ef_per_m"].values) == pytest.approx(605, abs=1)
    assert np.count_nonzero(out2["ef_per_m"].values) == pytest.approx(621, abs=1)

    # Pin some values
    da1 = out1["ef_per_m"].data
    filt1 = da1 > 0
    assert da1.where(filt1).mean().item() == pytest.approx(39581584, rel=1e-3)

    # In segment-free mode (generally and here), the mean nonzero EF is slightly lower
    da2 = out2["ef_per_m"].data
    filt2 = da2 > 0
    assert da2.where(filt2).mean().item() == pytest.approx(32914018, rel=1e-3)

    # For this example, we can say something about the distribution of EFs
    filt = filt1 & filt2
    v1 = da1.values[filt]
    v2 = da2.values[filt]
    xr.testing.assert_equal(filt, filt1)
    assert np.all(v1 > 0.5 * v2)
    assert np.all(v1 < 1.9 * v2)


@pytest.fixture
def syn_fl(instance_params: dict[str, Any], source: MetDataset) -> SyntheticFlight:
    """Return synthetic flight."""
    t_start = source.data["time"].values[0]
    t_stop = t_start + np.timedelta64(120, "m")
    bounds = {
        "longitude": source.data["longitude"].values,
        "latitude": source.data["latitude"].values,
        "level": source.data["level"].values,
        "time": np.array([t_start, t_stop]),
    }
    return SyntheticFlight(
        seed=5,
        bada4_path=BADA4_PATH,
        bounds=bounds,
        aircraft_type=CocipGrid.default_params.aircraft_type,
        u_wind=instance_params["met"]["eastward_wind"],
        v_wind=instance_params["met"]["northward_wind"],
    )


@pytest.mark.skipif(not BADA_AVAILABLE, reason="BADA not available")
def test_reasonable_syn_fl(syn_fl: SyntheticFlight, met_cocip1) -> None:
    """Check that syn_fl fixture is reasonable."""
    for _ in range(100):
        fl = syn_fl()
        fl.segment_angle
        u_wind = fl.intersect_met(met_cocip1["eastward_wind"])
        v_wind = fl.intersect_met(met_cocip1["northward_wind"])
        tas = fl.segment_true_airspeed(u_wind=u_wind, v_wind=v_wind)

        assert np.isnan(tas[-1])
        assert np.all(tas[:-1] < 250)
        assert np.all(tas[:-1] > 220)


@pytest.mark.skipif(not BADA_AVAILABLE, reason="BADA not available")
def test_geovector_source(syn_fl: SyntheticFlight, instance_params: dict[str, Any]) -> None:
    """Test `CocipGrid`  with GeoVectorDataset source."""
    # Call synthetic flight many times to create some random data.
    fls = [syn_fl() for _ in range(100)]
    source = GeoVectorDataset.sum(fls)

    # Pin the size
    assert source.size == 2608

    # Run the model
    gc = CocipGrid(**instance_params)
    out = gc.eval(source)
    assert isinstance(out, GeoVectorDataset)
    assert out.size == source.size
    assert "contrail_age" in out
    assert "ef_per_m" in out

    persistent = out["contrail_age"] > np.timedelta64(0, "ns")
    assert persistent.sum() == 72

    # Contrail age and positive EF are 1-1
    assert np.all(out["ef_per_m"][persistent] > 0)
    assert np.all(out["ef_per_m"][~persistent] == 0)

    # Pin the mean EF
    assert out["ef_per_m"].mean().item() == pytest.approx(875854, abs=10)
    assert out["ef_per_m"][persistent].mean().item() == pytest.approx(31725403, abs=10)


@pytest.mark.filterwarnings("ignore:invalid value encountered in remainder")
@pytest.mark.skipif(not BADA_AVAILABLE, reason="BADA not available")
def test_grid_against_flight(
    syn_fl: SyntheticFlight,
    instance_params: dict[str, Any],
    met_cocip1: MetDataset,
    rad_cocip1: MetDataset,
    bada_model: AircraftPerformance,
) -> None:
    """Confirm some agreement between traditional CoCiP and gridded CoCiP.

    Only if a waypoint is contrail forming, not the EF generated by waypoint.
    """
    # common cocip params
    params = {
        "max_age": instance_params["max_age"],
        "dt_integration": instance_params["dt_integration"],
        "humidity_scaling": ExponentialBoostHumidityScaling(rhi_adj=0.7),
        "interpolation_bounds_error": True,
    }

    # create the aircraft performance model from BADAFlight
    from pycontrails.ext.bada import BADAFlight

    bada_model = BADAFlight(bada4_path=instance_params["bada4_path"])

    # expect some persistent contrails
    n_fls_with_contrails = 0

    for _ in range(10):
        fl = syn_fl()
        fl["azimuth"] = fl.segment_azimuth()
        fl["true_airspeed"] = fl.segment_true_airspeed(
            fl.intersect_met(met_cocip1["eastward_wind"]),
            fl.intersect_met(met_cocip1["northward_wind"]),
        )

        cocip = Cocip(met_cocip1, rad_cocip1, params, aircraft_performance=bada_model)
        cocip_fl = cocip.eval(fl)
        cocip_fl_state = cocip_fl["cocip"].astype(bool)

        cg = CocipGrid(
            met=met_cocip1,
            rad=rad_cocip1,
            **params,
            bada4_path=instance_params["bada4_path"],
            verbose_outputs_formation="specific_humidity",
            verbose_outputs_evolution=True,
        )
        grid_fl = cg.eval(source=fl)
        grid_fl_state = grid_fl["ef_per_m"] != 0

        # Specifically check for humidity issues -- this has cropped up in the past
        q1 = cocip.source["specific_humidity"]
        q2 = grid_fl["specific_humidity"]
        np.testing.assert_array_equal(q1, q2)
        if hasattr(cocip, "_downwash_contrail"):
            q1 = cocip._downwash_contrail["specific_humidity"]
            q2 = cg.contrail.groupby("index")["specific_humidity"].first()

            # There can be slight inconsistencies in initially persistent.
            # This happens on one of the 10 flights. The Cocip model
            # predicts one more initially persistent waypoint compared with
            # the CocipGrid model. This is why we have some logic
            # below to check the sizes.
            if q1.size != q2.size:
                assert q1.size == q2.size + 1
                # Reassign q1, removing the extra waypoint
                q1 = cocip._downwash_contrail.dataframe.set_index("waypoint").loc[
                    q2.index, "specific_humidity"
                ]
            np.testing.assert_allclose(q1, q2, rtol=1e-8, atol=1e-5)

        # And there are at most three waypoint that doesn't show consistent
        # This is caused by continuity conventions
        assert np.sum(cocip_fl_state != grid_fl_state) <= 3
        if cocip_fl_state.sum():
            n_fls_with_contrails += 1

    assert n_fls_with_contrails == 8  # and we saw lots of persistent contrails


@pytest.mark.skipif(not BADA_AVAILABLE, reason="BADA not available")
def test_verbose_outputs_sac(instance_params: dict[str, Any], source: MetDataset):
    """Confirm the verbose_outputs parameter works with "sac" variable."""
    instance_params["verbose_outputs_formation"] = ["sac", "sdr"]

    model = CocipGrid(**instance_params)
    with pytest.warns(UserWarning, match="verbose_outputs"):
        ds = model.eval(source=source).data

    assert model.params["verbose_outputs_formation"] == {"sac"}
    assert isinstance(ds, xr.Dataset)

    assert "sac" in ds
    assert len(ds) == 3
    assert ds["sac"].sum() == 3199  # agrees with test_calc_first_contrail


@pytest.mark.skipif(not BADA_AVAILABLE, reason="BADA not available")
def test_verbose_outputs_formation(instance_params: dict[str, Any], source: MetDataset):
    """Confirm each verbose_outputs parameter is attached to results."""
    instance_params["verbose_outputs_formation"] = True
    model = CocipGrid(**instance_params)
    out = model.eval(source=source)

    expected = {
        "T_crit_sac",
        "persistent",
        "nvpm_ei_n",
        "sac",
        "engine_efficiency",
        "true_airspeed",
        "fuel_flow",
        "specific_humidity",
        "air_temperature",
        "rhi",
        "iwc",
    }
    assert model.params["verbose_outputs_formation"] == expected

    assert len(out) == len(expected) + 2  # verbose formation + 2 core (ef, age)
    for var in model.params["verbose_outputs_formation"]:
        assert var in out

    # Everything is a float
    ds = out.data
    for data in ds.values():
        assert data.dtype in ["float32", "float64"]

    # Pin a few values
    rel = 1e-4
    assert ds["T_crit_sac"].mean() == pytest.approx(222.54, rel=rel)
    assert ds["persistent"].sum() == 429
    assert (ds["contrail_age"] > 0).sum() == 235
    assert np.isfinite(ds["nvpm_ei_n"]).all()
    assert ds["nvpm_ei_n"].median() == pytest.approx(1.29718e15, rel=rel)
    assert ds["fuel_flow"].mean() == pytest.approx(0.6037, rel=rel)
    assert ds["rhi"].mean() == pytest.approx(0.6273, rel=rel)
    assert ds["iwc"].mean() == pytest.approx(4.08e-06, rel=rel)
