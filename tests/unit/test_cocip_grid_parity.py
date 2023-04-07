"""Ensure parity between Cocip and CocipGrid models."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from pycontrails.models.cocipgrid import CocipGrid
except ImportError:
    pytest.skip("CocipGrid not available", allow_module_level=True)

from pycontrails import Flight, MetDataset
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import ExponentialBoostHumidityScaling
from pycontrails.utils.synthetic_flight import SyntheticFlight
from tests import BADA4_PATH

# Found by running big for-loop over many seeds
# Each of these seeds creates a contrail-producing flight
# Only running on a subset of seeds to keep tests short
seeds = [35, 51, 184, 257, 324, 365, 409]
# 414  # this seed actually fails! The ones below succeed.
# 423
# 479
# 533
# 593
# 658
# 701
# 782
# 796
# 806
# 849
# 851
# 914
# 957
# 960


@pytest.fixture(scope="module")
def met(met_cocip1_module_scope: MetDataset):
    """Namespace convenience."""
    return met_cocip1_module_scope


@pytest.fixture(scope="module")
def rad(rad_cocip1_module_scope: MetDataset):
    """Namespace convenience."""
    return rad_cocip1_module_scope


@pytest.fixture(params=seeds)
def fl(met: MetDataset, request) -> Flight:
    """Create synthetic flight bounded by `met`."""
    # cocip_met has 13 time slices
    assert met["time"].size == 13

    # Explicitly slice longitude and latitude in order to keep trajectory in bounds
    # when running Cocip. In this test, we set interpolation_bounds_error=True
    bounds: dict[str, np.ndarray] = {
        "longitude": met["longitude"].values[1:-1],
        "latitude": met["latitude"].values[1:-1],
        "level": np.array([250]),
        "time": np.array([met["time"].values[0], met["time"].values[6]]),
    }

    syn = SyntheticFlight(
        bounds,
        "A320",
        u_wind=met["eastward_wind"],
        v_wind=met["northward_wind"],
        seed=request.param,
        bada4_path=BADA4_PATH,
    )
    return syn()


@pytest.mark.skipif(not BADA4_PATH.is_dir(), reason="BADA4 not available")
def test_parity(fl: Flight, met: MetDataset, rad: MetDataset):
    """Ensure substantial parity between `Cocip` and `CocipGrid`."""

    fl["azimuth"] = fl.segment_azimuth()

    # Run BADA up front
    from pycontrails.ext.bada import BADAFlight

    bada_model = BADAFlight(met=met, bada4_path=BADA4_PATH)
    fl = bada_model.eval(fl)

    # Confirm that fl has no additional variables
    assert len(fl.data) == 16

    model_params = {
        "dt_integration": np.timedelta64(2, "m"),  # keep small to ensure parity
        "max_age": np.timedelta64(1, "h"),
        "interpolation_bounds_error": True,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
    }
    cocip = Cocip(met=met, rad=rad, **model_params, aircraft_performance=bada_model)
    out1 = cocip.eval(fl)

    # source data has been copied on eval
    assert len(fl.data) == 16
    assert len(cocip.source.data) == 68
    downwash1 = cocip._downwash_contrail.dataframe.set_index("waypoint")

    cg = CocipGrid(
        met=met,
        rad=rad,
        **model_params,
        bada4_path=BADA4_PATH,
        verbose_outputs_evolution=True,
        verbose_outputs_formation=True,
    )
    out2 = cg.eval(fl)
    assert len(fl.data) == 16
    assert len(cg.source.data) == 25

    # Clunky but reliable way to access CocipGrid downwash data
    downwash2 = cg.contrail.groupby("index").first()

    assert len(downwash1) == len(downwash2)
    np.testing.assert_array_equal(downwash1.index, downwash2.index)

    # ------------------------------------------------------------
    # Confirm exact agreement on verbose formation outputs
    # ------------------------------------------------------------

    # Ignore anything fl came with -- those are already identical
    common_keys = set(cg.source).intersection(cocip.source).difference(fl)
    assert common_keys == {"contrail_age", "nvpm_ei_n", "sac", "specific_humidity", "rhi"}
    exclude_keys = ["contrail_age"]  # This is checked below
    for key in common_keys.difference(exclude_keys):
        np.testing.assert_array_equal(cocip.source[key], cg.source[key], err_msg=key)

    # ---------------------------------------------------------
    # Confirm very tight agreement between initially persistent
    # ---------------------------------------------------------

    common_keys = set(downwash1).intersection(downwash2)
    assert len(common_keys) == 43
    exclude_keys = ["segment_length", "dsn_dz", "time", "formation_time"]

    for key in common_keys.difference(exclude_keys):
        np.testing.assert_array_equal(downwash1[key], downwash2[key], err_msg=key)

    different_keys = [
        ("vertical_velocity", "lagrangian_tendency_of_air_pressure"),
        ("u_wind", "eastward_wind"),
        ("u_wind_lower", "eastward_wind_lower"),
        ("v_wind", "northward_wind"),
        ("v_wind_lower", "northward_wind_lower"),
    ]
    for key1, key2 in different_keys:
        msg = f"{key1} and {key2} do not agree"
        np.testing.assert_array_equal(downwash1[key1], downwash2[key2], err_msg=msg)

    # The segment_length variable is fundamentally different
    # And dsn_dz is different as a consequence of the Cocip continuity conventions
    assert np.all(downwash2["segment_length"] == cg.params["segment_length"])
    continuous = downwash1["continuous"]
    np.testing.assert_allclose(
        downwash1["dsn_dz"][continuous],
        downwash2["dsn_dz"][continuous],
        atol=1e-6,
    )
    assert np.all(downwash1["dsn_dz"][~continuous] == 0)
    assert np.all(downwash2["dsn_dz"][~continuous] != 0)

    # No negative EF
    assert not np.any(out1["ef"] < 0)
    assert not np.any(out2["ef_per_m"] < 0)

    # -------------------------------------------------
    # Confirm general agreement between model EF output
    # -------------------------------------------------

    # Convert ef_per_m -> ef_per_waypoint
    ef_per_waypoint = out2["ef_per_m"] * out1["segment_length"]

    # Ignore waypoints with small EF -- these often arise from continuity conventions
    filt = out1["ef"] > 1e10
    assert np.sum(filt) > 5  # at least 5 waypoints with substantial EF

    # Apart from these continuity issues, close agreement
    # This assertion is critical ... it is the glue holding together Cocip and CocipGrid
    # The rtol cannot be improved much beyond the threshold below
    # Primary differences are due to continuity conventions
    # Secondary differences are due to different specific_humidity values between the two models
    ef1 = out1["ef"][filt]
    ef2 = ef_per_waypoint[filt]
    np.testing.assert_allclose(ef1, ef2, rtol=0.1)

    # ----------------------------------------------------
    # Confirm general agreement between model contrail age
    # ----------------------------------------------------

    # Borrow the filter from above
    age1 = out1["contrail_age"][filt]
    age2 = out2["contrail_age"][filt]
    np.testing.assert_array_equal(age1, age2, err_msg="contrail_age")
