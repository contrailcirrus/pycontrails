"""Test PS model optimizer."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import pycontrails.models.ps_model.ps_model as ps
import pycontrails.models.ps_model.ps_optimizer as ps_opt
from pycontrails import Flight, MetDataset
from pycontrails.physics import units

from .conftest import get_static_path


@pytest.fixture(scope="module")
def met() -> MetDataset:
    return MetDataset(xr.open_dataset(get_static_path("met-era5-cocip1.nc")))


@pytest.fixture(scope="module")
def fl() -> Flight:
    date = "2019-01-01"
    times = (
        "02:23:00",
        "02:25:08",
        "02:28:33",
        "02:31:12",
        "02:33:12",
        "02:43:23",
        "02:47:47",
        "02:57:20",
        "03:05:35",
        "03:15:33",
        "03:25:35",
        "03:40:08",
        "03:41:30",
    )
    dt = [np.datetime64(f"{date}T{t}") for t in times]
    altitude_ft = [
        28800,
        30350,
        34950,
        35000,
        35000,
        35025,
        35000,
        35000,
        35000,
        34975,
        35000,
        30300,
        27800,
    ]
    lat = [
        58.979550,
        58.8452,
        58.5296,
        58.2673,
        58.0698,
        57.0453,
        56.9806,
        56.191,
        55.5648,
        55.4782,
        54.6092,
        53.1028,
        52.999434,
    ]
    lon = [
        -31.321450,
        -31.4629,
        -31.7836,
        -31.984,
        -32.124,
        -32.8288,
        -33.5288,
        -34.0278,
        -35.0063,
        -36.8239,
        -38.1261,
        -38.4852,
        -38.560234,
    ]
    atype = "A320"

    fl = Flight(
        time=dt,
        latitude=lat,
        longitude=lon,
        altitude=[units.ft_to_m(a) for a in altitude_ft],
        aircraft_type=atype,
        flight_id="0",
    )
    return fl.resample_and_fill()


@pytest.fixture(scope="module")
def cocip_grid(fl: Flight) -> MetDataset:
    grid = MetDataset.from_coords(
        level=units.ft_to_pl(np.arange(270, 410, 10) * 100),
        time=fl["time"],
        longitude=fl["longitude"],
        latitude=fl["latitude"],
    )
    grid["ef_per_m"] = xr.DataArray(np.zeros(grid.shape), coords=grid.coords)
    grid.data["ef_per_m"][:, :, 9, :] = 1e8
    grid.data["ef_per_m"][3:5, :, 2:5, :] = 5e8
    grid.data["ef_per_m"][4:5, :, 6, :] = -1e8

    return grid


def test_opt_mach(met: MetDataset) -> None:
    ps_mod = ps.PSFlight(met=met)

    atype = "A320"
    atyp_param = ps_mod.aircraft_engine_params[atype]
    alt_ft = 31000.0
    air_pressure = 28744.0
    aircraft_mass = 60000.0
    cost_index = [0.0, 10.0, 70.0, 200.0]
    air_temperature = 221.0
    headwind = 16.0

    test_vals = [0.71, 0.74, 0.82, 0.83]
    for tv, ci in zip(test_vals, cost_index):
        m_opt = ps_opt.opt_mach(
            atyp_param,
            ps_mod,
            atype,
            alt_ft,
            air_pressure,
            aircraft_mass,
            ci,
            air_temperature,
            headwind,
        )
        assert pytest.approx(m_opt) == tv

    # Too high for weight
    ci = 70.0
    alt_ft = 42000.0
    m_opt = ps_opt.opt_mach(
        atyp_param,
        ps_mod,
        atype,
        alt_ft,
        units.ft_to_pl(alt_ft),
        aircraft_mass,
        ci,
        air_temperature,
        headwind,
    )
    assert np.isnan(m_opt)


def test_optimizer_grid(fl: Flight, cocip_grid: MetDataset, met: MetDataset) -> None:
    aircraft_mass = 70000.0
    climb_rate = 1000.0

    # These min and max values are outside met domain
    min_alt_ft = 27000.0
    max_alt_ft = 42000.0
    grid = ps_opt._build_grid(
        fl, aircraft_mass, min_alt_ft, max_alt_ft, climb_rate, met, cocip_grid["ef_per_m"]
    )
    assert grid["nx"] == 19
    assert pytest.approx(grid["dx"]) == 48680.74420683
    assert grid["nz"] == 31
    assert pytest.approx(grid["dz"]) == 500.0
    point = ps_opt._get_grid_point(0, 0, grid)
    assert point["time"] == np.datetime64("2019-01-01T02:23:00")
    assert pytest.approx(point["altitude_ft"]) == 27000.0
    assert np.isnan(point["air_temperature"])

    # These match the met domain
    min_alt_ft = 31000.0
    max_alt_ft = 38000.0
    grid = ps_opt._build_grid(
        fl, aircraft_mass, min_alt_ft, max_alt_ft, climb_rate, met, cocip_grid["ef_per_m"]
    )
    assert grid["nz"] == 15
    point = ps_opt._get_grid_point(0, 0, grid)
    assert pytest.approx(point["headwind"]) == 16.017722576
    assert pytest.approx(point["ef_per_m"]) == 1e8


def test_bad_start(fl: Flight, cocip_grid: MetDataset, met: MetDataset) -> None:
    aircraft_mass = 70000.0
    climb_rate = 1000.0
    fl_restrict = 1000.0
    contrail_scale = 1.0
    cost_index = 1.0
    min_seg_time = 10

    # These min and max values are outside met domain
    min_alt_ft = 27000.0
    max_alt_ft = 42000.0
    with pytest.raises(ValueError, match="Start of search outside met domain."):
        ps_opt.run_search(
            fl,
            met,
            cocip_grid["ef_per_m"],
            aircraft_mass,
            contrail_scale,
            min_alt_ft,
            max_alt_ft,
            climb_rate,
            fl_restrict,
            min_seg_time,
            cost_index,
        )

    # Now aircraft weight is too high
    min_alt_ft = 31000.0
    max_alt_ft = 38000.0
    aircraft_mass = 130000.0
    with pytest.raises(ValueError, match="Aircraft cannot fly at starting altitude/weight."):
        ps_opt.run_search(
            fl,
            met,
            cocip_grid["ef_per_m"],
            aircraft_mass,
            contrail_scale,
            min_alt_ft,
            max_alt_ft,
            climb_rate,
            fl_restrict,
            min_seg_time,
            cost_index,
        )


def test_allowed_actions(fl: Flight, cocip_grid: MetDataset, met: MetDataset) -> None:
    aircraft_mass = 60000.0
    climb_rate = 1000.0
    fl_restrict = 1000.0
    contrail_scale = 1.0
    cost_index = 1.0
    min_seg_time = 10
    min_alt_ft = 31000.0
    max_alt_ft = 38000.0
    grid = ps_opt.run_search(
        fl,
        met,
        cocip_grid["ef_per_m"],
        aircraft_mass,
        contrail_scale,
        min_alt_ft,
        max_alt_ft,
        climb_rate,
        fl_restrict,
        min_seg_time,
        cost_index,
    )

    # Can cruise or climb
    seq = ps_opt._get_allowed_actions(grid, 0, 0, fl_restrict, min_seg_time)
    assert seq == [0, 1, 3]

    # Test at x=1, z=0 - last choice was cruise
    # Must continue cruise to meet segment time restriction
    seq = ps_opt._get_allowed_actions(grid, 1, 0, fl_restrict, min_seg_time)
    assert len(seq) == 1
    assert seq == [0]

    # Test at x=1, z=1 - last choice was climb
    # We must continue climb to fl_restrict level
    seq = ps_opt._get_allowed_actions(grid, 1, 1, fl_restrict, min_seg_time)
    assert seq == [2, 4]

    # Test at x=2, z=2 - last choice was climb
    # We are at fl_restrict so we can cruise
    seq = ps_opt._get_allowed_actions(grid, 2, 2, fl_restrict, min_seg_time)
    assert seq == [2, 3, 5]

    # Test at x=3, z=2 - last choice was climb
    # Must continue cruise to meet segment time restriction
    seq = ps_opt._get_allowed_actions(grid, 3, 2, fl_restrict, min_seg_time)
    assert seq == [2]

    # Test at x=6, z=2 - last choice was climb
    # Past time restriction so we can cruise, climb, descend
    seq = ps_opt._get_allowed_actions(grid, 6, 2, fl_restrict, min_seg_time)
    assert seq == [2, 3, 5, 1]

    # Test at x=7, z=7 - last choice was climb
    # Past time restriction so we can cruise, climb, descend
    seq = ps_opt._get_allowed_actions(grid, 10, 14, fl_restrict, min_seg_time)
    assert seq == [14, 11]
