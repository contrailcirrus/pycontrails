"""Support methods for validation scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from pycontrails import Flight
from pycontrails.models import Cocip
from pycontrails.physics import units

FLIGHT_TOLERANCE = 1e-4
CONTRAIL_TOLERANCE = 1e-3


def create_flight(flight_id: str, df_flight: pd.DataFrame) -> Flight:
    """
    Function to create a Flight class from a
    group of rows with the same Flight ID
    """

    # constant properties along the length of the flight
    attrs = {
        "flight_id": flight_id,
        # get first val of atyp col as aircraft type
        "aircraft_type": df_flight["ATYP"].values[0],
    }

    # # DISCUSSION: We could use the Aircraft class here
    # aircraft = Aircraft(aircraft_type=df_flight["ATYP"].values[0])

    # convert UTC timestamp to np.datetime64
    df_flight["time"] = pd.to_datetime(df_flight["date_time"])

    # get altitude in m
    df_flight["altitude"] = df_flight["alt_ft"] * 0.3048

    # rename a few columns for compatibility with `Flight` requirements
    df_flight = df_flight.rename(columns={"lon_deg": "longitude", "lat_deg": "latitude"})

    # clean up a few columns before building Flight class
    df_flight = df_flight.drop(columns=["Flight ID", "date_time", "ATYP", "alt_ft"])

    # create a Flight class
    return Flight(data=df_flight, attrs=attrs)


def validate_output(
    cocip: Cocip,
    pycocip_flight: pd.DataFrame | None,
    pycocip_flight_stats: pd.Series,
    pycocip_contrail: pd.DataFrame,
) -> None:
    # ---
    # Flight waypoint validation
    # ---
    flight = cocip.flight

    # Test flight waypoints
    if pycocip_flight is not None:
        assert_array_equal(flight["time"], pd.to_datetime(pycocip_flight["time"]), err_msg="time")
        assert_allclose(flight["longitude"], pycocip_flight["lon"], err_msg="longitude")
        assert_allclose(flight["latitude"], pycocip_flight["lat"], err_msg="latitude")
        assert_allclose(
            units.m_to_ft(flight.altitude), pycocip_flight["alt_ft"], err_msg="altitude"
        )
        assert_allclose(flight.air_pressure, pycocip_flight["pressure_pa"], err_msg="level")
        assert_allclose(
            flight["air_temperature"], pycocip_flight["temperature"], err_msg="air_temperature"
        )
        assert_allclose(flight["u_wind"], pycocip_flight["winds_u"], err_msg="u_wind")
        assert_allclose(flight["v_wind"], pycocip_flight["winds_v"], err_msg="v_wind")
        assert_allclose(
            flight["segment_length"], pycocip_flight["seg_length"], err_msg="segment_length"
        )
        assert_allclose(flight["sin_a"], pycocip_flight["sin_a"], err_msg="sin_a")
        assert_allclose(flight["cos_a"], pycocip_flight["cos_a"], err_msg="cos_a")
        assert_allclose(
            flight["true_airspeed"],
            pycocip_flight["tas_ms"],
            rtol=FLIGHT_TOLERANCE,
            err_msg="true_airspeed",
        )
        assert_allclose(
            flight["aircraft_mass"],
            pycocip_flight["amass"],
            rtol=FLIGHT_TOLERANCE,
            err_msg="aircraft_mass",
        )
        assert_allclose(
            flight["fuel_flow"],
            pycocip_flight["fuel_flow"],
            rtol=FLIGHT_TOLERANCE,
            err_msg="fuel_flow",
        )
        assert_allclose(
            flight["engine_efficiency"],
            pycocip_flight["ope"],
            rtol=FLIGHT_TOLERANCE,
            err_msg="engine_efficiency",
        )
        assert_allclose(
            flight["thrust"],
            pycocip_flight["thrust_setting"],
            rtol=FLIGHT_TOLERANCE,
            err_msg="thrust",
        )
        assert_allclose(
            pycocip_flight["bc_ei_n"],
            flight["nvpm_ei_n"],
            rtol=FLIGHT_TOLERANCE,
            err_msg="nvpm_ei_n",
        )
        assert np.all((flight["sac"] == 1) == pycocip_flight["has_initial_contrails"])
        assert_allclose(
            flight["T_critical_sac"],
            pycocip_flight["temp_lc"],
            rtol=FLIGHT_TOLERANCE,
            err_msg="T_critical_sac",
        )
        assert_allclose(
            flight["rhi"],
            pycocip_flight["initial_rhi"],
            rtol=CONTRAIL_TOLERANCE,
            err_msg="rhi",
        )
        # assert_allclose(flight.data["nox_ei"], pycocip_flight["nox_ei"])

    # Test flight stats
    assert flight.attrs["aircraft_type"] == pycocip_flight_stats["ATYP"]
    assert flight.attrs["fuel_data_source"] == pycocip_flight_stats["Fuel methodology"]
    assert flight.attrs["engine_name"] == pycocip_flight_stats["Engine name"]
    assert flight.attrs["bc_data_source"] == pycocip_flight_stats["BC methodology"]
    assert flight.attrs["aircraft_type_bada"] == pycocip_flight_stats["ATYP assumed"]

    assert flight["time"][0] == pd.to_datetime(pycocip_flight_stats["First wypt"])
    assert flight["time"][-1] == pd.to_datetime(pycocip_flight_stats["Last wypt"])
    assert_allclose(
        flight.length / 1000.0,
        pycocip_flight_stats["Total flight dist (km)"],
        rtol=FLIGHT_TOLERANCE,
        err_msg="length",
    )
    assert_allclose(
        flight.attrs["total_fuel_burn"],
        pycocip_flight_stats["Total fuel burn (kg)"],
        rtol=FLIGHT_TOLERANCE,
        err_msg="total_fuel_burn",
    )
    assert_allclose(
        np.nanmean(flight["nvpm_ei_n"]),
        pycocip_flight_stats["BC EI_n mean (kg-1)"],
        rtol=FLIGHT_TOLERANCE,
        err_msg="nvpm_ei_n mean",
    )
    assert_allclose(
        np.nanmean(flight["aircraft_mass"]),
        pycocip_flight_stats["Aircraft mass mean (kg)"],
        rtol=FLIGHT_TOLERANCE,
        err_msg="aircraft_mass mean",
    )
    assert_allclose(
        np.nanmean(flight["engine_efficiency"]),
        pycocip_flight_stats["OPE mean"],
        rtol=FLIGHT_TOLERANCE,
        err_msg="engine_efficiency mean",
    )

    # return if no contrail
    if not len(pycocip_contrail):
        assert cocip.contrail is None or not len(cocip.contrail)
        return

    # ---
    # Contrail waypoint validation
    # ---
    contrail = cocip.contrail
    assert contrail is not None

    # waypoints
    original_waypoints = contrail["waypoint"].apply(lambda w: flight["wypt_id"][w])
    assert_allclose(original_waypoints, pycocip_contrail["wypt_uid"], err_msg="original_waypoints")

    # location
    assert_allclose(
        contrail["latitude"], pycocip_contrail["lat"], rtol=CONTRAIL_TOLERANCE, err_msg="latitude"
    )
    assert_allclose(
        contrail["longitude"], pycocip_contrail["lon"], rtol=CONTRAIL_TOLERANCE, err_msg="longitude"
    )
    assert_allclose(
        units.m_to_ft(contrail["altitude"]),
        pycocip_contrail["alt_ft"].values,
        rtol=CONTRAIL_TOLERANCE,
        err_msg="altitude",
    )

    # continuity
    assert_array_equal(contrail["continuous"], pycocip_contrail["continuous"], err_msg="continuous")
    continuous = contrail["continuous"].values
    assert np.all(contrail[~continuous]["ef"] == 0)

    # downselect contrail for only continuous points
    # pycocip has incorrect values in these locations
    contrail = contrail[continuous]
    pycocip_contrail = pycocip_contrail[
        continuous
    ]  # using `continuous` here since we've already asserted that pycontrails/pycocip agree here

    # geometry
    assert_allclose(
        contrail["segment_length"],
        pycocip_contrail["seg_length"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="segment_length",
    )
    assert_allclose(
        contrail["sin_a"], pycocip_contrail["sin_a"], rtol=CONTRAIL_TOLERANCE, err_msg="sin_a"
    )
    assert_allclose(
        contrail["cos_a"], pycocip_contrail["cos_a"], rtol=CONTRAIL_TOLERANCE, err_msg="cos_a"
    )

    # met
    assert_allclose(
        contrail["air_temperature"],
        pycocip_contrail["temp"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="air_temperature",
    )
    assert_allclose(
        contrail["u_wind"], pycocip_contrail["winds_u"], rtol=CONTRAIL_TOLERANCE, err_msg="u_wind"
    )
    assert_allclose(
        contrail["v_wind"], pycocip_contrail["winds_v"], rtol=CONTRAIL_TOLERANCE, err_msg="v_wind"
    )
    assert_allclose(
        contrail["vertical_velocity"],
        pycocip_contrail["omega"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="vertical_velocity",
    )
    assert_allclose(
        contrail["specific_humidity"],
        pycocip_contrail["q"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="specific_humidity",
    )
    assert_allclose(
        contrail["rhi"], pycocip_contrail["rhi"], rtol=CONTRAIL_TOLERANCE, err_msg="rhi"
    )
    assert_allclose(
        contrail["dt_dz"], pycocip_contrail["dt_dz"], rtol=CONTRAIL_TOLERANCE, err_msg="dt_dz"
    )
    assert_allclose(
        contrail["ds_dz"], pycocip_contrail["ds_dz"], rtol=CONTRAIL_TOLERANCE, err_msg="ds_dz"
    )
    assert_allclose(
        contrail["dsn_dz"], pycocip_contrail["dsn_dz"], rtol=CONTRAIL_TOLERANCE, err_msg="dsn_dz"
    )
    assert_allclose(
        contrail["tau_cirrus"],
        pycocip_contrail["tau_cirrus"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="tau_cirrus",
    )

    # contrail properties
    assert_allclose(
        contrail["iwc"], pycocip_contrail["iwc"], rtol=CONTRAIL_TOLERANCE, err_msg="iwc"
    )
    assert_allclose(
        contrail["width"], pycocip_contrail["width"], rtol=CONTRAIL_TOLERANCE, err_msg="width"
    )
    assert_allclose(
        contrail["depth"], pycocip_contrail["depth"], rtol=CONTRAIL_TOLERANCE, err_msg="depth"
    )
    assert_allclose(
        contrail["plume_mass_per_m"],
        pycocip_contrail["plume_mass_per_m"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="plume_mass_per_m",
    )
    assert_allclose(
        contrail["r_ice_vol"],
        pycocip_contrail["r_ice_vol"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="r_ice_vol",
    )
    assert_allclose(
        contrail["tau_contrail"],
        pycocip_contrail["tau_contrail"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="tau_contrail",
    )
    assert_allclose(
        contrail["n_ice_per_m"],
        pycocip_contrail["n_ice_per_m"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="n_ice_per_m",
    )
    assert_allclose(
        contrail["n_ice_per_vol"],
        pycocip_contrail["n_ice_per_vol"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="n_ice_per_vol",
    )
    assert_allclose(
        contrail["n_ice_per_kg_air"],
        pycocip_contrail["n_ice_per_kg_air"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="n_ice_per_kg_air",
    )
    assert_allclose(
        contrail["age"] / np.timedelta64(1, "h"),
        pycocip_contrail["age_hrs"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="age",
    )
    assert_allclose(
        contrail["age_hours"],
        pycocip_contrail["age_hrs"],
        rtol=CONTRAIL_TOLERANCE,
        err_msg="age_hours",
    )

    # radiative properties
    assert_allclose(
        contrail["sdr"], pycocip_contrail["sdr"], rtol=CONTRAIL_TOLERANCE, err_msg="sdr"
    )
    assert_allclose(
        contrail["rsr"], pycocip_contrail["rsr"], rtol=CONTRAIL_TOLERANCE, err_msg="rsr"
    )
    assert_allclose(
        contrail["olr"], pycocip_contrail["olr"], rtol=CONTRAIL_TOLERANCE, err_msg="olr"
    )
    assert_allclose(
        contrail["rf_sw"], pycocip_contrail["rf_sw"], rtol=CONTRAIL_TOLERANCE, err_msg="rf_sw"
    )
    assert_allclose(
        contrail["rf_lw"], pycocip_contrail["rf_lw"], rtol=CONTRAIL_TOLERANCE, err_msg="rf_lw"
    )
    assert_allclose(
        contrail["rf_net"], pycocip_contrail["rf_net"], rtol=CONTRAIL_TOLERANCE, err_msg="rf_net"
    )

    # energy forcing
    assert_allclose(contrail["ef"], pycocip_contrail["ef"], rtol=CONTRAIL_TOLERANCE, err_msg="ef")
