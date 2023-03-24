"""
Benchmark test data for `Cocip` Model.

See README.md for setup instructions.
"""

from __future__ import annotations

import logging
import pathlib

import pandas as pd

from pycontrails import Fleet, Flight, MetDataset
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.ext.bada import BADAFlight
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import ExponentialBoostLatitudeCorrectionHumidityScaling
from pycontrails.physics import units

# set up logging
LOG = logging.getLogger("pycontrails")
logging.basicConfig(
    # filename="benchmark.log",
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# setup
INPUT_PATH = pathlib.Path("inputs")
OUTPUT_PATH = pathlib.Path("outputs")
FLIGHT_METADATA_FILE = "flight-metadata.pq"
FLIGHT_WAYPOINT_FILE = "flight-waypoints.pq"

# ----
# Load Flights
# ----


def load_flights() -> list[Flight]:
    """Load flight data from input file.

    Returns
    -------
    list[Flight]
        List of Flight objects loaded from csv file
    """
    LOG.info("Opening flight data")

    # load flight metadata
    df_flights_metadata = pd.read_parquet(INPUT_PATH / "flight" / FLIGHT_METADATA_FILE)
    df_flights_metadata.set_index("flight_id", inplace=True)

    # Load flight waypoints
    df_flights_waypoints = pd.read_parquet(INPUT_PATH / "flight" / FLIGHT_WAYPOINT_FILE)

    # get altitude in m
    df_flights_waypoints["altitude"] = units.ft_to_m(df_flights_waypoints["altitude_ft"])

    # load datetimes
    df_flights_waypoints["time"] = pd.to_datetime(df_flights_waypoints["time"])

    cols_wypt = ["flight_id", "longitude", "latitude", "altitude", "time"]
    flights = []

    # load metadata and create Flight class
    for flt_id, df_flt_wypts in df_flights_waypoints.groupby("flight_id"):
        # Aircraft and engine properties
        attrs = {
            "flight_id": flt_id,
            "aircraft_type": df_flights_metadata.loc[flt_id]["aircraft_type_icao"],
            "engine_name": df_flights_metadata.loc[flt_id]["engine_type_edb"],
            "engine_uid": df_flights_metadata.loc[flt_id]["engine_uid"],
            "load_factor": df_flights_metadata.loc[flt_id]["assumed_load_factor"],
        }

        # Create flight object
        fl = Flight(data=df_flt_wypts[cols_wypt], attrs=attrs)
        flights.append(fl)

    return flights


# ----
# Load ERA5
# ----


def load_ERA5() -> tuple[MetDataset, MetDataset]:
    """Load meteorology and radiation datasets.

    Returns
    -------
    tuple(MetDataset, MetDataset):
        Meteorology on pressure levels, Radiation on single level
    """
    LOG.info("Opening meteorology and radiation datasets")

    time = ("2020-01-01 00:00:00", "2020-01-02 08:00:00")
    pressure_levels = [
        1000,
        975,
        950,
        925,
        900,
        875,
        850,
        825,
        800,
        775,
        750,
        700,
        650,
        600,
        550,
        500,
        450,
        400,
        350,
        300,
        250,
        225,
        200,
        175,
        150,
        125,
        100,
    ]

    # load local ERA5 files for this domain
    # Note the files at "paths=" must be directly from ECMWF (not processed by pycontrails first)
    era5pl = ERA5(
        time=time,
        variables=Cocip.met_variables,
        pressure_levels=pressure_levels,
        paths=list(pathlib.Path(INPUT_PATH / "met").glob("*.nc")),
        cachestore=None,
    )
    era5sl = ERA5(
        time=time,
        variables=Cocip.rad_variables,
        paths=list(pathlib.Path(INPUT_PATH / "rad").glob("*.nc")),
        cachestore=None,
    )

    # create `MetDataset` from sources
    met = era5pl.open_metdataset(xr_kwargs=dict(parallel=False, format="NETCDF3_CLASSIC"))
    rad = era5sl.open_metdataset(xr_kwargs=dict(parallel=False, format="NETCDF3_CLASSIC"))

    return met, rad


# ----
# Cocip
# ----


def run_cocip() -> Cocip:
    """Run Cocip model on flights.

    Returns
    -------
    Cocip
        Evaluated cocip model
    """

    flights = load_flights()
    met, rad = load_ERA5()

    # create Aircraft Performance model
    aircraft_performance = BADAFlight(
        bada3_path=INPUT_PATH / ".." / "bada" / "bada3",
        bada4_path=INPUT_PATH / ".." / "bada" / "bada4",
        copy_source=False,
        n_iter=3,
    )

    # create Humidity Scaling model
    humidity_scaling = ExponentialBoostLatitudeCorrectionHumidityScaling(copy_source=False)

    params = {
        "max_age": pd.Timedelta(12, "h"),
        "dt_integration": pd.Timedelta(5, "m"),
        "radiative_heating_effects": True,
        "max_seg_length_m": 60_000,
        "aircraft_performance": aircraft_performance,
        "humidity_scaling": humidity_scaling,
        "process_emissions": True,
        "verbose_outputs": True,
    }

    cocip = Cocip(
        met=met,
        rad=rad,
        params=params,
    )

    LOG.info("Evaluating Cocip model")

    # returns list of Flight outputs
    # output_flights = cocip.eval(source=flights[14:15])
    _ = cocip.eval(source=flights)

    return cocip


# -------
# Results
# -------


def parse_flight_results(fleet: Fleet) -> pd.DataFrame:
    """Interpret fleet results from Cocip evaluation.

    Uses `Cocip.flight` output instead of list[Flight]

    Parameters
    ----------
    fleet : Fleet
        Cocip.flight output after Cocip.eval(...)

    Returns
    -------
    pd.DataFrame
    """
    # add extra dimensions
    fleet["altitude_ft"] = fleet.altitude_ft
    fleet["air_pressure"] = fleet.air_pressure
    # fleet["mach_number"] = units.tas_to_mach_number(
    #     fleet["true_airspeed"], fleet["air_temperature"]
    # )

    # create dataframe from fleet
    fleet_df = fleet.dataframe

    # convert timestamps into ints (unix epoch)
    fleet_df["time_unix"] = (
        pd.to_datetime(fleet["time"]) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta("1s")

    # convert timedelta into seconds
    fleet_df["contrail_age"] = fleet["contrail_age"] // pd.Timedelta("1s")

    # drop some excess columns
    fleet_df = fleet_df.drop(
        columns=[
            "co_ei",
            "hc_ei",
            "nox_ei",
            "nvpm_mass",
            "nvpm_ei_m",
            "nvpm_number",
            "thrust_setting",
        ]
    )

    # fill na with -9999
    # fleet_df = fleet_df.fillna(value=-9999)

    return fleet_df


def parse_contrail_results(contrail: pd.DataFrame) -> pd.DataFrame:
    """Interpret contrail results from Cocip evaluation.

    Uses `Cocip.contrail` output

    Parameters
    ----------
    contrail : pd.DataFrame
        Cocip.contrail output after Cocip.eval(...)

    Returns
    -------
    pd.DataFrame
    """

    contrail = contrail.reset_index(drop=True)

    # convert timestamps into ints (unix epoch)
    contrail["time_unix"] = (
        pd.to_datetime(contrail["time"]) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta("1s")
    contrail["formation_time_unix"] = (
        pd.to_datetime(contrail["formation_time"]) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta("1s")

    # convert timedelta into seconds
    contrail["dt_integration"] = contrail["dt_integration"] // pd.Timedelta("1s")
    contrail["age"] = contrail["age"] // pd.Timedelta("1s")

    # cast bools to ints
    contrail.loc[:, ["continuous", "persistent"]] = contrail.loc[
        :, ["continuous", "persistent"]
    ].astype(int)

    contrail = contrail.drop(columns=["age_hours", "cumul_heat", "cumul_differential_heat"])

    # fill na with -9999
    # contrail = contrail.fillna(value=-9999)

    return contrail
