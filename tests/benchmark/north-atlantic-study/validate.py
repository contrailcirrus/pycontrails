"""Cocip / pycocip regression testing.

See README.md for setup instructions
"""

from __future__ import annotations

# set up logging
import logging
import os
import pathlib

import pandas as pd
import support

from pycontrails.models.cocip import cocip as _cocip
from pycontrails.models.cocip import contrail_properties
from pycontrails.physics import constants, geo, thermo, units

# monkey patch deprecated methods
from tests import _deprecated

geo.segment_angle = _deprecated.segment_angle
thermo.c_pm = lambda x: constants.c_pd
contrail_properties.segment_length_ratio = _deprecated.segment_length_ratio
contrail_properties.mean_energy_flux_per_m = _deprecated.mean_energy_flux_per_m
_cocip.CONTINUITY_CONVENTION_PARAM = 0
units.m_to_pl = _deprecated.m_to_pl
units.pl_to_m = _deprecated.pl_to_m

from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models import Cocip

LOG = logging.getLogger("pycontrails")
logging.basicConfig(filename="regression.log", level=logging.INFO)

# setup

flights_filename = "flights.csv"
flight_id = None  # run all
# flight_id = "190101_155_ELY027"  # select single flight
# flight_id = "190101_152_DAL178"
# flight_id = "190101_134_DAL64"
# flight_id = "181231_1057_CKS244"
# flight_id = "181231_1058_UPS214"
# flight_id = "190101_10_UAL84"
# flight_id = "181231_1067_SVA020"
# flight_id = "190101_176_UAL972"

input_path = pathlib.Path(os.path.realpath(__file__)).parent / "inputs"
output_path = pathlib.Path(os.path.realpath(__file__)).parent / "outputs"
bada_path = input_path / "bada"

# TODO: attempt validation with CDS derived met data
time = ("2019-01-01 00:00:00", "2019-01-03 01:00:00")  # (start, end) inclusive time range
pressure_levels = [
    # 1000,
    # 975,
    # 950,
    # 925,
    # 900,
    # 875,
    # 850,
    # 825,
    # 800,
    # 775,
    # 750,
    # 700,
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
]  # select pressure levels

# _cache = cache.GCPCacheStore(
#     bucket="contrails-301217-ecmwf-data2",
#     cache_dir="era5",
#     read_only=True,
#     show_progress=True,
# )

# ----
# Load Flights
# ----

# load all flight waypoints
flight_waypoints = pd.read_csv(input_path / flights_filename)

# filter for single flight, if specified
if flight_id is not None:
    flight_waypoints = flight_waypoints[flight_waypoints["Flight ID"] == flight_id]

# create a list of flights from each Flight ID in dataframe
flights = [
    support.create_flight(flight_id, df_flight)
    for (flight_id, df_flight) in flight_waypoints.groupby("Flight ID")
]

# ----
# Load outputs
# ----

LOG.info("Loading output data")

# load pycocip output
pycocip_contrails = pd.read_csv(output_path / "contrails.csv")
pycocip_flights = pd.read_csv(output_path / "flights.csv")

# ----
# Load ERA5
# ----
LOG.info("Opening met")
path_prefix = pathlib.Path("inputs")

met_filepaths = [
    input_path / "ERA5_HRES_20190101_Met.nc",
    input_path / "ERA5_HRES_20190102_Met.nc",
    input_path / "ERA5_HRES_20190103_Met.nc",
]
rad_filepaths = [
    input_path / "ERA5_HRES_20190101_Rad.nc",
    input_path / "ERA5_HRES_20190102_Rad.nc",
    input_path / "ERA5_HRES_20190103_Rad.nc",
]

# define ERA5 sources
era5pl = ERA5(
    time=time, variables=Cocip.met_variables, pressure_levels=pressure_levels, path=met_filepaths
)
era5sl = ERA5(time=time, variables=Cocip.rad_variables, path=rad_filepaths)

# create `MetDataset` from sources
met = era5pl.open_metdataset(xr_kwargs=dict(parallel=False))
rad = era5sl.open_metdataset(xr_kwargs=dict(parallel=False))

# preload cocip met
if flight_id is None:
    LOG.info("Preloading cocip met")
    met, rad = Cocip.process_met_datasets(met, rad)


# ----
# Cocip
# ----
params = {
    "bada3_path": bada_path / "bada3",
    "bada4_path": bada_path / "bada4",
    "process_met": flight_id is not None,
    "downselect_met": flight_id is not None,
    "interpolation_fill_value": 0.0,
}

for fl in flights:
    LOG.info(f'Validating {fl.attrs["flight_id"]}')
    cocip = Cocip(met, rad=rad, params=params)
    cocip.eval(source=fl)

    # get output stats
    pycocip_flight_stats = pycocip_flights[
        pycocip_flights["Flight ID"] == fl.attrs["flight_id"]
    ].iloc[0]

    # get flight, if its included
    try:
        pycocip_flight = pd.read_csv(output_path / "flights" / f'{fl.attrs["flight_id"]}.csv')
    except FileNotFoundError:
        pycocip_flight = None

    # get contrail waypoints
    pycocip_contrail = pycocip_contrails[pycocip_contrails["Flight ID"] == fl.attrs["flight_id"]]

    # validate
    try:
        support.validate_output(cocip, pycocip_flight, pycocip_flight_stats, pycocip_contrail)
    except AssertionError as e:
        LOG.info(f'------- {fl.attrs["flight_id"]} invalid')
        LOG.info(e)
        if flight_id is not None:
            raise (e)
