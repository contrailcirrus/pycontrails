"""Support for reading and parsing the ch-aviation fleet database."""

from __future__ import annotations

import dataclasses
import functools
import os
import pathlib
from typing import Any

import pandas as pd

import pycontrails
from pycontrails.core.flight import Flight
from pycontrails.core.models import Model, ModelParams


@dataclasses.dataclass(frozen=True)
class AircraftChAviation:
    """Registered aircraft properties from ch-aviation database."""

    #: Aircraft tail number
    tail_number: str

    #: ICAO 24-bit address
    icao_address: str

    #: Manufacturer Serial Number, MSN
    msn: str

    #: Country of registration
    country_of_registration: str

    #: ICAO aircraft type designator
    aircraft_type_icao: str

    #: IATA aircraft type designator
    aircraft_type_iata: str

    #: Aircraft family
    aircraft_family: str

    #: Aircraft model
    aircraft_subfamily: str

    #: Aircraft manufacturer
    manufacturer: str

    #: Engine model
    engine_subtype: str

    #: Engine unique identification number from the ICAO Aircraft Emissions Databank
    engine_uid: str

    #: Engine manufacturer
    engine_manufacturer: str

    #: Number of engines
    n_engines: int

    #: Maximum take-off weight (MTOW), [:math:`kg`]
    mtow_kg: float

    #: Operator name
    operator_name: str

    #: Operator ICAO code
    operator_icao: str

    #: Operator IATA code
    operator_iata: str

    #: Operator type
    operator_type: str

    #: Registered aircraft usage, i.e., passenger/military
    aircraft_role: str

    #: Aircraft market group
    aircraft_market_group: str

    #: Number of seats
    n_seats: int

    #: Aircraft status
    status: str

    #: First flight date
    first_flight_date: pd.Timestamp

    #: Delivery date
    delivery_date: pd.Timestamp

    #: Cumulative reported hours
    cumulative_reported_hours: int

    #: Cumulative reported cycles
    cumulative_reported_cycles: int

    #: Cumulative reported hours, trailing twelve months
    cumulative_reported_hours_ttm: int

    #: Cumulative reported cycles, trailing twelve months
    cumulative_reported_cycles_ttm: int

    #: Cumulative statistics as of date
    cumulative_stats_as_of_date: pd.Timestamp

    #: Average annual utilization hours
    average_annual_hours: float

    #: Average daily utilization hours
    average_daily_hours: float

    #: Average daily utilization hours, trailing twelve months
    average_daily_hours_ttm: float

    #: Average annual utilization cycles
    average_annual_cycles: float

    #: Average statistics as of date
    average_stats_as_of_date: pd.Timestamp

    def aircraft_age_yrs(self, date: pd.Timestamp) -> float:
        """Estimate aircraft age in years at the provided date.

        Parameters
        ----------
        date : pd.Timestamp
            Date of interest

        Returns
        -------
        float
            Aircraft age in years at the provided date
        """
        return (date - self.delivery_date).days / 365.25


@dataclasses.dataclass(frozen=True)
class AirlineAircraftLookUp:
    """Estimated engine properties from airline-aircraft look-up tables."""

    #: ICAO aircraft type designator
    aircraft_type_icao: str

    #: Engine model
    engine_subtype: str

    #: Engine unique identification number from the ICAO Aircraft Emissions Databank
    engine_uid: str

    #: Operator name
    operator_name: str

    #: Operator IATA code
    operator_iata: str


@dataclasses.dataclass
class ChAviationParams(ModelParams):
    """Parameters for :class:`ChAviation` model."""

    #: Path to ch-aviation fleet database CSV file.
    fleet_database_path: str | pathlib.Path | None = None

    #: Path to airline-aircraft engine look-up table CSV file.
    airline_engine_lookup_path: str | pathlib.Path | None = None


def _ch_aviation_root_path() -> pathlib.Path:
    if (p := os.getenv("CH_AVIATION_ROOT_PATH")) is not None:
        return pathlib.Path(p)
    return pathlib.Path(*pycontrails.__path__).parents[1] / "ch-aviation"


class ChAviation(Model):
    """Support for querying the ch-aviation fleet database."""

    name = "ch-aviation"
    long_name = "ch-aviation fleet database"
    default_params = ChAviationParams

    #: Lookup dictionary of the form ``{tail_number: AircraftChAviation}``
    data: dict[str, AircraftChAviation]

    #: Lookup dictionary of the form ``{(airline_iata, aircraft_type_icao): AirlineAircraftLookUp}``
    airline_engines: dict[tuple[str, str], AirlineAircraftLookUp]

    source: Flight

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ) -> None:
        super().__init__(params=params, **params_kwargs)

        if not hasattr(self, "data"):
            fpath = self.params["fleet_database_path"]
            if fpath is None:
                fpath = _ch_aviation_root_path() / "20260318_fleet_database_processed.csv"
            if not pathlib.Path(fpath).is_file():
                raise FileNotFoundError(f"ch-aviation fleet database not found: {fpath}")

            type(self).data = _load_ch_fleet_database(fpath)

        if not hasattr(self, "airline_engines"):
            epath = self.params["airline_engine_lookup_path"]
            if epath is None:
                epath = _ch_aviation_root_path() / "20260318_airline_engine_look_up.csv"
            if not pathlib.Path(epath).is_file():
                raise FileNotFoundError(f"ch-aviation airline-aircraft look-up not found: {epath}")

            type(self).airline_engines = _load_airline_engine_look_up_tables(epath)

    def eval(self, source: Flight | None = None, **params: Any) -> Flight:
        """Extract specific aircraft properties for flight from ch-aviation database.

        Flight attribute must contain one of the following variables:
            - ``tail_number`` (mandatory),
            - ``icao_address`` (optional), or
            - ``airline_iata`` and ``aircraft_type`` (optional)

        The timestamp of the first waypoint is optional, but preferred.

        The following properties will be added to the flight attribute if the ``tail_number`` or
        ``icao_address`` is covered in the fleet database:
            - ``msn``
            - ``country_of_registration``
            - ``atyp_icao_ch_a``
            - ``atyp_iata_ch_a``
            - ``atyp_name_ch_a``
            - ``atyp_manufacturer``
            - ``engine_name``
            - ``engine_uid``
            - ``engine_manufacturer``
            - ``n_engines_ch_a``
            - ``amass_mtow``
            - ``operator_name``
            - ``operator_icao``
            - ``operator_iata``
            - ``operator_type``
            - ``aircraft_role``
            - ``aircraft_market_group``
            - ``n_seats``
            - ``status``
            - ``first_flight_date``
            - ``delivery_date``
            - ``aircraft_age_yrs``, if the timestamp of the first waypoint is provided in `source`
            - ``cumulative_reported_hours``
            - ``cumulative_reported_hours_ttm``
            - ``cumulative_reported_cycles``
            - ``cumulative_reported_cycles_ttm``
            - ``cumulative_stats_as_of_date``
            - ``average_annual_hours``
            - ``average_daily_hours``
            - ``average_daily_hours_ttm``
            - ``average_annual_cycles``
            - ``average_stats_as_of_date``

        The following properties will be added to the flight attribute if the ``tail_number`` and
        ``icao_address`` are not covered in ch-aviation, but ``airline_iata`` and ``aircraft_type``
        are available:
            - ``engine_name``
            - ``engine_uid``
            - ``operator_name``
            - ``operator_iata``

        Parameters
        ----------
        source : Flight
            Flight to evaluate

        Returns
        -------
        Flight
            Flight with attached aircraft properties
        """
        self.update_params(params)
        self.set_source(source)
        self.source = self.require_source_type(Flight)

        tail_number = self.source.get_constant("tail_number", None)
        icao_address = self.source.get_constant("icao_address", None)
        airline_iata = self.source.get_constant("airline_iata", None)

        # Early exit if no identifying information is provided
        if tail_number is None and icao_address is None and airline_iata is None:
            return self.source

        aircraft_props = self.registered_aircraft_properties(tail_number, icao_address)

        # If tail number is not available, then try to estimate from airline look-up tables
        if aircraft_props is None:
            try:
                airline_iata = self.source.get_constant("airline_iata")
                atyp_icao = self.source.get_constant("aircraft_type")
            except KeyError:
                # End evaluation if airline_iata and atyp_icao not provided
                return self.source

            engine_props = self.airline_engines.get((airline_iata, atyp_icao))
            if engine_props is None:
                return self.source

            # Set attributes, if they aren't already defined
            self.source.attrs.setdefault("engine_name", engine_props.engine_subtype)
            self.source.attrs.setdefault("engine_uid", engine_props.engine_uid)
            self.source.attrs.setdefault("operator_name", engine_props.operator_name)
            self.source.attrs.setdefault("operator_iata", engine_props.operator_iata)
            return self.source

        # Happy path: aircraft properties are available in ch-aviation
        self.source.attrs.setdefault("msn", aircraft_props.msn)
        self.source.attrs.setdefault(
            "country_of_registration", aircraft_props.country_of_registration
        )

        # Aircraft properties
        self.source.attrs.setdefault("atyp_icao_ch_a", aircraft_props.aircraft_type_icao)
        self.source.attrs.setdefault("atyp_iata_ch_a", aircraft_props.aircraft_type_iata)
        self.source.attrs.setdefault("atyp_name_ch_a", aircraft_props.aircraft_subfamily)
        self.source.attrs.setdefault("atyp_manufacturer", aircraft_props.manufacturer)

        # Engine properties
        self.source.attrs.setdefault("engine_name", aircraft_props.engine_subtype)
        self.source.attrs.setdefault("engine_uid", aircraft_props.engine_uid)
        self.source.attrs.setdefault("engine_manufacturer", aircraft_props.engine_manufacturer)
        self.source.attrs.setdefault("n_engines_ch_a", aircraft_props.n_engines)

        # Performance envelope
        self.source.attrs.setdefault("amass_mtow", aircraft_props.mtow_kg)

        # Operator properties
        self.source.attrs.setdefault("operator_name", aircraft_props.operator_name)
        self.source.attrs.setdefault("operator_icao", aircraft_props.operator_icao)
        self.source.attrs.setdefault("operator_iata", aircraft_props.operator_iata)
        self.source.attrs.setdefault("operator_type", aircraft_props.operator_type)
        self.source.attrs.setdefault("aircraft_role", aircraft_props.aircraft_role)
        self.source.attrs.setdefault("aircraft_market_group", aircraft_props.aircraft_market_group)
        self.source.attrs.setdefault("n_seats", aircraft_props.n_seats)

        # Aircraft status
        self.source.attrs.setdefault("status", aircraft_props.status)
        self.source.attrs.setdefault("first_flight_date", aircraft_props.first_flight_date)
        self.source.attrs.setdefault("delivery_date", aircraft_props.delivery_date)

        # Aircraft age
        date = pd.Timestamp(self.source["time"][0]) if self.source else pd.Timestamp("NaT")
        self.source.attrs.setdefault("aircraft_age_yrs", aircraft_props.aircraft_age_yrs(date))

        # Aircraft utilisation statistics
        self.source.attrs.setdefault(
            "cumulative_reported_hours", aircraft_props.cumulative_reported_hours
        )
        self.source.attrs.setdefault(
            "cumulative_reported_hours_ttm", aircraft_props.cumulative_reported_hours_ttm
        )
        self.source.attrs.setdefault(
            "cumulative_reported_cycles", aircraft_props.cumulative_reported_cycles
        )
        self.source.attrs.setdefault(
            "cumulative_reported_cycles_ttm", aircraft_props.cumulative_reported_cycles_ttm
        )
        self.source.attrs.setdefault(
            "cumulative_stats_as_of_date", aircraft_props.cumulative_stats_as_of_date
        )
        self.source.attrs.setdefault("average_annual_hours", aircraft_props.average_annual_hours)
        self.source.attrs.setdefault("average_daily_hours", aircraft_props.average_daily_hours)
        self.source.attrs.setdefault(
            "average_daily_hours_ttm", aircraft_props.average_daily_hours_ttm
        )
        self.source.attrs.setdefault("average_annual_cycles", aircraft_props.average_annual_cycles)
        self.source.attrs.setdefault(
            "average_stats_as_of_date", aircraft_props.average_stats_as_of_date
        )
        return self.source

    def registered_aircraft_properties(
        self,
        tail_number: str,
        icao_address: str | None = None,
    ) -> AircraftChAviation | None:
        """Get registered aircraft properties from ch-aviation fleet database.

        Parameters
        ----------
        tail_number: str
            Aircraft tail number
        icao_address: str
            ICAO 24-bit address (Hexcode)

        Returns
        -------
        AircraftChAviation | None
            Registered aircraft properties. If ``tail_number`` and ``icao_address`` are not
            available in the ch-aviation fleet database, None is returned.
        """
        # Search for tail number first, as it has the highest unique values in the fleet database
        aircraft = self.data.get(tail_number)

        if aircraft:
            return aircraft

        if icao_address is None:
            return None

        # If tail number is not available, try searching for the icao_address provided
        aircrafts = [ac for ac in self.data.values() if ac.icao_address == icao_address]
        if len(aircrafts) > 1:
            # We don't ever end up here with the 20260318_fleet_database_processed.csv data
            raise ValueError(f"Found multiple aircraft with icao_address={icao_address}")
        if len(aircrafts) == 1:
            return aircrafts[0]
        return None


def _row_to_ch_aviation(tup: Any) -> tuple[str, AircraftChAviation]:
    return tup.tail_number, AircraftChAviation(
        **{k.name: getattr(tup, k.name) for k in dataclasses.fields(AircraftChAviation)}
    )


@functools.cache
def _load_ch_fleet_database(path: str | pathlib.Path) -> dict[str, AircraftChAviation]:
    date_cols = ["First Flight", "Delivery Date", "As of date", "Last Updated"]
    df = pd.read_csv(
        path,
        parse_dates=date_cols,
        date_format="ISO8601",
        dtype={"Regional Partnership": str, "Lease Remarks": str},
    )

    # Defensive, unnecessary with the 20260318_fleet_database_processed.csv data
    if not df["Registration"].is_unique:
        raise ValueError("Duplicate Registration found in fleet database")

    all_nan_seats = (
        df["Seats Y"].isna()
        & df["Seats YP"].isna()
        & df["Seats W"].isna()
        & df["Seats C"].isna()
        & df["Seats F"].isna()
    )
    df["n_seats"] = (
        df["Seats Y"].fillna(0.0)
        + df["Seats YP"].fillna(0.0)
        + df["Seats W"].fillna(0.0)
        + df["Seats C"].fillna(0.0)
        + df["Seats F"].fillna(0.0)
    ).where(~all_nan_seats, other=float("nan"))
    df["average_daily_hours"] = (
        pd.to_timedelta(df["Avg. Daily Utilisation"] + ":00").dt.total_seconds() / 3600
    )
    df["average_daily_hours_ttm"] = (
        pd.to_timedelta(df["Avg. Daily Utilisation TTM"] + ":00").dt.total_seconds() / 3600
    )

    # Rename other columns to match AircraftChAviation field names
    columns = {
        "Registration": "tail_number",
        "Hexcode": "icao_address",
        "MSN": "msn",
        "Aircraft Register": "country_of_registration",
        "Aircraft ICAO": "aircraft_type_icao",
        "Aircraft IATA": "aircraft_type_iata",
        "Aircraft Family": "aircraft_family",
        "Aircraft Variant": "aircraft_subfamily",
        "Manufacturer": "manufacturer",
        "Engine Subtype": "engine_subtype",
        "ICAO Engine Emission Databank ID": "engine_uid",
        "Engine Manufacturer": "engine_manufacturer",
        "Number of Engines": "n_engines",
        "MTOW (kg)": "mtow_kg",
        "Operator": "operator_name",
        "Operator ICAO": "operator_icao",
        "Operator IATA": "operator_iata",
        "Operator Type": "operator_type",
        "Aircraft Role": "aircraft_role",
        "Aircraft Market Group": "aircraft_market_group",
        "Status": "status",
        "First Flight": "first_flight_date",
        "Delivery Date": "delivery_date",
        "Hours": "cumulative_reported_hours",
        "Hours TTM": "cumulative_reported_hours_ttm",
        "Cycles": "cumulative_reported_cycles",
        "Cycles TTM": "cumulative_reported_cycles_ttm",
        "As of date": "cumulative_stats_as_of_date",
        "Avg. Annual Hours": "average_annual_hours",
        "Avg. Annual Cycles": "average_annual_cycles",
        "Last Updated": "average_stats_as_of_date",
    }
    df = df.rename(columns=columns)

    return dict(_row_to_ch_aviation(tup) for tup in df.itertuples())


def _row_to_airline_lookup(tup: Any) -> tuple[tuple[str, str], AirlineAircraftLookUp]:
    return (tup.operator_iata, tup.aircraft_type_icao), AirlineAircraftLookUp(
        **{k.name: getattr(tup, k.name) for k in dataclasses.fields(AirlineAircraftLookUp)}
    )


@functools.cache
def _load_airline_engine_look_up_tables(
    path: pathlib.Path | str,
) -> dict[tuple[str, str], AirlineAircraftLookUp]:
    df = pd.read_csv(path)
    columns = {
        "Operator IATA": "operator_iata",
        "Aircraft ICAO": "aircraft_type_icao",
        "Engine Subtype": "engine_subtype",
        "ICAO Engine Emission Databank ID": "engine_uid",
        "Operator": "operator_name",
    }
    df = df.rename(columns=columns)

    # Defensive, unnecessary with the 20260318_airline_engine_look_up.csv data
    if df.duplicated(subset=["operator_iata", "aircraft_type_icao"]).any():
        raise ValueError("Duplicate (operator_iata, aircraft_type_icao) found in look-up table")

    return dict(_row_to_airline_lookup(tup) for tup in df.itertuples())
