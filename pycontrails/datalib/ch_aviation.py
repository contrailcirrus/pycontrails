"""Support for reading and parsing the ch-aviation fleet database."""

from __future__ import annotations

import dataclasses
import functools
import pathlib
from typing import Any

import numpy as np
import pandas as pd
from pycontrails.core.flight import Flight
from pycontrails.core.models import Model
from pycontrails.physics import units


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
    first_flight_date: pd.Timestamp | str

    #: Delivery date
    delivery_date: pd.Timestamp | str

    #: Aircraft age in years
    aircraft_age_yrs: float

    #: Cumulative reported hours
    cumulative_reported_hours: int

    #: Cumulative reported cycles
    cumulative_reported_cycles: int

    #: Cumulative reported hours, trailing twelve months
    cumulative_reported_hours_ttm: int

    #: Cumulative reported cycles, trailing twelve months
    cumulative_reported_cycles_ttm: int

    #: Cumulative statistics as of date
    cumulative_stats_as_of_date: pd.Timestamp | str

    #: Average annual utilization hours
    average_annual_hours: float

    #: Average daily utilization hours
    average_daily_hours: float

    #: Average daily utilization hours, trailing twelve months
    average_daily_hours_ttm: float

    #: Average annual utilization cycles
    average_annual_cycles: float

    #: Average statistics as of date
    average_stats_as_of_date: pd.Timestamp | str


class ChAviation(Model):
    """Support for querying the ch-aviation fleet database."""

    name = "ch-aviation"
    long_name = "ch-aviation fleet database"
    source: Flight
    data: pd.DataFrame
    airline_engines: pd.DataFrame

    def __init__(self, **params_kwargs: Any):
        super().__init__(**params_kwargs)
        if not hasattr(self, "data"):
            type(self).data = _load_ch_fleet_database()

    def eval(self, source: Flight | None = None, **params: Any) -> Flight:
        """Extract specific aircraft properties for flight from ch-aviation database.

        Flight attribute must contain one of the following variables:
            - ``tail_number`` (mandatory),
            - ``icao_address`` (optional), or
            - ``airline_iata`` and ``aircraft_type`` (optional)

        The timestamp of the first waypoint is optional.

        The following properties will be added to the flight attribute if the `tail_number` or
        `icao_address` is covered in ch-aviation:
        # TODO: Please update
            - ``country_of_registration``
            - ``atyp_name_ch_a``
            - ``atyp_icao_ch_a``
            - ``atyp_manufacturer``
            - ``atyp_modifiers``
            - ``engine_name``
            - ``engine_uid``
            - ``engine_manufacturer``
            - ``engine_propulsion_type``
            - ``n_engines_ch_a``
            - ``apu_name``
            - ``amass_mtow``
            - ``amass_mzfw``
            - ``amass_oew``
            - ``amass_mpl``
            - ``amass_fuel_capacity``
            - ``operator_name``
            - ``operator_icao``
            - ``operator_iata``
            - ``operator_type``
            - ``aircraft_usage``
            - ``aircraft_market_class``
            - ``n_seats``
            - ``status``
            - ``last_update``
            - ``aircraft_age``
            - ``cumulative_reported_hours``
            - ``cumulative_reported_cycles``
            - ``average_utilization_hours``

        The following properties will be added to the flight attribute if the `tail_number` is not
        covered in ch-aviation, but `airline_iata` and `aircraft_type` are available:
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

        # End evaluation if `tail_number` and `icao_address` is not provided
        try:
            tail_number = self.source.get_constant("tail_number")
        except KeyError:
            tail_number = None

        try:
            icao_address = self.source.get_constant("icao_address")
        except KeyError:
            icao_address = None

        if tail_number is None and icao_address is None:
            return self.source

        # This fails if self.source is empty
        t_first_wypt = self.source["time"][0]

        aircraft_props = self.registered_aircraft_properties(
            tail_number, icao_address, date=t_first_wypt
        )

        # Set attributes, if they aren't already defined
        # TODO: Check this logic and redo look-up tables
        if aircraft_props is None:
            # If tail number is not available, then try to estimate from airline look-up tables
            try:
                airline_iata = self.source.get_constant("airline_iata")
                atyp_icao = self.source.get_constant("aircraft_type")
            except KeyError:
                # End evaluation if `airline_iata` and `atyp_icao` not provided
                return self.source

            engine_props = self.airline_aircraft_engine_look_up(airline_iata, atyp_icao)

            if engine_props is None:
                return self.source

            # Set attributes, if they aren't already defined
            self.source.attrs.setdefault("engine_name", engine_props.engine_subseries)
            self.source.attrs.setdefault("engine_uid", engine_props.engine_uid)
            self.source.attrs.setdefault("operator_name", engine_props.operator_name)
            self.source.attrs.setdefault("operator_iata", engine_props.operator_iata)
            return self.source

        # TODO: Above not edited

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
        self.source.attrs.setdefault("aircraft_age_yrs", aircraft_props.aircraft_age_yrs)

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
        date: pd.Timestamp | None = None
    ) -> AircraftChAviation | None:
        """Get registered aircraft properties from ch-aviation fleet database.

        Parameters
        ----------
        tail_number: str
            Aircraft tail number
        icao_address: str
            ICAO 24-bit address (Hexcode)
        date: pd.Timestamp | None
            Date of flight or date of first waypoint. If None is provided, the most recent
            registered aircraft properties will be provided.

        Returns
        -------
        AircraftChAviation | None
            Registered aircraft properties. If ``tail_number`` and ``icao_address`` are not
            available in the ch-aviation fleet database, None is returned.
        """
        # Search for tail number first, as it has the highest unique values in the fleet database
        if self._check_tail_number_availability(tail_number, False):
            df_aircraft = self.data.loc[[tail_number]]

        # If tail number is not available, try searching for the icao_address if provided
        elif icao_address is not None:
            if not self._check_icao_address_availability(icao_address, False):
                return None
            df_aircraft = self.data[self.data["Hexcode"] == icao_address]

        # tail number and icao address is not available
        else:
            return None

        # Ensure that the data only contains one unique aircraft
        if len(df_aircraft) != 1:
            raise ValueError(
                f"Expected exactly 1 row for tail_number={tail_number} "
                f"and icao_address={icao_address}, found {len(df_aircraft)}"
            )

        df_aircraft = df_aircraft.iloc[0]

        return AircraftChAviation(
            # Registration properties
            tail_number=tail_number,
            icao_address=df_aircraft["Hexcode"],
            msn=df_aircraft["MSN"],
            country_of_registration=df_aircraft["Aircraft Register"],

            # Aircraft properties
            aircraft_type_icao=df_aircraft["Aircraft ICAO"],
            aircraft_type_iata=df_aircraft["Aircraft IATA"],
            aircraft_family=df_aircraft["Aircraft Family"],
            aircraft_subfamily=df_aircraft["Aircraft Variant"],
            manufacturer=df_aircraft["Manufacturer"],

            # Engine properties
            engine_subtype=df_aircraft["Engine Subtype"],
            engine_uid=df_aircraft["ICAO Engine Emission Databank ID"],
            engine_manufacturer=df_aircraft["Engine Manufacturer"],
            n_engines=df_aircraft["Number of Engines"],

            # Performance envelope
            mtow_kg=df_aircraft["MTOW (kg)"],

            # Operator properties
            operator_name=df_aircraft["Operator"],
            operator_icao=df_aircraft["Operator ICAO"],
            operator_iata=df_aircraft["Operator IATA"],
            operator_type=df_aircraft["Operator Type"],
            aircraft_role=df_aircraft["Aircraft Role"],
            aircraft_market_group=df_aircraft["Aircraft Market Group"],
            n_seats=(
                df_aircraft["Seats Y"]
                + df_aircraft["Seats YP"]
                + df_aircraft["Seats W"]
                + df_aircraft["Seats C"]
                + df_aircraft["Seats F"]
            ),

            # Aircraft status
            status=df_aircraft["Status"],
            first_flight_date=df_aircraft["First Flight"],
            delivery_date=df_aircraft["Delivery Date"],
            aircraft_age_yrs=(
                (date - df_aircraft["Delivery Date"]) / pd.Timedelta(days=365.25)
                if pd.notna(date) and pd.notna(df_aircraft["Delivery Date"]) else np.nan
            ),

            # Aircraft utilisation statistics
            cumulative_reported_hours=df_aircraft["Hours"],
            cumulative_reported_hours_ttm=df_aircraft["Hours TTM"],
            cumulative_reported_cycles=df_aircraft["Cycles"],
            cumulative_reported_cycles_ttm=df_aircraft["Cycles TTM"],
            cumulative_stats_as_of_date=df_aircraft["As of date"],

            average_annual_hours=df_aircraft["Avg. Annual Hours"],
            average_daily_hours=(
                pd.to_timedelta(df_aircraft["Avg. Daily Utilisation"], errors="coerce")
                .total_seconds() / 3600
                if pd.notna(df_aircraft["Avg. Daily Utilisation"]) else np.nan
            ),
            average_daily_hours_ttm=(
                pd.to_timedelta(df_aircraft["Avg. Daily Utilisation TTM"], errors="coerce")
                .total_seconds() / 3600
                if pd.notna(df_aircraft["Avg. Daily Utilisation TTM"]) else np.nan
            ),
            average_annual_cycles=df_aircraft["Avg. Annual Cycles"],
            average_stats_as_of_date=df_aircraft["Last Updated"],
        )

    def _check_tail_number_availability(
        self,
        tail_number: str,
        raise_error: bool = True,
    ) -> bool:
        """
        Check if the provided tail number is available in the ch-aviation fleet database.

        Setting ``raise_error`` to True allows functions in this class to be used independently
        outside of :meth:`eval`.

        Parameters
        ----------
        tail_number: str
            Aircraft tail number
        raise_error: bool
            Raise a KeyError if aircraft tail number is not available.

        Returns
        -------
        bool
            True if aircraft tail number is available in the ch-aviation fleet database.

        Raises
        ------
        KeyError
            If aircraft tail number is not available in the ch-aviation fleet database.
        """
        if tail_number not in self.data.index:
            if raise_error:
                raise KeyError(
                    f"Aircraft tail number ({tail_number}) is not available in the ch-aviation fleet database"
                )
            return False
        return True

    def _check_icao_address_availability(
        self,
        icao_address: str,
        raise_error: bool = True,
    ) -> bool:
        """
        Check if the provided icao address is available in the ch-aviation fleet database.

        Setting ``raise_error`` to True allows functions in this class to be used independently
        outside of :meth:`eval`.

        Parameters
        ----------
        icao_address: str
            ICAO 24-bit address (Hexcode)
        raise_error: bool
            Raise a KeyError if aircraft tail number is not available.

        Returns
        -------
        bool
            True if icao address is available in the ch-aviation fleet database.

        Raises
        ------
        KeyError
            If icao address is not available in the ch-aviation fleet database.
        """
        if icao_address not in self.data["Hexcode"].values:
            if raise_error:
                raise KeyError(
                    f"ICAO address ({icao_address}) is not available in the ch-aviation fleet database"
                )
            return False
        return True


@functools.cache
def _load_ch_fleet_database() -> pd.DataFrame:
    #temp_path = pathlib.Path(__file__).parent / "static" / "2024-cleaned-20250530.csv"
    temp_path = "C:/Users/Roger/OneDrive - Imperial College London/Aviation/Datasets/ch-aviation/20260318_fleet_database_processed.csv"
    df = pd.read_csv(temp_path, index_col="Registration")

    date_cols = ["First Flight", "Delivery Date", "As of date", "Last Updated"]
    df[date_cols] = df[date_cols].apply(pd.to_datetime, errors="coerce")

    df["ICAO Engine Emission Databank ID"] = df["ICAO Engine Emission Databank ID"].replace(
        np.nan, None
    )

    # Ensure no duplicate tail numbers
    if not df.index.is_unique:
        raise ValueError("Duplicate Registration found in fleet database")

    return df
