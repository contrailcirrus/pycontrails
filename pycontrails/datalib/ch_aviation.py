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
    aircraft_age: float

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

    def eval(self):
        # TODO: Def eval here
        return

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
                f"Expected exactly 1 row for tail_number={tail_number}, found {len(df_aircraft)}"
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
            # TODO: Check if date is nan
            aircraft_age=(date - df_aircraft["Delivery Date"]) if date is not None else np.nan,

            # Aircraft utilisation statistics
            cumulative_reported_hours=df_aircraft["Hours"],
            cumulative_reported_cycles=df_aircraft["Cycles"],
            cumulative_reported_hours_ttm=df_aircraft["Hours TTM"],
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
    #cirium_path = pathlib.Path(__file__).parent / "static" / "cirium-2024-cleaned-20250530.csv"
    temp_path = "C:/Users/Roger/OneDrive - Imperial College London/Aviation/Datasets/ch-aviation/20260318_fleet_database_processed.csv"
    df = pd.read_csv(
        temp_path,
        parse_dates=["First Flight", "Delivery Date", "As of date", "Last Updated", ],
        index_col="Registration"
    )
    df["ICAO Engine Emission Databank ID"] = df["ICAO Engine Emission Databank ID"].replace(
        np.nan, None
    )

    # Ensure no duplicate tail numbers
    if not df.index.is_unique:
        raise ValueError("Duplicate Registration found in fleet database")

    return df
