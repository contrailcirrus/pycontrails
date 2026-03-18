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

    #: Aircraft tail number -> Registration
    tail_number: str

    #: ICAO 24-bit address -> Hexcode
    tail_number: str

    #: Country of registration -> Aircraft Register
    country_of_registration: str

    #: ICAO aircraft type designator -> Aircraft ICAO
    aircraft_type_icao: str

    #: IATA aircraft type designator -> Aircraft IATA
    aircraft_type_iata: str

    #: Aircraft family -> Aircraft Family
    aircraft_family: str

    #: Aircraft model -> Aircraft Family
    aircraft_model: str

    #: Aircraft manufacturer -> Manufacturer
    manufacturer: str

    #: Engine model -> Engine Subtype
    engine_subtype: str

    #: Engine unique identification number from the ICAO Aircraft Emissions Databank -> ICAO Engine Emission Databank ID
    engine_uid: str

    #: Engine manufacturer -> Engine Manufacturer
    engine_manufacturer: str

    #: Number of engines -> Number of Engines
    n_engines: int

    #: Maximum take-off weight (MTOW), [:math:`kg`] -> MTOW (kg)
    mtow_kg: float

    #: Operator name -> Operator
    operator_name: str

    #: Operator ICAO code -> Operator ICAO
    operator_icao: str

    #: Operator IATA code -> Operator IATA
    operator_iata: str

    #: Operator type -> Operator Type
    operator_type: str

    #: Registered aircraft usage, i.e., passenger/military -> Aircraft Role
    usage: str

    #: Aircraft market group -> Aircraft Market Group
    market_group: str

    #: Number of seats -> "Seats Y" + "Seats YP" + "Seats W" + "Seats C" + "Seats F"
    n_seats: int

    #: Aircraft status (in-service/storage) -> "Status"
    status: str

    #: First flight date -> "First Flight"
    first_flight_date: pd.Timestamp | str

    #: Delivery date -> "Delivery Date"
    delivery_date: pd.Timestamp | str

    #: Date when the aircraft status is changed -> "As of date"
    status_change_date: pd.Timestamp | str

    #: Aircraft age (years)
    # TODO: This will be a derived quantity
    #age: float

    #: Cumulative reported hours -> "Hours"
    cumulative_reported_hours: int

    #: Cumulative reported cycles -> "Cycles"
    cumulative_reported_cycles: int

    #: Average utilization hours -> "Avg. Annual Hours"
    average_utilization_hours: float

    #: Average utilization cycles -> "Avg. Annual Cycles"
    average_utilization_cycles: float


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

    # TODO: Def eval here

    def registered_aircraft_properties(
        self, tail_number: str, date: pd.Timestamp | None = None
    ) -> AircraftChAviation | None:
        """Get registered aircraft properties from ch-aviation fleet database.

        Parameters
        ----------
        tail_number: str
            Aircraft tail number
        date: pd.Timestamp | None
            Date of flight or date of first waypoint. If None is provided, the most recent
            registered aircraft properties will be provided.

        Returns
        -------
        AircraftChAviation | None
            Registered aircraft properties. If ``tail_number`` is not available
            in the ch-aviation fleet database, None is returned.
        """
        if not self._check_tail_number_availability(tail_number, False):
            return None

        df_aircraft = self.data.loc[[tail_number]]

        # Select most recent data if date not provided
        # TODO: Need to deal with duplicate rows (these are all with fleet + historical)
        if len(df_aircraft) == 1 or date is None:
            df_aircraft = df_aircraft.iloc[-1]
        else:
            is_before_date = date > df_aircraft["Status_change_date"].values

            # If the status change date of all entries occurs after the date provided, use first row
            if np.all(~is_before_date):
                df_aircraft = df_aircraft.iloc[0]
            else:
                df_aircraft = df_aircraft[is_before_date].iloc[-1]

        return AircraftCirium(
            # Registration properties
            tail_number=tail_number,
            country_of_registration=df_aircraft["Country_Of_Registration"],
            # Aircraft properties
            aircraft_subfamily=df_aircraft["aircraft_subfamily"],
            aircraft_type_icao=df_aircraft["ICAO_ATYP"],
            manufacturer=df_aircraft["Manufacturer"],
            modifiers=df_aircraft["Modifiers"],
            # Engine properties
            engine_subseries=df_aircraft["Engine_Subseries"],
            engine_uid=df_aircraft["engine_uid_edb"],
            engine_manufacturer=df_aircraft["Engine_Manufacturer"],
            engine_propulsion_type=df_aircraft["enginepropulsiontypename"],
            n_engines=df_aircraft["Number_Of_Engines"],
            apu_subseries=df_aircraft["APU_Subseries"],
            # Performance envelope
            mtow_kg=units.lbs_to_kg(df_aircraft["MTOW_lbs"]),
            mzfw_kg=units.lbs_to_kg(df_aircraft["MZFW_lbs"]),
            oew_kg=units.lbs_to_kg(df_aircraft["OEW_lbs"]),
            max_payload_kg=units.lbs_to_kg(df_aircraft["Max_Payload_lbs"]),
            fuel_capacity_kg=(df_aircraft["Fuel_Capacity_gallons"] * 3.7854),
            # Operator properties
            operator_name=df_aircraft["Operator"],
            operator_icao=df_aircraft["Operator_ICAO"],
            operator_iata=df_aircraft["Operator_IATA"],
            operator_type=df_aircraft["Operator_Company_Type"],
            usage=df_aircraft["Usage"],
            market_class=df_aircraft["Market_class"],
            n_seats=df_aircraft["N_Seats"],
            # Aircraft status
            status=df_aircraft["Status"],
            status_change_date=df_aircraft["Status_change_date"],
            age=df_aircraft["Age"],
            cumulative_reported_hours=df_aircraft["Cumulative_Reported_Hours"],
            cumulative_reported_cycles=df_aircraft["Cumulative_Reported_Cycles"],
            average_utilization_hours=df_aircraft["Avg_Daily_Reported_Util"],
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


@functools.cache
def _load_ch_fleet_database() -> pd.DataFrame:
    #cirium_path = pathlib.Path(__file__).parent / "static" / "cirium-2024-cleaned-20250530.csv"
    temp_path = "C:/Users/Roger/OneDrive - Imperial College London/Aviation/Datasets/ch-aviation/20260209_fleet_database_processed.csv"
    df = pd.read_csv(
        temp_path,
        parse_dates=["First Flight", "Delivery Date", "As of date", ],
        index_col="Registration"
    )
    df["ICAO Engine Emission Databank ID"] = df["ICAO Engine Emission Databank ID"].replace(
        np.nan, None
    )
    return df
