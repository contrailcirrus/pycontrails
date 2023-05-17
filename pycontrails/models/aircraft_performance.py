"""Abstract classes for aircraft performance models."""

from __future__ import annotations

import abc
import dataclasses
import warnings
from typing import Any, NoReturn, overload

import numpy as np
import numpy.typing as npt

from pycontrails.core import flight
from pycontrails.core.fleet import Fleet
from pycontrails.core.flight import Flight
from pycontrails.core.fuel import Fuel, JetA
from pycontrails.core.met import MetDataset
from pycontrails.core.models import Model, ModelParams
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.physics import jet


@dataclasses.dataclass
class AircraftPerformanceParams(ModelParams):
    """Parameters for :class:`AircraftPerformance`."""

    #: Fuel type
    fuel: Fuel = dataclasses.field(default_factory=JetA)

    #: Whether to correct fuel flow to ensure it remains within the operational
    #: limits of the aircraft type.
    correct_fuel_flow: bool = True

    #: The number of iterations used to calculate aircraft mass and fuel flow.
    n_iter: int = 5


class AircraftPerformance(Model):
    """
    Support for standardizing aircraft performance methodologies.

    Currently just a container until additional models are implemented.
    """

    @abc.abstractmethod
    @overload
    def eval(self, source: Flight, **params: Any) -> Flight:
        ...

    @abc.abstractmethod
    @overload
    def eval(self, source: Fleet, **params: Any) -> Fleet:
        ...

    @abc.abstractmethod
    @overload
    def eval(self, source: list[Flight], **params: Any) -> list[Flight]:
        ...

    # This is only included for type consistency with parent. This will raise.
    @abc.abstractmethod
    @overload
    def eval(self, source: None = None, **params: Any) -> NoReturn:
        ...

    @abc.abstractmethod
    def eval(self, source: Flight | list[Flight] | None = None, **params: Any) -> Any:
        """Evaluate the aircraft performance model."""

    @abc.abstractmethod
    def check_aircraft_type_availability(
        self, aircraft_type: str, raise_error: bool = True
    ) -> bool:
        """Check if aircraft type designator is available in the model's database."""

    def simulate_fuel_and_performance(
        self,
        *,
        aircraft_type: str,
        altitude_ft: npt.NDArray[np.float_],
        time: npt.NDArray[np.datetime64],
        true_airspeed: npt.NDArray[np.float_],
        air_temperature: npt.NDArray[np.float_],
        aircraft_mass: npt.NDArray[np.float_] | float | None,
        thrust: npt.NDArray[np.float_] | float | None,
        engine_efficiency: npt.NDArray[np.float_] | float | None,
        fuel_flow: npt.NDArray[np.float_] | float | None,
        q_fuel: float,
        amass_ref: float,
        amass_oew: float,
        amass_mtow: float,
        amass_mpl: float,
        load_factor: float | None = None,
    ) -> AircraftPerformanceData:
        r"""
        Calculate aircraft mass, fuel mass flow rate, and overall propulsion efficiency.

        This method performs `n_iter` iterations, each of which calls
        :meth:`calculate_aircraft_performance`. Each successive iteration
        generates a better estimate for mass fuel flow rate and aircraft
        mass at each waypoint.

        Parameters
        ----------
        aircraft_type: str
            ICAO aircraft type designator (BADA 3), or long aircraft type designator (BADA 4).
        altitude_ft: npt.NDArray[np.float_]
            Altitude at each waypoint, [:math:`ft`]
        time: npt.NDArray[np.datetime64]
            Waypoint time in np.datetime64 format.
        true_airspeed: npt.NDArray[np.float_]
            True airspeed for each waypoint, [:math:`m s^{-1}`]
        air_temperature : npt.NDArray[np.float_]
            Ambient temperature for each waypoint, [:math:`K`]
        q_fuel : float
            Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].
        correct_fuel_flow : bool
            Passed into :meth:`calculate_aircraft_performance`
        model_choice : str
            Passed into :meth:`calculate_aircraft_performance`
        load_factor: None | float
            Aircraft load factor assumption (between 0 and 1). If None is given, the reference mass
            from the BADA database is used to get the mass of the specific aircraft type.
        n_iter : int, optional
            Number of iterations to ensure convergence of aircraft mass and fuel mass flow rate.
        aircraft_mass : npt.NDArray[np.float_] | float | None
            Override the aircraft_mass at each waypoint, [:math:`kg`].
        engine_efficiency : npt.NDArray[np.float_] | float | None
            Override the engine efficiency at each waypoint.
        fuel_flow : npt.NDArray[np.float_] | float | None
            Override the fuel flow at each waypoint, [:math:`kg s^{-1}`].
        thrust : npt.NDArray[np.float_] | float | None
            Override the thrust setting at each waypoint, [:math: `N`].
        amass_oew: float | None
            Aircraft operating empty weight, i.e. the basic weight of an aircraft including the crew and necessary
            equipment, but excluding usable fuel and payload, can be provided by Cirium [:math:`kg`]
        amass_mtow: float | None
            Aircraft maximum take-off weight, can be provided by Cirium, [:math:`kg`]
        amass_mpl: float | None
            Aircraft maximum payload, can be provided by Cirium, [:math:`kg`]

        Returns
        -------
        AircraftPerformanceData
            Results from the final iteration is returned.
        """

        # shortcut if aircraft mass is provided
        if aircraft_mass is not None:
            warnings.warn(
                "Parameter 'aircraft_mass' provided to 'simulate_fuel_and_performance' "
                f"is not None. Skipping {self.params['n_iter']} iterations and only calculating "
                "aircraft performance once."
            )

            # If fuel_flow is None and a non-constant aircraft_mass is provided
            # at each waypoint, then assume that the derivative with respect to
            # time is the fuel flow rate.
            if fuel_flow is None and isinstance(aircraft_mass, np.ndarray):
                d_aircraft_mass = np.diff(aircraft_mass)
                if np.any(d_aircraft_mass < 0.0):
                    if not np.all(d_aircraft_mass < 0.0):
                        warnings.warn(
                            "Aircraft mass is being used to compute fuel flow, but the "
                            "aircraft mass is not monotonically decreasing. This may "
                            "result in incorrect fuel flow calculations."
                        )
                    segment_duration = flight.segment_duration(time, dtype=aircraft_mass.dtype)
                    fuel_flow = -np.append(d_aircraft_mass, np.float32(np.nan)) / segment_duration

            return self.calculate_aircraft_performance(
                aircraft_type=aircraft_type,
                altitude_ft=altitude_ft,
                air_temperature=air_temperature,
                time=time,
                true_airspeed=true_airspeed,
                aircraft_mass=aircraft_mass,
                engine_efficiency=engine_efficiency,
                fuel_flow=fuel_flow,
                thrust=thrust,
                q_fuel=q_fuel,
            )

        # Variable aircraft_mass will change dynamically after each iteration
        if load_factor is None:
            aircraft_mass = amass_ref
        else:
            aircraft_mass = amass_mtow

        for _ in range(self.params["n_iter"]):
            aircraft_performance = self.calculate_aircraft_performance(
                aircraft_type=aircraft_type,
                altitude_ft=altitude_ft,
                air_temperature=air_temperature,
                time=time,
                true_airspeed=true_airspeed,
                aircraft_mass=aircraft_mass,
                engine_efficiency=engine_efficiency,
                fuel_flow=fuel_flow,
                thrust=thrust,
                q_fuel=q_fuel,
            )

            tot_reserve_fuel = jet.reserve_fuel_requirements(
                aircraft_performance.rocd,
                aircraft_performance.fuel_flow,
                aircraft_performance.fuel_burn,
            )

            aircraft_mass = jet.update_aircraft_mass(
                operating_empty_weight=amass_oew,
                ref_mass=amass_ref,
                max_takeoff_weight=amass_mtow,
                max_payload=amass_mpl,
                fuel_burn=aircraft_performance.fuel_burn,
                total_reserve_fuel=tot_reserve_fuel,
                load_factor=load_factor,
            )

        # Update aircraft mass to the latest fuel consumption estimate
        aircraft_performance.aircraft_mass = aircraft_mass

        return aircraft_performance

    @abc.abstractmethod
    def calculate_aircraft_performance(
        self,
        *,
        aircraft_type: str,
        altitude_ft: npt.NDArray[np.float_],
        air_temperature: npt.NDArray[np.float_],
        time: npt.NDArray[np.datetime64] | None,
        true_airspeed: npt.NDArray[np.float_] | float | None,
        aircraft_mass: npt.NDArray[np.float_] | float | None,
        engine_efficiency: npt.NDArray[np.float_] | float | None,
        fuel_flow: npt.NDArray[np.float_] | float | None,
        thrust: npt.NDArray[np.float_] | float | None,
        q_fuel: float,
    ) -> AircraftPerformanceData:
        r"""
        Calculate aircraft performance along a trajectory using the underlying BADA data.

        When `time` is not None, This method should be used for a single flight
        trajectory. Waypoints are coupled via the `time` parameter.

        This method computes the rate of climb and descent (ROTC) to determine
        flight phases: "cruise", "climb", and "descent". Performance metrics
        depend on this phase.

        When `time` is None, this method can be used to simulate flight performance
        over a grid of flight waypoints (ie, grid points). Each grid point is treated
        independently. If a precise 1-dimensional flight trajectory is known,
        the method :meth:`calculate_aircraft_performance` should be used instead.

        All gridpoints are assumed to be in a "cruise" phase of the flight. It
        would be possible to support a "climb" or "descent" phase, but this
        is not yet implemented.

        Derived values are repeated corrected to ensure that they remain realistic.
        To this end, values are clipped at BADA defined thresholds. These correction
        methods include:
        - true airspeed is bounded by the max Mach number from the BADA PTF data
        - fuel flow is bounded by the "lo" and "high" values in the BADA PTF data
        - engine efficiency is clipped at 0.5

        Parameters
        ----------
        atyp_bada : str
            ICAO aircraft type designator (BADA 3), or long aircraft type designator (BADA 4).
        altitude_ft : npt.NDArray[np.float_]
            Altitude at each waypoint, [:math:`ft`]
        air_temperature : npt.NDArray[np.float_]
            Ambient temperature for each waypoint, [:math:`K`]
        time: npt.NDArray[np.datetime64] | None
            Waypoint time in `np.datetime64` format. If `None`, only drag force
            will is used in thrust calculations (ie, no vertical change and constant
            horizontal change). In addition, aircraft is assumed to be in cruise.
        q_fuel : float
            Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].
        correct_fuel_flow : bool
            Clip fuel flow by BADA PTF nominal values.
        model_choice : str
            BADA4 currently supports "total_energy_model", "actions_model", and
            "ptf_nominals". BADA3 only supports "total_energy_model" and
            "ptf_nominals".
        true_airspeed : npt.NDArray[np.float_] | float | None
            True airspeed for each waypoint, [:math:`m s^{-1}`].
            If `None`, the nominal BADA cruise value is used.
        aircraft_mass : npt.NDArray[np.float_] | float | None
            Aircraft mass for each waypoint, [:math:`kg`].
            If `None`, the nominal BADA value is used.
        engine_efficiency : npt.NDArray[np.float_] | float | None
            Override the engine efficiency at each waypoint.
        fuel_flow : npt.NDArray[np.float_] | float | None
            Override the fuel flow at each waypoint, [:math:`kg s^{-1}`].
        thrust : npt.NDArray[np.float_] | float | None
            Override the thrust setting at each waypoint, [:math: `N`].

        Returns
        -------
        AircraftPerformanceData
            BADA derived performance metrics at each waypoint.
        """


class AircraftPerformanceGrid(Model):
    """
    Support for standardizing aircraft performance methodologies on a grid.

    Currently just a container until additional models are implemented.
    """

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset:
        ...

    @overload
    def eval(self, source: MetDataset | None = None, **params: Any) -> MetDataset:
        ...

    @abc.abstractmethod
    def eval(
        self, source: GeoVectorDataset | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | MetDataset:
        """Evaluate the aircraft performance model."""


@dataclasses.dataclass
class AircraftPerformanceData:
    """Store the aircraft performance metrics calculated from BADA.

    Parameters
    ----------
    fuel_flow : npt.NDArray[DTYPE]
        Fuel mass flow rate for each waypoint, [:math:`kg s^{-1}`]
    aircraft_mass : npt.NDArray[DTYPE]
        Aircraft mass for each waypoint, [:math:`kg`]
    true_airspeed : npt.NDArray[DTYPE]
        True airspeed at each waypoint, [:math: `m s^{-1}`]
    fuel_burn: npt.NDArray[DTYPE]
        Fuel consumption for each waypoint, [:math:`kg`]. Set to an array of
        all nan values if it cannot be computed (ie, working with gridpoints).
    thrust: npt.NDArray[DTYPE]
        Thrust force, [:math:`N`]
    engine_efficiency: npt.NDArray[DTYPE]
        Overall propulsion efficiency for each waypoint
    rocd : npt.NDArray[DTYPE]
        Rate of climb and descent, [:math:`ft min^{-1}`]
    """

    fuel_flow: npt.NDArray[np.float_]
    aircraft_mass: npt.NDArray[np.float_]
    true_airspeed: npt.NDArray[np.float_]
    fuel_burn: npt.NDArray[np.float_]
    thrust: npt.NDArray[np.float_]
    engine_efficiency: npt.NDArray[np.float_]
    rocd: npt.NDArray[np.float_]
