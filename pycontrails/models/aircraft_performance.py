"""Abstract classes for aircraft performance models."""

from __future__ import annotations

import abc
import dataclasses
import warnings
from typing import Any, NoReturn, overload

import numpy as np
import numpy.typing as npt

from pycontrails.core import flight
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataset
from pycontrails.core.models import Model, ModelParams, interpolate_met
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.physics import jet

#: Default load factor for aircraft performance models.
DEFAULT_LOAD_FACTOR = 0.7


@dataclasses.dataclass
class AircraftPerformanceParams(ModelParams):
    """Parameters for :class:`AircraftPerformance`."""

    #: Whether to correct fuel flow to ensure it remains within
    #: the operational limits of the aircraft type.
    correct_fuel_flow: bool = True

    #: The number of iterations used to calculate aircraft mass and fuel flow.
    #: The default value of 3 is sufficient for most cases.
    n_iter: int = 3


class AircraftPerformance(Model):
    """
    Support for standardizing aircraft performance methodologies.

    This class provides a :meth:`simulate_fuel_and_performance` method for
    iteratively calculating aircraft mass and fuel flow rate.

    The implementing class must bring :meth:`eval` and
    :meth:`calculate_aircraft_performance` methods. At runtime, these methods
    are intended to be chained together as follows:

    1. The :meth:`eval` method is called with a :class:`Flight`
    2. The :meth:`simulate_fuel_and_performance` method is called inside :meth:`eval`
       to iteratively calculate aircraft mass and fuel flow rate. If an aircraft
       mass is provided, the fuel flow rate is calculated once directly with a single
       call to :meth:`calculate_aircraft_performance`. If an aircraft mass is not
       provided, the fuel flow rate is calculated iteratively with multiple calls to
       :meth:`calculate_aircraft_performance`.
    """

    @abc.abstractmethod
    @overload
    def eval(self, source: Flight, **params: Any) -> Flight:
        ...

    @abc.abstractmethod
    @overload
    def eval(self, source: list[Flight], **params: Any) -> list[Flight]:
        ...

    # The source must be a Flight or list of Flights
    @abc.abstractmethod
    @overload
    def eval(self, source: None = None, **params: Any) -> NoReturn:
        ...

    @abc.abstractmethod
    def eval(self, source: Flight | list[Flight] | None = None, **params: Any) -> Any:
        """Evaluate the aircraft performance model."""

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
        n_iter: int,
        amass_oew: float,
        amass_mtow: float,
        amass_mpl: float,
        load_factor: float,
        takeoff_mass: float | None,
        **kwargs: Any,
    ) -> AircraftPerformanceData:
        r"""
        Calculate aircraft mass, fuel mass flow rate, and overall propulsion efficiency.

        This method performs ``n_iter`` iterations, each of
        which calls :meth:`calculate_aircraft_performance`. Each successive
        iteration generates a better estimate for mass fuel flow rate and aircraft
        mass at each waypoint.

        Parameters
        ----------
        aircraft_type: str
            Aircraft type designator used to query the underlying model database.
        altitude_ft: npt.NDArray[np.float_]
            Altitude at each waypoint, [:math:`ft`]
        time: npt.NDArray[np.datetime64]
            Waypoint time in np.datetime64 format.
        true_airspeed: npt.NDArray[np.float_]
            True airspeed for each waypoint, [:math:`m s^{-1}`]
        air_temperature : npt.NDArray[np.float_]
            Ambient temperature for each waypoint, [:math:`K`]
        aircraft_mass : npt.NDArray[np.float_] | float | None
            Override the aircraft_mass at each waypoint, [:math:`kg`].
        thrust : npt.NDArray[np.float_] | float | None
            Override the thrust setting at each waypoint, [:math: `N`].
        engine_efficiency : npt.NDArray[np.float_] | float | None
            Override the engine efficiency at each waypoint.
        fuel_flow : npt.NDArray[np.float_] | float | None
            Override the fuel flow at each waypoint, [:math:`kg s^{-1}`].
        q_fuel : float
            Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].
        amass_oew : float
            Aircraft operating empty weight, [:math:`kg`]. Used to determine
            the initial aircraft mass if ``takeoff_mass`` is not provided.
            This quantity is constant for a given aircraft type.
        amass_mtow : float
            Aircraft maximum take-off weight, [:math:`kg`]. Used to determine
            the initial aircraft mass if ``takeoff_mass`` is not provided.
            This quantity is constant for a given aircraft type.
        amass_mpl : float
            Aircraft maximum payload, [:math:`kg`]. Used to determine
            the initial aircraft mass if ``takeoff_mass`` is not provided.
            This quantity is constant for a given aircraft type.
        load_factor : float
            Aircraft load factor assumption (between 0 and 1). If unknown,
            a value of 0.7 is a reasonable default. Typically, this parameter
            is between 0.6 and 0.8. During the height of the COVID-19 pandemic,
            this parameter was often much lower.
        takeoff_mass : float | None, optional
            If known, the takeoff mass can be provided to skip the calculation
            in :func:`jet.initial_aircraft_mass`. In this case, the parameters
            ``load_factor``, ``amass_oew``, ``amass_mtow``, and ``amass_mpl`` are
            ignored.
        **kwargs : Any
            Additional keyword arguments are passed to :meth:`calculate_aircraft_performance`.

        Returns
        -------
        AircraftPerformanceData
            Results from the final iteration is returned.
        """

        # shortcut if aircraft mass is provided
        if aircraft_mass is not None:
            warnings.warn(
                "Parameter 'aircraft_mass' provided to 'simulate_fuel_and_performance' "
                f"is not None. Skipping {n_iter} iterations and only "
                "calculating aircraft performance once."
            )
            return self._simulate_fuel_and_performance_known_aircraft_mass(
                aircraft_type=aircraft_type,
                altitude_ft=altitude_ft,
                time=time,
                true_airspeed=true_airspeed,
                air_temperature=air_temperature,
                aircraft_mass=aircraft_mass,
                thrust=thrust,
                engine_efficiency=engine_efficiency,
                fuel_flow=fuel_flow,
                q_fuel=q_fuel,
                **kwargs,
            )

        return self._simulate_fuel_and_performance_unknown_aircraft_mass(
            aircraft_type=aircraft_type,
            altitude_ft=altitude_ft,
            time=time,
            true_airspeed=true_airspeed,
            air_temperature=air_temperature,
            thrust=thrust,
            engine_efficiency=engine_efficiency,
            fuel_flow=fuel_flow,
            q_fuel=q_fuel,
            n_iter=n_iter,
            amass_oew=amass_oew,
            amass_mtow=amass_mtow,
            amass_mpl=amass_mpl,
            load_factor=load_factor,
            takeoff_mass=takeoff_mass,
            **kwargs,
        )

    def _simulate_fuel_and_performance_known_aircraft_mass(
        self,
        *,
        aircraft_type: str,
        altitude_ft: npt.NDArray[np.float_],
        time: npt.NDArray[np.datetime64],
        true_airspeed: npt.NDArray[np.float_],
        air_temperature: npt.NDArray[np.float_],
        aircraft_mass: npt.NDArray[np.float_] | float,
        thrust: npt.NDArray[np.float_] | float | None,
        engine_efficiency: npt.NDArray[np.float_] | float | None,
        fuel_flow: npt.NDArray[np.float_] | float | None,
        q_fuel: float,
        **kwargs: Any,
    ) -> AircraftPerformanceData:
        # If fuel_flow is None and a non-constant aircraft_mass is provided
        # at each waypoint, then assume that the derivative with respect to
        # time is the fuel flow rate.
        if fuel_flow is None and isinstance(aircraft_mass, np.ndarray):
            d_aircraft_mass = np.diff(aircraft_mass)

            if np.any(d_aircraft_mass > 0.0):
                warnings.warn(
                    "There are increases in aircraft mass between waypoints. This is not expected."
                )

            # Only proceed if aircraft mass is decreasing somewhere
            # This excludes a constant aircraft mass
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
            **kwargs,
        )

    def _simulate_fuel_and_performance_unknown_aircraft_mass(
        self,
        *,
        aircraft_type: str,
        altitude_ft: npt.NDArray[np.float_],
        time: npt.NDArray[np.datetime64],
        true_airspeed: npt.NDArray[np.float_],
        air_temperature: npt.NDArray[np.float_],
        thrust: npt.NDArray[np.float_] | float | None,
        engine_efficiency: npt.NDArray[np.float_] | float | None,
        fuel_flow: npt.NDArray[np.float_] | float | None,
        q_fuel: float,
        n_iter: int,
        amass_oew: float,
        amass_mtow: float,
        amass_mpl: float,
        load_factor: float,
        takeoff_mass: float | None,
        **kwargs: Any,
    ) -> AircraftPerformanceData:
        # Variable aircraft_mass will change dynamically after each iteration
        # Set the initial aircraft mass depending on a possible load factor

        aircraft_mass: npt.NDArray[np.float_] | float
        if takeoff_mass is not None:
            aircraft_mass = takeoff_mass
        else:
            # The initial aircraft mass gets updated at each iteration
            # The exact value here is not important
            aircraft_mass = amass_oew + load_factor * (amass_mtow - amass_oew)

        for _ in range(n_iter):
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
                **kwargs,
            )

            # The max value in the BADA tables is 4.6 kg/s per engine.
            # Multiplying this by 4 engines and giving a buffer.
            if np.any(aircraft_performance.fuel_flow > 25.0):
                raise RuntimeError(
                    "Model failure: fuel mass flow rate is unrealistic and the "
                    "built-in guardrails are not working."
                )

            tot_reserve_fuel = jet.reserve_fuel_requirements(
                aircraft_performance.rocd,
                altitude_ft,
                aircraft_performance.fuel_flow,
                aircraft_performance.fuel_burn,
            )

            aircraft_mass = jet.update_aircraft_mass(
                operating_empty_weight=amass_oew,
                max_takeoff_weight=amass_mtow,
                max_payload=amass_mpl,
                fuel_burn=aircraft_performance.fuel_burn,
                total_reserve_fuel=tot_reserve_fuel,
                load_factor=load_factor,
                takeoff_mass=takeoff_mass,
            )

        # Update aircraft mass to the latest fuel consumption estimate
        # As long as the for-loop is entered, the aircraft mass will be
        # a numpy array.
        aircraft_performance.aircraft_mass = aircraft_mass  # type: ignore[assignment]

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
        aircraft_mass: npt.NDArray[np.float_] | float,
        engine_efficiency: npt.NDArray[np.float_] | float | None,
        fuel_flow: npt.NDArray[np.float_] | float | None,
        thrust: npt.NDArray[np.float_] | float | None,
        q_fuel: float,
        **kwargs: Any,
    ) -> AircraftPerformanceData:
        r"""
        Calculate aircraft performance along a trajectory.

        When ``time`` is not None, this method should be used for a single flight
        trajectory. Waypoints are coupled via the ``time`` parameter.

        This method computes the rate of climb and descent (ROCD) to determine
        flight phases: "cruise", "climb", and "descent". Performance metrics
        depend on this phase.

        When ``time`` is None, this method can be used to simulate flight performance
        over an arbitrary sequence of flight waypoints by assuming nominal flight
        characteristics. In this case, each point is treated independently and
        all points are assumed to be in a "cruise" phase of the flight.

        Parameters
        ----------
        aircraft_type : str
            Used to query the underlying model database for aircraft engine parameters.
        altitude_ft : npt.NDArray[np.float_]
            Altitude at each waypoint, [:math:`ft`]
        air_temperature : npt.NDArray[np.float_]
            Ambient temperature for each waypoint, [:math:`K`]
        time: npt.NDArray[np.datetime64] | None
            Waypoint time in ``np.datetime64`` format. If None, only drag force
            will is used in thrust calculations (ie, no vertical change and constant
            horizontal change). In addition, aircraft is assumed to be in cruise.
        true_airspeed : npt.NDArray[np.float_] | float | None
            True airspeed for each waypoint, [:math:`m s^{-1}`].
            If None, a nominal value is used.
        aircraft_mass : npt.NDArray[np.float_] | float
            Aircraft mass for each waypoint, [:math:`kg`].
        engine_efficiency : npt.NDArray[np.float_] | float | None
            Override the engine efficiency at each waypoint.
        fuel_flow : npt.NDArray[np.float_] | float | None
            Override the fuel flow at each waypoint, [:math:`kg s^{-1}`].
        thrust : npt.NDArray[np.float_] | float | None
            Override the thrust setting at each waypoint, [:math: `N`].
        q_fuel : float
            Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].
        **kwargs : Any
            Additional keyword arguments to pass to the model.

        Returns
        -------
        AircraftPerformanceData
            Derived performance metrics at each waypoint.
        """

    def ensure_true_airspeed_on_source(self) -> None:
        """Add ``true_airspeed`` field to :attr:`source` data if not already present."""

        if "true_airspeed" in self.source:
            return

        if not isinstance(self.source, Flight):
            raise TypeError("Model source must be a Flight to calculate true airspeed.")

        # Two step fallback: try to find u_wind and v_wind.
        try:
            u = interpolate_met(self.met, self.source, "eastward_wind", **self.interp_kwargs)
            v = interpolate_met(self.met, self.source, "northward_wind", **self.interp_kwargs)

        except (ValueError, KeyError):
            raise ValueError(
                "Variable `true_airspeed` not found. Include 'eastward_wind' and"
                " 'northward_wind' variables on `met`in model constructor, or define"
                " `true_airspeed` data on flight. This can be achieved by calling the"
                " `Flight.segment_true_airspeed` method."
            )

        self.source["true_airspeed"] = self.source.segment_true_airspeed(u, v)


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
    """Store the computed aircraft performance metrics.

    Parameters
    ----------
    fuel_flow : npt.NDArray[np.float_]
        Fuel mass flow rate for each waypoint, [:math:`kg s^{-1}`]
    aircraft_mass : npt.NDArray[np.float_]
        Aircraft mass for each waypoint, [:math:`kg`]
    true_airspeed : npt.NDArray[np.float_]
        True airspeed at each waypoint, [:math: `m s^{-1}`]
    fuel_burn: npt.NDArray[np.float_]
        Fuel consumption for each waypoint, [:math:`kg`]. Set to an array of
        all nan values if it cannot be computed (ie, working with gridpoints).
    thrust: npt.NDArray[np.float_]
        Thrust force, [:math:`N`]
    engine_efficiency: npt.NDArray[np.float_]
        Overall propulsion efficiency for each waypoint
    rocd : npt.NDArray[np.float_]
        Rate of climb and descent, [:math:`ft min^{-1}`]
    """

    fuel_flow: npt.NDArray[np.float_]
    aircraft_mass: npt.NDArray[np.float_]
    true_airspeed: npt.NDArray[np.float_]
    fuel_burn: npt.NDArray[np.float_]
    thrust: npt.NDArray[np.float_]
    engine_efficiency: npt.NDArray[np.float_]
    rocd: npt.NDArray[np.float_]
