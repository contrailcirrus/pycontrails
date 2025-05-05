"""Abstract interfaces for aircraft performance models."""

from __future__ import annotations

import abc
import dataclasses
import warnings
from typing import Any, Generic, NoReturn, overload

import numpy as np
import numpy.typing as npt

from pycontrails.core import flight, fuel
from pycontrails.core.fleet import Fleet
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import AirTemperature, EastwardWind, MetVariable, NorthwardWind
from pycontrails.core.models import Model, ModelParams, interpolate_met
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.physics import jet
from pycontrails.utils.types import ArrayOrFloat

#: Default load factor for aircraft performance models.
#: See :func:`pycontrails.physics.jet.aircraft_load_factor`
#: for a higher precision approach to estimating the load factor.
DEFAULT_LOAD_FACTOR = 0.83


# --------------------------------------
# Trajectory aircraft performance models
# --------------------------------------


@dataclasses.dataclass
class CommonAircraftPerformanceParams:
    """Params for :class:`AircraftPerformanceParams` and :class:`AircraftPerformanceGridParams`."""

    #: Account for "in-service" engine deterioration between maintenance cycles.
    #: Default value is set to +2.5% increase in fuel consumption.
    #: Reference:
    #: Gurrola Arrieta, M.D.J., Botez, R.M. and Lasne, A., 2024. An Engine Deterioration Model for
    #: Predicting Fuel Consumption Impact in a Regional Aircraft. Aerospace, 11(6), p.426.
    engine_deterioration_factor: float = 0.025


@dataclasses.dataclass
class AircraftPerformanceParams(ModelParams, CommonAircraftPerformanceParams):
    """Parameters for :class:`AircraftPerformance`."""

    #: Whether to correct fuel flow to ensure it remains within
    #: the operational limits of the aircraft type.
    correct_fuel_flow: bool = True

    #: The number of iterations used to calculate aircraft mass and fuel flow.
    #: The default value of 3 is sufficient for most cases.
    n_iter: int = 3

    #: Experimental. If True, fill waypoints below the lowest altitude met
    #: level with ISA temperature when interpolating "air_temperature" or "t".
    #: If the ``met`` data is not provided, the entire air temperature array
    #: is approximated with the ISA temperature. Enabling this does NOT
    #: remove any NaN values in the ``met`` data itself.
    fill_low_altitude_with_isa_temperature: bool = False

    #: Experimental. If True, fill waypoints below the lowest altitude met
    #: level with zero wind when computing true airspeed. In other words,
    #: approximate low-altitude true airspeed with the ground speed. Enabling
    #: this does NOT remove any NaN values in the ``met`` data itself.
    #: In the case that ``met`` is not provided, any missing values are
    #: filled with zero wind.
    fill_low_altitude_with_zero_wind: bool = False


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

    source: Flight
    met_variables: tuple[MetVariable, ...] = ()
    optional_met_variables: tuple[MetVariable, ...] = (AirTemperature, EastwardWind, NorthwardWind)
    default_params = AircraftPerformanceParams

    @overload
    def eval(self, source: Fleet, **params: Any) -> Fleet: ...

    @overload
    def eval(self, source: Flight, **params: Any) -> Flight: ...

    @overload
    def eval(self, source: None = ..., **params: Any) -> NoReturn: ...

    def eval(self, source: Flight | None = None, **params: Any) -> Flight:
        """Evaluate the aircraft performance model.

        Parameters
        ----------
        source : Flight
            Flight trajectory to evaluate. Can be a :class:`Flight` or :class:`Fleet`.
        params : Any
            Override :attr:`params` with keyword arguments.

        Returns
        -------
        Flight
            Flight trajectory with aircraft performance data.
        """
        self.update_params(params)
        self.set_source(source)
        self.source = self.require_source_type(Flight)
        self.downselect_met()
        self.set_source_met()
        self._cleanup_indices()

        # Calculate temperature and true airspeed if not included on source
        self.ensure_air_temperature_on_source()
        self.ensure_true_airspeed_on_source()

        if isinstance(self.source, Fleet):
            fls = [self.eval_flight(fl) for fl in self.source.to_flight_list()]
            self.source = Fleet.from_seq(fls, attrs=self.source.attrs, broadcast_numeric=False)
            return self.source

        self.source = self.eval_flight(self.source)
        return self.source

    @abc.abstractmethod
    def eval_flight(self, fl: Flight) -> Flight:
        """Evaluate the aircraft performance model on a single flight trajectory.

        The implementing model adds the following fields to the source flight:

        - ``aircraft_mass``: aircraft mass at each waypoint, [:math:`kg`]
        - ``fuel_flow``: fuel mass flow rate at each waypoint, [:math:`kg s^{-1}`]
        - ``thrust``: thrust at each waypoint, [:math:`N`]
        - ``engine_efficiency``: engine efficiency at each waypoint
        - ``rocd``: rate of climb or descent at each waypoint, [:math:`ft min^{-1}`]
        - ``fuel_burn``: fuel burn at each waypoint, [:math:`kg`]

        In addition, the following attributes are added to the source flight:

        - ``n_engine``: number of engines
        - ``wingspan``: wingspan, [:math:`m`]
        - ``max_mach``: maximum Mach number
        - ``max_altitude``: maximum altitude, [:math:`m`]
        - ``total_fuel_burn``: total fuel burn, [:math:`kg`]
        """

    def simulate_fuel_and_performance(
        self,
        *,
        aircraft_type: str,
        altitude_ft: npt.NDArray[np.floating],
        time: npt.NDArray[np.datetime64],
        true_airspeed: npt.NDArray[np.floating],
        air_temperature: npt.NDArray[np.floating],
        aircraft_mass: npt.NDArray[np.floating] | float | None,
        thrust: npt.NDArray[np.floating] | float | None,
        engine_efficiency: npt.NDArray[np.floating] | float | None,
        fuel_flow: npt.NDArray[np.floating] | float | None,
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
        altitude_ft: npt.NDArray[np.floating]
            Altitude at each waypoint, [:math:`ft`]
        time: npt.NDArray[np.datetime64]
            Waypoint time in ``np.datetime64`` format.
        true_airspeed: npt.NDArray[np.floating]
            True airspeed for each waypoint, [:math:`m s^{-1}`]
        air_temperature : npt.NDArray[np.floating]
            Ambient temperature for each waypoint, [:math:`K`]
        aircraft_mass : npt.NDArray[np.floating] | float | None
            Override the aircraft_mass at each waypoint, [:math:`kg`].
        thrust : npt.NDArray[np.floating] | float | None
            Override the thrust setting at each waypoint, [:math: `N`].
        engine_efficiency : npt.NDArray[np.floating] | float | None
            Override the engine efficiency at each waypoint.
        fuel_flow : npt.NDArray[np.floating] | float | None
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
        altitude_ft: npt.NDArray[np.floating],
        time: npt.NDArray[np.datetime64],
        true_airspeed: npt.NDArray[np.floating],
        air_temperature: npt.NDArray[np.floating],
        aircraft_mass: npt.NDArray[np.floating] | float,
        thrust: npt.NDArray[np.floating] | float | None,
        engine_efficiency: npt.NDArray[np.floating] | float | None,
        fuel_flow: npt.NDArray[np.floating] | float | None,
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
        altitude_ft: npt.NDArray[np.floating],
        time: npt.NDArray[np.datetime64],
        true_airspeed: npt.NDArray[np.floating],
        air_temperature: npt.NDArray[np.floating],
        thrust: npt.NDArray[np.floating] | float | None,
        engine_efficiency: npt.NDArray[np.floating] | float | None,
        fuel_flow: npt.NDArray[np.floating] | float | None,
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

        aircraft_mass: npt.NDArray[np.floating] | float
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
        altitude_ft: npt.NDArray[np.floating],
        air_temperature: npt.NDArray[np.floating],
        time: npt.NDArray[np.datetime64] | None,
        true_airspeed: npt.NDArray[np.floating] | float | None,
        aircraft_mass: npt.NDArray[np.floating] | float,
        engine_efficiency: npt.NDArray[np.floating] | float | None,
        fuel_flow: npt.NDArray[np.floating] | float | None,
        thrust: npt.NDArray[np.floating] | float | None,
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
        altitude_ft : npt.NDArray[np.floating]
            Altitude at each waypoint, [:math:`ft`]
        air_temperature : npt.NDArray[np.floating]
            Ambient temperature for each waypoint, [:math:`K`]
        time: npt.NDArray[np.datetime64] | None
            Waypoint time in ``np.datetime64`` format. If None, only drag force
            will is used in thrust calculations (ie, no vertical change and constant
            horizontal change). In addition, aircraft is assumed to be in cruise.
        true_airspeed : npt.NDArray[np.floating] | float | None
            True airspeed for each waypoint, [:math:`m s^{-1}`].
            If None, a nominal value is used.
        aircraft_mass : npt.NDArray[np.floating] | float
            Aircraft mass for each waypoint, [:math:`kg`].
        engine_efficiency : npt.NDArray[np.floating] | float | None
            Override the engine efficiency at each waypoint.
        fuel_flow : npt.NDArray[np.floating] | float | None
            Override the fuel flow at each waypoint, [:math:`kg s^{-1}`].
        thrust : npt.NDArray[np.floating] | float | None
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

    def ensure_air_temperature_on_source(self) -> None:
        """Add ``air_temperature`` field to :attr:`source` data if not already present.

        This function operates in-place. If ``air_temperature`` is not already present
        on :attr:`source`, it is calculated by interpolation from met data.
        """
        fill_with_isa = self.params["fill_low_altitude_with_isa_temperature"]

        if "air_temperature" in self.source:
            if not fill_with_isa:
                return
            _fill_low_altitude_with_isa_temperature(self.source, 0.0)
            return

        temp_available = self.met is not None and "air_temperature" in self.met

        if not temp_available:
            if fill_with_isa:
                self.source["air_temperature"] = self.source.T_isa()
                return
            msg = (
                "Cannot compute air temperature without providing met data that includes an "
                "'air_temperature' variable. Either include met data with 'air_temperature' "
                "in the model constructor, define 'air_temperature' data on the flight, or set "
                "'fill_low_altitude_with_isa_temperature' to True."
            )
            raise ValueError(msg)

        interpolate_met(self.met, self.source, "air_temperature", **self.interp_kwargs)

        if not fill_with_isa:
            return

        met_level_0 = self.met.data["level"][-1].item()  # type: ignore[union-attr]
        _fill_low_altitude_with_isa_temperature(self.source, met_level_0)

    def ensure_true_airspeed_on_source(self) -> None:
        """Add ``true_airspeed`` field to :attr:`source` data if not already present.

        This function operates in-place. If ``true_airspeed`` is not already present
        on :attr:`source`, it is calculated using :meth:`Flight.segment_true_airspeed`.
        """
        tas = self.source.get("true_airspeed")
        fill_with_groundspeed = self.params["fill_low_altitude_with_zero_wind"]

        if tas is not None:
            if not fill_with_groundspeed:
                return
            cond = np.isnan(tas)
            tas[cond] = self.source.segment_groundspeed()[cond]
            return

        # Use current cocip convention: eastward_wind on met, u_wind on source
        wind_available = ("u_wind" in self.source and "v_wind" in self.source) or (
            self.met is not None and "eastward_wind" in self.met and "northward_wind" in self.met
        )

        if not wind_available:
            if fill_with_groundspeed:
                tas = self.source.segment_groundspeed()
                self.source["true_airspeed"] = tas
                return
            msg = (
                "Cannot compute 'true_airspeed' without 'eastward_wind' and 'northward_wind' "
                "met data. Either include met data in the model constructor, define "
                "'true_airspeed' data on the flight, or set "
                "'fill_low_altitude_with_zero_wind' to True."
            )
            raise ValueError(msg)

        u = interpolate_met(self.met, self.source, "eastward_wind", "u_wind", **self.interp_kwargs)
        v = interpolate_met(self.met, self.source, "northward_wind", "v_wind", **self.interp_kwargs)

        if fill_with_groundspeed:
            if self.met is None:
                cond = np.isnan(u) & np.isnan(v)
            else:
                met_level_max = self.met.data["level"][-1].item()  # type: ignore[union-attr]
                cond = self.source.level > met_level_max

            # We DON'T overwrite the original u and v arrays already attached to the source
            u = np.where(cond, 0.0, u)
            v = np.where(cond, 0.0, v)

        out = self.source.segment_true_airspeed(u, v)
        self.source["true_airspeed"] = out


@dataclasses.dataclass
class AircraftPerformanceData:
    """Store the computed aircraft performance metrics.

    Parameters
    ----------
    fuel_flow : npt.NDArray[np.floating]
        Fuel mass flow rate for each waypoint, [:math:`kg s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating]
        Aircraft mass for each waypoint, [:math:`kg`]
    true_airspeed : npt.NDArray[np.floating]
        True airspeed at each waypoint, [:math: `m s^{-1}`]
    fuel_burn: npt.NDArray[np.floating]
        Fuel consumption for each waypoint, [:math:`kg`]. Set to an array of
        all nan values if it cannot be computed (ie, working with gridpoints).
    thrust: npt.NDArray[np.floating]
        Thrust force, [:math:`N`]
    engine_efficiency: npt.NDArray[np.floating]
        Overall propulsion efficiency for each waypoint
    rocd : npt.NDArray[np.floating]
        Rate of climb and descent, [:math:`ft min^{-1}`]
    """

    fuel_flow: npt.NDArray[np.floating]
    aircraft_mass: npt.NDArray[np.floating]
    true_airspeed: npt.NDArray[np.floating]
    fuel_burn: npt.NDArray[np.floating]
    thrust: npt.NDArray[np.floating]
    engine_efficiency: npt.NDArray[np.floating]
    rocd: npt.NDArray[np.floating]


# --------------------------------
# Grid aircraft performance models
# --------------------------------


@dataclasses.dataclass
class AircraftPerformanceGridParams(ModelParams, CommonAircraftPerformanceParams):
    """Parameters for :class:`AircraftPerformanceGrid`."""

    #: Fuel type
    fuel: fuel.Fuel = dataclasses.field(default_factory=fuel.JetA)

    #: ICAO code designating simulated aircraft type.
    #: Can be overridden by including ``aircraft_type`` attribute in source data
    aircraft_type: str = "B737"

    #: Mach number, [:math:`Ma`]
    #: If ``None``, a nominal cruise value is determined by the implementation.
    #: Can be overridden by including a ``mach_number`` key in source data
    mach_number: float | None = None

    #: Aircraft mass, [:math:`kg`]
    #: If ``None``, a nominal value is determined by the implementation.
    #: Can be overridden by including an ``aircraft_mass`` key in source data
    aircraft_mass: float | None = None


class AircraftPerformanceGrid(Model):
    """
    Support for standardizing aircraft performance methodologies on a grid.

    Currently just a container until additional models are implemented.
    """

    @overload
    @abc.abstractmethod
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset: ...

    @overload
    @abc.abstractmethod
    def eval(self, source: MetDataset | None = ..., **params: Any) -> MetDataset: ...

    @abc.abstractmethod
    def eval(
        self, source: GeoVectorDataset | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | MetDataset:
        """Evaluate the aircraft performance model."""


@dataclasses.dataclass
class AircraftPerformanceGridData(Generic[ArrayOrFloat]):
    """Store the computed aircraft performance metrics for nominal cruise conditions."""

    #: Fuel mass flow rate, [:math:`kg s^{-1}`]
    fuel_flow: ArrayOrFloat

    #: Engine efficiency, [:math:`0-1`]
    engine_efficiency: ArrayOrFloat


def _fill_low_altitude_with_isa_temperature(vector: GeoVectorDataset, met_level_max: float) -> None:
    """Fill low-altitude NaN values in ``air_temperature`` with ISA values.

    The ``air_temperature`` param is assumed to have been computed by
    interpolating against a gridded air temperature field that did not
    necessarily extend to the surface. This function fills points below the
    lowest altitude in the gridded data with ISA temperature values.

    This function operates in-place and modifies the ``air_temperature`` field.

    Parameters
    ----------
    vector : GeoVectorDataset
        GeoVectorDataset instance associated with the ``air_temperature`` data.
    met_level_max : float
        The maximum level in the met data, [:math:`hPa`].
    """
    air_temperature = vector["air_temperature"]
    is_nan = np.isnan(air_temperature)
    low_alt = vector.level > met_level_max
    cond = is_nan & low_alt

    t_isa = vector.T_isa()
    air_temperature[cond] = t_isa[cond]
