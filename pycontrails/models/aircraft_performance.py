"""Abstract classes for aircraft performance models."""

from __future__ import annotations

import abc
import dataclasses
import numpy as np
import numpy.typing as npt
from typing import Any, overload

from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataset
from pycontrails.core.models import Model, ModelParams
from pycontrails.core.vector import GeoVectorDataset


class AircraftPerformance(Model):
    """
    Support for standardizing aircraft performance methodologies.

    Currently just a container until additional models are implemented.
    """

    @overload
    def eval(self, source: Flight, **params: Any) -> Flight:
        ...

    # This is only included for type consistency with parent. This will raise.
    @overload
    def eval(self, source: None = None, **params: Any) -> Flight:
        ...

    @abc.abstractmethod
    def eval(self, source: Flight | None = None, **params: Any) -> Flight:
        """Evaluate the aircraft performance model."""


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
class AircraftPerformanceParams(ModelParams):
    """Parameters for :class:`AircraftPerformance`."""


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
