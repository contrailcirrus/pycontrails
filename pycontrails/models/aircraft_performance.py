"""Abstract classes for aircraft performance models."""

from __future__ import annotations

import abc
import dataclasses
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
