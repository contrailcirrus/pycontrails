"""Simulate aircraft performance using empirical historical data."""

from __future__ import annotations

import dataclasses
from typing import Any, NoReturn, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pycontrails.core.aircraft_performance import (
    AircraftPerformanceGrid,
    AircraftPerformanceGridParams,
)
from pycontrails.core.met import MetDataset
from pycontrails.core.vector import GeoVectorDataset


@dataclasses.dataclass
class EmpiricalGridParams(AircraftPerformanceGridParams):
    """Parameters for :class:`EmpiricalGrid`."""

    #: Random state to use for sampling
    random_state: int | np.random.Generator | None = None

    #: Empirical data to use for sampling. Must include columns:
    #: - altitude_ft
    #: - true_airspeed
    #: - aircraft_mass
    #: - fuel_flow
    #: - engine_efficiency
    #: - aircraft_type
    #: - wingspan
    #: If None, an error will be raised at runtime.
    data: pd.DataFrame | None = None


class EmpiricalGrid(AircraftPerformanceGrid):
    """Simulate aircraft performance using empirical historical data.

    For each altitude, the model samples from the empirical data to obtain
    hypothetical aircraft performance. The data is sampled with replacement,
    so the same data may be used multiple times.

    .. warning:: This model is experimental and will change in the future.

    .. versionadded:: 0.47.0
    """

    name = "empirical_grid"
    long_name = "Empirical Grid Aircraft Performance Model"

    source: GeoVectorDataset
    default_params = EmpiricalGridParams

    variables = (
        "true_airspeed",
        "aircraft_mass",
        "fuel_flow",
        "engine_efficiency",
        "aircraft_type",
        "wingspan",
    )

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset: ...

    @overload
    def eval(self, source: MetDataset | None = ..., **params: Any) -> NoReturn: ...

    def eval(
        self, source: GeoVectorDataset | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset:
        """Query the provided historical data and sample from it.

        Parameters
        ----------
        source : GeoVectorDataset, optional
            The source vector dataset to evaluate the model on. Presently, only
            altitude is used to query the ``data`` parameter.
        **params
            Parameters to update the model with.

        Returns
        -------
        GeoVectorDataset
            The evaluated vector dataset.
        """

        self.update_params(**params)
        self.set_source(source)
        self.require_source_type(GeoVectorDataset)

        altitude_ft = self.source.altitude_ft.copy()
        altitude_ft.round(-3, out=altitude_ft)  # round to flight levels

        # Fill the source with sampled data at each flight level
        self._sample(altitude_ft)

        return self.source

    def _query_data(self) -> pd.DataFrame:
        """Query ``self.params["data"]`` for the source aircraft type."""

        # Take only the columns that are not already in the source
        columns = [v for v in self.variables if v not in self.source]
        data = self.params["data"]
        if data is None:
            raise ValueError("No data provided")

        aircraft_type = self.source.attrs.get("aircraft_type", self.params["aircraft_type"])
        data = data.query(f"aircraft_type == '{aircraft_type}'")
        assert not data.empty, f"No data for aircraft type: {aircraft_type}"

        # Round to flight levels
        data.loc[:, "altitude_ft"] = data["altitude_ft"].round(-3)

        return data[["altitude_ft", *columns]].drop(columns=["aircraft_type"])

    def _sample(self, altitude_ft: npt.NDArray[np.floating]) -> None:
        """Sample the data and update the source."""

        df = self._query_data()
        grouped = df.groupby("altitude_ft")
        rng = self.params["random_state"]

        source = self.source
        for k in df:
            source[k] = np.full_like(altitude_ft, np.nan)

        for altitude, group in grouped:
            filt = altitude_ft == altitude
            n = filt.sum()
            if n == 0:
                continue

            sample = group.sample(n=n, replace=True, random_state=rng)
            for k, v in sample.items():
                source[k][filt] = v
