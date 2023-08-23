"""Simulate aircraft performance using empirical historical data."""

from __future__ import annotations

import dataclasses
from typing import Any, NoReturn, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

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

    #: Empirical data to use for sampling. Must have columns:
    #: - altitude_ft
    #: - true_airspeed
    #: - aircraft_mass
    #: - fuel_flow
    #: - engine_efficiency
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

    variables = "true_airspeed", "aircraft_mass", "fuel_flow", "engine_efficiency"

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset:
        ...

    @overload
    def eval(self, source: MetDataset | None = ..., **params: Any) -> NoReturn:
        ...

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
        altitude_ft.round(-3, out=altitude_ft)
        dtype = altitude_ft.dtype

        # Take only the columns that are not already in the source
        columns = sorted(set(self.variables).difference(self.source))

        # Initialize the variables in the source with NaNs
        self.source.update({k: np.full(len(self.source), np.nan, dtype=dtype) for k in columns})

        # Fill the variables with sampled data
        self._sample(altitude_ft, columns)

        return self.source

    def _get_grouped(self, columns: list[str]) -> DataFrameGroupBy:
        """Group the data by altitude and return the groupby object."""

        df = self.params["data"]
        if df is None:
            raise ValueError("No data provided")

        try:
            df = df[["altitude_ft"] + columns]
        except KeyError as e:
            raise ValueError(f"Column {e} not in data") from e

        # Round to flight levels
        df["altitude_ft"] = df["altitude_ft"].round(-3)
        return df.groupby("altitude_ft")

    def _sample(self, altitude_ft: npt.NDArray[np.float_], columns: list[str]) -> None:
        """Sample the data and update the source."""

        grouped = self._get_grouped(columns)  # move to init if the groupby is expensive
        rng = self.params["random_state"]

        for altitude, group in grouped:
            filt = altitude_ft == altitude
            n = filt.sum()
            if n == 0:
                continue

            sample = group.sample(n=n, replace=True, random_state=rng)
            for k, v in sample.items():
                self.source[k][filt] = v
