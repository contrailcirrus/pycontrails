"""Ice super-saturated regions (ISSR)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

import numpy as np

import pycontrails
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.core.met_var import AirTemperature, SpecificHumidity
from pycontrails.core.models import Model, ModelParams
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.physics import constants, thermo
from pycontrails.utils.types import ArrayLike, apply_nan_mask_to_arraylike


@dataclass
class ISSRParams(ModelParams):
    """Default ISSR model parameters."""

    #: RHI Threshold
    rhi_threshold: float = 1.0

    #: Humidity scaling
    humidity_scaling: HumidityScaling | None = None


class ISSR(Model):
    """Ice super-saturated regions over a :class:`Flight` trajectory or :class:`MetDataset` grid.

    This model calculates points where the relative humidity over ice is greater than 1.

    Parameters
    ----------
    met : MetDataset
        Dataset containing "air_temperature" and "specific_humidity" variables

    Examples
    --------
    >>> from datetime import datetime
    >>> from pycontrails.datalib.ecmwf import ERA5
    >>> from pycontrails.models.issr import ISSR
    >>> from pycontrails.models.humidity_scaling import ConstantHumidityScaling

    >>> # Get met data
    >>> time = datetime(2022, 3, 1, 0), datetime(2022, 3, 1, 2)
    >>> variables = ["air_temperature", "specific_humidity"]
    >>> pressure_levels = [200, 250, 300]
    >>> era5 = ERA5(time, variables, pressure_levels)
    >>> met = era5.open_metdataset()

    >>> # Instantiate and run model
    >>> scaling = ConstantHumidityScaling(rhi_adj=0.98)
    >>> model = ISSR(met, humidity_scaling=scaling)
    >>> issr = model.eval()
    >>> issr.proportion  # Get proportion of values with ice supersaturation
    0.114140917963

    >>> # Run with a lower threshold
    >>> issr2 = ISSR(met, rhi_threshold=0.95, humidity_scaling=scaling).eval()
    >>> issr2.proportion
    0.146647
    """

    name = "issr"
    long_name = "Ice super-saturated regions"
    met_variables = AirTemperature, SpecificHumidity
    default_params = ISSRParams

    @overload
    def eval(self, source: Flight, **params: Any) -> Flight:
        ...

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset:
        ...

    @overload
    def eval(self, source: MetDataset | None = None, **params: Any) -> MetDataArray:
        ...

    def eval(
        self, source: GeoVectorDataset | Flight | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | Flight | MetDataArray:
        """Evaluate ice super-saturated regions along flight trajectory or on meteorology grid.

        .. versionchanged:: 0.27.0

            Humidity scaling now handled automatically. This is controlled by
            model parameter ``humidity_scaling``.

        Parameters
        ----------
        source : GeoVectorDataset | Flight | MetDataset | None, optional
            Input GeoVectorDataset or Flight.
            If None, evaluates at the :attr:`met` grid points.
        **params : Any
            Overwrite model parameters before eval

        Returns
        -------
        GeoVectorDataset | Flight | MetDataArray
            Returns 1 in ISSR, 0 everywhere else.
            Returns `np.nan` if interpolating outside meteorology grid.

        Raises
        ------
        NotImplementedError
            Raises if input ``source`` is not supported.
        """

        self.update_params(params)
        self.set_source(source)

        if isinstance(self.source, GeoVectorDataset):
            self.downselect_met()
            self.source.setdefault("air_pressure", self.source.air_pressure)

        scale_humidity = (self.params["humidity_scaling"] is not None) and (
            "specific_humidity" not in self.source
        )
        self.set_source_met()

        # apply humidity scaling, warn if no scaling is provided for ECMWF data
        if scale_humidity:
            self.params["humidity_scaling"].eval(self.source, copy_source=False)

        issr_ = issr(
            air_temperature=self.source.data["air_temperature"],
            specific_humidity=self.source.data["specific_humidity"],
            air_pressure=self.source.data["air_pressure"],
            rhi=self.source.data.get("rhi", None),  # if rhi already known, pass it in
            rhi_threshold=self.params["rhi_threshold"],
        )
        self.source["issr"] = issr_

        # Tag output with additional attrs when source is MetDataset
        if isinstance(self.source, MetDataset):
            attrs: dict[str, Any] = {
                "description": self.long_name,
                "pycontrails_version": pycontrails.__version__,
            }
            if scale_humidity:
                for k, v in self.params["humidity_scaling"].description.items():
                    attrs[f"humidity_scaling_{k}"] = v
            if self.met is not None:
                attrs["met_source"] = self.met.attrs.get("met_source", "unknown")

            self.source["issr"].data.attrs.update(attrs)
            return self.source["issr"]

        # In GeoVectorDataset case, return source
        return self.source


def issr(
    air_temperature: ArrayLike,
    specific_humidity: ArrayLike | None = None,
    air_pressure: ArrayLike | None = None,
    rhi: ArrayLike | None = None,
    rhi_threshold: float = 1.0,
) -> ArrayLike:
    r"""Calculate ice super-saturated regions.

    Regions where the atmospheric relative humidity over ice is greater than 1.

    Parameters ``air_temperature``, ``specific_humidity``, ``air_pressure``,
    and ``rhi`` must have compatible shapes when defined.

    Either ``specific_humidity`` and ``air_pressure`` must both be provided, or
    ``rhi`` must be provided.

    Parameters
    ----------
    air_temperature : ArrayLike
        A sequence or array of temperature values, :math:`[K]`.
    specific_humidity : ArrayLike | None
        A sequence or array of specific humidity values, [:math:`kg_{H_{2}O} \ kg_{moist air}`]
        None by default.
    air_pressure : ArrayLike | None
        A sequence or array of atmospheric pressure values, [:math:`Pa`]. None by default.
    rhi : ArrayLike | None, optional
        A sequence of array of RHi values, if already known. If not provided, this function
        will compute RHi from `air_temperature`, `specific_humidity`, and `air_pressure`.
        None by default.
    rhi_threshold : float, optional
        Relative humidity over ice threshold for determining ISSR state

    Returns
    -------
    ArrayLike
        ISSR state of each point indexed by the parameters.
    """
    if rhi is None:
        if specific_humidity is None or air_pressure is None:
            raise TypeError(
                "If 'rhi' is not specified, both 'specific_humidity' "
                "and 'air_pressure' must be provided."
            )
        rhi = thermo.rhi(specific_humidity, air_temperature, air_pressure)

    # store nan values to refill after casting
    nan_mask = np.isnan(rhi)

    # compute issr as int
    sufficiently_cold = air_temperature < -constants.absolute_zero
    sufficiently_humid = rhi > rhi_threshold

    issr_ = (sufficiently_cold & sufficiently_humid).astype(rhi.dtype)

    return apply_nan_mask_to_arraylike(issr_, nan_mask)
