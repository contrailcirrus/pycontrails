"""
Persistent contrail regions (PCR = SAC & ISSR).

Equivalent to (SAC & ISSR)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

import numpy as np
import xarray as xr

from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import AirTemperature, SpecificHumidity
from pycontrails.core.models import Model
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.models import issr, sac
from pycontrails.physics import thermo
from pycontrails.utils.types import apply_nan_mask_to_arraylike


@dataclass
class PCRParams(sac.SACParams, issr.ISSRParams):
    """Persistent Contrail Regions (PCR) parameters."""


class PCR(Model):
    """Determine points with likely persistent contrails (PCR).

    Intersection of Ice Super Saturated Regions (ISSR) with regions in which the Schmidt-Appleman
    Criteria (SAC) is satisfied.

    Parameters
    ----------
    met : MetDataset
        Dataset containing "air_temperature", "specific_humidity" variables
    params : dict[str, Any], optional
        Override PCR model parameters with dictionary.
        See :class:`PCRParams` for model parameters.
    **params_kwargs
        Override PCR model parameters with keyword arguments.
        See :class:`PCRParams` for model parameters.
    """

    name = "pcr"
    long_name = "Persistent contrail regions"
    met_variables = AirTemperature, SpecificHumidity
    default_params = PCRParams

    @overload
    def eval(self, source: Flight, **params: Any) -> Flight: ...

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset: ...

    @overload
    def eval(self, source: MetDataset | None = ..., **params: Any) -> MetDataset: ...

    def eval(
        self, source: GeoVectorDataset | Flight | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | Flight | MetDataset:
        """Evaluate potential contrails regions of the :attr:`met` grid.

        Parameters
        ----------
        source : GeoVectorDataset | Flight | MetDataset | None, optional
            Input GeoVectorDataset or Flight.
            If None, evaluates at the :attr:`met` grid points.
        **params : Any
            Overwrite model parameters.

        Returns
        -------
        GeoVectorDataset | Flight | MetDataset
            Returns 1 in potential contrail regions, 0 everywhere else.
            Returns ``np.nan`` if interpolating outside meteorology grid.

        """

        self.update_params(params)
        self.set_source(source)
        issr_params = {k: v for k, v in self.params.items() if hasattr(issr.ISSR.default_params, k)}
        issr_model = issr.ISSR(self.met, params=issr_params, copy_source=False)
        issr_model.eval(self.source)

        sac_params = {k: v for k, v in self.params.items() if hasattr(sac.SAC.default_params, k)}
        # NOTE: met is not needed here: ISSR already used it
        sac_model = sac.SAC(met=None, params=sac_params, copy_source=False)
        sac_model.eval(self.source)

        pcr_ = _pcr_from_issr_and_sac(self.source.data["issr"], self.source.data["sac"])  # type: ignore[type-var]
        self.source["pcr"] = pcr_

        return self.source


def pcr[A: (np.ndarray, xr.DataArray)](
    air_temperature: A,
    specific_humidity: A,
    air_pressure: A,
    engine_efficiency: float | A,
    ei_h2o: float,
    q_fuel: float,
) -> tuple[A, A, A]:
    r"""Calculate regions of persistent contrail formation.

    Ice Super Saturated Regions (ISSR) where the Schmidt-Appleman Criteria (SAC) is satisfied.

    Parameters of type ``A`` must have compatible shapes.

    Parameters
    ----------
    air_temperature : A
        A sequence or array of temperature values, [:math:`K`]
    specific_humidity : A
        A sequence or array of specific humidity values, [:math:`kg_{H_{2}O} \ kg_{air}^{-1}`]
    air_pressure : A
        A sequence or array of atmospheric pressure values, [:math:`Pa`].
    engine_efficiency: float | A
        Engine efficiency, [:math:`0 - 1`]
    ei_h2o : float
        Emission index of water vapor, [:math:`kg \ kg^{-1}`]
    q_fuel : float
        Specific combustion heat of fuel combustion, [:math:`J \ kg^{-1} \ K^{-1}`]

    Returns
    -------
    pcr : A
        PCR state of each point indexed by the ``A``
        parameters.
    sac : A
        SAC state
    issr : A
        ISSR state
    """
    issr_ = issr.issr(air_temperature, specific_humidity, air_pressure)
    G = sac.slope_mixing_line(specific_humidity, air_pressure, engine_efficiency, ei_h2o, q_fuel)
    T_sat_liquid_ = sac.T_sat_liquid(G)
    rh = thermo.rh(specific_humidity, air_temperature, air_pressure)
    rh_crit_sac = sac.rh_critical_sac(air_temperature, T_sat_liquid_, G)
    sac_ = sac.sac(rh, rh_crit_sac)

    pcr_ = _pcr_from_issr_and_sac(issr_, sac_)
    return pcr_, sac_, issr_


def _pcr_from_issr_and_sac[A: (np.ndarray, xr.DataArray)](issr_: A, sac_: A) -> A:
    # store nan values to refill after casting
    nan_mask = np.isnan(issr_) | np.isnan(sac_)

    dtype = np.result_type(issr_, sac_)
    pcr_ = ((issr_ > 0.0) & (sac_ > 0.0)).astype(dtype)
    return apply_nan_mask_to_arraylike(pcr_, nan_mask)
