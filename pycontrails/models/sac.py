"""Schmidt-Appleman criteria (SAC)."""

from __future__ import annotations

import dataclasses
from typing import Any, overload

import numpy as np
import scipy.optimize

import pycontrails
from pycontrails.core.flight import Flight
from pycontrails.core.fuel import Fuel, JetA
from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.core.met_var import AirTemperature, SpecificHumidity
from pycontrails.core.models import Model, ModelParams
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.physics import constants, thermo
from pycontrails.utils.types import ArrayLike, ArrayScalarLike, apply_nan_mask_to_arraylike

# -----------------
# Models as classes
# -----------------


@dataclasses.dataclass
class SACParams(ModelParams):
    """Parameters for :class:`SAC`."""

    #: Jet engine efficiency, [:math:`0 - 1`]
    engine_efficiency: float = 0.3

    #: Fuel type.
    #: Overridden by Fuel provided on input ``source`` attributes
    fuel: Fuel = dataclasses.field(default_factory=JetA)

    #: Humidity scaling
    humidity_scaling: HumidityScaling | None = None


class SAC(Model):
    """Determine points where Schmidt-Appleman Criteria is satisfied.

    Parameters
    ----------
    met : MetDataset
        Dataset containing "air_temperature", "specific_humidity" variables.
    params : dict[str, Any], optional
        Override :class:`SACParams` with dictionary.
    **params_kwargs
        Override :class:`SACParams` with keyword arguments.
    """

    name = "sac"
    long_name = "Schmidt-Appleman contrail formation criteria"
    met_variables = AirTemperature, SpecificHumidity
    default_params = SACParams

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
        """Evaluate the Schmidt-Appleman criteria along flight trajectory or on meteorology grid.

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
            Returns 1 where SAC is satisfied, 0 everywhere else.
            Returns `np.nan` if interpolating outside meteorology grid.

        Raises
        ------
        NotImplementedError
            Raises if input ``source`` is not supported.
        """

        self.update_params(params)
        self.set_source(source)
        self.require_source_type((GeoVectorDataset, MetDataset))

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

        # Extract source data
        air_temperature = self.source.data["air_temperature"]
        specific_humidity = self.source.data["specific_humidity"]
        air_pressure = self.source.data["air_pressure"]
        engine_efficiency = self.get_source_param("engine_efficiency")

        # Flight class has fuel attribute, use this instead of params
        if isinstance(self.source, Flight):
            fuel = self.source.fuel
        else:
            fuel = self.get_source_param("fuel")
        assert isinstance(fuel, Fuel), "The fuel attribute must be of type Fuel"

        ei_h2o = fuel.ei_h2o
        q_fuel = fuel.q_fuel

        G = slope_mixing_line(specific_humidity, air_pressure, engine_efficiency, ei_h2o, q_fuel)
        T_sat_liquid_ = T_sat_liquid(G)
        rh_crit_sac = rh_critical_sac(air_temperature, T_sat_liquid_, G)
        rh = thermo.rh(specific_humidity, air_temperature, air_pressure)
        sac_ = sac(rh, rh_crit_sac)

        # Attaching some intermediate artifacts onto the source
        self.source["G"] = G
        self.source["T_sat_liquid"] = T_sat_liquid_
        self.source["rh"] = rh
        self.source["rh_critical_sac"] = rh_crit_sac
        self.source["sac"] = sac_

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
            if isinstance(engine_efficiency, (int, float)):
                attrs["engine_efficiency"] = engine_efficiency

            self.source["sac"].data.attrs.update(attrs)
            return self.source["sac"]

        # In GeoVectorDataset case, return source
        return self.source


# -------------------
# Models as functions
# -------------------


def slope_mixing_line(
    specific_humidity: ArrayLike,
    air_pressure: ArrayLike,
    engine_efficiency: float | ArrayLike,
    ei_h2o: float,
    q_fuel: float,
) -> ArrayLike:
    r"""Calculate the slope of the mixing line in a temperature-humidity diagram.

    This quantity is often notated with ``G`` in the literature.

    Parameters
    ----------
    specific_humidity : ArrayLike
        A sequence or array of specific humidity values, [:math:`kg_{H_{2}O} \ kg_{air}`]
    air_pressure : ArrayLike
        A sequence or array of atmospheric pressure values, [:math:`Pa`].
    engine_efficiency: float | ArrayLike
        Engine efficiency, [:math:`0 - 1`]
    ei_h2o : float
        Emission index of water vapor, [:math:`kg \ kg^{-1}`]
    q_fuel : float
        Specific combustion heat of fuel combustion, [:math:`J \ kg^{-1} \ K^{-1}`]

    Returns
    -------
    ArrayLike
        Slope of the mixing line in a temperature-humidity diagram, [:math:`Pa \ K^{-1}`]
    """
    c_pm = thermo.c_pm(specific_humidity)  # Often taken as 1004 (= constants.c_pd)
    return (ei_h2o * c_pm * air_pressure) / (constants.epsilon * q_fuel * (1.0 - engine_efficiency))


def T_sat_liquid(G: ArrayLike) -> ArrayLike:
    r"""Calculate temperature at which liquid saturation curve has slope G.

    Parameters
    ----------
    G : ArrayLike
        Slope of the mixing line in a temperature-humidity diagram.

    Returns
    -------
    ArrayLike
        Maximum threshold temperature for 100% relative humidity with respect to liquid,
        [:math:`K`]. This can also be interpreted as the temperature at which the liquid
        saturation curve has slope G.

    References
    ----------
    - :cite:`schumannConditionsContrailFormation1996`

    See Also
    --------
    :func:`T_sat_liquid_high_accuracy`

    Notes
    -----
    Defined (using notation T_LM) in :cite:`schumannConditionsContrailFormation1996`
    in the first full paragraph on page 10 as

      for T = T_LC, the mixing line just touches [is tangent to]
      the saturation curve. See equation (10).

    The formula used here is taken from equation (31).
    """
    # FIXME: Presently, mypy is not aware that numpy ufuncs will return `xr.DataArray``
    # when xr.DataArray is passed in. This will get fixed at some point in the future
    # as `numpy` their typing patterns, after which the "type: ignore" comment can
    # get ripped out.
    # We could explicitly check for `xr.DataArray` then use `xr.apply_ufunc`, but
    # this only renders our code more boilerplate and less performant.
    # This comment is pasted several places in `pycontrails` -- they should all be
    # addressed at the same time.
    log_ = np.log(G - 0.053)
    return -46.46 - constants.absolute_zero + 9.43 * log_ + 0.72 * log_**2  # type: ignore[return-value]  # noqa: E501


def _e_sat_liquid_prime(T: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate derivative of :func:`thermo.e_sat_liquid`.

    Parameters
    ----------
    T : ArrayScalarLike
        Temperature, [:math:`K`].

    Returns
    -------
    ArrayScalarLike
        Derivative of :func:`thermo.e_sat_liquid`, [:math:``Pa \ K^{-1}`].
    """
    d_inside = 6096.9385 / (T**2) - 0.02711193 + 1.673952 * 1e-5 * 2 * T + 2.433502 / T
    return thermo.e_sat_liquid(T) * d_inside


def T_sat_liquid_high_accuracy(
    G: ArrayLike,
    maxiter: int = 5,
) -> ArrayLike:
    """Calculate temperature at which liquid saturation curve has slope G.

    The function :func:`T_sat_liquid` gives a first order approximation to equation (10)
    of the Schumann paper referenced below. This function uses Newton's method to
    compute the numeric solution to (10).

    Parameters
    ----------
    G : ArrayLike
        Slope of the mixing line
    maxiter : int, optional
        Passed into :func:`scipy.optimize.newton`. Because ``T_sat_liquid`` is already
        fairly accurate, few iterations are needed for Newton's method to converge.
        By default, 5.

    Returns
    -------
    ArrayLike
        Maximum threshold temperature for 100% relative humidity with respect to liquid,
        [:math:`K`].

    References
    ----------
    - :cite:`schumannConditionsContrailFormation1996`

    See Also
    --------
    :func:`T_sat_liquid_high`
    """
    init_guess = T_sat_liquid(G)

    def func(T: ArrayLike) -> ArrayLike:
        """Equation (10) from Schumann 1996."""
        return _e_sat_liquid_prime(T) - G

    return scipy.optimize.newton(func, init_guess, maxiter=maxiter)


def rh_critical_sac(air_temperature: ArrayLike, T_sat_liquid: ArrayLike, G: ArrayLike) -> ArrayLike:
    r"""Calculate critical relative humidity threshold of contrail formation.

    Parameters
    ----------
    air_temperature : ArrayLike
        A sequence or array of temperature values, [:math:`K`]
    T_sat_liquid : ArrayLike,
        Maximum threshold temperature for 100% relative humidity with respect to liquid, [:math:`K`]
    G : ArrayLike
        Slope of the mixing line in a temperature-humidity diagram.

    Returns
    -------
    ArrayLike
        Critical relative humidity of contrail formation, [:math:`[0 - 1]`]

    References
    ----------
    - :cite:`ponaterContrailsComprehensiveGlobal2002`
    """
    e_sat_T_sat_liquid = thermo.e_sat_liquid(T_sat_liquid)  # always positive
    e_sat_T = thermo.e_sat_liquid(air_temperature)  # always positive

    # Below, `rh_crit` can be negative
    rh_crit = (G * (air_temperature - T_sat_liquid) + e_sat_T_sat_liquid) / e_sat_T

    # Per Ponater, section 2.3:
    # >>> The critical relative humidity `r_contr` can range from 0 to 1
    # The "first order" term `G * (air_temperature - T_sat_liquid)` can be negative.
    # Consequently, we clip rh_crit at 0 and 1.
    # After clipping, when rh_crit = 0 the SAC is guaranteed to be satisfied.

    # Per Ponater, beginning of section 2.3:
    # >>> "No contrails are possible if T > T_contr" <<<
    # In our notation, this inequality is `air_temperature > T_sat_liquid`
    # We set the corresponding rh_crit to infinity, indicating no SAC is possible

    # numpy case
    if isinstance(rh_crit, np.ndarray):
        rh_crit.clip(0.0, 1.0, out=rh_crit)  # clip in place
        rh_crit[air_temperature > T_sat_liquid] = np.inf
        return rh_crit

    # xarray case
    # FIXME: the two cases handle nans differently. Unfortunately, unit tests break
    # if I try to consolidate to a single condition
    rh_crit = rh_crit.clip(0.0, 1.0)
    return rh_crit.where(air_temperature <= T_sat_liquid, np.inf)


def sac(
    rh: ArrayLike,
    rh_crit_sac: ArrayLike,
) -> ArrayLike:
    r"""Points at which the Schmidt-Appleman Criteria is satisfied.

    Parameters of type :class:`ArrayLike` must have compatible shapes.

    Parameters
    ----------
    rh : ArrayLike
        Relative humidity values
    rh_crit_sac: ArrayLike
        Critical relative humidity threshold of contrail formation

    Returns
    -------
    ArrayLike
        SAC state of each point indexed by the :class:`ArrayLike` parameters.
        Returned array has floating ``dtype`` with values

            - 0.0 signifying SAC fails
            - 1.0 signifying SAC holds

        NaN entries of parameters propagate into the returned array.
    """
    nan_mask = np.isnan(rh) | np.isnan(rh_crit_sac)

    dtype = np.result_type(rh, rh_crit_sac)
    sac_ = (rh > rh_crit_sac).astype(dtype)
    return apply_nan_mask_to_arraylike(sac_, nan_mask)


def T_critical_sac(
    T_LM: ArrayLike,
    relative_humidity: ArrayLike,
    G: ArrayLike,
    maxiter: int = 10,
) -> ArrayLike:
    r"""Estimate temperature threshold for persistent contrail formation.

    This quantity is defined as ``T_LC`` in Schumann (see reference below). Equation (11)
    of this paper implicitly defines ``T_LC`` as the solution to the equation
    ::

        T_LC = T_LM - (e_L(T_LM) - rh * e_L(T_LC)) / G

    For relative humidity above 0.999, the corresponding entry from ``T_LM``
    is returned (page 10, top of the right-hand column). Otherwise, the solution
    to the equation above is approximated via Newton's method.

    Parameters
    ----------
    T_LM : ArrayLike
        Output of :func:`T_sat_liquid` calculation.
    relative_humidity : ArrayLike
        Relative humidity values
    G : ArrayLike
        Slope of the mixing line in a temperature-humidity diagram.
    maxiter : int, optional
        Passed into :func:`scipy.optimize.newton`. By default, 10.

    Returns
    -------
    ArrayLike
        Critical temperature threshold values.

    References
    ----------
    - :cite:`schumannConditionsContrailFormation1996`
    """
    # Near U = 1, Newton's method is slow to converge (I believe this is because
    # the function `func` has a double root at T_LM when U = 1, so Newton's method
    # is somewhat degenerate here)
    # But the answer is known in this case. This is discussed at the top of the
    # right hand column on page 10 in Schumann 1996.
    # We only apply Newton's method at points with rh bounded below 1 (scipy will
    # raise an error if Newton's method is not converging well).
    filt = (relative_humidity < 0.999) & np.isfinite(T_LM)
    if not np.any(filt):
        return T_LM

    U_filt = relative_humidity[filt]
    T_LM_filt = T_LM[filt]
    e_L_of_T_LM_filt = thermo.e_sat_liquid(T_LM_filt)
    G_filt = G[filt]

    def func(T: ArrayLike) -> ArrayLike:
        """Equation (11) from Schumann."""
        return T - T_LM_filt + (e_L_of_T_LM_filt - U_filt * thermo.e_sat_liquid(T)) / G_filt

    def fprime(T: ArrayLike) -> ArrayLike:
        return 1.0 - U_filt * _e_sat_liquid_prime(T) / G_filt

    # This initial guess should be less than T_LM.
    # For relative_humidity away from 1, Newton's method converges quickly, and so
    # any initial guess will work.
    init_guess = T_LM_filt - 1.0
    newton_approx = scipy.optimize.newton(
        func, init_guess, fprime=fprime, maxiter=maxiter, disp=False
    )

    # For relative_humidity > 0.999, we just use T_LM
    # We copy over the entire array T_LM here instead of using np.empty_like
    # in order to keep typing compatible with xarray types
    out = T_LM.copy()
    out[filt] = newton_approx
    return out
