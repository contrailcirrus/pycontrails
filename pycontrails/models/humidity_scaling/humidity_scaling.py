"""Support for humidity scaling methodologies."""

from __future__ import annotations

import abc
import dataclasses
import functools
import pathlib
import warnings
from typing import Any, NoReturn, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from overrides import overrides

from pycontrails.core import models
from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.physics import constants, thermo, units
from pycontrails.utils.types import ArrayLike


def _rhi_over_q(air_temperature: ArrayLike, air_pressure: ArrayLike) -> ArrayLike:
    """Compute the quotient ``RHi / q``."""
    return air_pressure * (constants.R_v / constants.R_d) / thermo.e_sat_ice(air_temperature)


class HumidityScaling(models.Model):
    """Support for standardizing humidity scaling methodologies.

    The method :meth:`scale` or :meth:`eval` should be called immediately
    after interpolation over meteorology data.

    .. versionadded:: 0.27.0
    """

    #: Variables required in addition to specific_humidity, air_temperature, and air_pressure
    #: These are either :class:`ModelParams` specific to scaling, or variables that should
    #: be extracted from :meth:`eval` parameter ``source``.
    scaler_specific_keys: tuple[str, ...] = tuple()

    @property
    @abc.abstractmethod
    def formula(self) -> str:
        """Serializable formula for humidity scaler."""

    @property
    def description(self) -> dict[str, Any]:
        """Get description for instance."""
        params = {k: v for k in self.scaler_specific_keys if (v := self.params.get(k)) is not None}
        return {"name": self.name, "formula": self.formula, **params}

    # Used by pycontrails.utils.json.NumpyEncoder
    # This ensures model parameters are serializable to JSON
    to_json = description

    @abc.abstractmethod
    def scale(
        self,
        specific_humidity: ArrayLike,
        air_temperature: ArrayLike,
        air_pressure: ArrayLike,
        **kwargs: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike]:
        r"""Compute scaled specific humidity and RHi.

        See docstring for the implementing subclass for specific methodology.

        Parameters
        ----------
        specific_humidity : ArrayLike
            Unscaled specific relative humidity, [:math:`kg \ kg^{-1}`]. Typically,
            this is interpolated meteorology data.
        air_temperature : ArrayLike
            Air temperature, [:math:`K`]. Typically, this is interpolated meteorology
            data.
        air_pressure : ArrayLike
            Pressure, [:math:`Pa`]
        kwargs : ArrayLike
            Other keyword-only variables and model parameters used by the formula.

        Returns
        -------
        specific_humidity : ArrayLike
            Scaled specific humidity.
        rhi : ArrayLike
            Scaled relative humidity over ice.

        See Also
        --------
        :meth:`eval`
        """

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset:
        ...

    @overload
    def eval(self, source: MetDataset, **params: Any) -> MetDataset:
        ...

    @overload
    def eval(self, source: None = ..., **params: Any) -> NoReturn:
        ...

    def eval(
        self, source: GeoVectorDataset | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | MetDataset:
        """Scale specific humidity values on ``source``.

        This method mutates the parameter ``source`` by modifying its
        "specific_humidity" variable and by attaching an "rhi" variable. Set
        model parameter ``copy_source=True`` to avoid mutating ``source``.

        Parameters
        ----------
        source : GeoVectorDataset | MetDataset
            Data with variables "specific_humidity", "air_temperature",
            and any variables defined by :attr:`scaler_specific_keys`.
        **params : Any
            Overwrite model parameters before eval

        Returns
        -------
        GeoVectorDataset | MetDataset
            Source data with updated "specific_humidity" and "rhi". If ``source``
            is :class:`GeoVectorDataset`, "air_pressure" data is also attached.

        See Also
        --------
        :meth:`scale`
        """
        self.update_params(params)
        if source is None:
            raise TypeError("Parameter source must be GeoVectorDataset or MetDataset")
        self.set_source(source)

        if "rhi" in self.source:
            warnings.warn(
                "Variable 'rhi' already found on source to be scaled. This "
                "is unexpected and may be the result of humidity scaling "
                "being applied more than once."
            )

        p: np.ndarray | xr.DataArray
        if isinstance(self.source, GeoVectorDataset):
            p = self.source.setdefault("air_pressure", self.source.air_pressure)
        else:
            p = self.source.data["air_pressure"]

        q = self.source.data["specific_humidity"]
        T = self.source.data["air_temperature"]
        kwargs = {k: self.get_source_param(k) for k in self.scaler_specific_keys}

        q, rhi = self.scale(q, T, p, **kwargs)
        self.source.update(specific_humidity=q, rhi=rhi)

        return self.source


@dataclasses.dataclass
class ConstantHumidityScalingParams(models.ModelParams):
    """Parameters for :class:`ConstantHumidityScaling`."""

    #: Scale specific humidity by dividing it with adjustment factor per
    #: :cite:`schumannContrailCirrusPrediction2012` eq. (9). Set to a constant
    #: 0.9 in :cite:`schumannContrailCirrusPrediction2012` to account for sub-scale
    #: variability of specific humidity. A value of 1.0 provides no scaling.
    rhi_adj: float = 0.97


class ConstantHumidityScaling(HumidityScaling):
    """Scale specific humidity by applying a constant uniform scaling.

    This scalar simply applies the transformation..

        rhi -> rhi / rhi_adj

    where ``rhi_adj`` is a constant specified by :attr:`params` or overridden by
    a variable or attribute on ``source`` in :meth:`eval`.

    The convention to divide by ``rhi_adj`` instead of considering the more natural
    product ``rhi_adj * rhi`` is somewhat arbitrary. In short, ``rhi_adj`` can be
    thought of as the critical threshold for supersaturation.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    - :cite:`reutterIceSupersaturatedRegions2020`
    """

    name = "constant_scale"
    long_name = "Constant humidity scaling"
    formula = "rhi -> rhi / rhi_adj"
    default_params = ConstantHumidityScalingParams
    scaler_specific_keys = ("rhi_adj",)

    @overrides
    def scale(
        self,
        specific_humidity: ArrayLike,
        air_temperature: ArrayLike,
        air_pressure: ArrayLike,
        **kwargs: Any,
    ) -> tuple[ArrayLike, ArrayLike]:
        rhi_adj = kwargs.get("rhi_adj", self.params["rhi_adj"])
        q = specific_humidity / rhi_adj
        rhi = thermo.rhi(q, air_temperature, air_pressure)
        return q, rhi


@dataclasses.dataclass
class ExponentialBoostHumidityScalingParams(ConstantHumidityScalingParams):
    """Parameters for :class:`ExponentialBoostHumidityScaling`."""

    #: Boost RHi values exceeding 1 as described in :cite:`teohAviationContrailClimate2022`.
    #: In :meth:`eval`, this can be overridden by a keyword argument with the same name.
    rhi_boost_exponent: float = 1.7

    #: Used to clip overinflated unrealistic RHi values.
    clip_upper: float = 1.7


class ExponentialBoostHumidityScaling(HumidityScaling):
    """Scale humidity by composing constant scaling with exponential boosting.

    This formula composes the transformations

    #. constant scaling: ``rhi -> rhi / rhi_adj``
    #. exponential boosting: ``rhi -> rhi ^ rhi_boost_exponent if rhi > 1``
    #. clipping: ``rhi -> min(rhi, clip_upper)``

    where ``rhi_adj``, ``rhi_boost_exponent``, and ``clip_upper`` are model :attr:`params`.

    References
    ----------
    - :cite:`teohAviationContrailClimate2022`

    See Also
    --------
    :class:`ExponentialBoostLatitudeCorrectionHumidityScaling`
    """

    name = "exponential_boost"
    long_name = "Constant humidity scaling composed with exponential boosting"
    formula = "rhi -> (rhi / rhi_adj) ^ rhi_boost_exponent"
    default_params = ExponentialBoostHumidityScalingParams
    scaler_specific_keys = "rhi_adj", "rhi_boost_exponent", "clip_upper"

    @overrides
    def scale(
        self,
        specific_humidity: ArrayLike,
        air_temperature: ArrayLike,
        air_pressure: ArrayLike,
        **kwargs: Any,
    ) -> tuple[ArrayLike, ArrayLike]:
        # Get coefficients
        rhi_adj = kwargs["rhi_adj"]
        rhi_boost_exponent = kwargs["rhi_boost_exponent"]
        clip_upper = kwargs["clip_upper"]

        # Compute uncorrected RHi
        rhi_over_q = _rhi_over_q(air_temperature, air_pressure)
        rhi = specific_humidity * rhi_over_q

        # Correct RHi
        rhi /= rhi_adj

        # Find ISSRs
        is_issr = rhi >= 1.0

        # Apply boosting to ISSRs
        if isinstance(rhi, xr.DataArray):
            rhi = rhi.where(~is_issr, rhi**rhi_boost_exponent)
            rhi = rhi.clip(max=clip_upper)

        else:
            boosted_rhi = rhi[is_issr] ** rhi_boost_exponent
            boosted_rhi.clip(max=clip_upper, out=boosted_rhi)
            rhi[is_issr] = boosted_rhi

        # Recompute specific_humidity from corrected rhi
        specific_humidity = rhi / rhi_over_q

        # Return the pair
        return specific_humidity, rhi


@dataclasses.dataclass
class ExponentialBoostLatitudeCorrectionHumidityScalingParams(models.ModelParams):
    """Parameters for :class:`ExponentialBoostLatitudeCorrectionHumidityScaling`."""

    #: Constants obtained by fitting a sigmoid curve to IAGOS data. These can be
    #: overridden by keyword arguments with the same name.
    rhi_a0: float = 0.062621
    rhi_a1: float = 0.45893
    rhi_a2: float = 39.254
    rhi_a3: float = 0.95224
    rhi_b0: float = 1.4706
    rhi_b1: float = 0.44312
    rhi_b2: float = 18.755
    rhi_b3: float = 1.4325


class ExponentialBoostLatitudeCorrectionHumidityScaling(HumidityScaling):
    """Correct RHi values derived from ECMWF ERA5 HRES.

    Unlike other RHi correction factors, this function applies a custom latitude-based
    term and has been tuned for global application.

    This formula composes the transformations

    #. constant scaling: ``rhi -> rhi / rhi_adj``
    #. exponential boosting: ``rhi -> rhi ^ rhi_boost_exponent if rhi > 1``
    #. clipping: ``rhi -> min(rhi, rhi_max)``

    where ``rhi_adj`` and ``rhi_boost_exponent`` depend on ``latitude`` to minimize
    error between ERA5 HRES and IAGOS in-situ data.

    For each waypoint, ``rhi_max`` ensures that the corrected RHi does not exceed the
    maximum value according to thermodynamics:

    - ``rhi_max = p_liq(T) / p_ice(T)`` for ``T > 235 K``,
      (Pruppacher and Klett, 1997)
    - ``rhi_max = 1.67 + (1.45 - 1.67) * (T - 190.) / (235. - 190.)`` for ``T < 235 K``
      (Karcher and Lohmann, 2002; Tompkins et al., 2007)

    The RHi correction addresses the known limitations of the ERA5 HRES humidity fields,
    ensuring that the ISSR coverage area and RHi-distribution is consistent with in-situ
    measurements from the IAGOS dataset. Generally, the correction:

    #. reduces the ISSR coverage area near the equator,
    #. increases the ISSR coverage area at higher latitudes, and
    #. accounts for localized regions with very high ice supersaturation (RHi > 120%).

    This methodology is an extension of Teoh et al. (2022) and has not yet been
    peer-reviewed/published.

    The ERA5 HRES <> IAGOS fitting uses a sigmoid curve to capture significant
    changes in tropopause height at 20 - 50 degrees latitude.

    The method :meth:`eval` requires a ``latitude`` keyword argument.

    References
    ----------
    - :cite:`teohAviationContrailClimate2022`
    - Kärcher, B. and Lohmann, U., 2002. A parameterization of cirrus cloud formation: Homogeneous
      freezing of supercooled aerosols. Journal of Geophysical Research: Atmospheres, 107(D2),
      pp.AAC-4.
    - Pruppacher, H.R. and Klett, J.D. (1997) Microphysics of Clouds and Precipitation. 2nd Edition,
      Kluwer Academic, Dordrecht, 954 p.
    - Tompkins, A.M., Gierens, K. and Rädel, G., 2007. Ice supersaturation in the ECMWF integrated
      forecast system. Quarterly Journal of the Royal Meteorological Society: A journal of the
      atmospheric sciences, applied meteorology and physical oceanography, 133(622), pp.53-63.

    See Also
    --------
    :class:`ExponentialBoostHumidityScaling`
    """

    name = "exponential_boost_latitude_customization"
    long_name = "Latitude specific humidity scaling composed with exponential boosting"
    formula = "rhi -> (rhi / rhi_adj) ^ rhi_boost_exponent"
    default_params = ExponentialBoostLatitudeCorrectionHumidityScalingParams
    scaler_specific_keys = (
        "latitude",
        "rhi_a0",
        "rhi_a1",
        "rhi_a2",
        "rhi_a3",
        "rhi_b0",
        "rhi_b1",
        "rhi_b2",
        "rhi_b3",
    )

    @overrides
    def scale(
        self,
        specific_humidity: ArrayLike,
        air_temperature: ArrayLike,
        air_pressure: ArrayLike,
        **kwargs: Any,
    ) -> tuple[ArrayLike, ArrayLike]:
        # Get sigmoid coefficients
        a0 = kwargs["rhi_a0"]
        a1 = kwargs["rhi_a1"]
        a2 = kwargs["rhi_a2"]
        a3 = kwargs["rhi_a3"]
        b0 = kwargs["rhi_b0"]
        b1 = kwargs["rhi_b1"]
        b2 = kwargs["rhi_b2"]
        b3 = kwargs["rhi_b3"]

        # Use the dtype of specific_humidity to determine the precision of the
        # the calculation. If working with gridded data here, latitude will have
        # float64 precision.
        latitude = kwargs["latitude"]
        if latitude.dtype != specific_humidity.dtype:
            latitude = latitude.astype(specific_humidity.dtype)
        lat_abs = np.abs(latitude)

        # Compute uncorrected RHi
        rhi_over_q = _rhi_over_q(air_temperature, air_pressure)
        rhi = specific_humidity * rhi_over_q

        # Calculate the rhi_adj factor and correct RHi
        rhi_adj = a0 / (1.0 + np.exp(a1 * (lat_abs - a2))) + a3
        rhi /= rhi_adj

        # Find ISSRs
        is_issr = rhi >= 1

        # Limit RHi to maximum value allowed by physics
        rhi_max = _calc_rhi_max(air_temperature)

        # Apply boosting to ISSRs
        if isinstance(rhi, xr.DataArray):
            boost_exponent = b0 / (1.0 + np.exp(b1 * (lat_abs - b2))) + b3
            rhi = rhi.where(~is_issr, rhi**boost_exponent)
            rhi = rhi.clip(max=rhi_max)

        else:
            # Calculate the optimal b coefficient over points in ISSRs
            boost_exponent = b0 / (1.0 + np.exp(b1 * (lat_abs[is_issr] - b2))) + b3

            # Apply boosting to ISSRs
            rhi[is_issr] = rhi[is_issr] ** boost_exponent
            rhi.clip(max=rhi_max, out=rhi)

        # Recompute specific_humidity from corrected rhi
        specific_humidity = rhi / rhi_over_q

        # Return the pair
        return specific_humidity, rhi


def _calc_rhi_max(air_temperature: ArrayLike) -> ArrayLike:
    p_ice: ArrayLike
    p_liq: ArrayLike

    if isinstance(air_temperature, xr.DataArray):
        p_ice = thermo.e_sat_ice(air_temperature)
        p_liq = thermo.e_sat_liquid(air_temperature)
        return xr.where(
            air_temperature < 235.0,
            1.67 + (1.45 - 1.67) * (air_temperature - 190.0) / (235.0 - 190.0),
            p_liq / p_ice,
        )

    low = air_temperature < 235
    air_temperature_low = air_temperature[low]
    air_temperature_high = air_temperature[~low]

    p_ice = thermo.e_sat_ice(air_temperature_high)
    p_liq = thermo.e_sat_liquid(air_temperature_high)

    out = np.empty_like(air_temperature)
    out[low] = 1.67 + (1.45 - 1.67) * (air_temperature_low - 190.0) / (235.0 - 190.0)
    out[~low] = p_liq / p_ice
    return out


@dataclasses.dataclass
class HumidityScalingByLevelParams(models.ModelParams):
    """Parameters for :class:`HumidityScalingByLevel`."""

    #: Fraction of troposphere for mid-troposphere humidity scaling.
    #: Default value suggested in :cite:`schumannContrailCirrusPrediction2012`.
    rhi_adj_mid_troposphere: float = 0.8

    #: Fraction of troposphere for stratosphere humidity scaling.
    #: Default value suggested in :cite:`schumannContrailCirrusPrediction2012`.
    rhi_adj_stratosphere: float = 1.0

    #: Adjustment factor for mid-troposphere humidity scaling. Default value
    #: of 0.8 taken from :cite:`schumannContrailCirrusPrediction2012`.
    mid_troposphere_threshold: float = 0.8

    #: Adjustment factor for stratosphere humidity scaling. Default value
    #: of 1.0 taken from :cite:`schumannContrailCirrusPrediction2012`.
    stratosphere_threshold: float = 1.0


class HumidityScalingByLevel(HumidityScaling):
    """Apply custom scaling to specific_humidity by pressure level.

    This implements the original humidity scaling scheme suggested in
    :cite:`schumannContrailCirrusPrediction2012`. In particular, see eq. (9)
    and the surrounding text, quoted below.

        Hence, the critical value RHic is usually taken different and below 100%
        in NWP models. In the ECMWF model, this value is..

            RHic = 0.8, (9)

        in the mid-troposphere, 1.0 in the stratosphere and follows
        a smooth transition with pressure altitude between these two
        values in the upper 20 % of the troposphere. For simplicity
        of further analysis, we divide the input value of q by RHic
        initially.

    See :class:`ConstantHumidityScaling` for the simple method described
    above.

    The diagram below shows the piecewise-linear ``rhi_adj`` factor by
    level. In particular, ``rhi_adj`` is constant at the stratosphere and above,
    linearly changes from the mid-troposphere to the stratosphere, and is
    constant at the mid-troposphere and below.

    ::

                     _________  stratosphere rhi_adj = 1.0
                    /
                   /
                  /
        _________/  mid-troposphere rhi_adj = 0.8

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """

    name = "constant_scale_by_level"
    long_name = "Constant humidity scaling by level"
    formula = "rhi -> rhi / rhi_adj"
    default_params = HumidityScalingByLevelParams
    scaler_specific_keys = (
        "rhi_adj_mid_troposphere",
        "rhi_adj_stratosphere",
        "mid_troposphere_threshold",
        "stratosphere_threshold",
    )

    @overrides
    def scale(
        self,
        specific_humidity: ArrayLike,
        air_temperature: ArrayLike,
        air_pressure: ArrayLike,
        **kwargs: Any,
    ) -> tuple[ArrayLike, ArrayLike]:
        rhi_adj_mid_troposphere = kwargs["rhi_adj_mid_troposphere"]
        rhi_adj_stratosphere = kwargs["rhi_adj_stratosphere"]
        mid_troposphere_threshold = kwargs["mid_troposphere_threshold"]
        stratosphere_threshold = kwargs["stratosphere_threshold"]

        thresholds = np.array([stratosphere_threshold, mid_troposphere_threshold])
        xp = units.m_to_pl(constants.h_tropopause * thresholds)  # type: ignore

        # np.interp expects a sorted parameter `xp`
        # The calculation will get bungled if this isn't the case
        if xp[0] > xp[1]:
            raise ValueError(
                "Attribute 'stratosphere_threshold' must exceed "
                "attribute 'mid_troposphere_threshold'."
            )

        level = air_pressure / 100.0
        fp = [rhi_adj_stratosphere, rhi_adj_mid_troposphere]
        rhi_adj = np.interp(level, xp=xp, fp=fp)

        q = specific_humidity / rhi_adj
        rhi = thermo.rhi(q, air_temperature, air_pressure)
        return q, rhi


@functools.cache
def _load_quantiles() -> pd.DataFrame:
    """Load precomputed ERA5 and IAGOS quantiles.

    Returns
    -------
    pd.DataFrame
        A DataFrame with 801 rows and 34 columns. The index is parameterized
        by the quantile, and the columns are parameterized by the met source
        and the interpolation methodology. The IAOGS quantiles are in the
        ``("iagos", "iagos")`` column.
    """
    path = pathlib.Path(__file__).parent / "quantiles" / "era5-quantiles.pq"
    return pd.read_parquet(path)


def histogram_matching(
    era5_rhi: npt.NDArray[np.float_],
    product_type: str,
    member: int | None,
    q_method: str,
) -> npt.NDArray[np.float_]:
    """Map ERA5-derived RHi to it's corresponding IAGOS quantile via histogram matching.

    This matching is performed on a **single** ERA5 ensemble member.

    Parameters
    ----------
    era5_rhi : npt.NDArray[np.float_]
        ERA5-derived RHi values for the given ensemble member.
    product_type : {"reanalysis", "ensemble_members"}
        The ERA5 product type.
    member : int | None
        The ERA5 ensemble member to use. Must be in the range ``[0, 10)``.
        Only used if ``product_type == "ensemble_members"``.
    q_method : {None, "cubic-spline", "log-q-log-p"}
        The interpolation method.

    Returns
    -------
    npt.NDArray[np.float_]
        The IAGOS quantiles corresponding to the ERA5-derived RHi values.
    """
    df = _load_quantiles()
    iagos_quantiles = df[("iagos", "iagos")]

    if product_type == "ensemble_members":
        col = f"ensemble{member}", q_method or "linear-q"
    elif product_type == "reanalysis":
        col = "reanalysis", q_method or "linear-q"
    else:
        raise ValueError(
            f"Invalid 'product_type' value '{product_type}'. "
            "Must be one of ['reanalysis', 'ensemble_members']."
        )

    try:
        era5_quantiles = df[col]
    except KeyError:
        models.raise_invalid_q_method_error(q_method)

    out = np.interp(era5_rhi, era5_quantiles, iagos_quantiles)

    # Preserve the dtype (np.interp returns float64)
    return out.astype(era5_rhi.dtype, copy=False)


def histogram_matching_all_members(
    era5_rhi_all_members: npt.NDArray[np.float_], member: int, q_method: str
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Recalibrate ERA5-derived RHi values to IAGOS quantiles by histogram matching.

    This recalibration requires values for **all** ERA5 ensemble members. Currently, the
    number of ensemble members is hard-coded to 10.

    Parameters
    ----------
    era5_rhi_all_members : npt.NDArray[np.float_]
        ERA5-derived RHi values for all ensemble members. This array should have shape ``(n, 10)``.
    member : int
        The ERA5 ensemble member to use. Must be in the range ``[0, 10)``.
    q_method : {None, "cubic-spline", "log-q-log-p"}
        The interpolation method.

    Returns
    -------
    ensemble_mean_rhi : npt.NDArray[np.float_]
        The mean RHi values after histogram matching over all ensemble members.
        This is an array of shape ``(n,)``.
    ensemble_member_rhi : npt.NDArray[np.float_]
        The RHi values after histogram matching for the given ensemble member.
        This is an array of shape ``(n,)``.
    """

    n_members = 10
    assert era5_rhi_all_members.shape[1] == n_members

    # Perform histogram matching on the given ensemble member
    ensemble_member_rhi = histogram_matching(
        era5_rhi_all_members[:, member], "ensemble_members", member, q_method
    )

    # Perform histogram matching on all other ensemble members
    # Add up the results into a single 'ensemble_mean_rhi' array
    ensemble_mean_rhi: npt.NDArray[np.float_] = 0.0  # type: ignore[assignment]
    for r in range(n_members):
        if r == member:
            ensemble_mean_rhi += ensemble_member_rhi
        else:
            ensemble_mean_rhi += histogram_matching(
                era5_rhi_all_members[:, r], "ensemble_members", r, q_method
            )

    # Divide by the number of ensemble members to get the mean
    ensemble_mean_rhi /= n_members

    return ensemble_mean_rhi, ensemble_member_rhi


def eckel_scaling(
    ensemble_mean_rhi: npt.NDArray[np.float_],
    ensemble_member_rhi: npt.NDArray[np.float_],
    q_method: str,
) -> npt.NDArray[np.float_]:
    """Apply Eckel scaling to the given RHi values.

    Parameters
    ----------
    ensemble_mean_rhi : npt.NDArray[np.float_]
        The ensemble mean RHi values. This should be a 1D array with the same shape as
        ``ensemble_member_rhi``.
    ensemble_member_rhi : npt.NDArray[np.float_]
        The RHi values for a single ensemble member.
    q_method : {None, "cubic-spline", "log-q-log-p"}
        The interpolation method.

    Returns
    -------
    npt.NDArray[np.float_]
        The scaled RHi values. Values are manually clipped at 0 to ensure
        only non-negative values are returned.

    References
    ----------
    :cite:`eckelCalibratedProbabilisticQuantitative1998`
    """

    # https://journals.ametsoc.org/view/journals/wefo/27/1/waf-d-11-00015_1.xml
    # https://doi.org/10.1016/B978-0-12-812372-0.00003-0

    if q_method is None:
        eckel_a = -6.365384483974193e-05
        eckel_c = 2.731157095387021
    elif q_method == "cubic-spline":
        eckel_a = -6.268024189244961e-05
        eckel_c = 2.6843757126334302
    elif q_method == "log-q-log-p":
        eckel_a = -5.8690498260424506e-05
        eckel_c = 2.679501356493337
    else:
        models.raise_invalid_q_method_error(q_method)

    out = (ensemble_mean_rhi - eckel_a) + eckel_c * (ensemble_member_rhi - ensemble_mean_rhi)
    out.clip(min=0.0, out=out)
    return out


@dataclasses.dataclass
class HistogramMatchingParams(models.ModelParams):
    """Parameters for :class:`HistogramMatching`."""

    #: The ERA5 product. Must be one of ``"reanalysis"`` or ``"ensemble_members"``.
    product_type: str = "reanalysis"

    #: The ERA5 ensemble member to use. Must be in the range ``[0, 10)``.
    #: Only used if ``product_type`` is ``"ensemble_members"``.
    member: int | None = None


class HistogramMatching(HumidityScaling):
    """Scale humidity by histogram matching to IAGOS RHi quantiles."""

    name = "histogram_matching"
    long_name = "IAGOS RHi histogram matching"
    formula = "era5_quantiles -> iagos_quantiles"
    default_params = HistogramMatchingParams

    @overrides
    def scale(  # type: ignore[override]
        self,
        specific_humidity: npt.NDArray[np.float_],
        air_temperature: npt.NDArray[np.float_],
        air_pressure: npt.NDArray[np.float_],
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        rhi_over_q = _rhi_over_q(air_temperature, air_pressure)
        rhi = rhi_over_q * specific_humidity

        rhi_1 = histogram_matching(
            rhi,
            self.params["product_type"],
            self.params["member"],
            self.params["interpolation_q_method"],
        )
        q_1 = rhi_1 / rhi_over_q

        return q_1, rhi_1


@dataclasses.dataclass
class HistogramMatchingWithEckelParams(models.ModelParams):
    """Parameters for :class:`HistogramMatchingWithEckel`.

    .. warning::
        Experimental. This may change or be removed in a future release.
    """

    #: A length-10 list of ERA5 ensemble members.
    #: Each element is a :class:`MetDataArray` holding specific humidity
    #: values for a single ensemble member. If None, a ValueError will be
    #: raised at model instantiation time. The order of the list must be
    #: consistent with the order of the ERA5 ensemble members.
    ensemble_specific_humidity: list[MetDataArray] | None = None

    #: The specific member used. Must be in the range [0, 10). If None,
    #: a ValueError will be raised at model instantiation time.
    member: int | None = None

    #: If a log transform has already been applied to each member of
    #: ``ensemble_specific_humidity``, set this to True.
    log_applied: bool = False


class HistogramMatchingWithEckel(HumidityScaling):
    """Scale humidity by histogram matching to IAGOS RHi quantiles.

    This method also applies the Eckel scaling to the recalibrated RHi values.

    Unlike other specific humidity scaling methods, this method requires met data
    and performs interpolation at evaluation time.

    .. warning::
        Experimental. This may change or be removed in a future release.

    References
    ----------
    :cite:`eckelCalibratedProbabilisticQuantitative1998`
    """

    name = "histogram_matching_with_eckel"
    long_name = "IAGOS RHi histogram matching with Eckel scaling"
    formula = "era5_quantiles -> iagos_quantiles -> recalibrated_rhi"
    default_params = HistogramMatchingWithEckelParams

    n_members = 10  # hard-coded elsewhere

    def _validate_params(self) -> None:
        member = self.params["member"]
        assert member in range(self.n_members)

        ensemble_specific_humidity = self.params["ensemble_specific_humidity"]
        assert len(ensemble_specific_humidity) == self.n_members
        for member, mda in enumerate(ensemble_specific_humidity):
            try:
                assert mda.data["number"] == member
            except KeyError:
                pass

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset:
        ...

    @overload
    def eval(self, source: MetDataset, **params: Any) -> NoReturn:
        ...

    @overload
    def eval(self, source: None = ..., **params: Any) -> NoReturn:
        ...

    def eval(
        self, source: GeoVectorDataset | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | MetDataset:
        """Scale specific humidity by histogram matching to IAGOS RHi quantiles.

        This method assumes ``source`` is equipped with the following variables:

        - air_temperature
        - specific_humidity: Humidity values for the ``params["member"]`` ERA5 ensemble member.
        """

        self.update_params(params)
        self._validate_params()
        self.set_source(source)
        self.source = self.require_source_type(GeoVectorDataset)

        if "rhi" in self.source:
            warnings.warn(
                "Variable 'rhi' already found on source to be scaled. This "
                "is unexpected and may be the result of humidity scaling "
                "being applied more than once."
            )

        # Create a 2D array of specific humidity values for all ensemble members
        # The specific humidity values for the current member are taken from the source
        # This matches patterns used in other humidity scaling methods
        # The remaining values are interpolated from the ERA5 ensemble members
        q = self.source.data["specific_humidity"]
        q2d = np.empty((len(self.source), self.n_members), dtype=q.dtype)

        interp_kwargs = self.interp_kwargs
        q_method = interp_kwargs.pop("q_method")

        ensemble_specific_humidity: list[MetDataArray] = self.params["ensemble_specific_humidity"]
        member: int = self.params["member"]
        log_applied: bool = self.params["log_applied"]

        for i, mda in enumerate(ensemble_specific_humidity):
            if i == member:
                q2d[:, i] = q
                continue

            q2d[:, i] = models.interpolate_gridded_specific_humidity(
                mda, self.source, q_method, log_applied, **interp_kwargs
            )

        p = self.source.setdefault("air_pressure", self.source.air_pressure)
        T = self.source.data["air_temperature"]

        q, rhi = self.scale(q2d, T, p)
        self.source.update(specific_humidity=q, rhi=rhi)

        return self.source

    @overrides
    def scale(  # type: ignore[override]
        self,
        specific_humidity: npt.NDArray[np.float_],
        air_temperature: npt.NDArray[np.float_],
        air_pressure: npt.NDArray[np.float_],
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Scale specific humidity values via histogram matching and Eckel scaling.

        Unlike the method on the base class, the method assumes each of the input
        arrays are :class:`np.ndarray` and not :class:`xr.DataArray` objects.

        Parameters
        ----------
        specific_humidity : npt.NDArray[np.float_]
            A 2D array of specific humidity values for all ERA5 ensemble members.
            The shape of this array must be ``(n, 10)``, where ``n`` is the number
            of observations and ``10`` is the number of ERA5 ensemble members.
        air_temperature : npt.NDArray[np.float_]
            A 1D array of air temperature values with shape ``(n,)``.
        air_pressure : npt.NDArray[np.float_]
            A 1D array of air pressure values with shape ``(n,)``.
        kwargs: Any
            Unused, kept for compatibility with the base class.

        Returns
        -------
        specific_humidity : npt.NDArray[np.float_]
            The recalibrated specific humidity values. A 1D array with shape ``(n,)``.
        rhi : npt.NDArray[np.float_]
            The recalibrated RHi values. A 1D array with shape ``(n,)``.
        """

        rhi_over_q = _rhi_over_q(air_temperature, air_pressure)
        rhi = rhi_over_q[:, np.newaxis] * specific_humidity

        q_method = self.params["interpolation_q_method"]

        ensemble_mean_rhi, ensemble_member_rhi = histogram_matching_all_members(
            rhi, self.params["member"], q_method
        )
        rhi_1 = eckel_scaling(ensemble_mean_rhi, ensemble_member_rhi, q_method)

        q_1 = rhi_1 / rhi_over_q

        return q_1, rhi_1
