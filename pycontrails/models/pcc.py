"""Probability of persistent contrail coverage (PCC)."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import xarray as xr

from pycontrails.core.fuel import Fuel, JetA
from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.core.met_var import AirTemperature, SpecificHumidity
from pycontrails.core.models import Model, ModelParams
from pycontrails.datalib.ecmwf.variables import SpecificCloudIceWaterContent
from pycontrails.models import sac
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.physics import thermo


@dataclasses.dataclass
class PCCParams(ModelParams):
    """PCC Model Parameters."""

    #: Cloud model
    #: Options include "Smith1990", "Sundqvist1989", "Slingo1980"
    cloud_model: str = "Smith1990"

    #: Critical RH Factor for the model to cirrus clouds
    rh_crit_factor: float = 0.7

    #: Fuel type
    fuel: Fuel = dataclasses.field(default_factory=JetA)

    #: Engine efficiency
    engine_efficiency: float = 0.35

    #: Humidity scaling
    humidity_scaling: HumidityScaling | None = None


class PCC(Model):
    r"""Potential Contrail Coverage Algorithm.

    Determines the potential of ambient atmosphere to allow contrail formation at grid points.

    Parameters
    ----------
    met : MetDataset
        Dataset containing  :attr:`met_variables` variables.
    surface : MetDataset
        Surface level dataset containing "air_pressure".
    params : dict[str, Any], optional
        Override PCC model parameters with dictionary.
        See :class:`PCCParams` for model parameters.
    **params_kwargs
        Override PCC model parameters with keyword arguments.
        See :class:`PCCParams` for model parameters.

    Notes
    -----
    Based on Ponater et al. (2002)
    """

    __slots__ = ("_cloud_model", "surface")
    name = "pcc"
    long_name = "Potential contrail coverage"
    met_variables = (AirTemperature, SpecificHumidity, SpecificCloudIceWaterContent)
    _cloud_model: Any
    source: MetDataset
    surface: MetDataset
    default_params = PCCParams

    def __init__(
        self,
        met: MetDataset,
        surface: MetDataset,
        params: dict[str, Any] = {},
        **params_kwargs: Any,
    ) -> None:
        super().__init__(met, params=params, **params_kwargs)

        # set cloud model by method
        self._cloud_model = getattr(self, self.params["cloud_model"], None)
        if self._cloud_model is None:
            raise ValueError(
                "Cloud model must be one of 'Smith1990', 'Sundqvist1989', 'Slingo1980'"
            )

        # make sure surface level dataset is valid
        if not isinstance(surface, MetDataset):
            raise ValueError("Surface air pressure required as input parameter `surface`")

        self.surface = surface.copy()
        self.surface.ensure_vars(["surface_air_pressure"])

    def eval(self, source: MetDataset | None = None, **params: Any) -> MetDataArray:
        """Evaluate PCC model.

        Currently only implemented to work on the :attr:`met` data input.

        Parameters
        ----------
        source : MetDataset | None, optional
           Input MetDataset.
            If None, evaluates at the :attr:`met` grid points.
        **params : Any
            Overwrite model parameters before eval


        Returns
        -------
        MetDataArray
            PCC model output
        """
        self.update_params(params)
        self.set_source(source)

        # apply humidity scaling
        scale_humidity = (self.params["humidity_scaling"] is not None) and (
            "specific_humidity" not in self.source
        )
        self.set_source_met()

        if scale_humidity:
            self.params["humidity_scaling"].eval(self.source, copy_source=False)

        return MetDataArray(self.b_contr(), name="pcc")

    def b_contr(self) -> xr.DataArray:
        """Calculate critical relative humidity threshold of contrail formation.

        Returns
        -------
        xr.DataArray
            Critical relative humidity of contrail formation, [:math:`[0 - 1]`]

        Notes
        -----
        Instead of using a prescribed threshold relative humidity for ``rh_crit_old``
        the threshold relative humidity now change with pressure.

        This equation is described in Roeckner et al. 1996, Eq.57
        THE ATMOSPHERIC GENERAL CIRCULATION MODEL ECHAM-4: MODEL DESCRIPTION AND
        SIMULATION OF PRESENT-DAY CLIMATE
        """
        sp = self.surface["surface_air_pressure"].data.loc[dict(level=-1)]  # surface air pressure

        def _apply_b_contr_plev(_ds: xr.Dataset) -> xr.Dataset:
            # cannot instantiate SAC object unless all 4 x, y, z, t dimensions are present
            # convert "level" coordinate to dimension
            _ds = _ds.expand_dims(dim="level")

            p = _ds["air_pressure"]

            G = sac.slope_mixing_line(
                _ds["specific_humidity"],
                _ds["air_pressure"],
                self.params["engine_efficiency"],
                self.params["fuel"].ei_h2o,
                self.params["fuel"].q_fuel,
            )
            T_sat_liquid = sac.T_sat_liquid(G)
            rh_crit_sac = sac.rh_critical_sac(_ds["air_temperature"], T_sat_liquid, G)

            # rh_crit_old = np.ones(p.shape) * self.rh_crit_factor
            rh_crit_old = 0.99 + (0.6 - 0.99) * np.exp(1 - (sp / p) ** 4)
            rh_crit_new = rh_crit_sac * rh_crit_old

            b_crit = self._cloud_model(
                _ds["air_temperature"],
                _ds["air_pressure"],
                _ds["specific_cloud_ice_water_content"],
                _ds["specific_humidity"],
                rh_crit_old,
                rh_crit_old,
            )
            b_crit_contr = self._cloud_model(
                _ds["air_temperature"],
                _ds["air_pressure"],
                _ds["specific_cloud_ice_water_content"],
                _ds["specific_humidity"],
                rh_crit_old,
                rh_crit_new,
            )
            b_crit_potential = b_crit_contr - b_crit

            b_crit_potential = xr.where((b_crit_potential > 1), 1, b_crit_potential)
            b_crit_potential = xr.where((b_crit_potential < 0), 0, b_crit_potential)

            # issue recombining groups arises if "level" is in dims
            # convert "level" dimension to coordinate
            b_crit_potential = b_crit_potential.squeeze("level")
            return b_crit_potential

        # apply calculation per pressure level
        return (
            self.source.data.groupby("level")  # type: ignore
            .map(_apply_b_contr_plev)
            .transpose(*self.source.dim_order)
        )

    def Smith1990(
        self,
        T: xr.DataArray,
        p: xr.DataArray,
        iwc: xr.DataArray,
        q: xr.DataArray,
        rh_crit_old: xr.DataArray,
        rh_crit_new: xr.DataArray,
    ) -> xr.DataArray:
        r"""Apply Smith Scheme described in Rap et al. (2009).

        Parameterization of contrails in the UK Met OfficeClimate Model;

        Parameters
        ----------
        T : :class:`xarray:DataArray`
            Air Temperature, [:math:`K`]
        p : :class:`xarray:DataArray`
            Air Pressure, [:math:`Pa`]
        iwc : :class:`xarray:DataArray`
            Cloud ice water content, [:math:`kg \ kg^{-1}`]
        q : :class:`xarray:DataArray`
            Specific humidity
        rh_crit_old : :class:`xarray:DataArray`
            Critical relative humidity, [:math:`[0 - 1]`]
        rh_crit_new : :class:`xarray:DataArray`
            Critical relative humidity, [:math:`[0 - 1]`]

        Returns
        -------
        :class:`xarray:DataArray`
            Probability of cirrus formation, [:math:`[0 - 1]`]
        """
        r = thermo.rh(q, T, p)
        q_sw = thermo.q_sat(T, p)
        b_crit = (1 - rh_crit_old) * q_sw
        Q_n = iwc / b_crit - (1 - r) / (1 - rh_crit_new)

        b_cirrus = xr.DataArray(np.zeros(Q_n.shape), coords=Q_n.coords)
        b_cirrus = xr.where((Q_n <= -1), 0, b_cirrus)
        b_cirrus = xr.where(
            (Q_n > -1) & (Q_n <= 0), 0.5 * (1 + Q_n.where((Q_n > -1) & (Q_n <= 0))) ** 2, b_cirrus
        )
        b_cirrus = xr.where(
            (Q_n > 0) & (Q_n <= 1), 1 - 0.5 * (1 - Q_n.where((Q_n > 0) & (Q_n <= 1))) ** 2, b_cirrus
        )
        return xr.where((Q_n > 1), 1, b_cirrus)

    def Slingo1980(
        self,
        T: xr.DataArray,
        p: xr.DataArray,
        iwc: xr.DataArray,
        q: xr.DataArray,
        rh_crit_old: xr.DataArray,
        rh_crit_new: xr.DataArray,
    ) -> xr.DataArray:
        r"""Apply Slingo scheme described in Wood and Field, 1999.

        Relationships between Total Water, Condensed Water, and Cloud Fraction in
        Stratiform Clouds Examined Using Aircraft Data

        Parameters
        ----------
        T : :class:`xarray:DataArray`
            Air Temperature, [:math:`K`]
        p : :class:`xarray:DataArray`
            Air Pressure, [:math:`Pa`]
        iwc : :class:`xarray:DataArray`
            Cloud ice water content, [:math:`kg \ kg^{-1}`]
        q : :class:`xarray:DataArray`
            Specific humidity
        rh_crit_old : :class:`xarray:DataArray`
            Critical relative humidity, [:math:`[0 - 1]`]
        rh_crit_new : :class:`xarray:DataArray`
            Critical relative humidity, [:math:`[0 - 1]`]

        Returns
        -------
        :class:`xarray:DataArray`
            Probability of cirrus formation, [:math:`[0 - 1]`]
        """
        r = thermo.rh(q, T, p)
        b_cirrus = ((r - rh_crit_new) / (1 - rh_crit_new)) ** 2
        b_cirrus = xr.where(r < rh_crit_new, 0, b_cirrus)
        return xr.where(r >= 1.0, 1.0, b_cirrus)

    def Sundqvist1989(
        self,
        T: xr.DataArray,
        p: xr.DataArray,
        iwc: xr.DataArray,
        q: xr.DataArray,
        rh_crit_old: xr.DataArray,
        rh_crit_new: xr.DataArray,
    ) -> xr.DataArray:
        r"""Apply Sundqvist scheme described in Ponater et al. (2002).

        Contrails in a comprehensive global climate model: Parameterization and radiative
        forcing results

        Parameters
        ----------
        T : :class:`xarray:DataArray`
            Air Temperature, [:math:`K`]
        p : :class:`xarray:DataArray`
            Air Pressure, [:math:`Pa`]
        iwc : :class:`xarray:DataArray`
            Cloud ice water content, [:math:`kg \ kg^{-1}`]
        q : :class:`xarray:DataArray`
            Specific humidity
        rh_crit_old : :class:`xarray:DataArray`
            Critical relative humidity, [:math:`[0 - 1]`]
        rh_crit_new : :class:`xarray:DataArray`
            Critical relative humidity, [:math:`[0 - 1]`]

        Returns
        -------
        :class:`xarray:DataArray`
            Probability of cirrus formation, [:math:`[0 - 1]`]
        """
        r = thermo.rh(q, T, p)

        # clipping ratio at 1 to prevent np.sqrt from taking negative arguments
        ratio = (r - rh_crit_new) / (1 - rh_crit_new)
        ratio = xr.where(ratio > 1, 1, ratio)

        b_cirrus = 1 - (1 - ratio) ** 0.5
        return xr.where(r < rh_crit_new, 0, b_cirrus)
