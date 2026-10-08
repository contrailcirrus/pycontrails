"""Unified Parametric Contrail Model (UPCOM)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, overload

import numpy as np
import xarray as xr

import pycontrails
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import AirTemperature, MetVariable, SpecificHumidity
from pycontrails.core.models import Model, ModelParams
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.physics import thermo
from pycontrails.utils.types import ArrayLike


@dataclass
class UPCOMParams(ModelParams):
    """UPCOM model parameters for contrail formation analysis."""

    # Ice supersaturation threshold
    rhi_threshold: float = 1.0

    # Schmidt-Appleman contrail formation parameters
    # Based on Schumann (1996) and Ponater et al. (2002)
    ei_h2o: float = 1.23  # Water vapor emission index [kg H2O / kg fuel]
    Q: float = 43.13e6  # Specific combustion heat [J/kg]
    eta: float = 0.3  # Propulsion efficiency (dimensionless)

    # Humidity scaling
    humidity_scaling: HumidityScaling | None = None

    # --- Operational mode ---

    #: If True, use only the hard-coded physical constants defined below.
    #: No import from pycontrails.physics.constants will occur.
    #: If False (default), constants are loaded from pycontrails.physics.constants
    #: on initialisation and the hard-coded defaults serve as documented fallback values.
    standalone: bool = False

    # --- Physical constants ---
    # Hard-coded defaults are numerically identical to pycontrails.physics.constants values.
    # When standalone=False (default), __post_init__ overwrites these with the pycontrails values.

    #: Absolute zero temperature [C].
    #: Used to convert Celsius to Kelvin: T_K = T_C - tzeroC.
    #: pycontrails source: constants.absolute_zero = -273.15
    tzeroC: float = -273.15

    #: Isobaric heat capacity of dry air [J kg-1 K-1].
    #: pycontrails source: constants.c_pd = 1004.0
    cp_air: float = 1004.0

    #: Ratio of gas constants for dry air to water vapour [dimensionless].
    #: epsilon = R_d / R_v = 287.05 / 461.51.
    #: pycontrails source: constants.epsilon = R_d / R_v
    epsilon: float = 287.05 / 461.51

    def __post_init__(self) -> None:
        """Load physical constants from pycontrails when not running in standalone mode.

        In standalone mode (``standalone=True``), the hard-coded field defaults above
        are used unchanged.  In the default mode (``standalone=False``), the values are
        overwritten from ``pycontrails.physics.constants`` so that UPCOM always stays
        in sync with the rest of the pycontrails library.
        """
        if not self.standalone:
            from pycontrails.physics import constants as _pc_constants  # noqa: PLC0415

            # Absolute zero [C]; equivalent hard-coded value: -273.15
            self.tzeroC = _pc_constants.absolute_zero
            # Isobaric heat capacity of dry air [J kg-1 K-1]; equivalent: 1004.0
            self.cp_air = _pc_constants.c_pd
            # Ratio R_d / R_v [dimensionless]; equivalent: 287.05 / 461.51
            self.epsilon = _pc_constants.epsilon


def sac(
    air_temperature: ArrayLike,
    rh_liquid: ArrayLike,
    air_pressure: ArrayLike,
    params: Mapping[str, Any] | UPCOMParams | None = None,
    *,
    return_diagnostics: bool = False,
) -> ArrayLike | tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Calculate the Schmidt-Appleman contrail formation criterion.

    Parameters
    ----------
    air_temperature : ArrayLike
        Air temperature, [K].
    rh_liquid : ArrayLike
        Relative humidity over liquid water.
    air_pressure : ArrayLike
        Air pressure, [Pa].
    params : Mapping[str, Any] | UPCOMParams, optional
        UPCOM parameters used to calculate the mixing-line slope and thresholds.
        Defaults to :class:`UPCOMParams`.
    return_diagnostics : bool, optional
        If True, also return ``G``, ``T_contr``, and ``RH_contr`` after the SAC flag.

    Returns
    -------
    ArrayLike | tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
        A 0/1 SAC flag, optionally followed by ``G``, ``T_contr``, and ``RH_contr``.
    """
    if params is None:
        params = UPCOMParams().as_dict()
    elif isinstance(params, UPCOMParams):
        params = params.as_dict()

    if hasattr(air_pressure, "broadcast_like"):
        air_pressure = air_pressure.broadcast_like(air_temperature)

    G = (params["ei_h2o"] * params["cp_air"] * air_pressure) / (
        params["epsilon"] * params["Q"] * (1.0 - params["eta"])
    )
    T_contr, RH_contr = calculate_contrail_temperature_and_rh(
        air_temperature,
        air_pressure,
        G,
        tzeroC=params["tzeroC"],
    )
    sac_flag = ((air_temperature <= T_contr) & (rh_liquid > RH_contr)).astype(
        air_temperature.dtype
    )

    if return_diagnostics:
        return sac_flag, G, T_contr, RH_contr
    return sac_flag


class UPCOM(Model):
    """Unified Parametric Contrail Model - Grid Analysis.

    This model identifies regions favorable for persistent contrail formation
    using the Schmidt-Appleman criterion (Schumann 1996, Ponater et al. 2002).

    The model calculates:
    - Ice supersaturated regions (ISSR) where RHi > threshold
    - Critical temperature and humidity thresholds for contrail formation
    - Regions where conditions favor persistent contrails

    Parameters
    ----------
    met : MetDataset
        Dataset containing "air_temperature" and "specific_humidity" variables

    Examples
    --------
    >>> from datetime import datetime
    >>> from pycontrails.datalib.ecmwf import ERA5
    >>> from pycontrails.models.upcom import UPCOM
    >>> from pycontrails.models.humidity_scaling import ConstantHumidityScaling

    >>> # Get met data
    >>> time = datetime(2022, 3, 1, 0), datetime(2022, 3, 1, 2)
    >>> variables = ["air_temperature", "specific_humidity"]
    >>> pressure_levels = [200, 250, 300]
    >>> era5 = ERA5(time, variables, pressure_levels)
    >>> met = era5.open_metdataset()

    >>> # Instantiate and run model
    >>> scaling = ConstantHumidityScaling(rhi_adj=0.98)
    >>> model = UPCOM(met, humidity_scaling=scaling)
    >>> out = model.eval()
    >>> persistent = out["potential_persistent_contrail"]

    References
    ----------
    - Schumann, U. (1996). On conditions for contrail formation from aircraft exhausts.
      Meteorologische Zeitschrift, 5(1), 4-23.
    - Ponater, M., Marquart, S., & Sausen, R. (2002). Contrails in a comprehensive
      global climate model: Parameterization and radiative forcing results.
      Journal of Geophysical Research, 107(D13), ACL 2-1.
    """

    name = "upcom"
    long_name = "Unified Parametric Contrail Model"
    met_variables: tuple[MetVariable, ...] = AirTemperature, SpecificHumidity
    default_params = UPCOMParams

    @overload
    def eval(self, source: Flight, **params: Any) -> Flight: ...

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset: ...

    @overload
    def eval(self, source: MetDataset | None = ..., **params: Any) -> MetDataset: ...

    def eval(
        self, source: GeoVectorDataset | Flight | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | Flight | MetDataset:
        """Evaluate contrail formation conditions along trajectory or on meteorology grid.

        Parameters
        ----------
        source : GeoVectorDataset | Flight | MetDataset | None, optional
            Input GeoVectorDataset or Flight.
            If None, evaluates at the :attr:`met` grid points.
        **params : Any
            Overwrite model parameters before eval

        Returns
        -------
        GeoVectorDataset | Flight | MetDataset
            Returns source with additional data variables:
            - ``rhi``: Relative humidity over ice
            - ``rh_liquid``: Relative humidity over liquid water
            - ``issr``: Ice supersaturated regions (1 where RHi > threshold, 0 elsewhere)
            - ``sac``: Schmidt-Appleman criterion (1 where both SAC thresholds are met)
            - ``G``: Schmidt-Appleman G parameter
            - ``T_contr``: Critical contrail formation temperature [K]
            - ``RH_contr``: Critical relative humidity threshold
            - ``potential_persistent_contrail``: Regions favorable for persistent contrails

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

        humidity_scaling = self.params["humidity_scaling"]
        scale_humidity = humidity_scaling is not None and "specific_humidity" not in self.source

        self.set_source_met()

        # Apply humidity scaling, warn if no scaling is provided for ECMWF data
        if scale_humidity:
            humidity_scaling.eval(self.source, copy_source=False)

        # Extract variables
        air_temperature = self.source.data["air_temperature"]
        specific_humidity = self.source.data["specific_humidity"]
        air_pressure = self.source.data["air_pressure"]
        
        # Broadcast air_pressure to match temperature dimensions
        # xarray arithmetic will handle broadcasting automatically
        # No need to manually broadcast - just use the coordinates as-is

        # Calculate relative humidity over ice
        rhi = thermo.rhi(specific_humidity, air_temperature, air_pressure)

        # Calculate relative humidity over liquid water
        rh_liquid = thermo.rh(specific_humidity, air_temperature, air_pressure)

        # Create ISSR mask
        issr = (rhi > self.params["rhi_threshold"]).astype(rhi.dtype)

        # Calculate the SAC flag and expose its diagnostics in the model output.
        sac_flag, G, T_contr, RH_contr = sac(
            air_temperature,
            rh_liquid,
            air_pressure,
            self.params,
            return_diagnostics=True,
        )

        # Potential persistence requires both ice supersaturation and SAC.
        potential_persistent_contrail = ((issr == 1) & (sac_flag == 1)).astype(rhi.dtype)

        # Update source with calculated fields and set proper attributes
        self.source.data["rhi"] = rhi
        self.source.data["rhi"].attrs = {
            "long_name": "Relative humidity over ice",
            "units": "dimensionless",
        }
        
        self.source.data["rh_liquid"] = rh_liquid
        self.source.data["rh_liquid"].attrs = {
            "long_name": "Relative humidity over liquid water",
            "units": "dimensionless",
        }
        
        self.source.data["issr"] = issr
        self.source.data["issr"].attrs = {
            "long_name": "Ice supersaturated region",
            "units": "dimensionless",
            "description": "1 where RHi > threshold, 0 elsewhere",
        }

        self.source.data["sac"] = sac_flag
        self.source.data["sac"].attrs = {
            "long_name": "Schmidt-Appleman criterion",
            "units": "dimensionless",
            "description": "1 where air_temperature < T_contr and rh_liquid > RH_contr",
        }
        
        self.source.data["G"] = G
        self.source.data["G"].attrs = {
            "long_name": "Schmidt-Appleman G parameter",
            "units": "dimensionless",
            "description": "Slope of mixing line in T-q diagram",
        }
        
        self.source.data["T_contr"] = T_contr
        self.source.data["T_contr"].attrs = {
            "long_name": "Critical contrail formation temperature",
            "standard_name": "contrail_formation_temperature",
            "units": "K",
            "description": "Temperature threshold for contrail formation (Schmidt-Appleman criterion)",
        }
        
        self.source.data["RH_contr"] = RH_contr
        self.source.data["RH_contr"].attrs = {
            "long_name": "Critical relative humidity threshold",
            "units": "dimensionless",
            "description": "RH over liquid water threshold for contrail formation",
        }
        
        self.source.data["potential_persistent_contrail"] = potential_persistent_contrail
        self.source.data["potential_persistent_contrail"].attrs = {
            "long_name": "Potential persistent contrail regions",
            "units": "dimensionless",
            "description": "1 where conditions favor persistent contrails, 0 elsewhere",
        }

        # Tag output with additional metadata attrs
        self.transfer_met_source_attrs()
        self.source.attrs["pycontrails_version"] = pycontrails.__version__
        self.source.attrs["upcom_ei_h2o"] = self.params["ei_h2o"]
        self.source.attrs["upcom_Q"] = self.params["Q"]
        self.source.attrs["upcom_eta"] = self.params["eta"]
        self.source.attrs["upcom_rhi_threshold"] = self.params["rhi_threshold"]

        if scale_humidity:
            for k, v in humidity_scaling.description.items():
                self.source.attrs[f"humidity_scaling_{k}"] = v

        return self.source


def calculate_contrail_temperature_and_rh(
    air_temperature: ArrayLike,
    air_pressure: ArrayLike,
    G: ArrayLike,
    tzeroC: float = -273.15,
) -> tuple[ArrayLike, ArrayLike]:
    """Calculate critical temperature and RH thresholds from G parameter.

    Based on the work of Schumann (1996) and Ponater et al. (2002).

    Parameters
    ----------
    air_temperature : ArrayLike
        Air temperature, [:math:`K`]
    air_pressure : ArrayLike
        Air pressure, [:math:`Pa`]
    G : ArrayLike
        Schmidt-Appleman G parameter (dimensionless), already broadcast to grid
    tzeroC : float, optional
        Absolute zero temperature [:math:`C`], used to convert Celsius to Kelvin
        via ``T_K = T_C - tzeroC``.  Defaults to ``-273.15``, matching
        ``pycontrails.physics.constants.absolute_zero``.

    Returns
    -------
    T_contr : ArrayLike
        Critical contrail formation temperature, [:math:`K`].
        NaN where G <= 0.053 (conditions not suitable for contrails)
    RH_contr : ArrayLike
        Critical relative humidity threshold (dimensionless, [0-1])

    References
    ----------
    - Schumann, U. (1996). On conditions for contrail formation from aircraft exhausts.
      Meteorologische Zeitschrift, 5(1), 4-23.
    - Ponater, M., Marquart, S., & Sausen, R. (2002). Contrails in a comprehensive
      global climate model: Parameterization and radiative forcing results.
      Journal of Geophysical Research, 107(D13), ACL 2-1.
    """

    # Critical contrail temperature (Eq. 6, Ponater et al. 2002)
    # Only valid when G > 0.053
    # Use xr.where to preserve xarray structure
    mask = G > 0.053
    
    # Calculate temperature threshold in Celsius where valid
    # xr.where preserves DataArray structure while np.where strips it
    log_term = xr.where(mask, np.log(G - 0.053), np.nan)
    T_contr_C = -46.46 + 9.43 * log_term + 0.72 * log_term**2

    # Convert Celsius to Kelvin: T_K = T_C - tzeroC  (tzeroC = -273.15 [C])
    T_contr = xr.where(mask, T_contr_C - tzeroC, np.nan)

    # Critical RH over liquid water
    # Use xarray arithmetic which handles broadcasting
#    esat_l = thermo.e_sat_liquid(air_temperature)
#    RH_contr = (G * (air_temperature - T_contr) + esat_l) / esat_l

    esat_l = thermo.e_sat_liquid(air_temperature)
    esat_contr = thermo.e_sat_liquid(T_contr)
    RH_contr = (G * (air_temperature - T_contr) + esat_contr) / esat_l

    # Clip RH_contr to [0, 1] - use xr.where to preserve structure
    RH_contr = xr.where(RH_contr < 0, 0.0, RH_contr)
    RH_contr = xr.where(RH_contr > 1, 1.0, RH_contr)

    return T_contr, RH_contr
