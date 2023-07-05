"""Calculate jet engine emissions using the ICAO Aircraft Emissions Databank (EDB).

Functions without a subscript "_" can be used independently outside .eval()
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
import warnings
from typing import Any, NoReturn, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pycontrails.core import flight
from pycontrails.core.flight import Flight
from pycontrails.core.fuel import SAFBlend
from pycontrails.core.interpolation import EmissionsProfileInterpolator
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import AirTemperature, SpecificHumidity
from pycontrails.core.models import Model, ModelParams
from pycontrails.models.emissions import black_carbon, ffm2
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.physics import constants, jet, units

_path_to_static = pathlib.Path(__file__).parent / "static"


@dataclasses.dataclass
class EmissionsParams(ModelParams):
    """:class:`Emissions` model parameters."""

    #: Default paths
    edb_engine_path: str | pathlib.Path = _path_to_static / "edb-gaseous-v28c-engines.csv"
    edb_nvpm_path: str | pathlib.Path = _path_to_static / "edb-nvpm-v28c-engines.csv"
    engine_uid_path: str | pathlib.Path = _path_to_static / "default-engine-uids.csv"

    #: Default nvpm_ei_n value if engine UID is not found
    default_nvpm_ei_n: float = 1e15

    #: Humidity scaling. If None, no scaling is applied.
    humidity_scaling: HumidityScaling | None = None

    #: If True, if an engine UID is not provided on the ``source.attrs``, use a
    #: default engine UID based on Teoh's analysis of aircraft engine pairs in
    #: 2019 - 2021 Spire data.
    use_default_engine_uid: bool = True


class Emissions(Model):
    """Emissions handling using ICAO Emissions Databank (EDB) and black carbon correlations.

    Parameters
    ----------
    met : MetDataset | None, optional
        Met data, by default None.
    params : dict[str, Any] | None, optional
        Model parameters, by default None.
    params_kwargs : Any
        Model parameters passed as keyword arguments.

    References
    ----------
    - :cite:`leeContributionGlobalAviation2021`
    - :cite:`schumannDehydrationEffectsContrails2015`
    - :cite:`stettlerGlobalCivilAviation2013`
    - :cite:`wilkersonAnalysisEmissionData2010`

    See Also
    --------
    :mod:`pycontrails.models.emissions.black_carbon`
    :mod:`pycontrails.models.emissions.ffm2`
    """

    name = "emissions"
    long_name = "ICAO Emissions Databank (EDB)"
    met_variables = AirTemperature, SpecificHumidity
    default_params = EmissionsParams

    source: Flight

    def __init__(
        self,
        met: MetDataset | None = None,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ) -> None:
        super().__init__(met, params, **params_kwargs)

        self.edb_engine_gaseous = get_engine_params_from_edb(self.params["edb_engine_path"])
        self.edb_engine_nvpm = get_engine_nvpm_profile_from_edb(self.params["edb_nvpm_path"])
        self.default_engines = get_default_aircraft_engine_mapping(self.params["engine_uid_path"])

    @overload
    def eval(self, source: Flight, **params: Any) -> Flight:
        ...

    @overload
    def eval(self, source: None = None, **params: Any) -> NoReturn:
        ...

    def eval(self, source: Flight | None = None, **params: Any) -> Flight:
        """Calculate the emissions data for ``source``.

        Parameter ``source`` must contain each of the variables:
            - air_temperature
            - specific_humidity
            - true_airspeed
            - fuel_flow

        In addition, ``source.attrs`` should contain variables
            - engine_uid
            - n_engine

        If 'engine_uid' is not provided in the flight attribute or not available in the ICAO EDB,
        constant emission indices will be assumed for NOx, CO, HC, and nvPM mass and number.

        The computed pollutants include carbon dioxide (CO2), nitrogen oxide (NOx),
        carbon monoxide (CO), hydrocarbons (HC), non-volatile particulate matter
        (nvPM) mass and number, sulphur oxides (SOx), sulphates (S) and organic carbon (OC).

        Parameters
        ----------
        source : Flight
            Flight to evaluate
        **params : Any
            Overwrite model parameters before eval

        Returns
        -------
        Flight
            Flight with attached emissions data

        Raises
        ------
        AttributeError
            Raised if :attr:`fuel` on ``flight`` is incompatible with the model parameter "fuel".
        KeyError
            Raised if ``flight`` already contains "thrust_setting" or "nvpm_ei_n" variables.
        """
        self.update_params(params)
        self.set_source(source)
        self.source = self.require_source_type(Flight)

        # Set air_temperature and specific_humidity if not already set
        scale_humidity = (self.params["humidity_scaling"] is not None) and (
            "specific_humidity" not in self.source
        )
        self.set_source_met()

        # Only enhance humidity if it wasn't already present on source
        if scale_humidity:
            self.params["humidity_scaling"].eval(self.source, copy_source=False)

        # Ensure that flight has the required variables defined as attrs or columns
        # We could support calculating true_airspeed in the same way as done by BADAFlight
        # Right now, the pathway to using this model is by chaining with BADAFlight,
        # so we don't need to support it yet.
        self.source.ensure_vars(("true_airspeed", "fuel_flow"))

        engine_uid = self.source.attrs.get("engine_uid")
        if (
            engine_uid is None
            and self.params["use_default_engine_uid"]
            and (aircraft_type := self.source.attrs.get("aircraft_type"))
        ):
            try:
                engine_uid = self.default_engines.at[aircraft_type, "engine_uid"]
                n_engine = self.default_engines.at[aircraft_type, "n_engine"]
            except KeyError:
                pass
            else:
                self.source.attrs.setdefault("engine_uid", engine_uid)
                self.source.attrs.setdefault("n_engine", n_engine)

        if engine_uid is None:
            warnings.warn(
                "No 'engine_uid' found on source attrs. A constant emissions will be used."
            )

        try:
            fuel_flow_per_engine = self.source.get_data_or_attr("fuel_flow_per_engine")
        except KeyError:
            # Try to keep vector and attrs data consistent here
            n_engine = self.source.attrs["n_engine"]
            try:
                ff = self.source["fuel_flow"]
            except KeyError:
                ff = self.source.attrs["fuel_flow"]
                fuel_flow_per_engine = ff / n_engine
                self.source.attrs["fuel_flow_per_engine"] = fuel_flow_per_engine
            else:
                fuel_flow_per_engine = ff / n_engine
                self.source["fuel_flow_per_engine"] = fuel_flow_per_engine

        # Attach thrust setting
        if "thrust_setting" not in self.source:
            try:
                edb_gaseous = self.edb_engine_gaseous[engine_uid]  # type: ignore[index]
            except KeyError:
                self.source["thrust_setting"] = np.full(shape=len(self.source), fill_value=np.nan)
            else:
                self.source["thrust_setting"] = get_thrust_setting(
                    edb_gaseous,
                    fuel_flow_per_engine=fuel_flow_per_engine,
                    air_pressure=self.source.air_pressure,
                    air_temperature=self.source["air_temperature"],
                    true_airspeed=self.source.get_data_or_attr("true_airspeed"),
                )

        self._gaseous_emission_indices(engine_uid)
        self._nvpm_emission_indices(engine_uid)
        self._total_pollutant_emissions()
        return self.source

    def _gaseous_emission_indices(self, engine_uid: str | None) -> None:
        """Calculate EI's for nitrogen oxide (NOx), carbon monoxide (CO) and hydrocarbons (HC).

        This method attaches the following variables to the underlying :attr:`flight`:

        - `nox_ei`
        - `co_ei`
        - `hc_ei`

        Parameters
        ----------
        engine_uid : str
            Engine unique identification number from the ICAO EDB
        """
        try:
            edb_gaseous = self.edb_engine_gaseous[engine_uid]  # type: ignore[index]
        except KeyError:
            self._gaseous_emissions_constant()
        else:
            self._gaseous_emissions_ffm2(edb_gaseous)

    def _gaseous_emissions_ffm2(self, edb_gaseous: EDBGaseous) -> None:
        """Calculate gaseous emissions using the FFM2 methodology.

        This method attaches the following variables to the underlying :attr:`flight`:

        - `nox_ei`
        - `co_ei`
        - `hc_ei`

        Parameters
        ----------
        edb_gaseous : EDBGaseous
            EDB gaseous data
        """
        self.source.attrs["gaseous_data_source"] = "FFM2"

        fuel_flow_per_engine = self.source.get_data_or_attr("fuel_flow_per_engine")
        true_airspeed = self.source.get_data_or_attr("true_airspeed")
        air_temperature = self.source["air_temperature"]

        # Emissions indices
        self.source["nox_ei"] = nitrogen_oxide_emissions_index_ffm2(
            edb_gaseous,
            fuel_flow_per_engine,
            true_airspeed,
            self.source.air_pressure,
            air_temperature,
            self.source["specific_humidity"],
        )

        self.source["co_ei"] = carbon_monoxide_emissions_index_ffm2(
            edb_gaseous,
            fuel_flow_per_engine,
            true_airspeed,
            self.source.air_pressure,
            air_temperature,
        )

        self.source["hc_ei"] = hydrocarbon_emissions_index_ffm2(
            edb_gaseous,
            fuel_flow_per_engine,
            true_airspeed,
            self.source.air_pressure,
            air_temperature,
        )

    def _gaseous_emissions_constant(self) -> None:
        """Fill gaseous emissions data with default values.

        This method attaches the following variables to the underlying :attr:`flight`:

        - `nox_ei`
        - `co_ei`
        - `hc_ei`

        Assumes constant emission indices for nitrogen oxide, carbon monoxide and
        hydrocarbon for a given aircraft-engine pair if data is not available in the ICAO EDB.

        - NOx EI = 15.14 g-NOx/kg-fuel (Table 1 of Lee et al., 2020)
        - CO EI = 3.61 g-CO/kg-fuel (Table 1 of Wilkerson et al., 2010),
        - HC EI = 0.520 g-HC/kg-fuel (Table 1 of Wilkerson et al., 2010)

        References
        ----------
        - :cite:`leeContributionGlobalAviation2021`
        - :cite:`wilkersonAnalysisEmissionData2010`
        """
        self.source.attrs["gaseous_data_source"] = "Constant"

        nox_ei = np.full(shape=len(self.source), fill_value=15.14)
        co_ei = np.full(shape=len(self.source), fill_value=3.61)
        hc_ei = np.full(shape=len(self.source), fill_value=0.520)

        self.source["nox_ei"] = nox_ei * 1e-3  # g-NOx/kg-fuel to kg-NOx/kg-fuel
        self.source["co_ei"] = co_ei * 1e-3  # g-CO/kg-fuel to kg-CO/kg-fuel
        self.source["hc_ei"] = hc_ei * 1e-3  # g-HC/kg-fuel to kg-HC/kg-fuel

    def _nvpm_emission_indices(self, engine_uid: str | None) -> None:
        """Calculate emission indices for nvPM mass and number.

        This method attaches the following variables to the underlying :attr:`source`.
            - nvpm_ei_m
            - nvpm_ei_n

        In addition, ``nvpm_data_source`` is attached to the ``source.attrs``.

        Parameters
        ----------
        engine_uid : str
            Engine unique identification number from the ICAO EDB
        """
        if "nvpm_ei_n" in self.source and "nvpm_ei_m" in self.source:
            return  # early exit if values already exist

        if (edb_nvpm := self.edb_engine_nvpm.get(engine_uid)) is not None:  # type: ignore[arg-type]
            nvpm_data_source, nvpm_ei_m, nvpm_ei_n = self._nvpm_emission_indices_edb(edb_nvpm)
        elif (edb_gaseous := self.edb_engine_gaseous.get(engine_uid)) is not None:  # type: ignore[arg-type]  # noqa: E501
            nvpm_data_source, nvpm_ei_m, nvpm_ei_n = self._nvpm_emission_indices_sac(edb_gaseous)
        else:
            if engine_uid is not None:
                warnings.warn(
                    f"Cannot find 'engine_uid' {engine_uid} in EDB. "
                    "A constant emissions will be used."
                )
            nvpm_data_source, nvpm_ei_m, nvpm_ei_n = self._nvpm_emission_indices_constant()

        # Adjust nvPM emission indices if SAF is used.
        if isinstance(self.source.fuel, SAFBlend) and self.source.fuel.pct_blend:
            pct_eim_reduction = black_carbon.nvpm_mass_ei_pct_reduction_due_to_saf(
                self.source.fuel.hydrogen_content, self.source["thrust_setting"]
            )
            pct_ein_reduction = black_carbon.nvpm_number_ei_pct_reduction_due_to_saf(
                self.source.fuel.hydrogen_content, self.source["thrust_setting"]
            )

            nvpm_ei_m *= 1.0 + pct_eim_reduction / 100.0
            nvpm_ei_n *= 1.0 + pct_ein_reduction / 100.0

        self.source.attrs["nvpm_data_source"] = nvpm_data_source
        self.source.setdefault("nvpm_ei_m", nvpm_ei_m)
        self.source.setdefault("nvpm_ei_n", nvpm_ei_n)

    def _nvpm_emission_indices_edb(
        self, edb_nvpm: EDBnvpm
    ) -> tuple[str, npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Calculate emission indices for nvPM mass and number.

        This method uses data from the ICAO EDB along with the T4/T2 methodology.

        Parameters
        ----------
        edb_nvpm : EDBnvpm
            EDB nvPM data.

        Returns
        -------
        nvpm_data_source : str
            Source of nvpm data.
        nvpm_ei_m : npt.NDArray[np.float_]
            Non-volatile particulate matter (nvPM) mass emissions index, [:math:`kg/kg_{fuel}`]
        nvpm_ei_n : npt.NDArray[np.float_]
            Black carbon number emissions index, [:math:`kg_{fuel}^{-1}`]

        References
        ----------
        - :cite:`teohTargetedUseSustainable2022`
        """
        nvpm_data_source = "ICAO EDB"

        # Emissions indices
        return nvpm_data_source, *get_nvpm_emissions_index_edb(
            edb_nvpm,
            true_airspeed=self.source.get_data_or_attr("true_airspeed"),
            air_temperature=self.source["air_temperature"],
            air_pressure=self.source.air_pressure,
            thrust_setting=self.source["thrust_setting"],
            q_fuel=self.source.fuel.q_fuel,
        )

    def _nvpm_emission_indices_sac(
        self, edb_gaseous: EDBGaseous
    ) -> tuple[str, npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Calculate EIs for nvPM mass and number assuming the profile of single annular combustors.

        nvPM EI_m is calculated using the FOX and ImFOX methods, while the nvPM EI_n
        is calculated using the Fractal Aggregates (FA) model.

        Parameters
        ----------
        edb_gaseous : EDBGaseous
            EDB gaseous data

        Returns
        -------
        nvpm_data_source : str
            Source of nvpm data.
        nvpm_ei_m : npt.NDArray[np.float_]
            Non-volatile particulate matter (nvPM) mass emissions index, [:math:`kg/kg_{fuel}`]
        nvpm_ei_n : npt.NDArray[np.float_]
            Black carbon number emissions index, [:math:`kg_{fuel}^{-1}`]

        References
        ----------
        - :cite:`stettlerGlobalCivilAviation2013`
        - :cite:`abrahamsonPredictiveModelDevelopment2016`
        - :cite:`teohTargetedUseSustainable2022`
        """
        nvpm_data_source = "FA Model"

        # calculate properties
        thrust_setting = self.source["thrust_setting"]
        fuel_flow_per_engine = self.source.get_data_or_attr("fuel_flow_per_engine")
        true_airspeed = self.source.get_data_or_attr("true_airspeed")
        air_temperature = self.source["air_temperature"]

        # Emissions indices
        nvpm_ei_m = nvpm_mass_emissions_index_sac(
            edb_gaseous,
            air_pressure=self.source.air_pressure,
            true_airspeed=true_airspeed,
            air_temperature=air_temperature,
            thrust_setting=thrust_setting,
            fuel_flow_per_engine=fuel_flow_per_engine,
            hydrogen_content=self.source.fuel.hydrogen_content,
        )
        nvpm_gmd = nvpm_geometric_mean_diameter_sac(
            edb_gaseous,
            air_pressure=self.source.air_pressure,
            true_airspeed=true_airspeed,
            air_temperature=air_temperature,
            thrust_setting=thrust_setting,
            q_fuel=self.source.fuel.q_fuel,
        )
        nvpm_ei_n = black_carbon.number_emissions_index_fractal_aggregates(nvpm_ei_m, nvpm_gmd)
        return nvpm_data_source, nvpm_ei_m, nvpm_ei_n

    def _nvpm_emission_indices_constant(
        self,
    ) -> tuple[str, npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """
        Assume constant emission indices for nvPM mass and number.

        (nvpm_ei_n = 1e15 /kg-fuel) for a given aircraft-engine pair if data
        is not available in the ICAO EDB.

        - nvpm_ei_m = 0.088 g-nvPM/kg-fuel (Table 2 of Stettler et al., 2013)
        - nvpm_ei_n = 1e15 /kg-fuel (Schumann et al., 2015)

        Returns
        -------
        nvpm_data_source : str
            Source of nvpm data.
        nvpm_ei_m : npt.NDArray[np.float_]
            Non-volatile particulate matter (nvPM) mass emissions index, [:math:`kg/kg_{fuel}`]
        nvpm_ei_n : npt.NDArray[np.float_]
            Black carbon number emissions index, [:math:`kg_{fuel}^{-1}`]

        References
        ----------
        - :cite:`stettlerGlobalCivilAviation2013`
        - :cite:`wilkersonAnalysisEmissionData2010`
        - :cite:`schumannDehydrationEffectsContrails2015`
        """
        nvpm_data_source = "Constant"
        nvpm_ei_m = np.full(shape=len(self.source), fill_value=(0.088 * 1e-3))  # g to kg
        nvpm_ei_n = np.full(shape=len(self.source), fill_value=self.params["default_nvpm_ei_n"])
        return nvpm_data_source, nvpm_ei_m, nvpm_ei_n

    def _total_pollutant_emissions(self) -> None:
        # Required variables
        # FIXME: These two variables are already calculated in BADA models
        dt_sec = flight.segment_duration(self.source["time"], dtype=self.source.altitude_ft.dtype)
        fuel_burn = jet.fuel_burn(self.source.get_data_or_attr("fuel_flow"), dt_sec)

        # TODO: these currently overwrite values and will throw warnings

        # Total emissions for each waypoint
        self.source["co2"] = fuel_burn * self.source.fuel.ei_co2
        self.source["h2o"] = fuel_burn * self.source.fuel.ei_h2o
        self.source["so2"] = fuel_burn * self.source.fuel.ei_so2
        self.source["sulphates"] = fuel_burn * self.source.fuel.ei_sulphates
        self.source["oc"] = fuel_burn * self.source.fuel.ei_oc
        self.source["nox"] = fuel_burn * self.source["nox_ei"]
        self.source["co"] = fuel_burn * self.source["co_ei"]
        self.source["hc"] = fuel_burn * self.source["hc_ei"]
        self.source["nvpm_mass"] = fuel_burn * self.source["nvpm_ei_m"]
        self.source["nvpm_number"] = fuel_burn * self.source["nvpm_ei_n"]

        # Total emissions for the flight
        self.source.attrs["total_co2"] = np.nansum(self.source["co2"])
        self.source.attrs["total_h2o"] = np.nansum(self.source["h2o"])
        self.source.attrs["total_so2"] = np.nansum(self.source["so2"])
        self.source.attrs["total_sulphates"] = np.nansum(self.source["sulphates"])
        self.source.attrs["total_oc"] = np.nansum(self.source["oc"])
        self.source.attrs["total_nox"] = np.nansum(self.source["nox"])
        self.source.attrs["total_co"] = np.nansum(self.source["co"])
        self.source.attrs["total_hc"] = np.nansum(self.source["hc"])
        self.source.attrs["total_nvpm_mass"] = np.nansum(self.source["nvpm_mass"])
        self.source.attrs["total_nvpm_number"] = np.nansum(self.source["nvpm_number"])

    def _check_edb_gaseous_availability(
        self,
        engine_uid: str,
        raise_error: bool = True,
    ) -> bool:
        """
        Check if the provided engine is available in the gaseous ICAO EDB.

        Setting ``raise_error`` to True allows functions in this class to be
        used independently outside of :meth:`eval`.

        Parameters
        ----------
        engine_uid: str
            Engine unique identification number from the ICAO EDB
        raise_error: bool
            Raise a KeyError if engine type is not available.

        Returns
        -------
        bool
            True if engine type is available in the gaseous ICAO EDB.

        Raises
        ------
        KeyError
            If engine type is not available in the gaseous ICAO EDB.
        """
        if engine_uid not in self.edb_engine_gaseous:
            if raise_error:
                raise KeyError(
                    f"Engine ({engine_uid}) is not available in the ICAO EDB gaseous database"
                )
            return False
        return True

    def _check_edb_nvpm_availability(
        self,
        engine_uid: str,
        raise_error: bool = True,
    ) -> bool:
        """
        Check if the provided engine is available in the nvPM ICAO EDB.

        Setting ``raise_error`` to True allows functions in this class to be
        used independently outside of :meth:`eval`.

        Parameters
        ----------
        engine_uid: str
            Engine unique identification number from the ICAO EDB
        raise_error: bool
            Raise a KeyError if engine type is not available.

        Returns
        -------
        bool
            True if engine type is available in the nvPM ICAO EDB.

        Raises
        ------
        KeyError
            If engine type is not available in the nvPM ICAO EDB.
        """
        if engine_uid not in self.edb_engine_nvpm:
            if raise_error:
                raise KeyError(
                    f"Engine ({engine_uid}) is not available in the ICAO EDB nvPM database"
                )
            return False
        return True


def nitrogen_oxide_emissions_index_ffm2(
    edb_gaseous: EDBGaseous,
    fuel_flow_per_engine: npt.NDArray[np.float_],
    true_airspeed: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    specific_humidity: None | npt.NDArray[np.float_] = None,
) -> npt.NDArray[np.float_]:
    """
    Estimate the nitrogen oxide (NOx) emissions index (EI) using the Fuel Flow Method 2 (FFM2).

    Parameters
    ----------
    edb_gaseous : EDBGaseous
        EDB gaseous data
    fuel_flow_per_engine: npt.NDArray[np.float_]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    true_airspeed: npt.NDArray[np.float_]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_pressure : npt.NDArray[np.float_]
        pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature : npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]
    specific_humidity: npt.NDArray[np.float_]
        specific humidity for each waypoint, [:math:`kg_{H_{2}O}/kg_{air}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Nitrogen oxide emissions index for each waypoint, [:math:`kg_{NO_{X}}/kg_{fuel}`]
    """
    res_nox = ffm2.estimate_nox(
        edb_gaseous.log_ei_nox_profile,
        fuel_flow_per_engine,
        true_airspeed,
        air_pressure,
        air_temperature,
        specific_humidity,
    )
    return res_nox * 1e-3  # g-NOx/kg-fuel to kg-NOx/kg-fuel


def carbon_monoxide_emissions_index_ffm2(
    edb_gaseous: EDBGaseous,
    fuel_flow_per_engine: npt.NDArray[np.float_],
    true_airspeed: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Estimate the carbon monoxide (CO) emissions index (EI) using the Fuel Flow Method 2 (FFM2).

    Parameters
    ----------
    edb_gaseous : EDBGaseous
        EDB gaseous data
    fuel_flow_per_engine: npt.NDArray[np.float_]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    true_airspeed: npt.NDArray[np.float_]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_pressure : npt.NDArray[np.float_]
        pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature : npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]

    Returns
    -------
    npt.NDArray[np.float_]
        Carbon monoxide emissions index for each waypoint, [:math:`kg_{CO}/kg_{fuel}`]
    """
    res_co = ffm2.estimate_ei(
        edb_gaseous.log_ei_co_profile,
        fuel_flow_per_engine,
        true_airspeed,
        air_pressure,
        air_temperature,
    )
    return res_co * 1e-3  # g-CO/kg-fuel to kg-CO/kg-fuel


def hydrocarbon_emissions_index_ffm2(
    edb_gaseous: EDBGaseous,
    fuel_flow_per_engine: npt.NDArray[np.float_],
    true_airspeed: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Estimate the hydrocarbon (HC) emissions index (EI) using the Fuel Flow Method 2 (FFM2).

    Parameters
    ----------
    edb_gaseous : EDBGaseous
        EDB gaseous data
    fuel_flow_per_engine: npt.NDArray[np.float_]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    true_airspeed: npt.NDArray[np.float_]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_pressure : npt.NDArray[np.float_]
        pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature : npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]

    Returns
    -------
    npt.NDArray[np.float_]
        Hydrocarbon emissions index for each waypoint, [:math:`kg_{HC}/kg_{fuel}`]
    """
    res_hc = ffm2.estimate_ei(
        edb_gaseous.log_ei_hc_profile,
        fuel_flow_per_engine,
        true_airspeed,
        air_pressure,
        air_temperature,
    )
    return res_hc * 1e-3  # g-HC/kg-fuel to kg-HC/kg-fuel


def get_nvpm_emissions_index_edb(
    edb_nvpm: EDBnvpm,
    true_airspeed: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    thrust_setting: npt.NDArray[np.float_],
    q_fuel: float,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    r"""Calculate nvPM mass emissions index (nvpm_ei_m) and number emissions index (nvpm_ei_n).

    Interpolate the non-volatile particulate matter (nvPM) mass and number emissions index from
    the emissions profile of a given engine type that is provided by the ICAO EDB.

    The non-dimensional thrust setting (t4_t2) is clipped to the minimum and maximum t4_t2 values
    that is estimated from the four ICAO EDB datapoints to prevent extrapolating the nvPM values.

    Parameters
    ----------
    edb_nvpm : EDBnvpm
        EDB nvPM data
    true_airspeed: npt.NDArray[np.float_]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    fuel_flow_per_engine: npt.NDArray[np.float_]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    air_temperature: npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]
    air_pressure: npt.NDArray[np.float_]
        pressure altitude at each waypoint, [:math:`Pa`]
    thrust_setting : npt.NDArray[np.float_]
        thrust setting
    q_fuel : float
        Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].

    Returns
    -------
    nvpm_ei_m : npt.NDArray[np.float_]
        Non-volatile particulate matter (nvPM) mass emissions index, [:math:`kg/kg_{fuel}`]
    nvpm_ei_n : npt.NDArray[np.float_]
        Black carbon number emissions index, [:math:`kg_{fuel}^{-1}`]
    """
    # Non-dimensionalized thrust setting
    t4_t2 = jet.thrust_setting_nd(
        true_airspeed,
        thrust_setting,
        air_temperature,
        air_pressure,
        edb_nvpm.pressure_ratio,
        q_fuel,
        cruise=True,
    )

    # Interpolate nvPM EI_m and EI_n
    nvpm_ei_m = edb_nvpm.nvpm_ei_m.interp(t4_t2)
    nvpm_ei_m = nvpm_ei_m * 1e-6  # mg-nvPM/kg-fuel to kg-nvPM/kg-fuel
    nvpm_ei_n = edb_nvpm.nvpm_ei_n.interp(t4_t2)
    return nvpm_ei_m, nvpm_ei_n


def nvpm_mass_emissions_index_sac(
    edb_gaseous: EDBGaseous,
    air_pressure: npt.NDArray[np.float_],
    true_airspeed: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    thrust_setting: npt.NDArray[np.float_],
    fuel_flow_per_engine: npt.NDArray[np.float_],
    hydrogen_content: float,
) -> npt.NDArray[np.float_]:
    """Estimate nvPM mass emission index for singular annular combustor (SAC) engines.

    Here, SAC should not be confused with the Schmidt-Appleman Criterion.

    The nvpm_ei_m for SAC is estimated as the mean between a lower bound (80% of the FOX-estimated
    nvpm_ei_m) and an upper bound (150% of ImFOX-estimated nvpm_ei_m).

    Parameters
    ----------
    edb_gaseous : EDBGaseous
        EDB gaseous data
    air_pressure: npt.NDArray[np.float_]
        pressure altitude at each waypoint, [:math:`Pa`]
    true_airspeed: npt.NDArray[np.float_]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_temperature: npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]
    thrust_setting : npt.NDArray[np.float_]
        thrust setting
    fuel_flow_per_engine: npt.NDArray[np.float_]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    hydrogen_content : float
        Engine unique identification number from the ICAO EDB

    Returns
    -------
    npt.NDArray[np.float_]
        nvPM mass emissions index, [:math:`kg/kg_{fuel}`]
    """
    nvpm_ei_m_fox = black_carbon.mass_emissions_index_fox(
        air_pressure,
        air_temperature,
        true_airspeed,
        fuel_flow_per_engine,
        thrust_setting,
        edb_gaseous.pressure_ratio,
    )
    nvpm_ei_m_imfox = black_carbon.mass_emissions_index_imfox(
        fuel_flow_per_engine, thrust_setting, hydrogen_content
    )
    nvpm_ei_m = 0.5 * (0.8 * nvpm_ei_m_fox + 1.5 * nvpm_ei_m_imfox)
    return nvpm_ei_m * 1e-6  # mg-nvPM/kg-fuel to kg-nvPM/kg-fuel


def nvpm_geometric_mean_diameter_sac(
    edb_gaseous: EDBGaseous,
    air_pressure: npt.NDArray[np.float_],
    true_airspeed: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    thrust_setting: npt.NDArray[np.float_],
    q_fuel: float,
) -> npt.NDArray[np.float_]:
    r"""
    Estimate nvPM geometric mean diameter for singular annular combustor (SAC) engines.

    Parameters
    ----------
    edb_gaseous : EDBGaseous
        EDB gaseous data
    air_pressure: npt.NDArray[np.float_]
        pressure altitude at each waypoint, [:math:`Pa`]
    true_airspeed: npt.NDArray[np.float_]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_temperature: npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]
    thrust_setting : npt.NDArray[np.float_]
        thrust setting
    q_fuel : float
        Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].

    Returns
    -------
    npt.NDArray[np.float_]
        nvPM geometric mean diameter, [:math:`m`]
    """
    nvpm_gmd = black_carbon.geometric_mean_diameter_sac(
        air_pressure,
        air_temperature,
        true_airspeed,
        thrust_setting,
        edb_gaseous.pressure_ratio,
        q_fuel,
        cruise=True,
    )
    return nvpm_gmd * 1e-9  # nm to m


def get_thrust_setting(
    edb_gaseous: EDBGaseous,
    fuel_flow_per_engine: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    true_airspeed: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Approximate the engine thrust setting at cruise conditions.

    The thrust setting is approximated by dividing the fuel mass flow rate
    by the maximum fuel mass flow rate, and clipped to 3% (0.03) and 100% (1)
    respectively to account for unrealistic values.

    Parameters
    ----------
    edb_gaseous : EDBGaseous
        EDB gaseous data
    fuel_flow_per_engine: npt.NDArray[np.float_]
        Fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    air_pressure: npt.NDArray[np.float_]
        Pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature: npt.NDArray[np.float_]
        Ambient temperature for each waypoint, [:math:`K`]
    true_airspeed: npt.NDArray[np.float_]
        True airspeed for each waypoint, [:math:`m s^{-1}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Engine thrust setting. Returns ``np.nan`` if engine data is
        not available in the ICAO EDB dataset.
    """
    theta_amb = jet.temperature_ratio(air_temperature)
    delta_amb = jet.pressure_ratio(air_pressure)
    mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)
    fuel_flow_per_engine = jet.equivalent_fuel_flow_rate_at_sea_level(
        fuel_flow_per_engine, theta_amb, delta_amb, mach_num
    )

    thrust_setting = fuel_flow_per_engine / edb_gaseous.ff_100
    thrust_setting.clip(0.03, 1, out=thrust_setting)  # clip in place
    return thrust_setting


def _row_to_edb_gaseous(tup: Any) -> tuple[str, EDBGaseous]:
    return tup.engine_uid, EDBGaseous(
        **{k.name: getattr(tup, k.name) for k in dataclasses.fields(EDBGaseous)}
    )


@dataclasses.dataclass(frozen=True)
class EDBGaseous:
    """Gaseous emissions data.

    -------------------------------------
    ENGINE IDENTIFICATION AND TYPE:
    -------------------------------------
    manufacturer: str
        engine manufacturer
    engine_name: str
        name of engine
    combustor: str
        description of engine combustor

    -------------------------------------
    ENGINE CHARACTERISTICS:
    -------------------------------------
    bypass_ratio: float
        engine bypass ratio
    pressure_ratio: float
        engine pressure ratio
    rated_thrust: float
        rated thrust of engine, [:math:`kN`]

    -------------------------------------
    FUEL CONSUMPTION:
    -------------------------------------
    ff_7: float
        fuel mass flow rate at 7% thrust setting, [:math:`kg s^{-1}`]
    ff_30: float
        fuel mass flow rate at 30% thrust setting, [:math:`kg s^{-1}`]
    ff_85: float
        fuel mass flow rate at 85% thrust setting, [:math:`kg s^{-1}`]
    ff_100: float
        fuel mass flow rate at 100% thrust setting, [:math:`kg s^{-1}`]

    -------------------------------------
    EMISSIONS:
    -------------------------------------
    ei_nox_7: float
        NOx emissions index at 7% thrust setting, [:math:`g_{NO_{X}}/kg_{fuel}`]
    ei_nox_30: float
        NOx emissions index at 30% thrust setting, [:math:`g_{NO_{X}}/kg_{fuel}`]
    ei_nox_85: float
        NOx emissions index at 85% thrust setting, [:math:`g_{NO_{X}}/kg_{fuel}`]
    ei_nox_100: float
        NOx emissions index at 100% thrust setting, [:math:`g_{NO_{X}}/kg_{fuel}`]

    ei_co_7: float
        CO emissions index at 7% thrust setting, [:math:`g_{CO}/kg_{fuel}`]
    ei_co_30: float
        CO emissions index at 30% thrust setting, [:math:`g_{CO}/kg_{fuel}`]
    ei_co_85: float
        CO emissions index at 85% thrust setting, [:math:`g_{CO}/kg_{fuel}`]
    ei_co_100: float
        CO emissions index at 100% thrust setting, [:math:`g_{CO}/kg_{fuel}`]

    ei_hc_7: float
        HC emissions index at 7% thrust setting, [:math:`g_{HC}/kg_{fuel}`]
    ei_hc_30: float
        HC emissions index at 30% thrust setting, [:math:`g_{HC}/kg_{fuel}`]
    ei_hc_85: float
        HC emissions index at 85% thrust setting, [:math:`g_{HC}/kg_{fuel}`]
    ei_hc_100: float
        HC emissions index at 100% thrust setting, [:math:`g_{HC}/kg_{fuel}`]

    sn_7: float
        smoke number at 7% thrust setting
    sn_30: float
        smoke number at 30% thrust setting
    sn_85: float
        smoke number at 85% thrust setting
    sn_100: float
        smoke number at 100% thrust setting
    sn_max: float
        maximum smoke number value across the range of thrust setting
    """

    # Engine identification and type
    manufacturer: str
    engine_name: str
    combustor: str

    # Engine characteristics
    bypass_ratio: float
    pressure_ratio: float
    rated_thrust: float

    # Fuel consumption
    ff_7: float
    ff_30: float
    ff_85: float
    ff_100: float

    # Emissions
    ei_nox_7: float
    ei_nox_30: float
    ei_nox_85: float
    ei_nox_100: float

    ei_co_7: float
    ei_co_30: float
    ei_co_85: float
    ei_co_100: float

    ei_hc_7: float
    ei_hc_30: float
    ei_hc_85: float
    ei_hc_100: float

    sn_7: float
    sn_30: float
    sn_85: float
    sn_100: float
    sn_max: float

    @property
    def log_ei_nox_profile(self) -> EmissionsProfileInterpolator:
        """Get the logarithmic emissions index profile for NOx emissions."""
        return ffm2.nitrogen_oxide_emissions_index_profile(
            ff_idle=self.ff_7,
            ff_approach=self.ff_30,
            ff_climb=self.ff_85,
            ff_take_off=self.ff_100,
            ei_nox_idle=self.ei_nox_7,
            ei_nox_approach=self.ei_nox_30,
            ei_nox_climb=self.ei_nox_85,
            ei_nox_take_off=self.ei_nox_100,
        )

    @property
    def log_ei_co_profile(self) -> EmissionsProfileInterpolator:
        """Get the logarithmic emissions index profile for CO emissions."""
        return ffm2.co_hc_emissions_index_profile(
            ff_idle=self.ff_7,
            ff_approach=self.ff_30,
            ff_climb=self.ff_85,
            ff_take_off=self.ff_100,
            ei_idle=self.ei_co_7,
            ei_approach=self.ei_co_30,
            ei_climb=self.ei_co_85,
            ei_take_off=self.ei_co_100,
        )

    @property
    def log_ei_hc_profile(self) -> EmissionsProfileInterpolator:
        """Get the logarithmic emissions index profile for HC emissions."""
        return ffm2.co_hc_emissions_index_profile(
            ff_idle=self.ff_7,
            ff_approach=self.ff_30,
            ff_climb=self.ff_85,
            ff_take_off=self.ff_100,
            ei_idle=self.ei_hc_7,
            ei_approach=self.ei_hc_30,
            ei_climb=self.ei_hc_85,
            ei_take_off=self.ei_hc_100,
        )


def _row_to_edb_nvpm(tup: Any) -> tuple[str, EDBnvpm]:
    return tup.engine_uid, EDBnvpm(
        **{k.name: getattr(tup, k.name) for k in dataclasses.fields(EDBnvpm)}
    )


@dataclasses.dataclass
class EDBnvpm:
    """A data class for EDB nvPM data.

    -------------------------------------
    ENGINE IDENTIFICATION AND TYPE:
    -------------------------------------
    manufacturer: str
        engine manufacturer
    engine_name: str
        name of engine
    combustor: str
        description of engine combustor

    -------------------------------------
    ENGINE CHARACTERISTICS:
    -------------------------------------
    pressure_ratio: float
        engine pressure ratio

    -------------------------------------
    nvPM EMISSIONS:
    -------------------------------------
    nvpm_ei_m: EmissionsProfileInterpolator
         non-volatile PM mass emissions index profile (mg/kg) vs.
         non-dimensionalized thrust setting (t4_t2)
    nvpm_ei_n: EmissionsProfileInterpolator
        non-volatile PM number emissions index profile (1/kg) vs.
        non-dimensionalized thrust setting (t4_t2)
    """

    # Engine identification and type
    manufacturer: str
    engine_name: str
    combustor: str

    # Engine characteristics
    pressure_ratio: float
    temp_min: float
    temp_max: float
    fuel_heat: float

    # Fuel consumption
    ff_7: float
    ff_30: float
    ff_85: float
    ff_100: float

    # Emissions
    nvpm_ei_m_7: float
    nvpm_ei_m_30: float
    nvpm_ei_m_85: float
    nvpm_ei_m_100: float

    nvpm_ei_n_7: float
    nvpm_ei_n_30: float
    nvpm_ei_n_85: float
    nvpm_ei_n_100: float

    @property
    def nvpm_ei_m(self) -> EmissionsProfileInterpolator:
        """Get the nvPM emissions index mass profile."""
        return _nvpm_emissions_profiles(
            pressure_ratio=self.pressure_ratio,
            combustor=self.combustor,
            temp_min=self.temp_min,
            temp_max=self.temp_max,
            fuel_heat=self.fuel_heat,
            ff_7=self.ff_7,
            ff_30=self.ff_30,
            ff_85=self.ff_85,
            ff_100=self.ff_100,
            nvpm_ei_m_7=self.nvpm_ei_m_7,
            nvpm_ei_m_30=self.nvpm_ei_m_30,
            nvpm_ei_m_85=self.nvpm_ei_m_85,
            nvpm_ei_m_100=self.nvpm_ei_m_100,
            nvpm_ei_n_7=self.nvpm_ei_n_7,
            nvpm_ei_n_30=self.nvpm_ei_n_30,
            nvpm_ei_n_85=self.nvpm_ei_n_85,
            nvpm_ei_n_100=self.nvpm_ei_n_100,
        )[0]

    @property
    def nvpm_ei_n(self) -> EmissionsProfileInterpolator:
        """Get the nvPM emissions index number profile."""
        return _nvpm_emissions_profiles(
            pressure_ratio=self.pressure_ratio,
            combustor=self.combustor,
            temp_min=self.temp_min,
            temp_max=self.temp_max,
            fuel_heat=self.fuel_heat,
            ff_7=self.ff_7,
            ff_30=self.ff_30,
            ff_85=self.ff_85,
            ff_100=self.ff_100,
            nvpm_ei_m_7=self.nvpm_ei_m_7,
            nvpm_ei_m_30=self.nvpm_ei_m_30,
            nvpm_ei_m_85=self.nvpm_ei_m_85,
            nvpm_ei_m_100=self.nvpm_ei_m_100,
            nvpm_ei_n_7=self.nvpm_ei_n_7,
            nvpm_ei_n_30=self.nvpm_ei_n_30,
            nvpm_ei_n_85=self.nvpm_ei_n_85,
            nvpm_ei_n_100=self.nvpm_ei_n_100,
        )[1]


@functools.cache
def _nvpm_emissions_profiles(
    pressure_ratio: float,
    combustor: str,
    temp_min: float,
    temp_max: float,
    fuel_heat: float,
    ff_7: float,
    ff_30: float,
    ff_85: float,
    ff_100: float,
    nvpm_ei_m_7: float,
    nvpm_ei_m_30: float,
    nvpm_ei_m_85: float,
    nvpm_ei_m_100: float,
    nvpm_ei_n_7: float,
    nvpm_ei_n_30: float,
    nvpm_ei_n_85: float,
    nvpm_ei_n_100: float,
) -> tuple[EmissionsProfileInterpolator, EmissionsProfileInterpolator]:
    # Extract fuel flow
    fuel_flow = np.array([ff_7, ff_30, ff_85, ff_100])
    fuel_flow_max = fuel_flow[-1]

    # Extract nvPM emissions arrays
    nvpm_ei_m = np.array([nvpm_ei_m_7, nvpm_ei_m_30, nvpm_ei_m_85, nvpm_ei_m_100])
    nvpm_ei_n = np.array([nvpm_ei_n_7, nvpm_ei_n_30, nvpm_ei_n_85, nvpm_ei_n_100])

    is_staged_combustor = combustor in ("DAC", "TAPS", "TAPS II")
    if is_staged_combustor:
        # In this case, all of our interpolators will have size 5
        fuel_flow = np.insert(fuel_flow, 2, (fuel_flow[1] * 1.001))
        nvpm_ei_n_lean_burn = np.mean(nvpm_ei_n[2:])
        nvpm_ei_n = np.r_[nvpm_ei_n[:2], [nvpm_ei_n_lean_burn] * 3]
        nvpm_ei_m_lean_burn = np.mean(nvpm_ei_m[2:])
        nvpm_ei_m = np.r_[nvpm_ei_m[:2], [nvpm_ei_m_lean_burn] * 3]

    thrust_setting = fuel_flow / fuel_flow_max
    avg_temp = (temp_min + temp_max) / 2.0

    t4_t2 = jet.thrust_setting_nd(
        true_airspeed=0.0,
        thrust_setting=thrust_setting,
        T=avg_temp,
        p=constants.p_surface,
        pressure_ratio=pressure_ratio,
        q_fuel=fuel_heat * 1e6,
        cruise=False,
    )

    nvpm_ei_m_interp = EmissionsProfileInterpolator(t4_t2, nvpm_ei_m)
    nvpm_ei_n_interp = EmissionsProfileInterpolator(t4_t2, nvpm_ei_n)
    return nvpm_ei_m_interp, nvpm_ei_n_interp


@functools.cache
def get_engine_params_from_edb(filepath: str | pathlib.Path) -> dict[str, EDBGaseous]:
    """Read EDB file into a dictionary of the form ``{engine_uid: gaseous_data}``.

    Parameters
    ----------
    filepath : str | pathlib.Path
        Path to EDB csv.

    Returns
    -------
    dict[str, EDBGaseous]
        Mapping from engine UID to gaseous emissions data for engine.
    """

    columns = {
        "UID No": "engine_uid",
        "Manufacturer": "manufacturer",
        "Engine Identification": "engine_name",
        "Combustor Description": "combustor",
        "B/P Ratio": "bypass_ratio",
        "Pressure Ratio": "pressure_ratio",
        "Rated Thrust (kN)": "rated_thrust",
        "Fuel Flow Idle (kg/sec)": "ff_7",
        "Fuel Flow App (kg/sec)": "ff_30",
        "Fuel Flow C/O (kg/sec)": "ff_85",
        "Fuel Flow T/O (kg/sec)": "ff_100",
        "NOx EI Idle (g/kg)": "ei_nox_7",
        "NOx EI App (g/kg)": "ei_nox_30",
        "NOx EI C/O (g/kg)": "ei_nox_85",
        "NOx EI T/O (g/kg)": "ei_nox_100",
        "CO EI Idle (g/kg)": "ei_co_7",
        "CO EI App (g/kg)": "ei_co_30",
        "CO EI C/O (g/kg)": "ei_co_85",
        "CO EI T/O (g/kg)": "ei_co_100",
        "HC EI Idle (g/kg)": "ei_hc_7",
        "HC EI App (g/kg)": "ei_hc_30",
        "HC EI C/O (g/kg)": "ei_hc_85",
        "HC EI T/O (g/kg)": "ei_hc_100",
        "SN Idle": "sn_7",
        "SN App": "sn_30",
        "SN C/O": "sn_85",
        "SN T/O": "sn_100",
        "SN Max": "sn_max",
    }

    df = pd.read_csv(filepath)
    df = df.rename(columns=columns)

    return dict(_row_to_edb_gaseous(tup) for tup in df.itertuples(index=False))


@functools.cache
def get_engine_nvpm_profile_from_edb(filepath: str | pathlib.Path) -> dict[str, EDBnvpm]:
    """Read EDB file into a dictionary of the form ``{engine_uid: npvm_data}``.

    Parameters
    ----------
    filepath : str | pathlib.Path
        Path to EDB csv.

    Returns
    -------
    dict[str, EDBnvpm]
        Mapping from aircraft type to nvPM data for engine.
    """
    columns = {
        "UID No": "engine_uid",
        "Manufacturer": "manufacturer",
        "Engine Identification": "engine_name",
        "Combustor Description": "combustor",
        "Pressure Ratio": "pressure_ratio",
        "Ambient Temp Min (K)": "temp_min",
        "Ambient Temp Max (K)": "temp_max",
        "Fuel Heat of Combustion (MJ/kg)": "fuel_heat",
        "Fuel Flow Idle (kg/sec)": "ff_7",
        "Fuel Flow App (kg/sec)": "ff_30",
        "Fuel Flow C/O (kg/sec)": "ff_85",
        "Fuel Flow T/O (kg/sec)": "ff_100",
        "nvPM EImass_SL Idle (mg/kg)": "nvpm_ei_m_7",
        "nvPM EImass_SL App (mg/kg)": "nvpm_ei_m_30",
        "nvPM EImass_SL C/O (mg/kg)": "nvpm_ei_m_85",
        "nvPM EImass_SL T/O (mg/kg)": "nvpm_ei_m_100",
        "nvPM EInum_SL Idle (#/kg)": "nvpm_ei_n_7",
        "nvPM EInum_SL App (#/kg)": "nvpm_ei_n_30",
        "nvPM EInum_SL C/O (#/kg)": "nvpm_ei_n_85",
        "nvPM EInum_SL T/O (#/kg)": "nvpm_ei_n_100",
    }

    df = pd.read_csv(filepath)
    df = df.rename(columns=columns)

    return dict(_row_to_edb_nvpm(tup) for tup in df.itertuples(index=False))


@functools.cache
def get_default_aircraft_engine_mapping(filepath: str | pathlib.Path) -> pd.DataFrame:
    """Read default aircraft type -> engine UID assignments.

    Parameters
    ----------
    filepath : str | pathlib.Path
        Path to default mapping.

    Returns
    -------
    pd.DataFrame
        A :class:`pd.DataFrame` whose index is available aircraft types with columns:

        - engine_uid
        - engine_name
        - n-engines
    """
    return pd.read_csv(filepath, index_col=0)
