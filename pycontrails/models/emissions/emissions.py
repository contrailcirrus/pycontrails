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

from pycontrails.core.flight import Flight
from pycontrails.core.fuel import Fuel, SAFBlend
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import AirTemperature, MetVariable, SpecificHumidity
from pycontrails.core.models import Model, ModelParams
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.models.emissions import gaseous, nvpm
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.physics import jet, units

_path_to_static = pathlib.Path(__file__).parent / "static"
EDB_ENGINE_PATH = _path_to_static / "edb-gaseous-v31-engines.csv"
EDB_NVPM_PATH = _path_to_static / "edb-nvpm-v31-engines.csv"
ENGINE_UID_PATH = _path_to_static / "default-engine-uids.csv"


@dataclasses.dataclass
class EmissionsParams(ModelParams):
    """:class:`Emissions` model parameters."""

    #: Default nvpm_ei_n value if engine UID is not found
    default_nvpm_ei_n: float = 1e15

    #: Humidity scaling. If None, no scaling is applied.
    humidity_scaling: HumidityScaling | None = None

    #: If True, if an engine UID is not provided on the ``source.attrs``, use a
    #: default engine UID based on Teoh's analysis of aircraft engine pairs in
    #: 2019 - 2021 Spire data.
    use_default_engine_uid: bool = True

    #: EXPERIMENTAL
    #: If True, use the alternative MEEM2/SCOPE11 method for nvPM EI calculations.
    #: https://doi.org/10.4271/2025-01-6000
    #: https://doi.org/10.1021/acs.est.8b04060
    #: If False (default), the nvPM methodology from GAIA (T4/T2 methodology,
    #: FOX, and ImFOX methods) will be used.
    use_meem: bool = False


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
    :mod:`pycontrails.models.emissions.nvpm`
    :mod:`pycontrails.models.emissions.gaseous`
    """

    name = "emissions"
    long_name = "ICAO Emissions Databank (EDB)"
    met_variables: tuple[MetVariable, ...] = AirTemperature, SpecificHumidity
    default_params = EmissionsParams

    source: GeoVectorDataset

    def __init__(
        self,
        met: MetDataset | None = None,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ) -> None:
        super().__init__(met, params, **params_kwargs)

        self.edb_engine_gaseous = load_edb_gaseous_database()
        self.edb_engine_nvpm = load_edb_nvpm_database()
        self.default_engines = load_default_aircraft_engine_mapping()

    @overload
    def eval(self, source: Flight, **params: Any) -> Flight: ...

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset: ...

    @overload
    def eval(self, source: None = ..., **params: Any) -> NoReturn: ...

    def eval(self, source: GeoVectorDataset | None = None, **params: Any) -> GeoVectorDataset:
        """Calculate the emissions data for ``source``.

        Parameter ``source`` must contain each of the variables:
            - air_temperature
            - specific_humidity
            - true_airspeed
            - fuel_flow

        If 'engine_uid' is not provided in ``source.attrs`` or not available in the ICAO EDB,
        constant emission indices will be assumed for NOx, CO, HC, and nvPM mass and number.

        The computed pollutants include carbon dioxide (CO2), nitrogen oxide (NOx),
        carbon monoxide (CO), hydrocarbons (HC), non-volatile particulate matter
        (nvPM) mass and number, sulphur oxides (SOx), sulphates (S) and organic carbon (OC).

        .. versionchanged:: 0.47.0
            Support GeoVectorDataset for the ``source`` parameter.

        Parameters
        ----------
        source : GeoVectorDataset
            Flight to evaluate
        **params : Any
            Overwrite model parameters before eval

        Returns
        -------
        GeoVectorDataset
            Flight with attached emissions data
        """
        self.update_params(params)
        self.set_source(source)
        self.source = self.require_source_type(GeoVectorDataset)

        # Set air_temperature and specific_humidity if not already set
        humidity_scaling = self.params["humidity_scaling"]
        scale_humidity = humidity_scaling is not None and "specific_humidity" not in self.source
        self.set_source_met()

        # Only enhance humidity if it wasn't already present on source
        if scale_humidity:
            humidity_scaling.eval(self.source, copy_source=False)

        # Ensure that flight has the required AP variables
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

        if "n_engine" not in self.source.attrs:
            aircraft_type = self.source.get_constant("aircraft_type", None)
            self.source.attrs["n_engine"] = self.default_engines.at[aircraft_type, "n_engine"]

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
                self.source["thrust_setting"] = np.full(len(self.source), np.nan, dtype=np.float32)
            else:
                self.source["thrust_setting"] = get_thrust_setting(
                    edb_gaseous,
                    fuel_flow_per_engine=fuel_flow_per_engine,
                    air_pressure=self.source.air_pressure,
                    air_temperature=self.source["air_temperature"],
                    true_airspeed=self.source.get_data_or_attr("true_airspeed"),
                )

        self._gaseous_emission_indices(engine_uid)
        if self.params["use_meem"]:
            self._nvpm_emission_indices_meem_scope11(engine_uid)
        else:
            self._nvpm_emission_indices_gaia(engine_uid)

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

    def _gaseous_emissions_ffm2(self, edb_gaseous: gaseous.EDBGaseous) -> None:
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
        self.source["nox_ei"] = (
            gaseous.estimate_nox_ffm2(
                edb_gaseous.log_ei_nox_profile,
                fuel_flow_per_engine,
                true_airspeed,
                self.source.air_pressure,
                air_temperature,
                self.source["specific_humidity"],
            )
            * 1e-3  # g-NOx/kg-fuel to kg-NOx/kg-fuel
        )

        self.source["co_ei"] = (
            gaseous.estimate_ei_co_hc_ffm2(
                edb_gaseous.log_ei_co_profile,
                fuel_flow_per_engine,
                true_airspeed,
                self.source.air_pressure,
                air_temperature,
            )
            * 1e-3  # g-CO/kg-fuel to kg-CO/kg-fuel
        )

        self.source["hc_ei"] = (
            gaseous.estimate_ei_co_hc_ffm2(
                edb_gaseous.log_ei_hc_profile,
                fuel_flow_per_engine,
                true_airspeed,
                self.source.air_pressure,
                air_temperature,
            )
            * 1e-3  # g-HC/kg-fuel to kg-HC/kg-fuel
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

        nox_ei = np.full(shape=len(self.source), fill_value=15.14, dtype=np.float32)
        co_ei = np.full(shape=len(self.source), fill_value=3.61, dtype=np.float32)
        hc_ei = np.full(shape=len(self.source), fill_value=0.520, dtype=np.float32)

        self.source["nox_ei"] = nox_ei * 1e-3  # g-NOx/kg-fuel to kg-NOx/kg-fuel
        self.source["co_ei"] = co_ei * 1e-3  # g-CO/kg-fuel to kg-CO/kg-fuel
        self.source["hc_ei"] = hc_ei * 1e-3  # g-HC/kg-fuel to kg-HC/kg-fuel

    def _nvpm_emission_indices_gaia(self, engine_uid: str | None) -> None:
        """Calculate nvPM mass and number emission indices using GAIA methodologies.

        This method attaches the following variables to the underlying :attr:`source`.
            - nvpm_ei_m
            - nvpm_ei_n

        In addition, ``nvpm_data_source`` is attached to the ``source.attrs``.

        Parameters
        ----------
        engine_uid : str
            Engine unique identification number from the ICAO EDB

        References
        ----------
        # TODO: Add to bibliography
        - (Teoh et al., 2024) https://doi.org/10.5194/acp-24-725-2024
        """
        if "nvpm_ei_n" in self.source and "nvpm_ei_m" in self.source:
            return  # early exit if values already exist

        if isinstance(self.source, Flight):
            fuel = self.source.fuel
        else:
            try:
                fuel = self.source.attrs["fuel"]
            except KeyError as exc:
                raise KeyError(
                    "If running 'Emissions' with a 'GeoVectorDataset' as source, "
                    "the fuel type must be provided in the attributes. "
                ) from exc

        edb_nvpm = self.edb_engine_nvpm.get(engine_uid) if engine_uid else None
        edb_gaseous = self.edb_engine_gaseous.get(engine_uid) if engine_uid else None

        if edb_nvpm is not None:
            nvpm_data = self._nvpm_emission_indices_t4_t2(edb_nvpm, fuel)
        elif edb_gaseous is not None:
            nvpm_data = self._nvpm_emission_indices_sac(edb_gaseous, fuel)
        else:
            if engine_uid is not None:
                warnings.warn(
                    f"Cannot find 'engine_uid' {engine_uid} in EDB. "
                    "A constant emissions will be used."
                )
            nvpm_data = self._nvpm_emission_indices_constant()

        nvpm_data_source, nvpm_ei_m, nvpm_ei_n = nvpm_data

        # Adjust nvPM emission indices if SAF is used.
        if isinstance(fuel, SAFBlend) and fuel.pct_blend:
            thrust_setting = self.source["thrust_setting"]
            pct_eim_reduction = nvpm.nvpm_mass_ei_pct_reduction_due_to_saf(
                fuel.hydrogen_content, thrust_setting
            )
            pct_ein_reduction = nvpm.nvpm_number_ei_pct_reduction_due_to_saf(
                fuel.hydrogen_content, thrust_setting
            )

            nvpm_ei_m *= 1.0 + pct_eim_reduction / 100.0
            nvpm_ei_n *= 1.0 + pct_ein_reduction / 100.0

        self.source.attrs["nvpm_data_source"] = nvpm_data_source
        self.source.setdefault("nvpm_ei_m", nvpm_ei_m)
        self.source.setdefault("nvpm_ei_n", nvpm_ei_n)

    def _nvpm_emission_indices_meem_scope11(self, engine_uid: str | None) -> None:
        """Calculate nvPM mass and number emission indices using MEEM2 and SCOPE11 methodologies.

        This method attaches the following variables to the underlying :attr:`source`.
            - nvpm_ei_m
            - nvpm_ei_n

        In addition, ``nvpm_data_source`` is attached to the ``source.attrs``.

        Parameters
        ----------
        engine_uid : str
            Engine unique identification number from the ICAO EDB

        References
        ----------
        # TODO: Add to bibliography
        - (Ahrens et al., 2025) https://doi.org/10.4271/2025-01-6000
        - (Agarwal et al., 2019) https://doi.org/10.1021/acs.est.8b04060
        """
        if "nvpm_ei_n" in self.source and "nvpm_ei_m" in self.source:
            return  # early exit if values already exist

        if isinstance(self.source, Flight):
            fuel = self.source.fuel
        else:
            try:
                fuel = self.source.attrs["fuel"]
            except KeyError as exc:
                raise KeyError(
                    "If running 'Emissions' with a 'GeoVectorDataset' as source, "
                    "the fuel type must be provided in the attributes. "
                ) from exc

        edb_nvpm = self.edb_engine_nvpm.get(engine_uid) if engine_uid else None
        edb_gaseous = self.edb_engine_gaseous.get(engine_uid) if engine_uid else None
        has_sn_data = (
            edb_gaseous
            and ~np.isnan(
                np.array(
                    [edb_gaseous.sn_7, edb_gaseous.sn_30, edb_gaseous.sn_85, edb_gaseous.sn_100]
                )
            ).all()
        )

        if edb_nvpm is not None:
            nvpm_data = self._nvpm_emission_indices_meem(edb_nvpm, fuel)
        elif edb_gaseous is not None and has_sn_data:
            nvpm_data = self._nvpm_emission_indices_scope11(edb_gaseous, fuel)
        else:
            if engine_uid is not None:
                warnings.warn(
                    f"Cannot find 'engine_uid' {engine_uid} in EDB. "
                    "A constant emissions will be used."
                )
            nvpm_data = self._nvpm_emission_indices_constant()

        nvpm_data_source, nvpm_ei_m, nvpm_ei_n = nvpm_data

        self.source.attrs["nvpm_data_source"] = nvpm_data_source
        self.source.setdefault("nvpm_ei_m", nvpm_ei_m)
        self.source.setdefault("nvpm_ei_n", nvpm_ei_n)

    def _nvpm_emission_indices_t4_t2(
        self, edb_nvpm: nvpm.EDBnvpm, fuel: Fuel
    ) -> tuple[str, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Calculate emission indices for nvPM mass and number.

        This method uses data from the ICAO EDB along with the T4/T2 methodology.

        Parameters
        ----------
        edb_nvpm : EDBnvpm
            EDB nvPM data.
        fuel : Fuel
            Fuel type.

        Returns
        -------
        nvpm_data_source : str
            Source of nvpm data.
        nvpm_ei_m : npt.NDArray[np.floating]
            Non-volatile particulate matter (nvPM) mass emissions index, [:math:`kg/kg_{fuel}`]
        nvpm_ei_n : npt.NDArray[np.floating]
            Black carbon number emissions index, [:math:`kg_{fuel}^{-1}`]

        References
        ----------
        - :cite:`teohTargetedUseSustainable2022`
        """
        nvpm_data_source = "ICAO EDB (T4/T2)"

        # Emissions indices
        return nvpm_data_source, *nvpm.estimate_nvpm_t4_t2(
            edb_nvpm,
            true_airspeed=self.source.get_data_or_attr("true_airspeed"),
            air_temperature=self.source["air_temperature"],
            air_pressure=self.source.air_pressure,
            thrust_setting=self.source["thrust_setting"],
            q_fuel=fuel.q_fuel,
        )

    def _nvpm_emission_indices_sac(
        self, edb_gaseous: gaseous.EDBGaseous, fuel: Fuel
    ) -> tuple[str, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Calculate EIs for nvPM mass and number assuming the profile of single annular combustors.

        nvPM EI_m is calculated using the FOX and ImFOX methods, while the nvPM EI_n
        is calculated using the Fractal Aggregates (FA) model.

        Parameters
        ----------
        edb_gaseous : EDBGaseous
            EDB gaseous data
        fuel : Fuel
            Fuel type.

        Returns
        -------
        nvpm_data_source : str
            Source of nvpm data.
        nvpm_ei_m : npt.NDArray[np.floating]
            Non-volatile particulate matter (nvPM) mass emissions index, [:math:`kg/kg_{fuel}`]
        nvpm_ei_n : npt.NDArray[np.floating]
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
            hydrogen_content=fuel.hydrogen_content,
        )
        nvpm_gmd = nvpm_geometric_mean_diameter_sac(
            edb_gaseous,
            air_pressure=self.source.air_pressure,
            true_airspeed=true_airspeed,
            air_temperature=air_temperature,
            thrust_setting=thrust_setting,
            q_fuel=fuel.q_fuel,
        )
        nvpm_ei_n = nvpm.number_emissions_index_fractal_aggregates(nvpm_ei_m, nvpm_gmd)
        return nvpm_data_source, nvpm_ei_m, nvpm_ei_n

    def _nvpm_emission_indices_meem(
        self, edb_nvpm: nvpm.EDBnvpm, fuel: Fuel
    ) -> tuple[str, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Calculate emission indices for nvPM mass and number using the MEEM2 methodology.

        Parameters
        ----------
        edb_nvpm : EDBnvpm
            EDB nvPM data.
        fuel : Fuel
            Fuel type.

        Returns
        -------
        nvpm_data_source : str
            Source of nvpm data.
        nvpm_ei_m : npt.NDArray[np.floating]
            nvPM mass emissions index, [:math:`kg/kg_{fuel}`]
        nvpm_ei_n : npt.NDArray[np.floating]
            nvPM number emissions index, [:math:`kg_{fuel}^{-1}`]

        References
        ----------
        # TODO: Add to bibliography
        - (Ahrens et al., 2025) https://doi.org/10.4271/2025-01-6000
        """
        nvpm_data_source = "ICAO EDB (MEEM2)"

        # Get MEEM nvPM emissions profile
        nvpm_ei_m_profile = nvpm.nvpm_mass_emission_profiles_meem(
            combustor=edb_nvpm.combustor,
            hydrogen_content=fuel.hydrogen_content,
            ff_7=edb_nvpm.ff_7,
            ff_30=edb_nvpm.ff_30,
            ff_85=edb_nvpm.ff_85,
            ff_100=edb_nvpm.ff_100,
            nvpm_ei_m_7=edb_nvpm.nvpm_ei_m_7,
            nvpm_ei_m_30=edb_nvpm.nvpm_ei_m_30,
            nvpm_ei_m_85=edb_nvpm.nvpm_ei_m_85,
            nvpm_ei_m_100=edb_nvpm.nvpm_ei_m_100,
            fifth_data_point_mass=edb_nvpm.nvpm_ei_m_use_max,
            nvpm_ei_m_30_no_sl=edb_nvpm.nvpm_ei_m_no_sl_30,
            nvpm_ei_m_85_no_sl=edb_nvpm.nvpm_ei_m_no_sl_85,
            nvpm_ei_m_max_no_sl=edb_nvpm.nvpm_ei_m_no_sl_max,
        )

        nvpm_ei_n_profile = nvpm.nvpm_number_emission_profiles_meem(
            combustor=edb_nvpm.combustor,
            hydrogen_content=fuel.hydrogen_content,
            ff_7=edb_nvpm.ff_7,
            ff_30=edb_nvpm.ff_30,
            ff_85=edb_nvpm.ff_85,
            ff_100=edb_nvpm.ff_100,
            nvpm_ei_n_7=edb_nvpm.nvpm_ei_n_7,
            nvpm_ei_n_30=edb_nvpm.nvpm_ei_n_30,
            nvpm_ei_n_85=edb_nvpm.nvpm_ei_n_85,
            nvpm_ei_n_100=edb_nvpm.nvpm_ei_n_100,
            fifth_data_point_number=edb_nvpm.nvpm_ei_n_use_max,
            nvpm_ei_n_30_no_sl=edb_nvpm.nvpm_ei_n_no_sl_30,
            nvpm_ei_n_85_no_sl=edb_nvpm.nvpm_ei_n_no_sl_85,
            nvpm_ei_n_max_no_sl=edb_nvpm.nvpm_ei_n_no_sl_max,
        )

        # Emissions indices
        return nvpm_data_source, *nvpm.estimate_nvpm_meem(
            nvpm_ei_m_profile=nvpm_ei_m_profile,
            nvpm_ei_n_profile=nvpm_ei_n_profile,
            fuel_flow_per_engine=self.source.get_data_or_attr("fuel_flow_per_engine"),
            true_airspeed=self.source.get_data_or_attr("true_airspeed"),
            air_pressure=self.source.air_pressure,
            air_temperature=self.source["air_temperature"],
            ff_7=edb_nvpm.ff_7,
            ff_100=edb_nvpm.ff_100,
        )

    def _nvpm_emission_indices_scope11(
        self, edb_gaseous: gaseous.EDBGaseous, fuel: Fuel
    ) -> tuple[str, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """
        Calculate emission indices for nvPM mass and number using the SCOPE11 and MEEM2 methodology.

        Parameters
        ----------
        edb_gaseous : EDBGaseous
            EDB gaseous data
        fuel : Fuel
            Fuel type.

        Returns
        -------
        nvpm_data_source : str
            Source of nvpm data.
        nvpm_ei_m : npt.NDArray[np.floating]
            nvPM mass emissions index, [:math:`kg/kg_{fuel}`]
        nvpm_ei_n : npt.NDArray[np.floating]
            nvPM number emissions index, [:math:`kg_{fuel}^{-1}`]

        References
        ----------
        # TODO: Add to bibliography
        - (Agarwal et al., 2019) https://doi.org/10.1021/acs.est.8b04060
        """
        nvpm_data_source = "ICAO EDB (SCOPE11-MEEM2)"

        # Use SCOPE11 to derive nvPM emissions profile
        smoke_number = np.array(
            [
                edb_gaseous.sn_7,
                edb_gaseous.sn_30,
                edb_gaseous.sn_85,
                edb_gaseous.sn_100,
            ]
        )
        afr = np.array([106.0, 83.0, 51.0, 45.0])  # Agarwal et al. (2019)
        thrust_setting = np.array([0.07, 0.30, 0.85, 1.00])

        nvpm_ei_m_scope = nvpm.mass_ei_scope11(
            sn=smoke_number, afr=afr, bypass_ratio=edb_gaseous.bypass_ratio
        )

        average_temp = 0.5 * (edb_gaseous.temp_min + edb_gaseous.temp_max)
        average_pressure = 0.5 * (edb_gaseous.pressure_min + edb_gaseous.pressure_max)

        nvpm_ei_n_scope = nvpm.number_ei_scope11(
            nvpm_ei_m_e=nvpm_ei_m_scope,
            sn=smoke_number,
            air_temperature=average_temp,
            air_pressure=average_pressure,
            thrust_setting=thrust_setting,
            afr=afr,
            q_fuel=fuel.q_fuel,
            bypass_ratio=edb_gaseous.bypass_ratio,
            pressure_ratio=edb_gaseous.pressure_ratio,
        )

        nvpm_ei_m_profile = nvpm.nvpm_mass_emission_profiles_meem(
            combustor=edb_gaseous.combustor,
            hydrogen_content=fuel.hydrogen_content,
            ff_7=edb_gaseous.ff_7,
            ff_30=edb_gaseous.ff_30,
            ff_85=edb_gaseous.ff_85,
            ff_100=edb_gaseous.ff_100,
            nvpm_ei_m_7=nvpm_ei_m_scope[0],
            nvpm_ei_m_30=nvpm_ei_m_scope[1],
            nvpm_ei_m_85=nvpm_ei_m_scope[2],
            nvpm_ei_m_100=nvpm_ei_m_scope[3],
        )

        nvpm_ei_n_profile = nvpm.nvpm_number_emission_profiles_meem(
            combustor=edb_gaseous.combustor,
            hydrogen_content=fuel.hydrogen_content,
            ff_7=edb_gaseous.ff_7,
            ff_30=edb_gaseous.ff_30,
            ff_85=edb_gaseous.ff_85,
            ff_100=edb_gaseous.ff_100,
            nvpm_ei_n_7=nvpm_ei_n_scope[0],
            nvpm_ei_n_30=nvpm_ei_n_scope[1],
            nvpm_ei_n_85=nvpm_ei_n_scope[2],
            nvpm_ei_n_100=nvpm_ei_n_scope[3],
        )

        return nvpm_data_source, *nvpm.estimate_nvpm_meem(
            nvpm_ei_m_profile=nvpm_ei_m_profile,
            nvpm_ei_n_profile=nvpm_ei_n_profile,
            fuel_flow_per_engine=self.source.get_data_or_attr("fuel_flow_per_engine"),
            true_airspeed=self.source.get_data_or_attr("true_airspeed"),
            air_pressure=self.source.air_pressure,
            air_temperature=self.source["air_temperature"],
            ff_7=edb_gaseous.ff_7,
            ff_100=edb_gaseous.ff_100,
        )

    def _nvpm_emission_indices_constant(
        self,
    ) -> tuple[str, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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
        nvpm_ei_m : npt.NDArray[np.floating]
            Non-volatile particulate matter (nvPM) mass emissions index, [:math:`kg/kg_{fuel}`]
        nvpm_ei_n : npt.NDArray[np.floating]
            Black carbon number emissions index, [:math:`kg_{fuel}^{-1}`]

        References
        ----------
        - :cite:`stettlerGlobalCivilAviation2013`
        - :cite:`wilkersonAnalysisEmissionData2010`
        - :cite:`schumannDehydrationEffectsContrails2015`
        """
        nvpm_data_source = "Constant"
        nvpm_ei_m = np.full(len(self.source), 0.088 * 1e-3, dtype=np.float32)  # g to kg
        nvpm_ei_n = np.full(len(self.source), self.params["default_nvpm_ei_n"], dtype=np.float32)
        return nvpm_data_source, nvpm_ei_m, nvpm_ei_n

    def _total_pollutant_emissions(self) -> None:
        if not isinstance(self.source, Flight):
            return

        dt_sec = self.source.segment_duration(self.source.altitude_ft.dtype)
        fuel_burn = jet.fuel_burn(self.source.get_data_or_attr("fuel_flow"), dt_sec)

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


def nvpm_mass_emissions_index_sac(
    edb_gaseous: gaseous.EDBGaseous,
    air_pressure: npt.NDArray[np.floating],
    true_airspeed: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
    fuel_flow_per_engine: npt.NDArray[np.floating],
    hydrogen_content: float,
) -> npt.NDArray[np.floating]:
    """Estimate nvPM mass emission index for singular annular combustor (SAC) engines.

    Here, SAC should not be confused with the Schmidt-Appleman Criterion.

    The nvpm_ei_m for SAC is estimated as the mean between a lower bound (80% of the FOX-estimated
    nvpm_ei_m) and an upper bound (150% of ImFOX-estimated nvpm_ei_m).

    Parameters
    ----------
    edb_gaseous : EDBGaseous
        EDB gaseous data
    air_pressure: npt.NDArray[np.floating]
        pressure altitude at each waypoint, [:math:`Pa`]
    true_airspeed: npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_temperature: npt.NDArray[np.floating]
        ambient temperature for each waypoint, [:math:`K`]
    thrust_setting : npt.NDArray[np.floating]
        thrust setting
    fuel_flow_per_engine: npt.NDArray[np.floating]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    hydrogen_content : float
        Engine unique identification number from the ICAO EDB

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM mass emissions index, [:math:`kg/kg_{fuel}`]
    """
    nvpm_ei_m_fox = nvpm.mass_emissions_index_fox(
        air_pressure,
        air_temperature,
        true_airspeed,
        fuel_flow_per_engine,
        thrust_setting,
        edb_gaseous.pressure_ratio,
    )
    nvpm_ei_m_imfox = nvpm.mass_emissions_index_imfox(
        fuel_flow_per_engine, thrust_setting, hydrogen_content
    )
    nvpm_ei_m = 0.5 * (0.8 * nvpm_ei_m_fox + 1.5 * nvpm_ei_m_imfox)
    return nvpm_ei_m * 1e-6  # mg-nvPM/kg-fuel to kg-nvPM/kg-fuel


def nvpm_geometric_mean_diameter_sac(
    edb_gaseous: gaseous.EDBGaseous,
    air_pressure: npt.NDArray[np.floating],
    true_airspeed: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
    q_fuel: float,
) -> npt.NDArray[np.floating]:
    r"""
    Estimate nvPM geometric mean diameter for singular annular combustor (SAC) engines.

    Parameters
    ----------
    edb_gaseous : EDBGaseous
        EDB gaseous data
    air_pressure: npt.NDArray[np.floating]
        pressure altitude at each waypoint, [:math:`Pa`]
    true_airspeed: npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_temperature: npt.NDArray[np.floating]
        ambient temperature for each waypoint, [:math:`K`]
    thrust_setting : npt.NDArray[np.floating]
        thrust setting
    q_fuel : float
        Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM geometric mean diameter, [:math:`m`]
    """
    nvpm_gmd = nvpm.geometric_mean_diameter_sac(
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
    edb_gaseous: gaseous.EDBGaseous,
    fuel_flow_per_engine: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    true_airspeed: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Approximate the engine thrust setting at cruise conditions.

    The thrust setting is approximated by dividing the fuel mass flow rate
    by the maximum fuel mass flow rate, and clipped to 3% (0.03) and 100% (1)
    respectively to account for unrealistic values.

    Parameters
    ----------
    edb_gaseous : EDBGaseous
        EDB gaseous data
    fuel_flow_per_engine: npt.NDArray[np.floating]
        Fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    air_pressure: npt.NDArray[np.floating]
        Pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature: npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`]
    true_airspeed: npt.NDArray[np.floating]
        True airspeed for each waypoint, [:math:`m s^{-1}`]

    Returns
    -------
    npt.NDArray[np.floating]
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
    thrust_setting.clip(0.03, 1.0, out=thrust_setting)  # clip in place
    return thrust_setting


# ---------------------------------------------------
# Functions to load ICAO EDB gaseous and nvPM dataset
# ---------------------------------------------------


@functools.cache
def load_edb_gaseous_database() -> dict[str, gaseous.EDBGaseous]:
    """Read EDB file into a dictionary of the form ``{engine_uid: gaseous_data}``.

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
        "Ambient Temp Min (K)": "temp_min",
        "Ambient Temp Max (K)": "temp_max",
        "Ambient Baro Min (kPa)": "pressure_min",
        "Ambient Baro Max (kPa)": "pressure_max",
    }

    df = pd.read_csv(EDB_ENGINE_PATH)
    df = df.rename(columns=columns)
    # Convert ambient pressure from kPa to Pa
    df[["pressure_min", "pressure_max"]] = df[["pressure_min", "pressure_max"]] * 1000
    return dict(_row_to_edb_gaseous(tup) for tup in df.itertuples(index=False))


def _row_to_edb_gaseous(tup: Any) -> tuple[str, gaseous.EDBGaseous]:
    return tup.engine_uid, gaseous.EDBGaseous(
        **{k.name: getattr(tup, k.name) for k in dataclasses.fields(gaseous.EDBGaseous)}
    )


@functools.cache
def load_edb_nvpm_database() -> dict[str, nvpm.EDBnvpm]:
    """Read EDB file into a dictionary of the form ``{engine_uid: npvm_data}``.

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
        # Fuel mass flow rate
        "Fuel Flow Idle (kg/sec)": "ff_7",
        "Fuel Flow App (kg/sec)": "ff_30",
        "Fuel Flow C/O (kg/sec)": "ff_85",
        "Fuel Flow T/O (kg/sec)": "ff_100",
        # System loss corrected nvPM mass EI
        "nvPM EImass_SL Idle (mg/kg)": "nvpm_ei_m_7",
        "nvPM EImass_SL App (mg/kg)": "nvpm_ei_m_30",
        "nvPM EImass_SL C/O (mg/kg)": "nvpm_ei_m_85",
        "nvPM EImass_SL T/O (mg/kg)": "nvpm_ei_m_100",
        # System loss corrected nvPM number EI
        "nvPM EInum_SL Idle (#/kg)": "nvpm_ei_n_7",
        "nvPM EInum_SL App (#/kg)": "nvpm_ei_n_30",
        "nvPM EInum_SL C/O (#/kg)": "nvpm_ei_n_85",
        "nvPM EInum_SL T/O (#/kg)": "nvpm_ei_n_100",
        # Variables required to use fifth nvPM data point for MEEM2
        "max_nvpm_ei_m_between_30_85": "nvpm_ei_m_use_max",
        "nvPM EImass App (mg/kg)": "nvpm_ei_m_no_sl_30",
        "nvPM EImass C/O (mg/kg)": "nvpm_ei_m_no_sl_85",
        "nvPM EImass Max (mg/kg)": "nvpm_ei_m_no_sl_max",
        "max_nvpm_ei_n_between_30_85": "nvpm_ei_n_use_max",
        "nvPM EInum App (#/kg)": "nvpm_ei_n_no_sl_30",
        "nvPM EInum C/O (#/kg)": "nvpm_ei_n_no_sl_85",
        "nvPM EInum Max (#/kg)": "nvpm_ei_n_no_sl_max",
    }

    df = pd.read_csv(EDB_NVPM_PATH)
    df = df.rename(columns=columns)
    df = df.astype({"nvpm_ei_m_use_max": bool, "nvpm_ei_n_use_max": bool})
    return dict(_row_to_edb_nvpm(tup) for tup in df.itertuples(index=False))


def _row_to_edb_nvpm(tup: Any) -> tuple[str, nvpm.EDBnvpm]:
    return tup.engine_uid, nvpm.EDBnvpm(
        **{k.name: getattr(tup, k.name) for k in dataclasses.fields(nvpm.EDBnvpm)}
    )


@functools.cache
def load_default_aircraft_engine_mapping() -> pd.DataFrame:
    """Read default aircraft type -> engine UID assignments.

    Returns
    -------
    pd.DataFrame
        A :class:`pd.DataFrame` whose index is available aircraft types with columns:

        - engine_uid
        - engine_name
        - n-engines
    """
    return pd.read_csv(ENGINE_UID_PATH, index_col=0)
