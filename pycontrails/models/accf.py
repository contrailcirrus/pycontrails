"""Algorithmic Climate Change Functions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, overload

import xarray as xr

import pycontrails
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import (
    AirTemperature,
    EastwardWind,
    Geopotential,
    NorthwardWind,
    RelativeHumidity,
    SpecificHumidity,
)
from pycontrails.core.models import Model, ModelParams
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.datalib import ecmwf
from pycontrails.utils import dependencies


def wide_body_jets() -> set[str]:
    """Return a set of wide body jets."""
    return {
        "A332",
        "A333",
        "A338",
        "A339",
        "A342",
        "A343",
        "A345",
        "A356",
        "A359",
        "A388",
        "B762",
        "B763",
        "B764",
        "B772",
        "B773",
        "B778",
        "B779",
        "B788",
        "B789",
    }


def regional_jets() -> set[str]:
    """Return a set of regional jets."""
    return {
        "CRJ1",
        "CRJ2",
        "CRJ7",
        "CRJ8",
        "CRJ9",
        "CRJX",
        "E135",
        "E145",
        "E170",
        "E190",
    }


@dataclass
class ACCFParams(ModelParams):
    """Default ACCF model parameters.

    See `config-user.yml` definition at
    https://github.com/dlr-pa/climaccf
    """

    lat_bound: tuple[float, float] | None = None
    lon_bound: tuple[float, float] | None = None

    efficacy: bool = True
    efficacy_option: str = "lee_2021"

    accf_v: str = "V1.0"

    ch4_scaling: float = 1.0
    co2_scaling: float = 1.0
    cont_scaling: float = 1.0
    h2o_scaling: float = 1.0
    o3_scaling: float = 1.0

    forecast_step: float | None = None

    sep_ri_rw: bool = False

    climate_indicator: str = "ATR"

    #: The horizontal resolution of the meteorological data in degrees.
    #: If None, it will be inferred from the ``met`` dataset for :class:`MetDataset`
    #: source, otherwise it will be set to 0.5.
    horizontal_resolution: float | None = None

    emission_scenario: str = "pulse"

    time_horizon: int = 20

    pfca: str = "PCFA-ISSR"

    merged: bool = True

    #: RHI Threshold
    issr_rhi_threshold: float = 0.9
    issr_temp_threshold: float = 235

    sac_ei_h2o: float = 1.25
    sac_q: float = 43000000.0
    sac_eta: float = 0.3

    nox_ei: str = "TTV"

    PMO: bool = False

    unit_K_per_kg_fuel: bool = False


class ACCF(Model):
    """Compute Algorithmic Climate Change Functions (ACCF).

    This class is a wrapper over the DLR / UMadrid library
    `climaccf <https://github.com/dlr-pa/climaccf>`_,
    `DOI: 10.5281/zenodo.6977272 <https://doi.org/10.5281/zenodo.6977272>`_

    Parameters
    ----------
    met : MetDataset
        Dataset containing "air_temperature" and "specific_humidity" variables
    surface : MetDataset, optional
        Dataset containing "surface_solar_downward_radiation" and
        "top_net_thermal_radiation" variables

    References
    ----------
    - :cite:`dietmullerDlrpaClimaccfDataset2022`
    - :cite:`dietmullerPythonLibraryComputing2022`

    """

    name = "accr"
    long_name = "algorithmic climate change functions"
    met_variables = (
        AirTemperature,
        SpecificHumidity,
        ecmwf.PotentialVorticity,
        Geopotential,
        (RelativeHumidity, ecmwf.RelativeHumidity),
        NorthwardWind,
        EastwardWind,
    )
    sur_variables = (ecmwf.SurfaceSolarDownwardRadiation, ecmwf.TopNetThermalRadiation)
    default_params = ACCFParams

    # This variable won't get used since we are not writing the output
    # anywhere, but the library will complain if it's not defined
    path_lib = "./"

    def __init__(
        self,
        met: MetDataset,
        surface: MetDataset | None = None,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ) -> None:
        # Normalize ECMWF variables
        variables = self.ecmwf_met_variables()
        met = met.standardize_variables(variables)

        # If relative humidity is in percentage, convert to a proportion
        if met["relative_humidity"].attrs.get("units") == "%":
            met.data["relative_humidity"] /= 100.0
            met.data["relative_humidity"].attrs["units"] = "1"

        # Ignore humidity scaling warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pycontrails.core.models")
            super().__init__(met, params=params, **params_kwargs)

        if surface:
            surface = surface.copy()
            surface = surface.standardize_variables(self.sur_variables)
            surface.data = _rad_instantaneous_to_accumulated(surface.data)
            self.surface = surface

    @overload
    def eval(self, source: Flight, **params: Any) -> Flight: ...

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset: ...

    @overload
    def eval(self, source: MetDataset | None = ..., **params: Any) -> MetDataset: ...

    def eval(
        self, source: GeoVectorDataset | Flight | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | Flight | MetDataset:
        """Evaluate accfs along flight trajectory or on meteorology grid.

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
            Returns `np.nan` if interpolating outside meteorology grid.

        Raises
        ------
        NotImplementedError
            Raises if input ``source`` is not supported.
        """
        try:
            from climaccf.accf import GeTaCCFs
        except ModuleNotFoundError as e:
            dependencies.raise_module_not_found_error(
                name="ACCF.eval method",
                package_name="climaccf",
                module_not_found_error=e,
                pycontrails_optional_package="accf",
            )

        self.update_params(params)
        self.set_source(source)

        if isinstance(self.source, GeoVectorDataset):
            self.downselect_met()
            if hasattr(self, "surface"):
                self.surface = self.source.downselect_met(self.surface)

        if self.params["horizontal_resolution"] is None:
            if isinstance(self.source, MetDataset):
                # Overwrite horizontal resolution to match met
                longitude = self.source.data["longitude"].values
                latitude = self.source.data["latitude"].values
                if longitude.size > 1:
                    hres = abs(longitude[1] - longitude[0])
                    self.params["horizontal_resolution"] = float(hres)
                elif latitude.size > 1:
                    hres = abs(latitude[1] - latitude[0])
                    self.params["horizontal_resolution"] = float(hres)
                else:
                    self.params["horizontal_resolution"] = 0.5
            else:
                self.params["horizontal_resolution"] = 0.5

        p_settings = _get_accf_config(self.params)

        self.set_source_met()
        self._generate_weather_store(p_settings)

        # check aircraft type and set in config if needed
        if self.params["nox_ei"] != "TTV":
            if isinstance(self.source, Flight):
                ac = self.source.attrs["aircraft_type"]
                if ac in wide_body_jets():
                    p_settings["ac_type"] = "wide-body"
                elif ac in regional_jets():
                    p_settings["ac_type"] = "regional"
                else:
                    p_settings["ac_type"] = "single-aisle"
            else:
                p_settings["ac_type"] = "wide-body"

        clim_imp = GeTaCCFs(self)
        clim_imp.get_accfs(**p_settings)
        aCCFs, _ = clim_imp.get_xarray()

        # assign ACCF outputs to source
        skip = {
            v[0].short_name if isinstance(v, tuple) else v.short_name
            for v in (*self.met_variables, *self.sur_variables)
        }
        maCCFs = MetDataset(aCCFs)
        for key, arr in maCCFs.data.items():
            # skip met variables
            if key in skip:
                continue

            assert isinstance(key, str)
            if isinstance(self.source, GeoVectorDataset):
                self.source[key] = self.source.intersect_met(maCCFs[key])
            else:
                self.source[key] = arr

        self.transfer_met_source_attrs()
        self.source.attrs["pycontrails_version"] = pycontrails.__version__

        return self.source

    def _generate_weather_store(self, p_settings: dict[str, Any]) -> None:
        from climaccf.weather_store import WeatherStore

        # The library does not call the coordinates by name, it just slices the
        # underlying data array, so we need to put them in the expected order.
        # It also needs variables to have the ECMWF short name
        if isinstance(self.met, MetDataset):
            ds_met = self.met.data.transpose("time", "level", "latitude", "longitude")
            name_dict = {
                v[0].standard_name if isinstance(v, tuple) else v.standard_name: v[0].short_name
                if isinstance(v, tuple)
                else v.short_name
                for v in self.met_variables
            }
            ds_met = ds_met.rename(name_dict)
        else:
            ds_met = None

        if hasattr(self, "surface"):
            ds_sur = self.surface.data.squeeze().transpose("time", "latitude", "longitude")
            name_dict = {v.standard_name: v.short_name for v in self.sur_variables}
            ds_sur = ds_sur.rename(name_dict)
        else:
            ds_sur = None

        ws = WeatherStore(
            ds_met,
            ds_sur,
            ll_resolution=p_settings["horizontal_resolution"],
            forecast_step=p_settings["forecast_step"],
        )

        if p_settings["lat_bound"] and p_settings["lon_bound"]:
            ws.reduce_domain(
                {
                    "latitude": p_settings["lat_bound"],
                    "longitude": p_settings["lon_bound"],
                }
            )

        self.ds = ws.get_xarray()
        self.variable_names = ws.variable_names
        self.pre_variable_names = ws.pre_variable_names
        self.coordinate_names = ws.coordinate_names
        self.pre_coordinate_names = ws.pre_coordinate_names
        self.coordinates_bool = ws.coordinates_bool
        self.aCCF_bool = ws.aCCF_bool
        self.axes = ws.axes
        self.var_xr = ws.var_xr


def _get_accf_config(params: dict[str, Any]) -> dict[str, Any]:
    # a good portion of these will get ignored since we are not producing an
    # output file, but the library will complain if they aren't defined
    return {
        "lat_bound": params["lat_bound"],
        "lon_bound": params["lon_bound"],
        "time_bound": None,
        "horizontal_resolution": params["horizontal_resolution"],
        "forecast_step": params["forecast_step"],
        "NOx_aCCF": True,
        "NOx_EI&F_km": params["nox_ei"],
        "output_format": "netCDF",
        "mean": False,
        "std": False,
        "merged": params["merged"],
        "aCCF-V": params["accf_v"],
        "efficacy": params["efficacy"],
        "efficacy-option": params["efficacy_option"],
        "emission_scenario": params["emission_scenario"],
        "climate_indicator": params["climate_indicator"],
        "TimeHorizon": params["time_horizon"],
        "ac_type": "wide-body",
        "sep_ri_rw": params["sep_ri_rw"],
        "PMO": params["PMO"],
        "aCCF-scalingF": {
            "CH4": params["ch4_scaling"],
            "CO2": params["co2_scaling"],
            "Cont.": params["cont_scaling"],
            "H2O": params["h2o_scaling"],
            "O3": params["o3_scaling"],
        },
        "unit_K/kg(fuel)": params["unit_K_per_kg_fuel"],
        "PCFA": params["pfca"],
        "PCFA-ISSR": {
            "rhi_threshold": params["issr_rhi_threshold"],
            "temp_threshold": params["issr_temp_threshold"],
        },
        "PCFA-SAC": {
            "EI_H2O": params["sac_ei_h2o"],
            "Q": params["sac_q"],
            "eta": params["sac_eta"],
        },
        "Chotspots": False,
        "hotspots_binary": True,
        "color": "Reds",
        "geojson": False,
        "save_path": "./",
        "save_format": "netCDF",
    }


def _rad_instantaneous_to_accumulated(ds: xr.Dataset) -> xr.Dataset:
    """Convert instantaneous radiation to accumulated radiation."""

    for name, da in ds.items():
        try:
            unit = da.attrs["units"]
        except KeyError as e:
            msg = (
                f"Radiation data contains '{name}' variable "
                "but units are not specified. Provide units in the "
                f"rad['{name}'].attrs passed into ACCF."
            )
            raise KeyError(msg) from e

        if unit == "J m**-2":
            continue
        if unit != "W m**-2":
            msg = f"Unexpected units '{unit}' for '{name}'. Expected 'J m**-2' or 'W m**-2'."
            raise ValueError(msg)

        # Convert from W m**-2 to J m**-2
        ds[name] = da.assign_attrs(units="J m**-2") * 3600.0

    return ds
