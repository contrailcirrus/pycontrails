"""Thin python wrapper over native APCEMM interface."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
from typing import Any

import numpy as np
import xarray as xr

from pycontrails.core import GeoVectorDataset, MetDataset, models
from pycontrails.models.apcemm import utils
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.physics import constants, thermo

_path_to_static = pathlib.Path(__file__).parent / "static"
YAML_TEMPLATE = _path_to_static / "apcemm_yaml_template.yaml"

#: Assumed default APCEMM root directory
APCEMM_DEFAULT_ROOT = os.path.expanduser("~/APCEMM")


@dataclasses.dataclass
class APCEMMYaml:
    """APCEMM YAML file generation.

    Physical parameters in this class are defined using MKS units, and units are converted
    as required at the time of YAML file generation.

    This class exposes many but not all of the parameters in APCEMM YAML files. To request
    that additional parameters be exposed,
    `open an issue on GitHub <https://github.com/contrailcirrus/pycontrails/issues>`__
    and provide:
    - a short description of each parameter (suitable for a docstring), and
    - a sensible default value for each parameter.
    """

    #: Number of APCEMM threads
    n_threads: int = 1

    #: Output directory name
    output_directory: str = "out"

    #: Overwrite existing output directories
    overwrite_output: bool = True

    #: Enable APCEMM netCDF outputs
    do_apcemm_nc_output: bool = True

    #: Indices of aerosol species to include in APCEMM netCDF output
    apcemm_nc_output_species: tuple[int] = (1,)

    #: APCEMM netCDF output frequency
    dt_apcemm_nc_output: np.timedelta64 = np.timedelta64(1, "m")

    #: Path to background conditions input file (distributed with APCEMM)
    input_background_conditions: str = os.path.join(APCEMM_DEFAULT_ROOT, "input_data", "init.txt")

    #: Path to engine emissions input file (distributed with APCEMM)
    input_engine_emissions: str = os.path.join(APCEMM_DEFAULT_ROOT, "input_data", "ENG_EI.txt")

    #: Maximum APCEMM simulation time
    max_age: np.timedelta64 = np.timedelta64(20, "h")

    #: Initial longitude [WGS84]
    longitude: float = 0.0

    #: Initial latitude [WGS84]
    latitude: float = 0.0

    #: Day of year at model initialization
    day_of_year: int = 1

    #: Fractional hour of day at model initialization
    hour_of_day: float = 0.0

    #: Background NOx volume mixing ratio [nondim]
    nox_vmr: float = 5100e-12

    #: Background HNO3 volume mixing ratio [nondim]
    hno3_vmr: float = 81.5e-12

    #: Background O3 volume mixing ratio [nondim]
    o3_vmr: float = 100e-9

    #: Background CO volume mixing ratio [nondim]
    co_vmr: float = 40e-9

    #: Background CH4 volume mixing ratio [nondim]
    ch4_vmr: float = 1.76e-6

    #: Background SO2 volume mixing ratio [nondim]
    so2_vmr: float = 7.25e-12

    #: Initial pressure [:math:`Pa`]
    air_pressure: float = 24_000.0

    #: Initial air temperature [:math:`K`]
    air_temperature: float = 220.0

    #: Initial RH over liquid water [dimensionless]
    rhw: float = 0.85

    #: Horizontal diffusion coefficient [:math:`m^2/s`]
    horiz_diff: float = 15.0

    #: Vertical diffusion coefficient [:math:`m^2/s`]
    vert_diff: float = 0.15

    #: Initial contrail-normal wind shear [:math:`1/s`]
    normal_shear: float = 0.0015

    #: Initial Brunt-Vaisala frequency [:math:`1/s`]
    brunt_vaisala_frequency: float = 0.008

    #: APCEMM transport timestep
    dt_apcemm_transport: np.timedelta64 = np.timedelta64(1, "m")

    #: Engine NOx emissions index [:math:`kg(NO2)/kg`]
    nox_ei: float = 10.0e-3

    #: Engine CO emissions index [:math:`kg/kg`]
    co_ei = 0.3e-3

    #: Engine unburned hydrocarbons emissions index [:math:`kg/kg`]
    hc_ei = 0.04e-3

    #: Engine SO2 emissions index [:math:`kg/kg`]
    so2_ei = 1.0e-3

    #: Engine SO2 to SO4 conversion factor [dimensionless]
    so2_to_so4_conversion: float = 0.02

    #: Engine soot emissions index [:math:`kg/kg`]
    nvpm_ei_m = 0.05e-3

    #: Emitted soot aerosol radius [:math:`m`]
    soot_radius: float = 1.7e-8

    #: Fuel flow [:math:`kg/s`]
    fuel_flow: float = 0.7

    #: Aircraft mass [:math:`kg`]
    aircraft_mass: float = 60_000.0

    #: Aircraft true airspeed [:math:`m/s`]
    true_airspeed: float = 240.0

    #: Number of engines
    n_engine: int = 2

    #: Wingspan [:math:`m`]
    wingspan: float = 35.0

    #: Engine exhaust exit temperature [:math:`K`]
    exhaust_exit_temp: float = 550.0

    #: Engine bypass area [:math:`m^2`]
    bypass_area: float = 1.0

    #: Enable gravitational settling
    do_gravitational_setting: bool = True

    #: Enable solid coagulation
    do_solid_coagulation: bool = True

    #: Enable liquid coagulation
    do_liquid_coagulation: bool = False

    #: Enable ice growth
    do_ice_growth: bool = True

    #: APCEMM coagulation timestep
    dt_apcemm_coagulation: np.timedelta64 = np.timedelta64(1, "m")

    #: APCEMM ice growth timestep
    dt_apcemm_ice_growth: np.timedelta64 = np.timedelta64(1, "m")

    #: Path to meteorology input
    input_met_file: str = "input.nc"

    #: Time step of input met data
    dt_input_met: np.timedelta64 = np.timedelta64(1, "h")

    #: If True, include plume transport by vertical winds.
    #: Currently produces strange behavior in APCEMM output,
    #: so default setting (False) is recommended.
    vertical_advection: bool = False

    #: Number of horizontal gridpoints
    nx: int = 200

    #: Number of vertical gridpoints
    ny: int = 180

    #: Initial distance to right edge of domain [:math:`m`]
    xlim_right: float = 1000.0

    #: Initial distance to left edge of domain [:math:`m`]
    xlim_left: float = 1000.0

    #: Initial distance to top of domain [:math:`m`]
    ylim_up: float = 300.0

    #: Initial distance to bottom of domain [:math:`m`]
    ylim_down: float = 1500.0

    #: Offset applied to initial contrail depth [:math:`m`]
    initial_contrail_depth_offset: float = 0.0

    #: Scale factor applied to initial contrail depth [dimensionless]
    initial_contrail_depth_scale_factor: float = 1.0

    #: Offset applied to initial contrail width [:math:`m`]
    initial_contrail_width_offset: float = 0.0

    #: Scale factor applied to initial contrail width [dimensionless]
    initial_contrail_width_scale_factor: float = 1.0

    def generate_yaml(self) -> str:
        """Generate YAML file from parameters."""

        with open(YAML_TEMPLATE) as f:
            template = f.read()

        return template.format(
            n_threads_int=self.n_threads,
            output_folder_str=self.output_directory,
            overwrite_output_bool=_yaml_bool(self.overwrite_output),
            input_background_conditions_str=self.input_background_conditions,
            input_engine_emissions_str=self.input_engine_emissions,
            max_age_hr=self.max_age / np.timedelta64(1, "h"),
            temperature_k=self.air_temperature,
            rhw_percent=self.rhw * 100,
            horiz_diff_coeff_m2_per_s=self.horiz_diff,
            vert_diff_coeff_m2_per_s=self.vert_diff,
            pressure_hpa=self.air_pressure / 100,
            wind_shear_per_s=self.normal_shear,
            brunt_vaisala_frequency_per_s=self.brunt_vaisala_frequency,
            longitude_deg=self.longitude,
            latitude_deg=self.latitude,
            emission_day_dayofyear=self.day_of_year,
            emission_time_hourofday=self.hour_of_day,
            nox_ppt=self.nox_vmr * 1e12,
            hno3_ppt=self.hno3_vmr * 1e12,
            o3_ppb=self.o3_vmr * 1e9,
            co_ppb=self.co_vmr * 1e9,
            ch4_ppm=self.ch4_vmr * 1e6,
            so2_ppt=self.so2_vmr * 1e12,
            nox_g_per_kg=self.nox_ei * 1000,
            co_g_per_kg=self.co_ei * 1000,
            uhc_g_per_kg=self.hc_ei * 1000,
            so2_g_per_kg=self.so2_ei * 1000,
            so2_to_so4_conv_percent=self.so2_to_so4_conversion * 100,
            soot_g_per_kg=self.nvpm_ei_m * 1000,
            soot_radius_m=self.soot_radius,
            total_fuel_flow_kg_per_s=self.fuel_flow,
            aircraft_mass_kg=self.aircraft_mass,
            flight_speed_m_per_s=self.true_airspeed,
            num_of_engines_int=self.n_engine,
            wingspan_m=self.wingspan,
            core_exit_temp_k=self.exhaust_exit_temp,
            exit_bypass_area_m2=self.bypass_area,
            transport_timestep_min=self.dt_apcemm_transport / np.timedelta64(1, "m"),
            gravitational_settling_bool=_yaml_bool(self.do_gravitational_setting),
            solid_coagulation_bool=_yaml_bool(self.do_solid_coagulation),
            liquid_coagulation_bool=_yaml_bool(self.do_liquid_coagulation),
            ice_growth_bool=_yaml_bool(self.do_ice_growth),
            coag_timestep_min=self.dt_apcemm_coagulation / np.timedelta64(1, "m"),
            ice_growth_timestep_min=self.dt_apcemm_ice_growth / np.timedelta64(1, "m"),
            met_input_file_path_str=self.input_met_file,
            time_series_data_timestep_hr=self.dt_input_met / np.timedelta64(1, "h"),
            init_vert_veloc_from_met_data_bool=_yaml_bool(self.vertical_advection),
            vert_veloc_time_series_input_bool=_yaml_bool(self.vertical_advection),
            interpolate_vert_veloc_met_data_bool=_yaml_bool(self.vertical_advection),
            save_aerosol_timeseries_bool=_yaml_bool(self.do_apcemm_nc_output),
            aerosol_indices_list_int=",".join(str(i) for i in self.apcemm_nc_output_species),
            save_frequency_min=self.dt_apcemm_nc_output / np.timedelta64(1, "m"),
            nx_pos_int=self.nx,
            ny_pos_int=self.ny,
            xlim_right_pos_m=self.xlim_right,
            xlim_left_pos_m=self.xlim_left,
            ylim_up_pos_m=self.ylim_up,
            ylim_down_pos_m=self.ylim_down,
            base_contrail_depth_m=self.initial_contrail_depth_offset,
            contrail_depth_scaling_factor_nondim=self.initial_contrail_depth_scale_factor,
            base_contrail_width_m=self.initial_contrail_width_offset,
            contrail_width_scaling_factor_nondim=self.initial_contrail_width_scale_factor,
        )


def _yaml_bool(param: bool) -> str:
    """Convert boolean to T/F for YAML file."""
    return "T" if param else "F"


@dataclasses.dataclass
class APCEMMMet:
    """APCEMM meteorology dataset generation."""

    #: Trajectory time coordinates.
    #: Will be flattened before use if not 1-dimensional.
    time: np.ndarray

    #: Trajectory longitude coordinates [WGS84]
    #: Defines the longitude of the trajectory at each time
    #: and should have the same shape as :attr:`time`
    #: Will be flattened before use if not 1-dimensional.
    longitude: np.ndarray

    #: Trajectory latitude coordinates [WGS84]
    #: Defines the latitude of the trajectory at each time
    #: and should have the same shape as :attr:`time`
    #: Will be flattened before use if not 1-dimensional.
    latitude: np.ndarray

    #: Azimuth [:math:`\deg`]
    #: Defines the azimuth of a contrail segment at each point along
    #: the trajectory and should have the same shape as :attr:`time`.
    #: Note that the azimuth defined the orientation of the contrail
    #: segment, not the direction of the trajectory motion vector.
    #: Will be flattened before use if not 1-dimensional.
    azimuth: np.ndarray

    #: Pressure levels [:math:`Pa`]
    #: Defines the pressure levels at which meteorology will be generated.
    #: The same set of pressure levels is used at every point along the trajectory.
    #: Will be flattened before use if not 1-dimensional.
    air_pressure: np.ndarray

    #: Humidity scaling applied to specific humidity
    humidity_scaling: HumidityScaling | None = None

    #: Altitude difference used to approximate vertical derivatives [:math:`m`]
    dz_m: float = 200.0

    def generate_met_source(self) -> GeoVectorDataset:
        """Generate APCEMM meteorology dataset coordinates.

        Returns
        -------
        GeoVectorDataset
            Coordinates and segment azimuth at each pressure level and each point along
            the trajectory. Note that segment azimuth is defined at each pressure level
            but is a function of along-trajectory location only.
        """
        # Flatten
        time = self.time.ravel()
        longitude = self.longitude.ravel()
        latitude = self.latitude.ravel()
        azimuth = self.azimuth.ravel()
        level = self.air_pressure.ravel() / 1e2

        # Broadcast to required shape and create vector
        shape = (time.size, level.size)
        time = np.broadcast_to(time[:, np.newaxis], shape).ravel()
        longitude = np.broadcast_to(longitude[:, np.newaxis], shape).ravel()
        latitude = np.broadcast_to(latitude[:, np.newaxis], shape).ravel()
        azimuth = np.broadcast_to(azimuth[:, np.newaxis], shape).ravel()
        level = np.broadcast_to(level[np.newaxis, :], shape).ravel()
        vector = GeoVectorDataset(
            data={"azimuth": azimuth},
            longitude=longitude,
            latitude=latitude,
            level=level,
            time=time,
        )
        return vector

    def generate_met(
        self, vector: GeoVectorDataset, met: MetDataset, interp_kwargs: dict[str, Any]
    ) -> xr.Dataset:
        """Create xarray Dataset for APCEMM netCDF input file.

        Parameters
        ----------
        vector: GeoVectorDataset
            Vector
        """

        scale_humidity = self.humidity_scaling is not None and "specific_humidity" not in vector

        for met_key in (
            "air_temperature",
            "eastward_wind",
            "northward_wind",
            "specific_humidity",
            "lagrangian_tendency_of_air_pressure",
        ):
            models.interpolate_met(met, vector, met_key, **interp_kwargs)

        air_pressure_lower = thermo.pressure_dz(
            vector["air_temperature"], vector.air_pressure, self.dz_m
        )
        lower_level = air_pressure_lower / 100.0
        for met_key in ("eastward_wind", "northward_wind"):
            vector_key = f"{met_key}_lower"
            models.interpolate_met(
                met, vector, met_key, vector_key, **interp_kwargs, level=lower_level
            )

        if scale_humidity and self.humidity_scaling is not None:
            self.humidity_scaling.eval(vector, copy_source=False)

        vector["normal_shear"] = utils.normal_wind_shear(
            vector["eastward_wind"],
            vector["eastward_wind_lower"],
            vector["northward_wind"],
            vector["northward_wind_lower"],
            vector["azimuth"],
            self.dz_m,
        )

        nlev = met["air_pressure"].size
        ntime = len(vector) // nlev
        shape = (ntime, nlev)
        pressure = met["air_pressure"].values
        altitude = met["altitude"].values
        time = np.unique(vector["time"])
        time = (time - time[0]) / np.timedelta64(1, "h")
        temperature = vector["air_temperature"].reshape(shape).T
        qv = vector["specific_humidity"].reshape(shape).T
        rhi = (
            thermo.rhi(vector["specific_humidity"], vector["air_temperature"], vector.air_pressure)
            .reshape(shape)
            .T
        )
        shear = vector["normal_shear"].reshape(shape).T
        shear[-1, :] = shear[-2, :]  # lowest level will be nan
        omega = vector["lagrangian_tendency_of_air_pressure"].reshape(shape).T
        virtual_temperature = temperature * (1 + qv / constants.epsilon) / (1 + qv)
        density = pressure[:, np.newaxis] / (constants.R_d * virtual_temperature)
        w = -omega / (density * constants.g)

        return xr.Dataset(
            data_vars={
                "pressure": (("altitude",), pressure / 1e2, {"units": "hPa"}),
                "temperature": (("altitude", "time"), temperature, {"units": "K"}),
                "relative_humidity_ice": (("altitude", "time"), 1e2 * rhi, {"units": "percent"}),
                "shear": (("altitude", "time"), shear, {"units": "s**-1"}),
                "w": (("altitude", "time"), w, {"units": "m s**-1"}),
            },
            coords={
                "altitude": ("altitude", altitude / 1e3, {"units": "km"}),
                "time": ("time", time, {"units": "hours"}),
            },
        )


def run(apcemm: str, rundir: str, stdout_log: str, stderr_log: str) -> None:
    """
    Run APCEMM executable.

    Parameters
    ----------
    apcemm : str
        Path to APCEMM executable

    rundir : str
        Path to APCEMM simulation directory. This directory must at minimum
        contain an ``input.yaml`` file along with any other assets required
        for the simulation.

    stdout_log : str
        Path to file used to log APCEMM stdout

    stderr_log  : str
        Path to file used to log APCEMM stderr

    Raises
    ------
    ChildProcessError
        APCEMM exits with a non-zero return code.
    """

    with open(stdout_log, "w") as stdout, open(stderr_log, "w") as stderr:
        result = subprocess.run(
            [apcemm, "input.yaml"], stdout=stdout, stderr=stderr, cwd=rundir, check=False
        )
        if result.returncode != 0:
            msg = (
                f"APCEMM simulation in {rundir} "
                f"exited with return code {result.returncode}. "
                f"Check logs at {stdout_log} and {stderr_log}."
            )
            raise ChildProcessError(msg)
