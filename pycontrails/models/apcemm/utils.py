"""APCEMM interface utility functions."""

from __future__ import annotations

import pathlib
import subprocess
from typing import Any

import numpy as np
import xarray as xr

from pycontrails.core import GeoVectorDataset, MetDataset, met_var, models
from pycontrails.models.apcemm.inputs import APCEMMInput
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.physics import constants, thermo, units
from pycontrails.utils.types import ArrayScalarLike

_path_to_static = pathlib.Path(__file__).parent / "static"
YAML_TEMPLATE = _path_to_static / "apcemm_yaml_template.yaml"


def generate_apcemm_input_yaml(params: APCEMMInput) -> str:
    """Generate YAML file from  APCEMM input parameters.

    Parameters
    ----------
    params : APCEMMInput
        :class:`APCEMMInput` instance with parameters for input YAML file.

    Return
    ------
    str
        Contents of input YAML file generated from :param:`params`.
    """

    with open(YAML_TEMPLATE) as f:
        template = f.read()

    return template.format(
        n_threads_int=params.n_threads,
        output_folder_str=params.output_directory,
        overwrite_output_bool=_yaml_bool(params.overwrite_output),
        input_background_conditions_str=params.input_background_conditions,
        input_engine_emissions_str=params.input_engine_emissions,
        max_age_hr=params.max_age / np.timedelta64(1, "h"),
        temperature_k=params.air_temperature,
        rhw_percent=params.rhw * 100,
        horiz_diff_coeff_m2_per_s=params.horiz_diff,
        vert_diff_coeff_m2_per_s=params.vert_diff,
        pressure_hpa=params.air_pressure / 100,
        wind_shear_per_s=params.normal_shear,
        brunt_vaisala_frequency_per_s=params.brunt_vaisala_frequency,
        longitude_deg=params.longitude,
        latitude_deg=params.latitude,
        emission_day_dayofyear=params.day_of_year,
        emission_time_hourofday=params.hour_of_day,
        nox_ppt=params.nox_vmr * 1e12,
        hno3_ppt=params.hno3_vmr * 1e12,
        o3_ppb=params.o3_vmr * 1e9,
        co_ppb=params.co_vmr * 1e9,
        ch4_ppm=params.ch4_vmr * 1e6,
        so2_ppt=params.so2_vmr * 1e12,
        nox_g_per_kg=params.nox_ei * 1000,
        co_g_per_kg=params.co_ei * 1000,
        uhc_g_per_kg=params.hc_ei * 1000,
        so2_g_per_kg=params.so2_ei * 1000,
        so2_to_so4_conv_percent=params.so2_to_so4_conversion * 100,
        soot_g_per_kg=params.nvpm_ei_m * 1000,
        soot_radius_m=params.soot_radius,
        total_fuel_flow_kg_per_s=params.fuel_flow,
        aircraft_mass_kg=params.aircraft_mass,
        flight_speed_m_per_s=params.true_airspeed,
        num_of_engines_int=params.n_engine,
        wingspan_m=params.wingspan,
        core_exit_temp_k=params.core_exit_temp,
        exit_bypass_area_m2=params.core_exit_area,
        transport_timestep_min=params.dt_apcemm_transport / np.timedelta64(1, "m"),
        gravitational_settling_bool=_yaml_bool(params.do_gravitational_setting),
        solid_coagulation_bool=_yaml_bool(params.do_solid_coagulation),
        liquid_coagulation_bool=_yaml_bool(params.do_liquid_coagulation),
        ice_growth_bool=_yaml_bool(params.do_ice_growth),
        coag_timestep_min=params.dt_apcemm_coagulation / np.timedelta64(1, "m"),
        ice_growth_timestep_min=params.dt_apcemm_ice_growth / np.timedelta64(1, "m"),
        met_input_file_path_str=params.input_met_file,
        time_series_data_timestep_hr=params.dt_input_met / np.timedelta64(1, "h"),
        save_aerosol_timeseries_bool=_yaml_bool(params.do_apcemm_nc_output),
        aerosol_indices_list_int=",".join(str(i) for i in params.apcemm_nc_output_species),
        save_frequency_min=params.dt_apcemm_nc_output / np.timedelta64(1, "m"),
        nx_pos_int=params.nx,
        ny_pos_int=params.ny,
        xlim_right_pos_m=params.xlim_right,
        xlim_left_pos_m=params.xlim_left,
        ylim_up_pos_m=params.ylim_up,
        ylim_down_pos_m=params.ylim_down,
        base_contrail_depth_m=params.initial_contrail_depth_offset,
        contrail_depth_scaling_factor_nondim=params.initial_contrail_depth_scale_factor,
        base_contrail_width_m=params.initial_contrail_width_offset,
        contrail_width_scaling_factor_nondim=params.initial_contrail_width_scale_factor,
    )


def _yaml_bool(param: bool) -> str:
    """Convert boolean to T/F for YAML file."""
    return "T" if param else "F"


def generate_apcemm_input_met(
    time: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    azimuth: np.ndarray,
    altitude: np.ndarray,
    met: MetDataset,
    humidity_scaling: HumidityScaling,
    dz_m: float,
    interp_kwargs: dict[str, Any],
) -> xr.Dataset:
    r"""Create xarray Dataset for APCEMM meteorology netCDF file.

    This dataset contains a sequence of atmospheric profiles along the
    Lagrangian trajectory of an advected flight segment. The along-trajectory
    dimension is parameterized by time (rather than latitude and longitude),
    so the dataset coordinates are air pressure and time.

    Parameters
    ----------
    time : np.ndarray
        Time coordinates along the Lagrangian trajectory of the advected flight segment.
        Values must be coercible to ``np.datetime64`` by :class:`GeoVectorDataset`.
        Will be flattened before use if not 1-dimensional.
    longitude : np.ndarray
        Longitude [WGS84] along the Lagrangian trajectory of the advected flight segment.
        Defines the longitude of the trajectory at each time and should have the
        same shape as :param:`time`
        Will be flattened before use if not 1-dimensional.
    latitude : np.ndarray
        Latitude [WGS84] along the Lagrangian trajectory of the advected flight segment.
        Defines the longitude of the trajectory at each time and should have the
        same shape as :param:`time`
        Will be flattened before use if not 1-dimensional.
    azimuth : np.ndarray
        Azimuth [:math:`\deg`] of the advected flight segment at each point along its
        Lagrangian trajectory. Note that the azimuth defines the orientation of the
        advected segment itself, and not the direction in which advection is transporting
        the segment. The azimuth is used to convert horizontal winds into segment-normal
        wind shear. Must have the same shape as :param:`time`.
        Will be flattened before use if not 1-dimensional.
    altitude : np.ndarray
        Defines altitudes [:math:`m`] on which atmospheric profiles are computed.
        Profiles are defined using the same set of altitudes at every point
        along the Lagrangian trajectory of the advected flight segment. Note that
        this parameter does not have to have the same shape as :param:`time`.
    met : MetDataset
        Meteorology used to generate the sequence of atmospheric profiles. Must contain:
        - air temperature [:math:`K`]
        - specific humidity [:math:`kg/kg`]
        - geopotential height [:math:`m`]
        - eastward wind [:math:`m/s`]
        - northward wind [:math:`m/s`]
        - vertical velocity [:math:`Pa/s`]
    humidity_scaling : HumidityScaling
        Humidity scaling applied to specific humidity in :param:`met` before
        generating atmospheric profiles.
    dz_m : float
        Altitude difference [:math:`m`] used to approximate vertical derivatives
        when computing wind shear.

    Returns
    -------
    xr.Dataset
        Meteorology dataset in required format for APCEMM input.
    """

    # Ensure that altitudes are sorted ascending
    altitude = np.sort(altitude)

    # Check for required fields in met
    vars = (
        met_var.AirTemperature,
        met_var.SpecificHumidity,
        met_var.GeopotentialHeight,
        met_var.EastwardWind,
        met_var.NorthwardWind,
        met_var.VerticalVelocity,
    )
    met.ensure_vars(vars)
    met.standardize_variables(vars)

    # Flatten input arrays
    time = time.ravel()
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    azimuth = azimuth.ravel()
    altitude = altitude.ravel()

    # Estimate pressure levels close to target altitudes
    # (not exact because this assumes the ISA temperature profile)
    pressure = units.m_to_pl(altitude) * 1e2

    # Broadcast to required shape and create vector for initial interpolation
    # onto original pressure levels at target horizontal location.
    shape = (time.size, altitude.size)
    time = np.broadcast_to(time[:, np.newaxis], shape).ravel()
    longitude = np.broadcast_to(longitude[:, np.newaxis], shape).ravel()
    latitude = np.broadcast_to(latitude[:, np.newaxis], shape).ravel()
    azimuth = np.broadcast_to(azimuth[:, np.newaxis], shape).ravel()
    level = np.broadcast_to(pressure[np.newaxis, :] / 1e2, shape).ravel()
    vector = GeoVectorDataset(
        data={"azimuth": azimuth},
        longitude=longitude,
        latitude=latitude,
        level=level,
        time=time,
    )

    # Downselect met before interpolation
    met = vector.downselect_met(met)

    # Interpolate meteorology data onto vector
    scale_humidity = humidity_scaling is not None and "specific_humidity" not in vector
    for met_key in (
        "air_temperature",
        "eastward_wind",
        "geopotential_height",
        "northward_wind",
        "specific_humidity",
        "lagrangian_tendency_of_air_pressure",
    ):
        models.interpolate_met(met, vector, met_key, **interp_kwargs)

    # Interpolate winds at lower level for shear calculation
    air_pressure_lower = thermo.pressure_dz(vector["air_temperature"], vector.air_pressure, dz_m)
    lower_level = air_pressure_lower / 100.0
    for met_key in ("eastward_wind", "northward_wind"):
        vector_key = f"{met_key}_lower"
        models.interpolate_met(met, vector, met_key, vector_key, **interp_kwargs, level=lower_level)

    # Apply humidity scaling
    if scale_humidity and humidity_scaling is not None:
        humidity_scaling.eval(vector, copy_source=False)

    # Compute RHi and segment-normal shear
    vector.setdefault(
        "rhi",
        thermo.rhi(vector["specific_humidity"], vector["air_temperature"], vector.air_pressure),
    )
    vector.setdefault(
        "normal_shear",
        normal_wind_shear(
            vector["eastward_wind"],
            vector["eastward_wind_lower"],
            vector["northward_wind"],
            vector["northward_wind_lower"],
            vector["azimuth"],
            dz_m,
        ),
    )

    # Reshape interpolated fields to (time, level).
    nlev = altitude.size
    ntime = len(vector) // nlev
    shape = (ntime, nlev)
    time = np.unique(vector["time"])
    time = (time - time[0]) / np.timedelta64(1, "h")
    temperature = vector["air_temperature"].reshape(shape)
    qv = vector["specific_humidity"].reshape(shape)
    z = vector["geopotential_height"].reshape(shape)
    rhi = vector["rhi"].reshape(shape)
    shear = vector["normal_shear"].reshape(shape)
    shear[:, -1] = shear[:, -2]  # lowest level will be nan
    omega = vector["lagrangian_tendency_of_air_pressure"].reshape(shape)
    virtual_temperature = temperature * (1 + qv / constants.epsilon) / (1 + qv)
    density = pressure[np.newaxis, :] / (constants.R_d * virtual_temperature)
    w = -omega / (density * constants.g)

    # Interpolate fields to target altitudes profile-by-profile
    # to obtain 2D arrays with dimensions (time, altitude).
    temperature_on_z = np.zeros(shape, dtype=temperature.dtype)
    rhi_on_z = np.zeros(shape, dtype=rhi.dtype)
    shear_on_z = np.zeros(shape, dtype=shear.dtype)
    w_on_z = np.zeros(shape, dtype=w.dtype)

    # Fields should already be on pressure levels close to target
    # altitudes, so this just uses linear interpolation and constant
    # extrapolation on fields expected by APCEMM.
    # NaNs are preserved at the start and end of interpolated profiles
    # but removed in interiors.
    def interp(z: np.ndarray, z0: np.ndarray, f0: np.ndarray) -> np.ndarray:
        # mask nans
        mask = np.isnan(z0) | np.isnan(f0)
        if np.all(mask):
            msg = (
                "Found all-NaN profile during APCEMM meterology input file creation. "
                "MetDataset may have insufficient spatiotemporal coverage."
            )
            raise ValueError(msg)
        z0 = z0[~mask]
        f0 = f0[~mask]

        # interpolate
        assert np.all(np.diff(z0) > 0)  # expect increasing altitudes
        fi = np.interp(z, z0, f0, left=f0[0], right=f0[-1])

        # restore nans at start and end of profile
        if mask[0]:  # nans at top of profile
            fi[z > z0.max()] = np.nan
        if mask[-1]:  # nans at end of profile
            fi[z < z0.min()] = np.nan
        return fi

    # The manual for loop is unlikely to be a bottleneck since a
    # substantial amount of work is done within each iteration.
    for i in range(ntime):
        temperature_on_z[i, :] = interp(altitude, z[i, :], temperature[i, :])
        rhi_on_z[i, :] = interp(altitude, z[i, :], rhi[i, :])
        shear_on_z[i, :] = interp(altitude, z[i, :], shear[i, :])
        w_on_z[i, :] = interp(altitude, z[i, :], w[i, :])

    # APCEMM also requires initial pressure profile
    pressure_on_z = interp(altitude, z[0, :], pressure)

    # Create APCEMM input dataset.
    # Transpose require because APCEMM expects (altitude, time) arrays.
    return xr.Dataset(
        data_vars={
            "pressure": (("altitude",), pressure_on_z.astype("float32") / 1e2, {"units": "hPa"}),
            "temperature": (
                ("altitude", "time"),
                temperature_on_z.astype("float32").T,
                {"units": "K"},
            ),
            "relative_humidity_ice": (
                ("altitude", "time"),
                1e2 * rhi_on_z.astype("float32").T,
                {"units": "percent"},
            ),
            "shear": (("altitude", "time"), shear_on_z.astype("float32").T, {"units": "s**-1"}),
            "w": (("altitude", "time"), w_on_z.astype("float32").T, {"units": "m s**-1"}),
        },
        coords={
            "altitude": ("altitude", altitude.astype("float32") / 1e3, {"units": "km"}),
            "time": ("time", time, {"units": "hours"}),
        },
    )


def run(
    apcemm_path: pathlib.Path | str, input_yaml: str, rundir: str, stdout_log: str, stderr_log: str
) -> None:
    """
    Run APCEMM executable.

    Parameters
    ----------
    apcemm_path : pathlib.Path | str
        Path to APCEMM executable.
    input_yaml : str
        Path to APCEMM input yaml file.
    rundir : str
        Path to APCEMM simulation directory.
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
            [apcemm_path, input_yaml], stdout=stdout, stderr=stderr, cwd=rundir, check=False
        )
        if result.returncode != 0:
            msg = (
                f"APCEMM simulation in {rundir} "
                f"exited with return code {result.returncode}. "
                f"Check logs at {stdout_log} and {stderr_log}."
            )
            raise ChildProcessError(msg)


def normal_wind_shear(
    u_hi: ArrayScalarLike,
    u_lo: ArrayScalarLike,
    v_hi: ArrayScalarLike,
    v_lo: ArrayScalarLike,
    azimuth: ArrayScalarLike,
    dz: float,
) -> ArrayScalarLike:
    r"""Compute segment-normal wind shear from wind speeds at lower and upper levels.

    Parameters
    ----------
    u_hi : ArrayScalarLike
        Eastward wind at upper level [:math:`m/s`]
    u_lo : ArrayScalarLike
        Eastward wind at lower level [:math:`m/s`]
    v_hi : ArrayScalarLike
        Northward wind at upper level [:math:`m/s`]
    v_lo : ArrayScalarLike
        Northward wind at lower level [:math:`m/s`]
    azimuth : ArrayScalarLike
        Segment azimuth [:math:`\deg`]
    dz : float
        Distance between upper and lower level [:math:`m`]

    Returns
    -------
    ArrayScalarLike
        Segment-normal wind shear [:math:`1/s`]
    """
    du_dz = (u_hi - u_lo) / dz
    dv_dz = (v_hi - v_lo) / dz
    az_radians = units.degrees_to_radians(azimuth)
    sin_az = np.sin(az_radians)
    cos_az = np.cos(az_radians)
    return sin_az * dv_dz - cos_az * du_dz


def soot_radius(
    nvpm_ei_m: ArrayScalarLike, nvpm_ei_n: ArrayScalarLike, rho_bc: float = 1770.0
) -> ArrayScalarLike:
    """Calculate mean soot radius from mass and number emissions indices.

    Parameters
    ----------
    nvpm_ei_m : ArrayScalarLike
        Soot mass emissions index [:math:`kg/kg`]
    nvpm_ei_n : ArrayScalarLike
        Soot number emissions index [:math:`1/kg`]
    rho_bc : float, optional
        Density of black carbon [:math:`kg/m^3`]. By default, 1770.
    """
    return ((3.0 * nvpm_ei_m) / (4.0 * np.pi * rho_bc * nvpm_ei_n)) ** (1.0 / 3.0)
