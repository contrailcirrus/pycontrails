"""Thin python wrapper over native APCEMM interface."""

from __future__ import annotations

import dataclasses
import os
import pathlib

import numpy as np

#: Assumed default APCEMM root directory
APCEMM_DEFAULT_ROOT = os.path.expanduser("~/APCEMM")


@dataclasses.dataclass
class APCEMMInput:
    """Parameters for APCEMM input YAML file.

    Physical parameters in this class are defined using MKS units, and units are converted
    as required at the time of YAML file generation.

    Parameters without default values must be provided each time a YAML file is generated.
    :class:`APCEMM` provides a convenient interface for determining parameter values
    based in flight and meterology data.

    Parameters with default values are set to sensible defaults, based in part on
    values provided in example input YAML files in the
    `APCEMM GitHub repository <https://github.com/MIT-LAE/APCEMM>`__, but users may
    with to override default parameters when configuring APCEMM simulations.

    This class exposes many but not all of the parameters in APCEMM YAML files. To request
    that additional parameters be exposed,
    `open an issue on GitHub <https://github.com/contrailcirrus/pycontrails/issues>`__
    and provide:
    - a short description of each parameter (suitable for a docstring), and
    - a sensible default value for each parameter.
    """

    # =================================
    # Parameters without default values
    # =================================

    #: Initial longitude [WGS84]
    longitude: float

    #: Initial latitude [WGS84]
    latitude: float

    #: Day of year at model initialization
    day_of_year: int

    #: Fractional hour of day at model initialization
    hour_of_day: float

    #: Initial pressure [:math:`Pa`]
    air_pressure: float

    #: Initial air temperature [:math:`K`]
    air_temperature: float

    #: Initial RH over liquid water [dimensionless]
    rhw: float

    #: Initial contrail-normal wind shear [:math:`1/s`]
    normal_shear: float

    #: Initial Brunt-Vaisala frequency [:math:`1/s`]
    brunt_vaisala_frequency: float

    #: Engine NOx emissions index [:math:`kg(NO2)/kg`]
    nox_ei: float

    #: Engine CO emissions index [:math:`kg/kg`]
    co_ei: float

    #: Engine unburned hydrocarbons emissions index [:math:`kg/kg`]
    hc_ei: float

    #: Engine SO2 emissions index [:math:`kg/kg`]
    so2_ei: float

    #: Engine soot emissions index [:math:`kg/kg`]
    nvpm_ei_m: float

    #: Emitted soot aerosol radius [:math:`m`]
    soot_radius: float

    #: Fuel flow [:math:`kg/s`]
    fuel_flow: float

    #: Aircraft mass [:math:`kg`]
    aircraft_mass: float

    #: Aircraft true airspeed [:math:`m/s`]
    true_airspeed: float

    #: Number of engines
    n_engine: int

    #: Wingspan [:math:`m`]
    wingspan: float

    #: Engine core exit temperature [:math:`K`]
    core_exit_temp: float

    #: Engine core exit area [:math:`m^2`]
    core_exit_area: float

    # ==============================
    # Parameters with default values
    # ==============================

    #: Number of APCEMM threads
    n_threads: int = 1

    #: Maximum APCEMM simulation time
    max_age: np.timedelta64 = np.timedelta64(20, "h")

    #: Output directory name (relative to APCEMM simulation directory)
    output_directory: pathlib.Path | str = "out"

    #: Overwrite existing output directories
    overwrite_output: bool = True

    #: Enable APCEMM netCDF outputs
    do_apcemm_nc_output: bool = True

    #: Indices of aerosol species to include in APCEMM netCDF output
    apcemm_nc_output_species: tuple[int] = (1,)

    #: APCEMM netCDF output frequency
    dt_apcemm_nc_output: np.timedelta64 = np.timedelta64(1, "m")

    #: Path to background conditions input file (distributed with APCEMM)
    input_background_conditions: pathlib.Path | str = os.path.join(
        APCEMM_DEFAULT_ROOT, "input_data", "init.txt"
    )

    #: Path to engine emissions input file (distributed with APCEMM)
    input_engine_emissions: pathlib.Path | str = os.path.join(
        APCEMM_DEFAULT_ROOT, "input_data", "ENG_EI.txt"
    )

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

    #: Horizontal diffusion coefficient [:math:`m^2/s`]
    horiz_diff: float = 15.0

    #: Vertical diffusion coefficient [:math:`m^2/s`]
    vert_diff: float = 0.15

    #: Engine SO2 to SO4 conversion factor [dimensionless]
    so2_to_so4_conversion: float = 0.02

    #: APCEMM transport timestep
    dt_apcemm_transport: np.timedelta64 = np.timedelta64(1, "m")

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
    input_met_file: pathlib.Path | str = "input.nc"

    #: Time step of input met data
    dt_input_met: np.timedelta64 = np.timedelta64(1, "h")

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
