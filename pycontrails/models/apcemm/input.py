"""Thin python wrapper over native APCEMM interface."""

from __future__ import annotations

import dataclasses
import os

import numpy as np

#: Assumed default APCEMM root directory
APCEMM_DEFAULT_ROOT = os.path.expanduser("~/APCEMM")


@dataclasses.dataclass
class APCEMMInput:
    """Parameters for APCEMM input YAML file.

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
