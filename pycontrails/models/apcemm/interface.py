"""Thin python wrapper over native APCEMM interface."""

from __future__ import annotations

import dataclasses
import os
import subprocess
from typing import Any

import numpy as np
import xarray as xr

from pycontrails.core import GeoVectorDataset, MetDataset, models
from pycontrails.models.apcemm import utils
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.physics import constants, thermo


@dataclasses.dataclass
class APCEMMYaml:
    """APCEMM YAML file generation.

    Physical parameters in this class are defined using MKS units, and units are converted
    as required at the time of YAML file generation.
    """

    #: Number of APCEMM threads
    n_threads: int = 1

    #: APCEMM root directory, required to find engine emissions database and
    #: atmospheric background conditions files distributed with APCEMM
    apcemm_root: str = os.path.expanduser("~/APCEMM")

    #: Maximum APCEMM simulation time
    max_age: np.timedelta64 = np.timedelta64(20, "h")

    #: Day of year at model initialization
    day_of_year: int = 1

    #: Fractional hour of day at model initialization
    hour_of_day: float = 0.0

    #: Initial longitude [WGS84]
    longitude: float = 0.0

    #: Initial latitude [WGS84]
    latitude: float = 0.0

    #: Initial pressure [:math:`Pa`]
    air_pressure: float = 24_000.0

    #: Initial air temperature [:math:`K`]
    air_temperature: float = 220.0

    #: Initial RH over liquid water [dimensionless]
    rhw: float = 0.85

    #: Initial contrail-normal wind shear [:math:`1/s`]
    normal_shear: float = 0.0015

    #: Initial Brunt-Vaisala frequency [:math:`1/s`]
    brunt_vaisala_frequency: float = 0.008

    #: Time step of input met data
    dt_input_met: np.timedelta64 = np.timedelta64(1, "h")

    #: Engine NOx emissions index [:math:`kg(NO2)/kg`]
    nox_ei: float = 10.0e-3

    #: Engine CO emissions index [:math:`kg/kg`]
    co_ei = 0.3e-3

    #: Engine unburned hydrocarbons emissions index [:math:`kg/kg`]
    hc_ei = 0.04e-3

    #: Engine SO2 emissions index [:math:`kg/kg`]
    so2_ei = 1.0e-3

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

    #: If True, include plume transport by vertical winds.
    #: APCEMM implementation of vertical advection is currently buggy,
    #: so default setting (False) is recommended.
    vertical_advection: bool = False

    def generate_yaml(self) -> str:
        """Generate YAML file from parameters."""

        return f"""SIMULATION MENU:
        
    OpenMP Num Threads (positive int): {self.n_threads}
    
    PARAM SWEEP SUBMENU:
        Parameter sweep (T/F): T
        Run Monte Carlo (T/F): F
        Num Monte Carlo runs (int): -1
    
    OUTPUT SUBMENU:
        Output folder (string): out
        Overwrite if folder exists (T/F): T
    
    Use threaded FFT (T/F): F
    
    FFTW WISDOM SUBMENU:
        Use FFTW WISDOM (T/F): F
        Dir w/ write permission (string): n/a
    
    Input background condition (string): "{os.path.join(
        self.apcemm_root, 'input_data', 'init.txt')}"
    Input engine emissions (string): "{os.path.join(
        self.apcemm_root, 'input_data', 'ENG_EI.txt')}"
    
    SAVE FORWARD RESULTS SUBMENU:
        Save forward results (T/F): F
        netCDF filename format (string): n/a
    
    ADJOINT OPTIMIZATION SUBMENU:
        Turn on adjoint optim. (T/F): F
        netCDF filename format (string): n/a
    
    BOX MODEL SUBMENU:
        Run box model (T/F): F
        netCDF filename format (string): n/a            
    
PARAMETER MENU:
    
    Plume Process [hr] (double): {self.max_age/np.timedelta64(1, "h")}
    
    METEOROLOGICAL PARAMETERS SUBMENU:
        Temperature [K] (double): {self.air_temperature}
        R.Hum. wrt water [%] (double): {1e2*self.rhw}
        Pressure [hPa] (double): {self.air_pressure/1e2}
        Horiz. diff. coeff. [m^2/s] (double): 15.0
        Verti. diff. [m^2/s] (double): 0.15
        Wind shear [1/s] (double): {self.normal_shear}
        Brunt-Vaisala Frequency [s^-1] (double): {self.brunt_vaisala_frequency}
    
    LOCATION AND TIME SUBMENU:
        LON [deg] (double): {self.longitude}
        LAT [deg] (double): {self.latitude}
        Emission day [1-365] (int): {self.day_of_year}
        Emission time [hr] (double): {self.hour_of_day}
           
    BACKGROUND MIXING RATIOS SUBMENU:
        NOx [ppt] (double): 5100
        HNO3 [ppt] (double): 81.5
        O3 [ppb] (double): 100
        CO [ppb] (double): 40
        CH4 [ppm] (double): 1.76
        SO2 [ppt] (double): 7.25
    
    EMISSION INDICES SUBMENU:
        NOx [g(NO2)/kg_fuel] (double): {1e3*self.nox_ei}
        CO [g/kg_fuel] (double): {1e3*self.co_ei}
        UHC [g/kg_fuel] (double): {1e3*self.hc_ei}
        SO2 [g/kg_fuel] (double): {1e3*self.so2_ei}
        SO2 to SO4 conv [%] (double): 2
        Soot [g/kg_fuel] (double): {1e3*self.nvpm_ei_m}
        
    Soot Radius [m] (double): {self.soot_radius}
    Total fuel flow [kg/s] (double): {self.fuel_flow}
    Aircraft mass [kg] (double): {self.aircraft_mass}
    Flight speed [m/s] (double): {self.true_airspeed}
    Num. of engines [2/4] (int): {self.n_engine}
    Wingspan [m] (double): {self.wingspan}
    Core exit temp. [K] (double): {self.exhaust_exit_temp}
    Exit bypass area [m^2] (double): {self.bypass_area}
    
TRANSPORT MENU:
    
    Turn on Transport (T/F): T
    Fill Negative Values (T/F): T
    Transport Timestep [min] (double): 1
    
    PLUME UPDRAFT SUBMENU:
        Turn on plume updraft (T/F): F
        Updraft timescale [s] (double): -1
        Updraft veloc. [cm/s] (double): -1
    
CHEMISTRY MENU:
    
    Turn on Chemistry (T/F): F
    Perform hetero. chem. (T/F): F
    Chemistry Timestep [min] (double): -1
    Photolysis rates folder (string): n/a
    
AEROSOL MENU:
    
    Turn on grav. settling (T/F): T
    Turn on solid coagulation (T/F): T
    Turn on liquid coagulation (T/F): F
    Coag. timestep [min] (double): 1
    Turn on ice growth (T/F): T
    Ice growth timestep [min] (double): 1
    
METEOROLOGY MENU:
    
    METEOROLOGICAL INPUT SUBMENU:
        Use met. input (T/F): T
        Met input file path (string): "input.nc"
        Time series data timestep [hr] (double): {self.dt_input_met/np.timedelta64(1, "h")}
        Init temp. from met. (T/F): T
        Temp. time series input (T/F): T
        Interpolate temp. met. data (T/F): T
        Init RH from met. (T/F): T
        RH time series input (T/F): T
        Interpolate RH met. data (T/F): T
        Init wind shear from met. (T/F): T
        Wind shear time series input (T/F): T
        Interpolate shear met. data (T/F): T
        Init vert. veloc. from met. data (T/F): {'T' if self.vertical_advection else 'F'}
        Vert. veloc. time series input (T/F): {'T' if self.vertical_advection else 'F'}
        Interpolate vert. veloc. met. data (T/F): {'T' if self.vertical_advection else 'F'}
    
        HUMIDITY SCALING OPTIONS:
            Humidity modification scheme (none / constant / scaling): none
            Constant RHi [%] (double): -1
            Humidity scaling constant a (double): -1
            Humidity scaling constant b (double): -1
    
    
    IMPOSE MOIST LAYER DEPTH SUBMENU:
        Impose moist layer depth (T/F): F
        Moist layer depth [m] (double): -1
        Subsaturated air RHi [%] (double): -1
    
    IMPOSE LAPSE RATE SUBMENU:
        Impose lapse rate (T/F): F
        Lapse rate [K/m] (T/F): -1
        
    Add diurnal variations (T/F): F
            
    TEMPERATURE PERTURBATION SUBMENU:
        Enable Temp. Pert. (T/F): F
        Temp. Perturb. Amplitude (double): -1
        Temp. Perturb. Timescale (min): -1
    
DIAGNOSTIC MENU:
    
    netCDF filename format (string): trac_avg.apcemm.hhmm
    
    SPECIES TIMESERIES SUBMENU:
        Save species timeseries (T/F): F
        Inst timeseries file (string): n/a
        Species indices to include (list of ints): -1
        Save frequency [min] (double): -1
    
    AEROSOL TIMESERIES SUBMENU:
        Save aerosol timeseries (T/F): T
        Inst timeseries file (string): ts_aerosol_hhmm.nc
        Aerosol indices to include (list of ints): 1
        Save frequency [min] (double): 1
    
    PRODUCTION & LOSS SUBMENU:
        Turn on P/L diag (T/F): F
        Save O3 P/L (T/F): F
    
ADVANCED OPTIONS MENU:
    
    GRID SUBMENU:
        NX (positive int): 200
        NY (positive int): 180
        XLIM_RIGHT (positive double): 1000
        XLIM_LEFT (positive double): 1000
        YLIM_UP (positive double): 300
        YLIM_DOWN (positive double): 1500
    INITIAL CONTRAIL SIZE SUBMENU:
        Base Contrail Depth [m] (double): 0.0
        Contrail Depth Scaling Factor [-] (double): 1.0
        Base Contrail Width [m] (double): 0.0
        Contrail Width Scaling Factor [-] (double): 1.0
    Ambient Lapse Rate [K/km] (double): -1
    Tropopause Pressure [Pa] (double): -1
"""


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
