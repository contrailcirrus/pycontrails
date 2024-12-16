"""Pycontrails :class:`Model` interface to APCEMM."""

from __future__ import annotations

import dataclasses
import glob
import logging
import multiprocessing as mp
import os
import pathlib
import shutil
from typing import Any, NoReturn, overload

import numpy as np
import pandas as pd

from pycontrails.core import cache, models
from pycontrails.core.aircraft_performance import AircraftPerformance
from pycontrails.core.flight import Flight
from pycontrails.core.fuel import Fuel, JetA
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import (
    AirTemperature,
    EastwardWind,
    Geopotential,
    GeopotentialHeight,
    NorthwardWind,
    SpecificHumidity,
    VerticalVelocity,
)
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.models.apcemm import utils
from pycontrails.models.apcemm.inputs import APCEMMInput
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.models.emissions import Emissions
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.models.ps_model import PSFlight
from pycontrails.physics import constants, geo, thermo

logger = logging.getLogger(__name__)

#: Minimum altitude


@dataclasses.dataclass
class APCEMMParams(models.ModelParams):
    """Default parameters for the pycontrails :class:`APCEMM` interface."""

    #: Maximum contrail age
    max_age: np.timedelta64 = np.timedelta64(20, "h")

    #: Longitude buffer for Lagrangian trajectory calculation [WGS84]
    met_longitude_buffer: tuple[float, float] = (10.0, 10.0)

    #: Latitude buffer for Lagrangian trajectory calculation [WGS84]
    met_latitude_buffer: tuple[float, float] = (10.0, 10.0)

    #: Level buffer for Lagrangian trajectory calculation [:math:`hPa`]
    met_level_buffer: tuple[float, float] = (40.0, 40.0)

    #: Timestep for Lagrangian trajectory calculation
    dt_lagrangian: np.timedelta64 = np.timedelta64(30, "m")

    #: Sedimentation rate for Lagrangian trajectories [:math:`Pa/s`]
    lagrangian_sedimentation_rate: float = 0.0

    #: Time step of meteorology in generated APCEMM input file.
    dt_input_met: np.timedelta64 = np.timedelta64(1, "h")

    #: Altitude coordinates [:math:`m`] for meteorology in generated APCEMM input file.
    #: If not provided, uses estimated altitudes for levels in input :class:`Metdataset`.
    altitude_input_met: list[float] | None = None

    #: Humidity scaling
    humidity_scaling: HumidityScaling | None = None

    #: Altitude difference for vertical derivative calculations [:math:`m`]
    dz_m: float = 200.0

    #: ICAO aircraft identifier
    aircraft_type: str = "B738"

    #: Engine UID. If not provided, uses the default engine UID
    #: for the :attr:`aircraft_type`.
    engine_uid: str | None = None

    #: Aircraft performance model
    aircraft_performance: AircraftPerformance = dataclasses.field(default_factory=PSFlight)

    #: Fuel type
    fuel: Fuel = dataclasses.field(default_factory=JetA)

    #: List of flight waypoints to simulate in APCEMM.
    #: By default, runs a simulation for every waypoint.
    waypoints: list[int] | None = None

    #: If defined, use to override ``input_background_conditions`` and
    #: ``input_engine_emissions`` in :class:`APCEMMInput` assuming that
    #: ``apcemm_root`` points to the root of the APCEMM git repository.
    apcemm_root: pathlib.Path | str | None = None

    #: If True, delete existing run directories before running APCEMM simulations.
    #: If False (default), raise an exception if a run directory already exists.
    overwrite: bool = False

    #: Name of output directory within run directory
    output_directory: str = "out"

    #: Number of threads to use within individual APCEMM simulations.
    apcemm_threads: int = 1

    #: Number of individual APCEMM simulations to run in parallel.
    n_jobs: int = 1


class APCEMM(models.Model):
    """Run APCEMM as a pycontrails :class:`Model`.

    This class acts as an adapter between the pycontrails :class:`Model` interface
    (shared with other contrail models) and APCEMM's native interface.

    `APCEMM <https://github.com/MIT-LAE/APCEMM>`__ was developed at the
    `MIT Laboratory for Aviation and the Environment <https://lae.mit.edu/>`__
    and is described in :cite:`fritzRolePlumescaleProcesses2020`.

    Parameters
    ----------
    met : MetDataset
        Pressure level dataset containing :attr:`met_variables` variables.
        See *Notes* for variable names by data source.
    apcemm_path : pathlib.Path | str
        Path to APCEMM executable. See *Notes* for information about
        acquiring and compiling APCEMM.
    apcemm_root : pathlib.Path | str, optional
        Path to APCEMM root directory, used to set ``input_background_conditions`` and
        ``input_engine_emissions`` based on the structure of the
        `APCEMM GitHub repository <https://github.com/MIT-LAE/APCEMM>`__.
        If not provided, pycontrails will use the default paths defined in :class:`APCEMMInput`.
    apcemm_input_params : APCEMMInput, optional
        Value for APCEMM input parameters defined in :class:`APCEMMInput`. If provided, values
        for ``input_background_condition`` or ``input_engine_emissions`` will override values
        set based on ``apcemm_root``. Attempting to provide values for input parameters
        that are determined automatically by this interface will result in an error.
        See *Notes* for detailed information about YAML file generation.
    cachestore : CacheStore, optional
        :class:`CacheStore` used to store APCEMM run directories.
        If not provided, uses a :class:`DiskCacheStore`.
        See *Notes* for detailed information about the file structure for APCEMM
        simulations.
    params : dict[str,Any], optional
        Override APCEMM model parameters with dictionary.
        See :class:`APCEMMParams` for model parameters.
    **params_kwargs : Any
        Override Cocip model parameters with keyword arguments.
        See :class:`APCEMMParams` for model parameters.

    Notes
    -----
    **Meteorology**

    APCEMM requires temperature, humidity, gepotential height, and winds.
    Geopotential height is required because APCEMM expects meteorological fields
    on height rather than pressure surfaces. See :attr:`met_variables` for the
    list of required variables.

    .. list-table:: Variable keys for pressure level data
        :header-rows: 1

        * - Parameter
          - ECMWF
          - GFS
        * - Air Temperature
          - ``air_temperature``
          - ``air_temperature``
        * - Specific Humidity
          - ``specific_humidity``
          - ``specific_humidity``
        * - Geopotential/Geopotential Height
          - ``geopotential``
          - ``geopotential_height``
        * - Eastward wind
          - ``eastward_wind``
          - ``eastward_wind``
        * - Northward wind
          - ``northward_wind``
          - ``northward_wind``
        * - Vertical velocity
          - ``lagrangian_tendency_of_air_pressure``
          - ``lagrangian_tendency_of_air_pressure``

    **Acquiring and compiling APCEMM**

    Users are responsible for acquiring and compiling APCEMM. The APCEMM source code is
    available through `GitHub <https://github.com/MIT-LAE/APCEMM>`__, and instructions
    for compiling APCEMM are available in the repository.

    Note that APCEMM does not provide versioned releases, and future updates may break
    this interface. To guarantee compatibility between this interface and APCEMM,
    users should use commit
    `9d8e1ee <https://github.com/MIT-LAE/APCEMM/commit/9d8e1eeaa61cbdee1b1d03c65b5b033ded9159e4>`__
    from the APCEMM repository.

    **Configuring APCEMM YAML files**

    :class:`APCEMMInput` provides low-level control over the contents of YAML files used
    as APCEMM input. YAML file contents can be controlled by passing custom parameters
    in a dictionary through the ``apcemm_input_params`` parameter. Note, however, that
    :class:`APCEMM` sets a number of APCEMM input parameters automatically, and attempting
    to override any automatically-determined parameters using ``apcemm_input_params``
    will result in an error. A list of automatically-determined parameters is available in
    :attr:`dynamic_yaml_params`.

    **Simulation initialization, execution, and postprocessing**

    This interface initializes, runs, and postprocesses APCEMM simulations in four stages:

    1. A :class:`DryAdvection` model is used to generate trajectories for contrails
       initialized at each flight waypoint. This is a necessary preprocessing step because
       APCEMM is a Lagrangian model and does not explicitly track changes in plume
       location over time. This step also provides time-dependent azimuths that define the
       orientation of advected contrails, which is required to compute contrail-normal
       wind shear from horizontal winds.
       Results from the trajectory calculation are stored in :attr:`trajectories`.
    2. Model parameters and results from the trajectory calculation are used to generate
       YAML files with APCEMM input parameters and netCDF files with meteorology data
       used by APCEMM simulations. A separate pair of files is generated for each
       waypoint processed by APCEMM. Files are saved as ``apcemm_waypoint_<i>/input.yaml``
       and ``apcemm_waypoint_<i>/input.nc`` in the model :attr:`cachestore`,
       where ``<i>`` is replaced by the index of each simulated flight waypoint.
    3. A separate APCEMM simulation is run in each run directory inside the model
       :attr:`cachestore`. Simulations are independent and can be run in parallel
       (controlled by the ``n_jobs`` parameter in :class:`APCEMMParams`). Standard output
       and error streams from each simulation are saved in ``apcemm_waypoint_<i>/stdout.log``
       and ``apcemm_waypoint_<i>/stderr.log``, and APCEMM output is saved
       in a subdirectory specified by the ``output_directory`` model parameter ("out" by default).
    4. APCEMM simulation output is postprocessed. After postprocessing:

    - A ``status`` column is attached to the ``Flight`` returned by :meth:`eval`.
      This column contains ``"NoSimulation"`` for waypoints where no simulation
      was run and the contents of the APCEMM ``status_case0`` output file for
      other waypoints.
    - A :class:`pd.DataFrame` is created and stored in :attr:`vortex`. This dataframe
      contains time series output from the APCEMM "early plume model" of the aircraft
      exhaust plume and downwash vortex, read from ``Micro_000000.out`` output files
      saved by APCEMM.
    - If APCEMM simulated at least one persistent contrail, A :class:`pd.DataFrame` is
      created and stored in :attr:`contrail`. This dataframe contains paths to netCDF
      files, saved at prescribed time intervals during the APCEMM simulation, and can be
      used to open APCEMM output (e.g., using :func:`xr.open_dataset`) for further analysis.

    **Numerics**

    APCEMM simulations are initialized at flight waypoints and represent the evolution of the
    cross-section of contrails formed at each waypoint. APCEMM does not explicitly model the length
    of contrail segments and does not include any representation of deformation by divergent flow.
    APCEMM output represents properties of cross-sections of contrails formed at flight waypoints,
    not properties of contrail segments that form between flight waypoints. Unlike :class:`Cocip`,
    output produced by this interface does not include trailing NaN values.

    **Known limitations**

    - Engine core exit temperature and bypass area are not provided as output by pycontrails
      aircraft performance models and are currently set to static values in APCEMM input files.
      These parameters will be computed dynamically in a future release.
    - APCEMM does not compute contrail radiative forcing internally. Radiative forcing must be
      computed offline by the user. Tools for radiative forcing calculations may be included
      in a future version of the interface.
    - APCEMM currently produces different results in simulations that do not read vertical
      velocity data from netCDF input files and in simulations that read vertical velocities
      that are set to 0 everywhere (see https://github.com/MIT-LAE/APCEMM/issues/17).
      Reading of vertical velocity data from netCDF input files will be disabled in this
      interface until this issue is resolved.

    References
    ----------
    - :cite:`fritzRolePlumescaleProcesses2020`
    """

    __slots__ = (
        "_trajectory_downsampling",
        "apcemm_input_params",
        "apcemm_path",
        "cachestore",
        "contrail",
        "trajectories",
        "vortex",
    )

    name = "apcemm"
    long_name = "Interface to APCEMM plume model"
    met_variables = (
        AirTemperature,
        SpecificHumidity,
        (Geopotential, GeopotentialHeight),
        EastwardWind,
        NorthwardWind,
        VerticalVelocity,
    )
    default_params = APCEMMParams

    #: Met data is not optional
    met: MetDataset
    met_required = True

    #: Path to APCEMM executable
    apcemm_path: pathlib.Path

    #: Overridden APCEMM input parameters
    apcemm_input_params: dict[str, Any]

    #: CacheStore for APCEMM run directories
    cachestore: cache.CacheStore

    #: Last flight processed in :meth:`eval`
    source: Flight

    #: Output from trajectory calculation
    trajectories: GeoVectorDataset | None

    #: Time series output from the APCEMM early plume model
    vortex: pd.DataFrame | None

    #: Paths to APCEMM netCDF output at prescribed time intervals
    contrail: pd.DataFrame | None

    #: Downsampling factor from trajectory time resolution to
    #: APCEMM met input file resolution
    _trajectory_downsampling: int

    def __init__(
        self,
        met: MetDataset,
        apcemm_path: pathlib.Path | str,
        apcemm_root: pathlib.Path | str | None = None,
        apcemm_input_params: dict[str, Any] | None = None,
        cachestore: cache.CacheStore | None = None,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ) -> None:
        super().__init__(met, params=params, **params_kwargs)
        self._ensure_geopotential_height()

        if isinstance(apcemm_path, str):
            apcemm_path = pathlib.Path(apcemm_path)
        self.apcemm_path = apcemm_path

        if cachestore is None:
            cache_root = cache._get_user_cache_dir()
            cache_dir = f"{cache_root}/apcemm"
            cachestore = cache.DiskCacheStore(cache_dir=cache_dir)
        self.cachestore = cachestore

        # Validate overridden input parameters
        apcemm_input_params = apcemm_input_params or {}
        if apcemm_root is not None:
            apcemm_input_params = {
                "input_background_conditions": os.path.join(apcemm_root, "input_data", "init.txt"),
                "input_engine_emissions": os.path.join(apcemm_root, "input_data", "ENG_EI.txt"),
            } | apcemm_input_params
        cannot_override = set(apcemm_input_params.keys()).intersection(self.dynamic_yaml_params)
        if len(cannot_override) > 0:
            msg = (
                f"Cannot override APCEMM input parameters {cannot_override}, "
                "as these parameters are set automatically by the APCEMM interface."
            )
            raise ValueError(msg)
        self.apcemm_input_params = apcemm_input_params

    @overload
    def eval(self, source: Flight, **params: Any) -> Flight: ...

    @overload
    def eval(self, source: None = ..., **params: Any) -> NoReturn: ...

    def eval(self, source: Flight | None = None, **params: Any) -> Flight:
        """Set up and run APCEMM simulations initialized at flight waypoints.

        Simulates the formation and evolution of contrails from a Flight
        using the APCEMM plume model described in Fritz et. al. (2020)
        :cite:`fritzRolePlumescaleProcesses2020`.

        Parameters
        ----------
        source : Flight | None
            Input Flight to model.
        **params : Any
            Overwrite model parameters before eval.

        Returns
        -------
        Flight | NoReturn
            Flight with exit status of APCEMM simulations. Detailed APCEMM outputs are attached
            to model :attr:`vortex` and :attr:`contrail` attributes (see :class:`APCEMM` notes
            for details).

        References
        ----------
        - :cite:`fritzRolePlumescaleProcesses2020`
        """

        self.update_params(params)
        self.set_source(source)
        self.source = self.require_source_type(Flight)

        # Assign waypoints to flight if not already present
        if "waypoint" not in self.source:
            self.source["waypoint"] = np.arange(self.source.size)

        logger.info("Attaching APCEMM initial conditions to source")
        self.attach_apcemm_initial_conditions()

        logger.info("Computing Lagrangian trajectories")
        self.compute_lagrangian_trajectories()

        # Select waypoints to run in APCEMM
        # Defaults to all waypoints, but allows user to select a subset
        waypoints = self.params["waypoints"]
        if waypoints is None:
            waypoints = list(self.source["waypoints"])

        # Generate input files (serial)
        logger.info("Generating APCEMM input files")  # serial
        for waypoint in waypoints:
            rundir = self.apcemm_file(waypoint)
            if self.cachestore.exists(rundir) and not self.params["overwrite"]:
                msg = f"APCEMM run directory already exists at {rundir}"
                raise ValueError(msg)
            if self.cachestore.exists(rundir) and self.params["overwrite"]:
                shutil.rmtree(rundir)
            self.generate_apcemm_input(waypoint)

        # Run APCEMM (parallelizable)
        logger.info("Running APCEMM")
        self.run_apcemm(waypoints)

        # Process output (serial)
        logger.info("Postprocessing APCEMM output")
        self.process_apcemm_output()

        return self.source

    def attach_apcemm_initial_conditions(self) -> None:
        """Compute fields required for APCEMM initial conditions and attach to :attr:`source`.

        This modifies :attr:`source` by attaching quantities derived from meterology
        data and aircraft performance calculations.
        """

        self._attach_apcemm_time()
        self._attach_initial_met()
        self._attach_aircraft_performance()

    def compute_lagrangian_trajectories(self) -> None:
        """Calculate Lagrangian trajectories using a :class:`DryAdvection` model.

        Lagrangian trajectories provide the expected time-dependent location
        (longitude, latitude, and altitude) and orientation (azimuth) of
        contrails formed by the input source. This information is used to
        extract time series of meteorological profiles at the contrail location
        from input meteorology data, and to compute contrail-normal horizontal shear
        from horizontal winds.

        The length of Lagrangian trajectories is set by the ``max_age`` parameter,
        and trajectories are integrated using a time step set by the ``dt_lagrangian``
        parameter. Contrails are advected both horizontally and vertically, and a
        fixed sedimentation velocity (set by the ``sedimentation_rate`` parameter)
        can be included to represent contrail sedimentation.

        Results of the trajectory calculation are attached to :attr:`trajectories`.
        """

        buffers = {
            f"{coord}_buffer": self.params[f"met_{coord}_buffer"]
            for coord in ("longitude", "latitude", "level")
        }
        buffers["time_buffer"] = (0, self.params["max_age"] + self.params["dt_lagrangian"])
        met = self.source.downselect_met(self.met, **buffers)
        model = DryAdvection(
            met=met,
            dt_integration=self.params["dt_lagrangian"],
            max_age=self.params["max_age"],
            sedimentation_rate=self.params["lagrangian_sedimentation_rate"],
        )
        self.trajectories = model.eval(self.source)

    def generate_apcemm_input(self, waypoint: int) -> None:
        """Generate APCEMM yaml and netCDF input files for a single waypoint.

        For details about generated input files, see :class:`APCEMM` notes.

        Parameters
        ----------
        waypoint : int
            Waypoint for which to generate input files.
        """

        self._gen_apcemm_yaml(waypoint)
        self._gen_apcemm_nc(waypoint)

    def run_apcemm(self, waypoints: list[int]) -> None:
        """Run APCEMM over multiple waypoints.

        Multiple waypoints will be processed in parallel if the :class:`APCEMM`
        ``n_jobs`` parameter is set to a value larger than 1.

        Parameters
        ----------
        waypoints : list[int]
            List of waypoints at which to initialize simulations.
        """

        # run in series
        if self.params["n_jobs"] == 1:
            for waypoint in waypoints:
                utils.run(
                    apcemm_path=self.apcemm_path,
                    input_yaml=self.apcemm_file(waypoint, "input.yaml"),
                    rundir=self.apcemm_file(waypoint),
                    stdout_log=self.apcemm_file(waypoint, "stdout.log"),
                    stderr_log=self.apcemm_file(waypoint, "stderr.log"),
                )

        # run in parallel
        else:
            with mp.Pool(self.params["n_jobs"]) as p:
                args = (
                    (
                        self.apcemm_path,
                        self.apcemm_file(waypoint, "input.yaml"),
                        self.apcemm_file(waypoint),
                        self.apcemm_file(waypoint, "stdout.log"),
                        self.apcemm_file(waypoint, "stderr.log"),
                    )
                    for waypoint in waypoints
                )
                p.starmap(utils.run, args)

    def process_apcemm_output(self) -> None:
        """Process APCEMM output.

        After processing, a ``status`` column will be attached to
        :attr:`source`, and additional output data will be attached
        to :attr:`vortex` and :attr:`contrail`. For details about
        contents of APCEMM output files, see :class:`APCEMM` notes.
        """

        output_directory = self.params["output_directory"]

        statuses: list[str] = []
        vortexes: list[pd.DataFrame] = []
        contrails: list[pd.DataFrame] = []

        for _, row in self.source.dataframe.iterrows():
            waypoint = row["waypoint"]

            # Mark waypoint as skipped if no APCEMM simulation ran
            if waypoint not in self.params["waypoints"]:
                statuses.append("NoSimulation")
                continue

            # Otherwise, record status of APCEMM simulation
            with open(
                self.apcemm_file(waypoint, os.path.join(output_directory, "status_case0"))
            ) as f:
                status = f.read().strip()
                statuses.append(status)

            # Get waypoint initialization time
            base_time = row["time"]

            # Convert contents of wake vortex output to pandas dataframe
            # with elapsed times converted to absolute times
            vortex = pd.read_csv(
                self.apcemm_file(waypoint, os.path.join(output_directory, "Micro000000.out")),
                skiprows=[1],
            ).rename(columns=lambda x: x.strip())
            time = (base_time + pd.to_timedelta(vortex["Time [s]"], unit="s")).rename("time")
            waypoint_col = pd.Series(np.full((len(vortex),), waypoint), name="waypoint")
            vortex = pd.concat(
                (waypoint_col, time, vortex.drop(columns="Time [s]")), axis="columns"
            )
            vortexes.append(vortex)

            # Record paths to contrail output (netCDF files) in pandas dataframe
            # get paths to contrail output
            files = sorted(
                glob.glob(
                    self.apcemm_file(
                        waypoint, os.path.join(output_directory, "ts_aerosol_case0_*.nc")
                    )
                )
            )
            if len(files) == 0:
                continue
            time = []
            path = []
            for file in files:
                elapsed_hours = pd.to_timedelta(file[-7:-5] + "h")
                elapsed_minutes = pd.to_timedelta(file[-5:-3] + "m")
                elapsed_time = elapsed_hours + elapsed_minutes
                time.append(base_time + elapsed_time)
                path.append(file)
            waypoint_col = pd.Series(np.full((len(time),), waypoint), name="waypoint")
            contrail = pd.DataFrame.from_dict(
                {
                    "waypoint": waypoint_col,
                    "time": time,
                    "path": path,
                }
            )
            contrails.append(contrail)

        # Attach status to self
        self.source["status"] = statuses

        # Attach wake vortex and contrail outputs to model
        self.vortex = pd.concat(vortexes, axis="index", ignore_index=True)
        if len(contrails) > 0:  # only present if APCEMM simulates persistent contrails
            self.contrail = pd.concat(contrails, axis="index", ignore_index=True)

    @property
    def dynamic_yaml_params(self) -> set[str]:
        """Set of :class:`APCEMMInput` attributes set dynamically by this model.

        Other :class:`APCEMMInput` attributes can be set statically by passing
        parameters in ``apcemm_input_params`` to the :class:`APCEMM` constructor.
        """
        return {
            "max_age",
            "day_of_year",
            "hour_of_day",
            "longitude",
            "latitude",
            "air_pressure",
            "air_temperature",
            "rhw",
            "normal_shear",
            "brunt_vaisala_frequency",
            "dt_input_met",
            "nox_ei",
            "co_ei",
            "hc_ei",
            "so2_ei",
            "nvpm_ei_m",
            "soot_radius",
            "fuel_flow",
            "aircraft_mass",
            "true_airspeed",
            "n_engine",
            "wingspan",
            "output_directory",
        }

    def apcemm_file(self, waypoint: int, name: str | None = None) -> str:
        """Get path to file from an APCEMM simulation initialized at a specific waypoint.

        Parameters
        ----------
        waypoint : int
            Segment index
        name : str, optional
            If provided, the path to the file relative to the APCEMM simulation
            root directory.

        Returns
        -------
        str
            Path to a file in the APCEMM simulation root directory, if ``name``
            is provided, or the path to the APCEMM simulation root directory otherwise.
        """
        rpath = f"apcemm_waypoint_{waypoint}"
        if name is not None:
            rpath = os.path.join(rpath, name)
        return self.cachestore.path(rpath)

    def _ensure_geopotential_height(self) -> None:
        """Ensure that :attr:`self.met` contains geopotential height."""
        geopotential = Geopotential.standard_name
        geopotential_height = GeopotentialHeight.standard_name

        if geopotential not in self.met and geopotential_height not in self.met:
            msg = f"APCEMM MetDataset must contain either {geopotential} or {geopotential_height}."
            raise ValueError(msg)

        if geopotential_height not in self.met:
            self.met.update({geopotential_height: self.met[geopotential].data / constants.g})

    def _attach_apcemm_time(self) -> None:
        """Attach day of year and fractional hour of day.

        Mutates :attr:`self.source` by adding the following keys if not already present:
        - ``day_of_year``
        - ``hour_of_day``
        """

        self.source.setdefault(
            "day_of_year",
            # APCEMM doesn't accept 366 on leap years
            self.source.dataframe["time"].dt.dayofyear.clip(upper=365),
        )
        self.source.setdefault(
            "hour_of_day",
            self.source.dataframe["time"].dt.hour
            + self.source.dataframe["time"].dt.minute / 60
            + self.source.dataframe["time"].dt.second / 3600,
        )

    def _attach_initial_met(self) -> None:
        """Attach meteorological fields for APCEMM initialization.

        Mutates :attr:`source` by adding the following keys if not already present:
        - ``air_temperature``
        - ``eastward_wind``
        - ``northward_wind``
        - ``specific_humidity``
        - ``air_temperature_lower``
        - ``eastward_wind_lower``
        - ``northward_wind_lower``
        - ``rhw``
        - ``brunt_vaisala_frequency``
        - ``normal_shear``
        """
        humidity_scaling = self.params["humidity_scaling"]
        scale_humidity = humidity_scaling is not None and "specific_humidity" not in self.source

        # Downselect met before interpolation.
        # Need buffer in downward direction for calculation of vertical derivatives,
        # but not in any other directions.
        level_buffer = 0, self.params["met_level_buffer"][1]
        met = self.source.downselect_met(self.met, level_buffer=level_buffer)

        # Interpolate meteorology data onto vector
        for met_key in ("air_temperature", "eastward_wind", "northward_wind", "specific_humidity"):
            models.interpolate_met(met, self.source, met_key, **self.interp_kwargs)

        # Interpolate fields at lower levels for vertical derivative calculation
        air_pressure_lower = thermo.pressure_dz(
            self.source["air_temperature"], self.source.air_pressure, self.params["dz_m"]
        )
        lower_level = air_pressure_lower / 100.0
        for met_key in ("air_temperature", "eastward_wind", "northward_wind"):
            source_key = f"{met_key}_lower"
            models.interpolate_met(
                met, self.source, met_key, source_key, **self.interp_kwargs, level=lower_level
            )

        # Apply humidity scaling
        if scale_humidity:
            humidity_scaling.eval(self.source, copy_source=False)

        # Compute RH over liquid water
        self.source.setdefault(
            "rhw",
            thermo.rh(
                self.source["specific_humidity"],
                self.source["air_temperature"],
                self.source.air_pressure,
            ),
        )

        # Compute Brunt-Vaisala frequency
        dT_dz = thermo.T_potential_gradient(
            self.source["air_temperature"],
            self.source.air_pressure,
            self.source["air_temperature_lower"],
            air_pressure_lower,
            self.params["dz_m"],
        )
        self.source.setdefault(
            "brunt_vaisala_frequency",
            thermo.brunt_vaisala_frequency(
                self.source.air_pressure, self.source["air_temperature"], dT_dz
            ),
        )

        # Compute azimuth
        # Use forward and backward differences for first and last waypoints
        # and centered differences elsewhere
        ileft = [0, *range(self.source.size - 1)]
        iright = [*range(1, self.source.size), self.source.size - 1]
        lon0 = self.source["longitude"][ileft]
        lat0 = self.source["latitude"][ileft]
        lon1 = self.source["longitude"][iright]
        lat1 = self.source["latitude"][iright]
        self.source.setdefault("azimuth", geo.azimuth(lon0, lat0, lon1, lat1))

        # Compute normal shear
        self.source.setdefault(
            "normal_shear",
            utils.normal_wind_shear(
                self.source["eastward_wind"],
                self.source["eastward_wind_lower"],
                self.source["northward_wind"],
                self.source["northward_wind_lower"],
                self.source["azimuth"],
                self.params["dz_m"],
            ),
        )

    def _attach_aircraft_performance(self) -> None:
        """Attach aircraft performance and emissions parameters.

        Mutates :attr:`source evaluating the aircraft performance model provided by
        the ``aircraft_performance`` model parameter and a :class:`Emissions` models. In addition:
            - MetDatasetutates :attr:`source` by adding the following keys if not already present:
            - ``soot_radius``
        - Mutates :attr:`source.attrs` by adding the following keys if not already present:
            - ``so2_ei``
        """

        ap_model = self.params["aircraft_performance"]
        emissions = Emissions()
        humidity_scaling = self.params["humidity_scaling"]
        scale_humidity = humidity_scaling is not None and "specific_humidity" not in self.source

        # Ensure required met data is present.
        # No buffers needed for interpolation!
        vars = ap_model.met_variables + ap_model.optional_met_variables + emissions.met_variables
        met = self.source.downselect_met(self.met)
        met.ensure_vars(vars)
        met.standardize_variables(vars)
        for var in vars:
            models.interpolate_met(met, self.source, var.standard_name, **self.interp_kwargs)

        # Apply humidity scaling
        if scale_humidity:
            humidity_scaling.eval(self.source, copy_source=False)

        # Ensure flight has aircraft type, fuel, and engine UID if defined
        self.source.attrs.setdefault("aircraft_type", self.params["aircraft_type"])
        self.source.attrs.setdefault("fuel", self.params["fuel"])
        if self.params["engine_uid"]:
            self.source.attrs.setdefault("engine_uid", self.params["engine_uid"])

        # Run performance and emissions calculations
        ap_model.eval(self.source, copy_source=False)
        emissions.eval(self.source, copy_source=False)

        # Attach additional required quantities
        soot_radius = utils.soot_radius(self.source["nvpm_ei_m"], self.source["nvpm_ei_n"])
        self.source.setdefault("soot_radius", soot_radius)
        self.source.attrs.setdefault("so2_ei", self.source.attrs["fuel"].ei_so2)

    def _gen_apcemm_yaml(self, waypoint: int) -> None:
        """Generate APCEMM yaml file.

        Parameters
        ----------
        waypoint : int
            Waypoint for which to generate the yaml file.
        """

        # Collect parameters determined by this interface
        dyn_params = _combine_prioritized(
            self.dynamic_yaml_params,
            [
                self.source.dataframe.loc[waypoint],  # flight waypoint
                self.source.attrs,  # flight attributes
                self.params,  # class parameters
            ],
        )

        # Combine with other overridden parameters
        params = self.apcemm_input_params | dyn_params

        # We should be setting these parameters based on aircraft data,
        # but we don't currently have an easy way to do this.
        # For now, stubbing in static values.
        params = params | {"core_exit_temp": 550.0, "core_exit_area": 1.0}

        # Generate and write YAML file
        yaml = APCEMMInput(**params)
        yaml_contents = utils.generate_apcemm_input_yaml(yaml)
        path = self.apcemm_file(waypoint, "input.yaml")
        with open(path, "w") as f:
            f.write(yaml_contents)

    def _gen_apcemm_nc(self, waypoint: int) -> None:
        """Generate APCEMM meteorology netCDF file.

        Parameters
        ----------
        waypoint : int
            Waypoint for which to generate the meteorology file.
        """
        # Extract trajectories of advected contrails, include initial position
        columns = ["longitude", "latitude", "time", "azimuth"]
        if self.trajectories is None:
            msg = (
                "APCEMM meteorology input generation requires precomputed trajectories. "
                "To compute trajectories, call `compute_lagrangian_trajectories`."
            )
            raise ValueError(msg)
        tail = self.trajectories.dataframe
        tail = tail[tail["waypoint"] == waypoint][columns]
        head = self.source.dataframe
        head = head[head["waypoint"] == waypoint][columns]
        traj = pd.concat((head, tail), axis="index").reset_index()

        # APCEMM requires atmospheric profiles at even time intervals,
        # but the time coordinates of the initial position plus subsequent
        # trajectory may not be evenly spaced. To fix this, we interpolate
        # horizontal location and azimuth to an evenly-spaced set of time
        # coordinates.
        time = traj["time"].values
        n_profiles = int(self.params["max_age"] / self.params["dt_input_met"]) + 1
        tick = np.timedelta64(1, "s")
        target_elapsed = np.linspace(
            0, (n_profiles - 1) * self.params["dt_input_met"] / tick, n_profiles
        )
        target_time = time[0] + target_elapsed * tick
        elapsed = (traj["time"] - traj["time"][0]) / tick

        # Need to deal with antimeridian crossing.
        # Detecting antimeridian crossing follows Flight.resample_and_fill,
        # but rather than applying a variable shift we just convert longitudes to
        # [0, 360) before interpolating flights that cross the antimeridian,
        # and un-convert any longitudes above 180 degree after interpolation.
        lon = traj["longitude"].values
        min_pos = np.min(lon[lon > 0], initial=np.inf)
        max_neg = np.max(lon[lon < 0], initial=-np.inf)
        if (180 - min_pos) + (180 + max_neg) < 180 and min_pos < np.inf and max_neg > -np.inf:
            lon = np.where(lon < 0, lon + 360, lon)
        interp_lon = np.interp(target_elapsed, elapsed, lon)
        interp_lon = np.where(interp_lon > 180, interp_lon - 360, interp_lon)

        interp_lat = np.interp(target_elapsed, elapsed, traj["latitude"].values)
        interp_az = np.interp(target_elapsed, elapsed, traj["azimuth"].values)

        if self.params["altitude_input_met"] is None:
            altitude = self.met["altitude"].values
        else:
            altitude = np.array(self.params["altitude_input_met"])

        ds = utils.generate_apcemm_input_met(
            time=target_time,
            longitude=interp_lon,
            latitude=interp_lat,
            azimuth=interp_az,
            altitude=altitude,
            met=self.met,
            humidity_scaling=self.params["humidity_scaling"],
            dz_m=self.params["dz_m"],
            interp_kwargs=self.interp_kwargs,
        )

        path = self.apcemm_file(waypoint, "input.nc")
        ds.to_netcdf(path)


def _combine_prioritized(keys: set[str], sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine dictionary keys from prioritized list of source dictionaries.

    Parameters
    ----------
    keys : set[str]
        Set of keys to attempt to extract from source dictionary.
    sources : list[dict[str, Any]]
        List of dictionaries from which to attempt to extract key-value pairs.
        If the key is in the first dictionary, it will be set in the returned dictionary
        with the corresponding value. Otherwise, the method will fall on the remaining
        dictionaries in the order provided.

    Returns
    -------
    dict[str, Any]
        Dictionary containing key-value pairs from :param:`sources`.

    Raises
    ------
    ValueError
        Any key is not found in any dictionary in ``sources``.
    """
    dest = {}
    for key in keys:
        for source in sources:
            if key in source:
                dest[key] = source[key]
                break
        else:
            msg = f"Key {key} not found in any source dictionary."
            raise ValueError(msg)
    return dest
