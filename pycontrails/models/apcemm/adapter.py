"""Adapter between APCEMM and pycontrails :class:`Model` interfaces."""

import dataclasses
import glob
import logging
import multiprocessing as mp
import os
import shutil
from typing import Any, NoReturn, overload

import numpy as np
import pandas as pd

from pycontrails.core import models
from pycontrails.core.aircraft_performance import AircraftPerformance
from pycontrails.core.cache import CacheStore, DiskCacheStore
from pycontrails.core.flight import Flight
from pycontrails.core.fuel import Fuel, JetA
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import (
    AirTemperature,
    EastwardWind,
    NorthwardWind,
    SpecificHumidity,
    VerticalVelocity,
)
from pycontrails.models.apcemm import utils
from pycontrails.models.apcemm.interface import APCEMMMet, APCEMMYaml, run
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.models.emissions import Emissions
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.models.ps_model import PSFlight
from pycontrails.physics import thermo

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class APCEMMParams(models.ModelParams):
    """Default parameters for the pycontrails :class:`APCEMM` adapter."""

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
    #: Must be a multiple of :attr:`dt_lagrangian`.
    dt_input_met: np.timedelta64 = np.timedelta64(1, "h")

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
    aircraft_performance: AircraftPerformance = PSFlight()

    #: Fuel type
    fuel: Fuel = JetA()

    #: List of segments (indexed from start of flight) to simulate in APCEMM.
    #: By default, runs a simulation for every segment.
    segments: list[int] | None = None

    #: If defined, override the ``apcemm_root`` value in :class:`APCEMMYaml`
    apcemm_root: str | None = None

    #: If True, delete existing run directories before running APCEMM simulations.
    #: If False (default), raise an exception if a run directory already exists.
    overwrite: bool = False

    #: Number of threads to use within individual APCEMM simulations.
    apcemm_threads: int = 1

    #: Number of individual APCEMM simulations to run in parallel.
    n_jobs: int = 1


class APCEMM(models.Model):
    """Run APCEMM as a pycontrails :class:`Model`.

    This class acts as an adapter between the pycontrails :class:`Model` interface
    (shared with other contrail models) and APCEMM's native interface.
    `APCEMM <https://github.com/MIT-LAE/APCEMM>__` was developed at the
    `MIT Laboratory for Aviation and the Environment <https://lae.mit.edu/>`__
    and is described in :cite:`fritzRolePlumescaleProcesses2020`.

    Parameters
    ----------
    met : MetDataset
        Pressure level dataset containing :attr:`met_variables` variables.
        See *Notes* for variable names by data source.
    apcemm : str
        Path to APCEMM executable. See *Notes* for information about
        acquiring and compiling APCEMM.
    apcemm_root : str, optional
        Path to APCEMM root directory. If not provided, pycontrails will use the
        default path defined in :class:`APCEMMYaml`.
    yaml : APCEMMYaml, optional
        Configuration for YAML file generated as APCEMM input. If provided, the
        value of the ``apcemm_root`` attribute in this instance will override
        the value of :param:`apcemm_root` if the latter is provided.
        See *Notes* for detailed information about YAML file generation.
    cachestore : CacheStore, optional
        :class:`CacheStore` used to store APCEMM run directories. If not
        provided, uses the default pycontrails :class:`DiskCacheStore`.
        See *Notes* for detailed information about the file structure for APCEMM
        simulations.
    params : dict[str, Any], optional
        Override APCEMM model parameters with dictionary.
        See :class:`APCEMMParams` for model parameters.
    **params_kwargs : Any
        Override Cocip model parameters with keyword arguments.
        See :class:`APCEMMParams` for model parameters.

    Notes
    -----
    **Meteorology**

    APCEMM requires temperature, humidity, and winds. See :attr:`met_variables` for the list
    of required variables.

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

    :class:`APCEMMYaml` provides low-level contrail over the contents of YAML files used
    as APCEMM input. YAML file contents can be controlled by passing an instance of
    :class:`APCEMMYaml` with customized parameters to the :class:`APCEMM` constructor.
    Note, however, that :class:`APCEMM` will overwrite YAML parameters that are controlled
    by the model parameters defined in :class:`APCEMMParams`. A list of overwritten YAML
    parameters is available in :attr:`dynamic_yaml_params`.

    **Simulation initialization, execution, and postprocessing**

    This interface initializes, runs, and postprocesses APCEMM simulations in four stages:

    1. A :class:`DryAdvection` model is used to generate trajectories for contrails
       formed from each flight segment. This is a necessary preprocessing step because
       APCEMM is a Lagrangian model and does not explicitly track changes in plume
       location over time.
       Results from the trajectory calculation are stored in :attr:`trajectories`.
    2. Model parameters and results from the trajectory calculation are used to generate
       YAML files with APCEMM input parameters and netCDF files with meteorology data
       used by APCEMM simulations. A separate pair of files is generated for each
       segment processed by APCEMM. Files are saved as ``apcemm_<segment>/input.yaml``
       and ``apcemm_<segment>/input.nc`` in the model :attr:`cachestore`,
       where ``<segment>`` is replaced by the index of each simulated flight segment.
    3. A separate APCEMM simulation is run in each run directory inside the model
       :attr:`cachestore`. Simulations are independent and can be run in parallel
       (controlled by the ``n_jobs`` parameter in :class:`APCEMMParams`.) Standard output
       and error streams from each simulation are saved in ``apcemm_<segment>/stdout.log``
       and ``apcemm_<segment>/stderr.log``, and APCEMM output is saved ``apcemm_<segment>/out``
       directories.
    4. APCEMM simulation output is postprocessed. After postprocessing:

    - A ``status`` column is attached to the ``Flight`` returned by :meth:`eval`.
      This column contains ``"NoSimulation"`` for segments where no simulation
      was run and the contents of the APCEMM ``status_case0`` output file for
      other segments.
    - A :class:`pd.DataFrame` is created and stored in :attr:`vortex`. This dataframe
      contains the content of APCEMM ``Micro_000000.out`` output files, which
      contains time series starting during the early wake vorted stage and continuing
      through the end of the simulation.
    - A :class:`pd.DataFrame` is created and stored in :attr:`contrail` provided APCEMM
      simulated at least one persistent contrail. This dataframe contains paths to netCDF
      files, saved at prescribed time intervals during the APCEMM simulation, and can be
      used to open APCEMM output (e.g., using :func:`xr.open_dataset`) for further analysis.

    References
    ----------
    - :cite:`fritzRolePlumescaleProcesses2020`
    """

    __slots__ = (
        "apcemm",
        "yaml",
        "cachestore",
        "trajectories",
        "vortex",
        "contrail",
        "_trajectory_downsampling",
    )

    name = "apcemm"
    long_name = "Interface to APCEMM plume model"
    met_variables = AirTemperature, SpecificHumidity, EastwardWind, NorthwardWind, VerticalVelocity
    default_params = APCEMMParams

    #: Met data is not optional
    met: MetDataset
    met_required = True

    #: Path to APCEMM executable
    apcemm: str

    #: APCEMM YAML input configuration
    yaml: APCEMMYaml

    #: CacheStore for APCEMM run directories
    cachestore: CacheStore

    #: Last flight processed in :meth:`eval`
    source: Flight

    #: Output from trajectory calculation
    trajectories: Flight | None

    #: APCEMM time series output, starting from early vortex stage
    vortex: pd.DataFrame | None

    #: Paths to APCEMM netCDF output at prescribed time intervals
    contrail: pd.DataFrame | None

    #: Downsampling factor from trajectory time resolution to
    #: APCEMM met input file resolution
    _trajectory_downsampling: int

    def __init__(
        self,
        met: MetDataset,
        apcemm: str,
        apcemm_root: str | None = None,
        yaml: APCEMMYaml | None = None,
        cachestore: DiskCacheStore | None = None,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ) -> None:
        super().__init__(met, params=params, **params_kwargs)

        self.apcemm = apcemm

        if cachestore is None:
            cachestore = DiskCacheStore(allow_clear=True)
        self.cachestore = cachestore

        self._trajectory_downsampling = self._validate_downsampling()

        if yaml is not None:
            self.yaml = yaml
        elif apcemm_root is not None:
            self.yaml = APCEMMYaml(apcemm_root=apcemm_root)
        else:
            self.yaml = APCEMMYaml()

    @overload
    def eval(self, source: Flight, **params: Any) -> Flight: ...

    @overload
    def eval(self, source: None = ..., **params: Any) -> NoReturn: ...

    def eval(self, source: Flight | None = None, **params: Any) -> Flight | NoReturn:
        """Run APCEMM simulation on flight.

        Simulates the formation and evolution of contrails from a Flight
        using the APCEMM plume model described in Fritz et. al. (2020)
        :cite:`schumannRolePlumescaleProcesses2020`.

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

        logger.info("Attaching APCEMM initial conditions to source")
        self.attach_apcemm_initial_conditions()

        logger.info("Computing Lagrangian trajectories")
        self.compute_lagrangian_trajectories()

        # Select segments to run in APCEMM
        # Defaults to all segments, but allows user to select a subset
        segments = self.params["segments"]
        if segments is None:
            segments = sorted(self.source.dataframe.index.unique())

        # Generate input files (serial)
        logger.info("Generating APCEMM input files")  # serial
        for segment in segments:
            rundir = self.apcemm_file(segment)
            if self.cachestore.exists(rundir) and not self.params["overwrite"]:
                msg = f"APCEMM run directory already exists at {rundir}"
                raise ValueError(msg)
            if self.cachestore.exists(rundir) and self.params["overwrite"]:
                shutil.rmtree(rundir)
            self.generate_apcemm_input(segment)

        # Run APCEMM (parallelizable)
        logger.info("Running APCEMM")
        self.run_apcemm(segments)

        # Process output (serial)
        logger.info("Postprocessing APCEMM output")
        self.process_apcemm_output()

        return self.source

    def attach_apcemm_initial_conditions(self) -> None:
        """Compute fields required for APCEMM initial conditions and attach to source.

        This modifies :attr:`source` by attaching quantities derived from meterology
        data and aircraft performance calculations.
        """

        self._attach_apcemm_time()
        self._attach_initial_met()
        self._attach_aircraft_performance()

    def compute_lagrangian_trajectories(self) -> None:
        """Calculate Lagrangian trajectories using a :class:`DryAdvection` model.

        Results of the trajectory calculation are attached to :attr:`trajectories`.
        """

        buffers = {
            f"{coord}_buffer": self.params[f"met_{coord}_buffer"]
            for coord in ("longitude", "latitude", "level")
        }
        buffers["time_buffer"] = (0, self.params["max_age"] + self.params["dt_lagrangian"])
        met = self.source.downselect_met(self.met, **buffers, copy=False)
        model = DryAdvection(
            met=met,
            dt_integration=self.params["dt_lagrangian"],
            max_age=self.params["max_age"],
        )
        self.trajectories = model.eval(self.source)

    def generate_apcemm_input(self, segment: int) -> None:
        """Generate APCEMM yaml and netCDF input files for a single segment.

        For details about generated input files, see :class:`APCEMM` notes.

        Parameters
        ----------
        segment : int
            Segment for which to generate input files.
        """

        self._gen_apcemm_yaml(segment)
        self._gen_apcemm_nc(segment)

    def run_apcemm(self, segments: list[int]) -> None:
        """Run APCEMM over a list of segments.

        Multiple segments will be processed in parallel if the interface parameter ``n_jobs``
        is set to a value larger than 1.

        Parameters
        ----------
        segments : list[int]
            List of segments for which to run simulations.
        """

        # run in series
        if self.params["n_jobs"] == 1:
            for segment in segments:
                run(
                    apcemm=self.apcemm,
                    rundir=self.apcemm_file(segment),
                    stdout_log=self.apcemm_file(segment, "stdout.log"),
                    stderr_log=self.apcemm_file(segment, "stderr.log"),
                )

        # run in parallel
        else:
            with mp.Pool(self.params["n_jobs"]) as p:
                args = (
                    (
                        self.apcemm,
                        self.apcemm_file(segment),
                        self.apcemm_file(segment, "stdout.log"),
                        self.apcemm_file(segment, "stderr.log"),
                    )
                    for segment in segments
                )
                p.starmap(run, args)

    def process_apcemm_output(self) -> None:
        """Process APCEMM output.

        After processing, a ``status`` column will be attached to
        :attr:`source`, and additional output data will be attached
        to :attr:`vortex` and :attr:`contrail`. For details about
        contents of APCEMM output files, see :class:`APCEMM` notes.
        """

        statuses: list[str] = []
        vortexes: list[pd.DataFrame] = []
        contrails: list[pd.DataFrame] = []

        for segment in self.source.dataframe.index[:-1]:  # no segment starting at last waypoint

            # Mark segment as skipped if no APCEMM simulation ran
            if segment not in self.params["segments"]:
                statuses.append("NoSimulation")
                continue

            # Otherwise, record status of APCEMM simulation
            with open(self.apcemm_file(segment, "out/status_case0")) as f:
                status = f.read().strip()
                statuses.append(status)

            # Get segment initialization time, estimated as
            # average of waypoint times at segment start and end
            base_time = self.source.dataframe.iloc[segment : segment + 1]["time"].mean()

            # Convert contents of wake vortex output to pandas dataframe
            # with elapsed times converted to absolute times
            vortex = pd.read_csv(
                self.apcemm_file(segment, "out/Micro000000.out"), skiprows=[1]
            ).rename(columns=lambda x: x.strip())
            time = (base_time + pd.to_timedelta(vortex["Time [s]"], unit="s")).rename("time")
            waypoint = pd.Series(np.full((len(vortex),), segment), name="waypoint")
            vortex = pd.concat((waypoint, time, vortex.drop(columns="Time [s]")), axis="columns")
            vortexes.append(vortex)

            # Record paths to contrail output (netCDF files) in pandas dataframe
            # get paths to contrail output
            files = sorted(glob.glob(f"{self.apcemm_file(segment, 'out')}/ts_aerosol_case0_*.nc"))
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
            waypoint = pd.Series(np.full((len(time),), segment), name="waypoint")
            contrail = pd.DataFrame.from_dict(
                {
                    "waypoint": waypoint,
                    "time": time,
                    "path": path,
                }
            )
            contrails.append(contrail)

        # Mark status of final waypoint as n/a (no segment)
        statuses.append("N/A")

        # Attach status to self
        self.source["status"] = statuses

        # Attach wake vortex and contrail outputs to model
        self.vortex = pd.concat(vortexes, axis="index", ignore_index=True)
        if len(contrails) > 0:  # only present if APCEMM simulates persistent contrails
            self.contrail = pd.concat(contrails, axis="index", ignore_index=True)

    @property
    def dynamic_yaml_params(self) -> list[str]:
        """List of :class:`APCEMMYaml` attributes set dynamically by this interface.

        Other :class:`APCEMMYaml` attributes can be set statically by passing a custom instance of
        :class:`APCEMMYaml` when the interface is instantiated.
        """
        return [
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
        ]

    def apcemm_file(self, segment: int, name: str | None = None) -> str:
        """Get path to file from an APCEMM simulation of a specific segment.

        Parameters
        ----------
        segment : int
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
        rpath = f"apcemm_{segment}"
        if name is not None:
            rpath = os.path.join(rpath, name)
        return self.cachestore.path(rpath)

    def _validate_downsampling(self) -> int:
        """Validate combination of dt_lagrangian and dt_input_met."""
        if self.params["dt_input_met"] < self.params["dt_lagrangian"]:
            msg = (
                f"Timestep for Lagrangian trajectories "
                f"({self.params['dt_lagrangian']}) "
                f"must be no longer than timestep for APCEMM meteorology input "
                f"({self.params['dt_input_met']})."
            )
            raise ValueError(msg)
        if self.params["dt_input_met"] % self.params["dt_lagrangian"] != 0:
            msg = (
                f"Timestep for Lagrangian trajectories "
                f"({self.params['dt_lagrangian']}) "
                f"must evenly divide timestep for APCEMM meteorology input "
                f"({self.params['dt_input_met']})."
            )
            raise ValueError(msg)
        return self.params["dt_input_met"] // self.params["dt_lagrangian"]

    def _attach_apcemm_time(self) -> None:
        """Attach day of year and fractional hour of day.

        Mutates :attr:`self.source` by adding ``day_of_year`` and ``hour_of_day``
        keys if not already present.
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

        Mutates :attr:`self.source` by adding ``rhw``, ``brunt_vaisala_frequency``,
        ``azimuth``, and ``normal_shear`` keys if not already present.
        """

        humidity_scaling = self.params["humidity_scaling"]
        scale_humidity = humidity_scaling is not None and "specific_humidity" not in self.source

        # Downselect met before interpolation.
        # Need buffer in downward direction for calculation of vertical derivatives,
        # but not in any other directions.
        level_buffer = 0, self.params["met_level_buffer"][1]
        met = self.source.downselect_met(self.met, level_buffer=level_buffer)

        for met_key in ("air_temperature", "eastward_wind", "northward_wind", "specific_humidity"):
            models.interpolate_met(met, self.source, met_key, **self.interp_kwargs)

        air_pressure_lower = thermo.pressure_dz(
            self.source["air_temperature"], self.source.air_pressure, self.params["dz_m"]
        )
        lower_level = air_pressure_lower / 100.0
        for met_key in ("air_temperature", "eastward_wind", "northward_wind"):
            source_key = f"{met_key}_lower"
            models.interpolate_met(
                met, self.source, met_key, source_key, **self.interp_kwargs, level=lower_level
            )

        if scale_humidity:
            humidity_scaling.eval(self.source, copy_source=False)

        self.source.setdefault(
            "rhw",
            thermo.rh(
                self.source["specific_humidity"],
                self.source["air_temperature"],
                self.source.air_pressure,
            ),
        )

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

        self.source.setdefault("azimuth", self.source.segment_azimuth())
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

        Mutates :attr:`self.source` by adding running aircraft performance and emissions
        models. In addition, adds ``aircraft_type``, ``fuel``, ``so2_ei``,
        ``engine_uid``, and ``soot_radius`` keys if not already present.
        """

        self.source.attrs.setdefault("aircraft_type", self.params["aircraft_type"])
        self.source.attrs.setdefault("fuel", self.params["fuel"])
        self.source.attrs.setdefault("so2_ei", self.source.attrs["fuel"].ei_so2)
        if self.params["engine_uid"]:
            self.source.attrs.setdefault("engine_uid", self.params["engine_uid"])

        ap_model = self.params["aircraft_performance"]
        ap_model.eval(self.source, copy_source=False)

        emissions = Emissions()
        emissions.eval(self.source, copy_source=False)

        self.source["soot_radius"] = utils.soot_radius(
            self.source["nvpm_ei_m"], self.source["nvpm_ei_n"]
        )

    def _gen_apcemm_yaml(self, segment: int) -> None:
        """Generate APCEMM yaml file.

        Parameters
        ----------
        segment : int
            Segment for which to generate the yaml file.
        """

        # Update APCEMM yaml parameters
        _set_with_fallbacks(
            self.yaml,
            self.dynamic_yaml_params,
            [
                self.source.dataframe.loc[segment],  # flight segment
                self.source.attrs,  # flight attributes
                self.params,  # class parameters
            ],
        )

        # Generate and write YAML file
        yaml_contents = self.yaml.generate_yaml()
        path = self.apcemm_file(segment, "input.yaml")
        with open(path, "w") as f:
            f.write(yaml_contents)

    def _gen_apcemm_nc(self, segment: int) -> None:
        """Generate APCEMM meteorology netCDF file.

        Parameters
        ----------
        segment : int
            Segment for which to generate the meteorology file.
        """
        columns = ["longitude", "latitude", "time", "azimuth"]
        if self.trajectories is None:
            msg = (
                "APCEMM meteorology input generation requires preocmputed trajectories. "
                "To compute trajectories, call `compute_lagrangian_trajectories`."
            )
            raise ValueError(msg)
        tail = self.trajectories.dataframe
        tail = tail[tail["waypoint"] == segment][columns]
        head = self.source.dataframe
        head = head[head.index == segment][columns]
        traj = pd.concat((head, tail), axis="index").reset_index()

        step = self._trajectory_downsampling
        nc = APCEMMMet(
            time=traj["time"].values[0::step],
            longitude=traj["longitude"].values[0::step],
            latitude=traj["latitude"].values[0::step],
            azimuth=traj["azimuth"].values[0::step],
            air_pressure=1e2 * self.met["level"].values,
            humidity_scaling=self.params["humidity_scaling"],
            dz_m=self.params["dz_m"],
        )

        nc_source = nc.generate_met_source()
        met = nc_source.downselect_met(self.met, copy=False)
        ds = nc.generate_met(nc_source, met, self.interp_kwargs)

        path = self.apcemm_file(segment, "input.nc")
        ds.to_netcdf(path)


def _set_with_fallbacks(instance: object, attrs: list[str], sources: list[dict]) -> None:
    """Set class instance attributes from prioritized list of dictionaries.

    Parameters
    ----------
    instance : object
        Class instance.

    attrs : list[str]
        List of instance attributes to attempt to set.

    sources : list[dict]
        List of dictionaries to attempt to set instance attributes from.
        If the attribute is present as a key in the first dictionary, it
        will be set to the corresponding value. Otherwise, the method
        will fall back to the rest of the list of dictionaries.

    Raises
    ------
    ValueError
        Any element of ``attrs`` is not a instance attribute or is not found
        in any dictionary in ``sources``.
    """
    for attr in attrs:
        if not hasattr(instance, attr):
            msg = f"Attribute {attr} does not exist in instance of type {type(instance)}."
            raise ValueError(msg)

        for source in sources:
            if attr in source:
                setattr(instance, attr, source[attr])
                break
        else:
            msg = f"Attribute {attr} not found in any source dictionary."
            raise ValueError(msg)
