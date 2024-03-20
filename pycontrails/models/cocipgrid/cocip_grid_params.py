"""Default :class:`CocipGrid` parameters."""

from __future__ import annotations

import dataclasses

from pycontrails.core.aircraft_performance import AircraftPerformanceGrid
from pycontrails.core.fuel import Fuel, JetA
from pycontrails.models.cocip.cocip_params import CocipParams


@dataclasses.dataclass
class CocipGridParams(CocipParams):
    """Default parameters for :class:`CocipGrid`."""

    # ---------
    # Algorithm
    # ---------

    #: Approximate size of a typical :class:`numpy.ndarray` used with in CoCiP calculations.
    #: The 4-dimensional array defining the waypoint is raveled and split into
    #: batches of this size.
    #: A smaller number for this parameter will reduce memory footprint at the
    #: expense of a longer compute time.
    target_split_size: int = 100_000

    #: Additional boost to target split size before SAC is computed. For typical meshes, only
    #: 10% of waypoints will survive SAC and initial downwash filtering. Accordingly, this parameter
    #: magnifies mesh split size before SAC is computed. See :attr:`target_split_size`.
    target_split_size_pre_SAC_boost: float = 3.0

    #: Display ``tqdm`` progress bar showing batch evaluation progress.
    show_progress: bool = True

    # ------------------
    # Simulated Aircraft
    # ------------------

    #: Nominal segment length to place at each grid point [:math:`m`]. Round-off error
    #: can be problematic with a small nominal segment length and a large
    #: :attr:`dt_integration` parameter. On the other hand,
    #: too large of a nominal segment length diminishes the "locality" of the grid point.
    #:
    #:     .. versionadded:: 0.32.2
    #:
    #:     EXPERIMENTAL: If None, run CoCiP in "segment-free"
    #:     mode. This mode does not include any terms involving segments (wind shear,
    #:     segment length, any derived terms). See :attr:`azimuth`
    #:     and :attr:`dsn_dz_factor` for more details.
    segment_length: float | None = 1000.0

    #: Fuel type
    fuel: Fuel = dataclasses.field(default_factory=JetA)

    #: ICAO code designating simulated aircraft type. Needed for the
    #: :attr:`aircraft_performance` and :class:`Emissions` models.
    aircraft_type: str = "B737"

    #: Engine unique identification number for the ICAO Aircraft Emissions Databank (EDB)
    #: If None, an assumed engine_uid is used in :class:`Emissions`.
    engine_uid: str | None = None

    #: Navigation bearing [:math:`\deg`] measured in clockwise direction from
    #: true north, by default 0.0.
    #:
    #:    .. versionadded:: 0.32.2
    #:
    #:    EXPERIMENTAL: If None, run CoCiP in "segment-free"
    #:    mode. This mode does not include any terms involving segments (wind shear,
    #:    segment_length, any derived terms), unless :attr:`dsn_dz_factor`
    #:    is non-zero.
    azimuth: float | None = 0.0

    #: Experimental parameter used to approximate ``dsn_dz`` from ``ds_dz`` via
    #: ``dsn_dz = ds_dz * dsn_dz_factor``.
    #: A value of 0.0 disables any normal wind shear effects.
    #: An initial unpublished experiment suggests that
    #: ``dsn_dz_factor = 0.665`` adequately approximates the mean EF predictions
    #: of :class:`CocipGrid` over all azimuths.
    #:
    #:     .. versionadded:: 0.32.2
    dsn_dz_factor: float = 0.0

    #: --------------------
    #: Aircraft Performance
    #: --------------------

    #: Aircraft wingspan, [:math:`m`]. If included in :attr:`CocipGrid.source`,
    #: this parameter is unused. Otherwise, if this parameter is None, the
    #: :attr:`aircraft_performance` model is used to estimate the wingspan.
    wingspan: float | None = None

    #: Nominal aircraft mass, [:math:`kg`]. If included in :attr:`CocipGrid.source`,
    #: this parameter is unused. Otherwise, if this parameter is None, the
    #: :attr:`aircraft_performance` model is used to estimate the aircraft mass.
    aircraft_mass: float | None = None

    #: Cruising true airspeed, [:math:`m \ s^{-1}`]. If included in :attr:`CocipGrid.source`,
    #: this parameter is unused. Otherwise, if this parameter is None, the
    #: :attr:`aircraft_performance` model is used to estimate the true airspeed.
    true_airspeed: float | None = None

    #: Nominal engine efficiency, [:math:`0 - 1`]. If included in :attr:`CocipGrid.source`,
    #: this parameter is unused. Otherwise, if this parameter is None, the
    #: :attr:`aircraft_performance` model is used to estimate the engine efficiency.
    engine_efficiency: float | None = None

    #: Nominal fuel flow, [:math:`kg \ s^{-1}`]. If included in :attr:`CocipGrid.source`,
    #: this parameter is unused. Otherwise, if this parameter is None, the
    #: :attr:`aircraft_performance` model is used to estimate the fuel flow.
    fuel_flow: float | None = None

    #: Aircraft performance model. Required unless ``source`` or ``params``
    #: provide all of the following variables:
    #:
    #: - wingspan
    #: - true_airspeed (or mach_number)
    #: - fuel_flow
    #: - engine_efficiency
    #: - aircraft_mass
    #:
    #: If None and :attr:`CocipGrid.source` or :class:`CocipGridParams` do not provide
    #: the above variables, a ValueError is raised. See :class:`PSGrid` for an open-source
    #: implementation of a :class:`AircraftPerformanceGrid` model.
    aircraft_performance: AircraftPerformanceGrid | None = None

    # ------------
    # Model output
    # ------------

    #: Attach additional formation specific data to the output. If True, attach
    #: all possible formation data. See :mod:`pycontrails.models.cocipgrid.cocip_grid`
    #: for a list of supported formation data.
    verbose_outputs_formation: bool | set[str] = False

    #: Attach contrail evolution data to :attr:`CocipGrid.contrail_list`. Requires
    #: substantial memory overhead.
    verbose_outputs_evolution: bool = False
