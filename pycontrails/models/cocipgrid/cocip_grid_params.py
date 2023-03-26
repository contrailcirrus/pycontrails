"""Default CocipGrid parameters."""

from __future__ import annotations

import dataclasses

import numpy as np

from pycontrails.core.fuel import Fuel, JetA

# from pycontrails.models.aircraft_performance import AircraftPerformanceGrid
from pycontrails.models.cocip.cocip_params import CocipParams

try:
    from pycontrails.ext.bada import BADAParams
except ImportError as e:
    raise ImportError(
        'CocipGrid requires BADA extension. Install with `pip install "pycontrails-bada @'
        ' git+ssh://git@github.com/contrailcirrus/pycontrails-bada.git"`'
    ) from e


@dataclasses.dataclass
class CocipGridParams(CocipParams, BADAParams):
    """Default parameters for :class:`CocipGrid`."""

    # ---------
    # Algorithm
    # ---------

    #: Approximate size of a typical `np.array` used with in CoCiP calculations.
    #: The 4-dimensional array defining the waypoint is raveled and split into
    #: batches of this size.
    #: A smaller number for this parameter will reduce memory footprint at the
    #: expense of a longer compute time.
    target_split_size: int = 100_000

    #: Additional boost to target split size before SAC is computed. For typical meshes, only
    #: 10% of waypoints will survive SAC and initial downwash filtering. Accordingly, this parameter
    #: magnifies mesh split size before SAC is computed.
    target_split_size_pre_SAC_boost: float = 3.0

    #: Time slice for loading met interpolators into memory, by default np.timedelta64(1, "h").
    #: Must be a multiple of `np.timedelta64(1, "h")`. The higher the multiple, the more
    #: memory consumed by the method `eval`. If None, the full met dataset is loaded
    #: at once.
    # TODO: Move this to CocipParams once that model is ready for fancy time handling
    met_slice_dt: np.timedelta64 | None = np.timedelta64(1, "h")

    #: Display `tqdm` progress bar showing batch evaluation progress.
    show_progress: bool = True

    # ------------------
    # Simulated Aircraft
    # ------------------

    #: Nominal segment length to place at each grid point [unit `m`]. Round-off error
    #: can be problematic with a small nominal segment length and a large
    #: `dt_integration` parameter. On the other hand, too large of a nominal segment
    #: length diminishes the "locality" of the grid point.
    #: .. versionadded 0.32.2:: EXPERIMENTAL: If None, run CoCiP in "segment-free"
    #: mode. This mode does not include any terms involving segments (wind shear,
    #: segment_length, any derived terms). See :attr:`CocipGridParams.azimuth`
    #: and :attr:`CocipGridParams.dsn_dz_factor` for more details.
    segment_length: float | None = 1000.0

    #: Fuel type
    fuel: Fuel = dataclasses.field(default_factory=JetA)

    #: ICAO code designating simulated aircraft type. Needed to query BADA database.
    aircraft_type: str = "B737"

    #: Engine unique identification number for the ICAO Aircraft Emissions Databank (EDB)
    #: If None, the assumed engine_uid from BADA is used.
    engine_uid: str | None = None

    #: Nominal aircraft wingspan [:math:`m`].
    #: By default, use nominal value derived from BADA.
    wingspan: float | None = None

    #: Navigation bearing [:math:`\deg`] measured in clockwise direction from
    #: true north, by default 0.0.
    #: .. versionadded 0.32.2:: EXPERIMENTAL: If None, run CoCiP in "segment-free"
    #: mode. This mode does not include any terms involving segments (wind shear,
    #: segment_length, any derived terms), unless :attr:`CocipGridParams.dsn_dz_factor`
    #: is non-zero.
    azimuth: float | None = 0.0

    #: Experimental parameter used to approximate ``dsn_dz`` from ``ds_dz`` via
    #: ``dsn_dz = ds_dz * dsn_dz_factor``. A value of 0.0 disables any normal
    #: wind shear effects. An initial unpublished experiment suggests that
    #: ``dsn_dz_factor = 0.665`` adequately approximates the mean EF predictions
    #: of :class:`CocipGrid` over all azimuths.
    #: .. versionadded 0.32.2::
    dsn_dz_factor: float = 0.0

    #: Nominal aircraft mass [:math:`kg`].
    #: By default, use nominal value derived from BADA.
    aircraft_mass: float | None = None

    #: Cruising true airspeed [:math:`m * s^{-1}`].
    #: By default, use nominal value derived from BADA.
    true_airspeed: float | None = None

    #: Nominal engine efficiency, [:math:`0 - 1`].
    #: By default, use values derived from BADA.
    engine_efficiency: float | None = None

    #: Nominal fuel flow, [:math:`kg s^{-1}`].
    #: By default, use values derived from BADA.
    fuel_flow: float | None = None

    #: Nominal thrust, [:math:`N`].
    #: By default, use values derived from BADA.
    thrust: float | None = None

    # #: Aircraft performance model
    # aircraft_performance: AircraftPerformanceGrid | None = None

    # ------------
    # Model output
    # ------------

    #: Attach additional formation specific data to the output. If True, attach
    #: all possible formation data. See :func:`contrail_grid._supported_verbose_outputs_formation`
    #: for a list of supported formation data.
    verbose_outputs_formation: bool | set[str] = False

    #: Attach contrail evolution data to :attr:`CocipGrid.contrail_list`. Requires
    #: substantial memory overhead.
    verbose_outputs_evolution: bool = False
