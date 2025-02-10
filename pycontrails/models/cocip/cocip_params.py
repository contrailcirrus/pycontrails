"""Default parameters for CoCiP models.

Used by :class:`Cocip` and :class:`CocipGrid`.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt

from pycontrails.core.aircraft_performance import AircraftPerformance
from pycontrails.core.models import AdvectionBuffers
from pycontrails.models.emissions.emissions import EmissionsParams
from pycontrails.models.humidity_scaling import HumidityScaling


def _radius_threshold_um() -> npt.NDArray[np.float32]:
    return np.array([5.0, 9.5, 23.0, 190.0, 310.0], dtype=np.float32)


def _habit_distributions() -> npt.NDArray[np.float32]:
    return np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0],
            [0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.4, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.15, 0.35, 0.0, 0.0],
            [0.0, 0.45, 0.45, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.03, 0.97, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def _habits() -> npt.NDArray[np.str_]:
    return np.array(
        [
            "Sphere",
            "Solid column",
            "Hollow column",
            "Rough aggregate",
            "Rosette-6",
            "Plate",
            "Droxtal",
            "Myhre",
        ]
    )


@dataclasses.dataclass
class CocipParams(AdvectionBuffers):
    """Model parameters required by the CoCiP models."""

    # -------------------------
    # Implementation parameters
    # -------------------------

    #: Determines whether :meth:`Cocip.process_emissions` runs on model :meth:`Cocip.eval`
    #: Set to ``False`` when input Flight includes emissions data.
    process_emissions: bool = True

    #: Apply Euler's method with a fixed step size of ``dt_integration``. Advected waypoints
    #: are interpolated against met data once each ``dt_integration``.
    dt_integration: np.timedelta64 = np.timedelta64(30, "m")

    #: Difference in altitude between top and bottom layer for stratification calculations,
    #: [:math:`m`]. Used to approximate derivative of "lagrangian_tendency_of_air_pressure" layer.
    dz_m: float = 200.0

    #: Vertical resolution (m) associated to met data.
    #: Constant below applies to ECMWF data.
    effective_vertical_resolution: float = 2000.0

    #: Smoothing parameters for true airspeed.
    #: Only used for Flight models.
    #: Passed directly to :func:`scipy.signal.savgol_filter`.
    #: See :meth:`pycontrails.Flight.segment_true_airspeed` for details.
    smooth_true_airspeed: bool = True
    smooth_true_airspeed_window_length: int = 7
    smooth_true_airspeed_polyorder: int = 1

    #: Humidity scaling
    humidity_scaling: HumidityScaling | None = None

    #: Experimental. If ``True``, attempt to reduce memory consumption during
    #: aircraft performance and initial contrail formation/persistent calculations
    #: by calling :meth:`MetDataArray.interpolate` with ``lowmem=True``.
    #:
    #: **IMPORTANT**:
    #:
    #: * Memory optimizations used when ``proprocess_lowmem=True`` are designed for
    #:   meteorology backed by dask arrays with a chunk size of 1 along
    #:   the time dimension. This option may degrade performance if dask if not used
    #:   or if chunks contain more than a single time step.
    #: * The impact on runtime of setting ``preprocess_lowmem=True`` depends on how
    #:   meteorology data is chunked. Runtime is likely to increase if meteorology
    #:   data is chunked in time only, but may decrease if meteorology data is also
    #:   chunked in longitude, latitude, and level.
    #: * Changes to data access patterns with ``preprocess_lowmem=True`` alter locations
    #:   where interpolation is in- vs out-of-bounds. As a consequence,
    #:   Cocip output with ``preprocess_lowmem=True`` is only guaranteed to match output
    #:   with ``preprocess_lowmem=False`` when run with ``interpolation_bounds_error=True``
    #:   to ensure no out-of-bounds interpolation occurs.
    #:
    #: .. versionadded:: 0.52.3
    preprocess_lowmem: bool = False

    # --------------
    # Downselect met
    # --------------

    #: Compute ``"tau_cirrus"`` variable in pressure-level met data during model
    #: initialization. Must be one of ``"auto"``, ``True``, or ``False``. If set to
    #: ``"auto"``, ``"tau_cirrus"`` will be computed during model initialization
    #: iff the met data is dask-backed. Otherwise, it will be computed during model
    #: evaluation after the met data is downselected.
    #:
    #: .. versionadded:: 0.47.1
    compute_tau_cirrus_in_model_init: bool | str = "auto"

    # ---------
    # Filtering
    # ---------

    #: Filter out waypoints if the don't satisfy the SAC criteria
    #: Note that the SAC algorithm will still be run to calculate
    #: ``T_critical_sac`` for use estimating initial ice particle number.
    #: Passing in a non-default value is unusual, but is included
    #: to allow for false negative calibration and model uncertainty studies.
    filter_sac: bool = True

    #: Filter out waypoints if they don't satisfy the initial persistent criteria
    #: Passing in a non-default value is unusual, but is included
    #: to allow for false negative calibration and model uncertainty studies.
    filter_initially_persistent: bool = True

    #: Continue evolving contrail waypoints ``persistent_buffer`` beyond
    #: end of contrail life.
    #: Passing in a non-default value is unusual, but is included
    #: to allow for false negative calibration and model uncertainty studies.
    persistent_buffer: np.timedelta64 | None = None

    # -------
    # Outputs
    # -------

    #: Add additional values to the flight and contrail that are not explicitly
    #: necessary for calculation.
    #: See also :attr:`CocipGridParams.verbose_outputs_formation` and
    #: :attr:`CocipGridParams.verbose_outputs_evolution`.
    verbose_outputs: bool = False

    #: Add additional metric of ATR20 and global yearly mean RF to model output.
    #: These are not standard CoCiP outputs but based on the derivation used
    #: in the first supplement to :cite:`yinPredictingClimateImpact2023`. ATR20 is defined
    #: as the average temperature response over a 20 year horizon.
    #:
    #: .. versionadded:: 0.50.0
    compute_atr20: bool = False

    #: Constant factor used to convert global- and year-mean RF, [:math:`W m^{-2}`],
    #: to ATR20, [:math:`K`], given by :cite:`yinPredictingClimateImpact2023`.
    #:
    #: .. versionadded:: 0.50.0
    global_rf_to_atr20_factor: float = 0.0151

    # ----------------
    # Model parameters
    # ----------------

    #: Initial wake vortex depth scaling factor.
    #: This factor scales max contrail downward displacement after the wake vortex phase
    #: to set the initial contrail depth.
    #: Denoted :math:`C_{D0}` in eq (14) in :cite:`schumannContrailCirrusPrediction2012`.
    initial_wake_vortex_depth: float = 0.5

    #: Sedimentation impact factor. Denoted by :math:`f_{T}` in eq. (35) of
    #: :cite:`schumannContrailCirrusPrediction2012`.
    #: Schumann describes this as "an important adjustable parameter", and sets
    #: it to 0.1 in the original publication. In :cite:`schumannAviationinducedCirrusRadiation2013`,
    #: a value of 0.5 is referenced after comparing CoCiP predictions to observations.
    sedimentation_impact_factor: float = 0.5

    #: Default ``nvpm_ei_n`` value if no data provided and emissions calculations fails.
    default_nvpm_ei_n: float = EmissionsParams.default_nvpm_ei_n

    #: Parameter denoted by :math:`n` in eq. (39) of :cite:`schumannContrailCirrusPrediction2012`.
    wind_shear_enhancement_exponent: float = 0.5

    #: Multiply flight black carbon number by enhancement factor.
    #: A value of 1.0 provides no scaling.
    #: Primarily used to support uncertainty estimation.
    nvpm_ei_n_enhancement_factor: float = 1.0

    #: Lower bound for ``nvpm_ei_n`` to account for ambient aerosol
    #: particles for newer engines, [:math:`kg^{-1}`]
    min_ice_particle_number_nvpm_ei_n: float = 1e13

    #: Upper bound for contrail plume depth, constraining it to realistic values.
    #: CoCiP only uses the ambient conditions at the mid-point of the Gaussian plume,
    #: and the edges could be in subsaturated conditions and sublimate. Important when
    #: :attr:`radiative_heating_effects` is enabled.
    max_depth: float = 1500.0

    #: Experimental. Improved ice crystal number survival fraction in the wake vortex phase.
    #: Implement :cite:`lottermoserHighResolutionEarlyContrails2025`, who developed a
    #: parametric model that estimates the survival fraction of the contrail ice crystal
    #: number after the wake vortex phase based on the results from large eddy simulations.
    #: This replicates Fig. 4 of :cite:`karcherFormationRadiativeForcing2018`.
    #:
    #:  .. versionadded:: 0.50.1
    #:  .. versionchanged:: 0.54.7
    unterstrasser_ice_survival_fraction: bool = False

    #: Experimental. Radiative heating effects on contrail cirrus properties.
    #: Terrestrial and solar radiances warm the contrail ice particles and cause
    #: convective turbulence. This effect is expected to enhance vertical mixing
    #: and reduce the lifetime of contrail cirrus. This parameter is experimental,
    #: and the CoCiP implementation of this parameter may change.
    #:
    #:  .. versionadded:: 0.28.9
    radiative_heating_effects: bool = False

    #: Experimental. Radiative effects due to contrail-contrail overlapping
    #: Account for change in local contrail shortwave and longwave radiative forcing
    #: due to contrail-contrail overlapping.
    #:
    #: .. versionadded:: 0.45
    contrail_contrail_overlapping: bool = False

    #: Experimental. Contrail-contrail overlapping altitude interval
    #: If :attr:`contrail_contrail_overlapping` is set to True, contrails will be grouped into
    #: altitude intervals, and all contrails within each altitude interval are treated as one
    #: contrail layer where they do not overlap.
    dz_overlap_m: float = 500.0

    #: Radius threshold for regime bins, [:math:`\mu m`]
    #: This is the row index label for ``habit_distributions``.
    #: See Table 2 in :cite:`schumannEffectiveRadiusIce2011`.
    radius_threshold_um: np.ndarray = dataclasses.field(default_factory=_radius_threshold_um)

    #: Particle habit (shape) types.
    #: This is the column index label for ``habit_distributions``.
    #: See Table 2 in :cite:`schumannEffectiveRadiusIce2011`.
    habits: np.ndarray = dataclasses.field(default_factory=_habits)

    #: Mix of ice particle habits in each radius regime.
    #: Rows indexes are ``radius_threshold_um`` elements.
    #: Columns indexes are ``habits`` particle habit type.
    #: See Table 2 from :cite:`schumannEffectiveRadiusIce2011`.
    habit_distributions: np.ndarray = dataclasses.field(default_factory=_habit_distributions)

    #: Scale shortwave radiative forcing.
    #: Primarily used to support uncertainty estimation.
    rf_sw_enhancement_factor: float = 1.0

    #: Scale longwave radiative forcing.
    #: Primarily used to support uncertainty estimation.
    rf_lw_enhancement_factor: float = 1.0

    # ---------------------------------------
    # Conditions for end of contrail lifetime
    # ---------------------------------------

    #: Minimum altitude domain in simulation, [:math:`m`]
    #: If set to ``None``, this check is disabled.
    min_altitude_m: float | None = 6000.0

    #: Maximum altitude domain in simulation, [:math:`m`]
    #: If set to ``None``, this check is disabled.
    max_altitude_m: float | None = 13000.0

    #: Maximum contrail segment length in simulation to prevent unrealistic values, [:math:`m`].
    max_seg_length_m: float = 40000.0

    #: Max age of contrail evolution.
    max_age: np.timedelta64 = np.timedelta64(20, "h")

    #: Minimum contrail optical depth.
    min_tau: float = 1e-6

    #: Maximum contrail optical depth to prevent unrealistic values.
    max_tau: float = 1e10

    #: Minimum contrail ice particle number per volume of air.
    min_n_ice_per_m3: float = 1e3

    #: Maximum contrail ice particle number per volume of air to prevent unrealistic values.
    max_n_ice_per_m3: float = 1e20


@dataclasses.dataclass
class CocipFlightParams(CocipParams):
    """Flight specific CoCiP parameters."""

    #: Aircraft performance model
    aircraft_performance: AircraftPerformance | None = None
