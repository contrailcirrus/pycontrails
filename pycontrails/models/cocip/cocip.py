"""Top level CoCiP classes and methods."""

from __future__ import annotations

import logging
import sys
import warnings
from collections.abc import Sequence
from typing import Any, Literal, NoReturn, overload

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pycontrails.core import met_var, models
from pycontrails.core.aircraft_performance import AircraftPerformance
from pycontrails.core.fleet import Fleet
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import MetVariable
from pycontrails.core.models import Model, interpolate_met
from pycontrails.core.vector import GeoVectorDataset, VectorDataDict
from pycontrails.datalib import ecmwf, gfs
from pycontrails.models import sac, tau_cirrus
from pycontrails.models.cocip import (
    contrail_properties,
    radiative_forcing,
    radiative_heating,
    unterstrasser_wake_vortex,
    wake_vortex,
    wind_shear,
)
from pycontrails.models.cocip.cocip_params import CocipFlightParams
from pycontrails.models.emissions.emissions import Emissions
from pycontrails.physics import constants, geo, thermo, units

logger = logging.getLogger(__name__)


class Cocip(Model):
    r"""Contrail Cirrus Prediction Model (CoCiP).

    Published by Ulrich Schumann *et. al.*
    (`DLR Institute of Atmospheric Physics <https://www.dlr.de/pa/en/>`_)
    in :cite:`schumannContrailCirrusPrediction2012`, :cite:`schumannParametricRadiativeForcing2012`.

    Parameters
    ----------
    met : MetDataset
        Pressure level dataset containing :attr:`met_variables` variables.
        See *Notes* for variable names by data source.
    rad : MetDataset
        Single level dataset containing top of atmosphere radiation fluxes.
        See *Notes* for variable names by data source.
    params : dict[str, Any], optional
        Override Cocip model parameters with dictionary.
        See :class:`CocipFlightParams` for model parameters.
    **params_kwargs : Any
        Override Cocip model parameters with keyword arguments.
        See :class:`CocipFlightParams` for model parameters.

    Notes
    -----
    **Inputs**

    The required meteorology variables depend on the data source. :class:`Cocip`
    supports data-source-specific variables from ECMWF models (HRES, ERA5) and the NCEP GFS, plus
    a generic set of model-agnostic variables.

    See :attr:`met_variables` and :attr:`rad_variables` for the list of required variables
    to the ``met`` and ``rad`` parameters, respectively.
    When an item in one of these arrays is a :class:`tuple`, variable keys depend on data source.

    A warning will be raised if meteorology data is from a source not currently supported by
    a pycontrails datalib. In this case it is the responsibility of the user to ensure that
    meteorology data is formatted correctly. The warning can be suppressed with a context manager:

    .. code-block:: python
        :emphasize-lines: 2,3

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning, message="Unknown provider")
            cocip = Cocip(met, rad, ...)

    The current list of required variables (labelled by ``"standard_name"``):

    .. list-table:: Variable keys for pressure level data
        :header-rows: 1

        * - Parameter
          - ECMWF
          - GFS
          - Generic
        * - Air Temperature
          - ``air_temperature``
          - ``air_temperature``
          - ``air_temperature``
        * - Specific Humidity
          - ``specific_humidity``
          - ``specific_humidity``
          - ``specific_humidity``
        * - Eastward wind
          - ``eastward_wind``
          - ``eastward_wind``
          - ``eastward_wind``
        * - Northward wind
          - ``northward_wind``
          - ``northward_wind``
          - ``northward_wind``
        * - Vertical velocity
          - ``lagrangian_tendency_of_air_pressure``
          - ``lagrangian_tendency_of_air_pressure``
          - ``lagrangian_tendency_of_air_pressure``
        * - Ice water content
          - ``specific_cloud_ice_water_content``
          - ``ice_water_mixing_ratio``
          - ``mass_fraction_of_cloud_ice_in_air``

    .. list-table:: Variable keys for single-level radiation data
        :header-rows: 1

        * - Parameter
          - ECMWF
          - GFS
          - Generic
        * - Top solar radiation
          - ``top_net_solar_radiation``
          - ``toa_upward_shortwave_flux``
          - ``toa_net_downward_shortwave_flux``
        * - Top thermal radiation
          - ``top_net_thermal_radiation``
          - ``toa_upward_longwave_flux``
          - ``toa_outgoing_longwave_flux``

    **Modifications**

    This implementation differs from original CoCiP (Fortran) implementation in a few places:

    - This model uses aircraft performance and emissions models to calculate nvPM, fuel flow,
      and overall propulsion efficiency, if not already provided.
    - As described in :cite:`teohAviationContrailClimate2022`, this implementation sets
      the initial ice particle activation rate to be a function of
      the difference between the ambient temperature and the critical SAC threshold temperature.
      See :func:`pycontrails.models.sac.T_critical_sac`.
    - Isobaric heat capacity calculation.
      The original model uses a constant value of 1004 :math:`J \ kg^{-1} \ K^{-1}`,
      whereas this model calculates isobaric heat capacity as a function of specific humidity.
      See :func:`pycontrails.physics.thermo.c_pm`.
    - Solar direct radiation.
      The original algorithm uses ECMWF radiation variable `tisr` (top incident solar radiation)
      as solar direct radiation value.
      This implementation calculates the theoretical solar direct radiation at
      any arbitrary point in the atmosphere.
      See :func:`pycontrails.physics.geo.solar_direct_radiation`.
    - Segment angle.
      The segment angle calculations for flights and contrail segments have been updated
      to use more precise spherical geometry instead of a triangular approximation.
      As the triangle approaches zero, the two calculations agree.
      See :func:`pycontrails.physics.geo.segment_angle`.
    - Integration.
      This implementation consistently uses left-Riemann sums
      in the time integration of contrail segments.
    - Segment length ratio.
      Instead of taking a geometric mean between contrail segments before/after advection,
      a simple ratio is computed.
      See :func:`contrail_properties.segment_length_ratio`.
    - Segment energy flux.
      This implementation does not average spatially contiguous contrail segments when calculating
      the mean energy flux for the segment of interest.
      See :func:`contrail_properties.mean_energy_flux_per_m`.

    This implementation is regression tested against
    results from :cite:`teohAviationContrailClimate2022`.

    **Outputs**

    NaN values may appear in model output. Specifically, ``np.nan`` values are used to indicate:

    - Flight waypoint or contrail waypoint is not contained with the :attr:`met` domain.
    - The variable was NOT computed during the model evaluation. For example, at flight waypoints
      not producing any persistent contrails, "radiative" variables (``rsr``, ``olr``, ``rf_sw``,
      ``rf_lw``, ``rf_net``) are not computed. Consequently, the corresponding values in the output
      of :meth:`eval` are NaN. One exception to this rule is found on ``ef`` (energy forcing)
      `contrail_age` predictions. For these two "cumulative" variables, waypoints not producing
      any persistent contrails are assigned 0 values.

    References
    ----------
    - :cite:`schumannDeterminationContrailsSatellite1990`
    - :cite:`schumannContrailCirrusPrediction2010`
    - :cite:`voigtInsituObservationsYoung2010`
    - :cite:`schumannPotentialReduceClimate2011`
    - :cite:`schumannContrailCirrusPrediction2012`
    - :cite:`schumannParametricRadiativeForcing2012`
    - :cite:`schumannContrailsVisibleAviation2012`
    - :cite:`schumannEffectiveRadiusIce2011`
    - :cite:`schumannDehydrationEffectsContrails2015`
    - :cite:`teohMitigatingClimateForcing2020`
    - :cite:`schumannAviationContrailCirrus2021`
    - :cite:`schumannAirTrafficContrail2021`
    - :cite:`teohAviationContrailClimate2022`

    See Also
    --------
    :class:`CocipFlightParams`
    :mod:`wake_vortex`
    :mod:`contrail_properties`
    :mod:`radiative_forcing`
    :mod:`humidity_scaling`
    :class:`Emissions`
    :mod:`sac`
    :mod:`tau_cirrus`
    """

    __slots__ = (
        "_downwash_contrail",
        "_downwash_flight",
        "_sac_flight",
        "contrail",
        "contrail_dataset",
        "contrail_list",
        "rad",
        "timesteps",
    )

    name = "cocip"
    long_name = "Contrail Cirrus Prediction Model"
    default_params = CocipFlightParams
    met_variables = (
        met_var.AirTemperature,
        met_var.SpecificHumidity,
        met_var.EastwardWind,
        met_var.NorthwardWind,
        met_var.VerticalVelocity,
        (
            met_var.MassFractionOfCloudIceInAir,
            ecmwf.SpecificCloudIceWaterContent,
            gfs.CloudIceWaterMixingRatio,
        ),
    )

    #: Required single-level top of atmosphere radiation variables.
    #: Variable keys depend on data source (e.g. ECMWF, GFS).
    rad_variables = (
        (
            met_var.TOANetDownwardShortwaveFlux,
            ecmwf.TopNetSolarRadiation,
            gfs.TOAUpwardShortwaveRadiation,
        ),
        (
            met_var.TOAOutgoingLongwaveFlux,
            ecmwf.TopNetThermalRadiation,
            gfs.TOAUpwardLongwaveRadiation,
        ),
    )

    #: Minimal set of met variables needed to run the model after pre-processing.
    #: The intention here is that ``ciwc`` is unnecessary after
    #: ``tau_cirrus`` has already been calculated.
    processed_met_variables = (
        met_var.AirTemperature,
        met_var.SpecificHumidity,
        met_var.EastwardWind,
        met_var.NorthwardWind,
        met_var.VerticalVelocity,
        tau_cirrus.TauCirrus,
    )

    #: Additional met variables used to support outputs
    #:
    #: .. versionchanged:: 0.48.0
    #:   Moved Geopotential from :attr:`met_variables` to :attr:`optional_met_variables`
    optional_met_variables = (
        (met_var.Geopotential, met_var.GeopotentialHeight),
        (
            met_var.CloudAreaFractionInAtmosphereLayer,
            ecmwf.CloudAreaFractionInLayer,
            gfs.TotalCloudCoverIsobaric,
        ),
    )

    #: Met data is not optional
    met: MetDataset
    met_required = True

    #: Radiation data formatted as a :class:`MetDataset` at a single pressure level [-1]
    rad: MetDataset

    #: Last Flight modeled in :meth:`eval`
    source: Flight | Fleet

    #: List of :class:`GeoVectorDataset` contrail objects - one for each timestep
    contrail_list: list[GeoVectorDataset]

    #: Contrail evolution output from model.
    #:
    #: Set to None when no contrails are formed.
    #: Otherwise, this is a :class:`pandas.DataFrame` describing the evolution of the contrail.
    #: Columns include:
    #:
    #: - ``waypoint``: The index of the waypoint in the original flight creating
    #:   the contrail. This can be used to join the contrail DataFrame to the :attr:`source`.
    #: - ``formation_time``: Time of contrail formation. Agrees with the ``time`` column
    #:   in :attr:`source`.
    #: - ``continuous``: Boolean indicating whether the contrail is continuous or not.
    #: - ``persistent``: Boolean indicating whether the contrail is persistent or not.
    #:   A contrail segment is considered continuous if both the current and the next
    #:   contrail waypoint at the same time step persist.
    #: - ``segment_length``: Length of the contrail segment, [:math:`m`].
    #: - ``sin_a``, ``cos_a``: Sine and cosine of the segment angle.
    #: - ``width``, ``depth``: Contrail width and depth, [:math:`m`].
    #: - ``sigma_yz``: The ``yz`` component of the covariance matrix, [:math:`m^{2}`].
    #:   See :func:`contrail_properties.plume_temporal_evolution`.
    #: - ``q_sat``: Saturation specific humidity over ice, [:math:`kg \ kg^{-1}`].
    #: - ``n_ice_per_m``: Number of ice particles per distance, [:math:`m^{-1}`].
    #: - ``iwc``: Ice water content, [:math:`kg_{ice} kg_{air}^{-1}`].
    #: - ``tau_contrail``: Optical depth of the contrail. See
    #:   :func:`contrail_properties.contrail_optical_depth`.
    #: - ``rf_sw``, ``rf_lw``, ``rf_net``: Shortwave, longwave, and net instantaneous
    #:   radiative forcing, [:math:`W \ m^{-2}`] at the contrail waypoint.
    #: - ``ef``: Energy forcing, [:math:`J`] at the contrail waypoint. See
    #:   :func:`contrail_properties.energy_forcing`.
    contrail: pd.DataFrame | None

    #: :class:`xr.Dataset` representation of contrail evolution.
    contrail_dataset: xr.Dataset | None

    #: Array of :class:`numpy.datetime64` time steps for contrail evolution
    timesteps: npt.NDArray[np.datetime64]

    #: Parallel copy of flight waypoints after SAC filter applied
    _sac_flight: Flight

    #: Parallel copy of flight waypoints after wake vortex downwash applied
    _downwash_flight: Flight

    #: GeoVectorDataset representation of :attr:`downwash_flight`
    _downwash_contrail: GeoVectorDataset

    def __init__(
        self,
        met: MetDataset,
        rad: MetDataset,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ) -> None:
        # call Model init
        super().__init__(met, params=params, **params_kwargs)

        compute_tau_cirrus = self.params["compute_tau_cirrus_in_model_init"]
        self.met, self.rad = process_met_datasets(met, rad, compute_tau_cirrus)

        # initialize outputs to None
        self.contrail = None
        self.contrail_dataset = None

    # ----------
    # Public API
    # ----------

    @overload
    def eval(self, source: Fleet, **params: Any) -> Fleet: ...

    @overload
    def eval(self, source: Flight, **params: Any) -> Flight: ...

    @overload
    def eval(self, source: Sequence[Flight], **params: Any) -> list[Flight]: ...

    @overload
    def eval(self, source: None = ..., **params: Any) -> NoReturn: ...

    def eval(
        self,
        source: Flight | Sequence[Flight] | None = None,
        **params: Any,
    ) -> Flight | list[Flight]:
        """Run CoCiP simulation on flight.

        Simulates the formation and evolution of contrails from a Flight
        using the contrail cirrus prediction model (CoCiP) from Schumann (2012)
        :cite:`schumannContrailCirrusPrediction2012`.

        .. versionchanged:: 0.25.11

            Previously, any waypoint not surviving the wake vortex downwash phase of CoCiP
            was assigned a nan-value in the ``ef`` array within the model
            output. This is no longer the case. Instead, energy forcing is set to 0.0
            for all waypoints which fail to produce persistent contrails. In particular,
            nan values in the ``ef`` array are only used to indicate an
            out-of-met-domain waypoint. The same convention is now used for output variables
            ``contrail_age`` and ``cocip`` as well.

        .. versionchanged::0.26.0

            :attr:`met` and :attr:`rad` down-selection is now handled automatically.

        Parameters
        ----------
        source : Flight | Sequence[Flight] | None
            Input Flight(s) to model.
        **params : Any
            Overwrite model parameters before eval.

        Returns
        -------
        Flight | list[Flight] | NoReturn
            Flight(s) with updated Contrail data. The model parameter "verbose_outputs"
            determines the variables on the return flight object.

        References
        ----------
        - :cite:`schumannContrailCirrusPrediction2012`
        """

        self.update_params(params)
        self.set_source(source)
        self.source = self.require_source_type(Flight)
        return_flight_list = isinstance(self.source, Fleet) and isinstance(source, Sequence)

        self._set_timesteps()

        # Downselect met for CoCiP initialization
        # We only need to buffer in the negative vertical direction,
        # which is the positive direction for level
        logger.debug("Downselect met for Cocip initialization")
        level_buffer = 0, self.params["met_level_buffer"][1]
        met = self.source.downselect_met(self.met, level_buffer=level_buffer)
        met = add_tau_cirrus(met)

        # Prepare flight for model
        self._process_flight(met)

        # Save humidity scaling type to output attrs
        # NOTE: Do this after _process_flight because that method automatically
        # broadcasts all numeric source params.
        humidity_scaling = self.params["humidity_scaling"]
        if humidity_scaling is not None:
            for k, v in humidity_scaling.description.items():
                self.source.attrs[f"humidity_scaling_{k}"] = v

        if isinstance(self.source, Fleet):
            label = f"fleet of {self.source.n_flights} flights"
        else:
            label = f"flight {self.source['flight_id'][0]}"

        logger.debug("Finding initial linear contrails with SAC")
        self._find_initial_contrail_regions()

        if not self._sac_flight:
            logger.debug("No linear contrails formed by %s", label)
            return self._fill_empty_flight_results(return_flight_list)

        self._simulate_wake_vortex_downwash(met)

        self._find_initial_persistent_contrails(met)

        if not self._downwash_flight:
            logger.debug("No persistent contrails formed by flight %s", label)
            return self._fill_empty_flight_results(return_flight_list)

        self.contrail_list = []
        self._simulate_contrail_evolution()

        if not self.contrail_list:
            logger.debug("No contrails formed by %s", label)
            return self._fill_empty_flight_results(return_flight_list)

        logger.debug("Complete contrail simulation for %s", label)

        self._cleanup_indices()
        self._bundle_results()

        if return_flight_list:
            return self.source.to_flight_list()  # type: ignore[attr-defined]

        return self.source

    @classmethod
    def generic_rad_variables(cls) -> tuple[MetVariable, ...]:
        """Return a model-agnostic list of required radiation variables.

        Returns
        -------
        tuple[MetVariable]
            List of model-agnostic variants of required variables
        """
        available = set(met_var.MET_VARIABLES)
        return tuple(models._find_match(required, available) for required in cls.rad_variables)

    @classmethod
    def ecmwf_rad_variables(cls) -> tuple[MetVariable, ...]:
        """Return an ECMWF-specific list of required radiation variables.

        Returns
        -------
        tuple[MetVariable]
            List of ECMWF-specific variants of required variables
        """
        available = set(ecmwf.ECMWF_VARIABLES)
        return tuple(models._find_match(required, available) for required in cls.rad_variables)

    @classmethod
    def gfs_rad_variables(cls) -> tuple[MetVariable, ...]:
        """Return a GFS-specific list of required radiation variables.

        Returns
        -------
        tuple[MetVariable]
            List of GFS-specific variants of required variables
        """
        available = set(gfs.GFS_VARIABLES)
        return tuple(models._find_match(required, available) for required in cls.rad_variables)

    def _set_timesteps(self) -> None:
        """Set the :attr:`timesteps` based on the ``source`` time range.

        This method is called in :meth:`eval` before the flight is processed.
        """
        if isinstance(self.source, Fleet):
            # time not sorted in Fleet instance
            tmin = self.source["time"].min()
            tmax = self.source["time"].max()
        else:
            tmin = self.source["time"][0]
            tmax = self.source["time"][-1]

        tmin = pd.to_datetime(tmin)
        tmax = pd.to_datetime(tmax)
        dt = pd.to_timedelta(self.params["dt_integration"])

        t_start = tmin.ceil(dt)
        t_end = tmax.floor(dt) + self.params["max_age"] + dt
        self.timesteps = np.arange(t_start, t_end, dt)

    # -------------
    # Model Methods
    # -------------

    def _process_flight(self, met: MetDataset) -> None:
        """Prepare :attr:`self.source` for use in model eval.

        Missing flight values should be prefilled before calling this method.
        This method modifies :attr:`self.source`.

        .. versionchanged:: 0.35.2

            No longer broadcast all numeric source params. Instead, numeric
            source params can be accessed with :meth:`Flight.get_data_or_attr`.

        Parameters
        ----------
        met : MetDataset
            Meteorology data

        Raises
        ------
        ValueError
            If non-sequential waypoints are found in ``self.source["waypoint"]``
            If there is no intersection between met domain and :attr:`source`.

        See Also
        --------
        :method:`Flight.resample_and_fill`
        """
        logger.debug("Pre-processing flight parameters")

        # STEP 1: Check for core columns
        # Attach level and altitude to avoid some redundancy
        self.source.setdefault("altitude", self.source.altitude)
        self.source.setdefault("level", self.source.level)
        self.source.setdefault("air_pressure", self.source.air_pressure)

        core_columns = ("longitude", "latitude", "altitude", "time")
        for col in core_columns:
            if np.isnan(self.source[col]).any():
                raise ValueError(
                    f"Parameter `flight` must not contain NaN values in {col} field."
                    "Call method `resample_and_fill` to clean up flight trajectory."
                )

        # STEP 2: Check for waypoints
        # Note: Fleet instances always have a waypoint column
        if "waypoint" not in self.source:
            # Give flight a range index
            self.source["waypoint"] = np.arange(self.source.size)
        elif not isinstance(self.source, Fleet) and not np.all(
            np.diff(self.source["waypoint"]) == 1
        ):
            # If self.source is a usual Flight (not the subclass Fleet)
            # and has "waypoint" data, we want to give a warning if the waypoint
            # data has an usual index. CoCiP uses the waypoint for its continuity
            # convention, so the waypoint data is critical.
            msg = (
                "Found non-sequential waypoints in flight key 'waypoint'. "
                "The CoCiP algorithm requires flight data key 'waypoint' "
                "to contain sequential waypoints if defined."
            )
            raise ValueError(msg)

        # STEP 3: Test met domain for some overlap
        # We attach the intersection to the source.
        # This is used in the function _fill_empty_flight_results
        intersection = self.source.coords_intersect_met(met)
        self.source["_met_intersection"] = intersection
        logger.debug(
            "Fraction of flight waypoints intersecting met domain: %s / %s",
            intersection.sum(),
            self.source.size,
        )
        if not intersection.any():
            msg = (
                "No intersection between flight waypoints and met domain. "
                "Rerun Cocip with met overlapping flight."
            )
            raise ValueError(msg)

        # STEP 4: Begin met interpolation
        # Unfortunately we use both "u_wind" and "eastward_wind" to refer to the
        # same variable, so the logic gets a bit more complex.
        humidity_scaling = self.params["humidity_scaling"]
        scale_humidity = humidity_scaling is not None and "specific_humidity" not in self.source
        verbose_outputs = self.params["verbose_outputs"]

        interp_kwargs = self.interp_kwargs
        if self.params["preprocess_lowmem"]:
            interp_kwargs["lowmem"] = True
        interpolate_met(met, self.source, "air_temperature", **interp_kwargs)
        interpolate_met(met, self.source, "specific_humidity", **interp_kwargs)
        interpolate_met(met, self.source, "eastward_wind", "u_wind", **interp_kwargs)
        interpolate_met(met, self.source, "northward_wind", "v_wind", **interp_kwargs)

        if scale_humidity:
            humidity_scaling.eval(self.source, copy_source=False)

        # if humidity_scaling isn't defined, add rhi to source for verbose_outputs
        elif verbose_outputs:
            self.source["rhi"] = thermo.rhi(
                self.source["specific_humidity"],
                self.source["air_temperature"],
                self.source.air_pressure,
            )

        # Cache extra met properties for post-analysis
        if verbose_outputs:
            interpolate_met(met, self.source, "tau_cirrus", **interp_kwargs)

            # handle ECMWF/GFS/generic ciwc variables
            if (key := "specific_cloud_ice_water_content") in met:  # noqa: SIM114
                interpolate_met(met, self.source, key, **interp_kwargs)
            elif (key := "ice_water_mixing_ratio") in met:  # noqa: SIM114
                interpolate_met(met, self.source, key, **interp_kwargs)
            elif (key := "mass_fraction_of_cloud_ice_in_air") in met:
                interpolate_met(met, self.source, key, **interp_kwargs)

            self.source["rho_air"] = thermo.rho_d(
                self.source["air_temperature"], self.source.air_pressure
            )
            self.source["sdr"] = geo.solar_direct_radiation(
                self.source["longitude"], self.source["latitude"], self.source["time"]
            )

        # STEP 5: Calculate segment-specific properties if they are not already attached
        if "true_airspeed" not in self.source:
            self.source["true_airspeed"] = self.source.segment_true_airspeed(
                u_wind=self.source["u_wind"],
                v_wind=self.source["v_wind"],
                smooth=self.params["smooth_true_airspeed"],
                window_length=self.params["smooth_true_airspeed_window_length"],
            )
        if "segment_length" not in self.source:
            self.source["segment_length"] = self.source.segment_length()

        max_ = self.source.max_distance_gap
        lim_ = self.params["max_seg_length_m"]
        if max_ > 0.9 * lim_:
            warnings.warn(
                "Flight trajectory has segment lengths close to or exceeding the "
                "'max_seg_length_m' parameter. Evolved contrail segments may reach "
                "their end of life artificially early. Either resample the flight "
                "with the 'resample_and_fill' method (recommended), or use a larger "
                "'max_seg_length_m' parameter. Current values: "
                f"max_seg_length_m={lim_}, max_seg_length_on_trajectory={max_}"
            )

        # if either angle is not provided, re-calculate both
        # this is the one case where we will overwrite a flight data key
        if "sin_a" not in self.source or "cos_a" not in self.source:
            self.source["sin_a"], self.source["cos_a"] = self.source.segment_angle()

        # STEP 6: Calculate emissions if requested, or if keys don't exist in Flight
        if self.params["process_emissions"]:
            self._process_emissions()

        # STEP 7: Ensure that flight has the required variables defined as attrs or columns
        self.source.ensure_vars(_emissions_variables())

    def _process_emissions(self) -> None:
        """Process flight emissions.

        See :class:`Emissions`.

        We should consider supporting OpenAP (https://github.com/TUDelft-CNS-ATM/openap)
        and alternate performance models in the future.
        """
        logger.debug("Processing flight emissions")

        # Call aircraft performance and Emissions models as needed
        # NOTE: None of these sub-models actually do any met interpolation -- self.source already
        # has all of the required met variables attached. Therefore, we don't need to worry about
        # being consistent with passing in Cocip's interp_kwargs and humidity_scaling into
        # the sub-models.
        emissions = Emissions()
        ap_model = self.params["aircraft_performance"]

        # Run against a list of flights (Fleet)
        if isinstance(self.source, Fleet):
            # Rip the Fleet apart, run BADA on each, then reassemble
            logger.debug("Separately running aircraft performance on each flight in fleet")
            fls = self.source.to_flight_list(copy=False)
            fls = [_eval_aircraft_performance(ap_model, fl) for fl in fls]

            # In Fleet-mode, always call emissions
            logger.debug("Separately running emissions on each flight in fleet")
            fls = [_eval_emissions(emissions, fl) for fl in fls]

            # Broadcast numeric AP and emissions variables back to Fleet.data
            for fl in fls:
                fl.broadcast_attrs(_emissions_variables(), raise_error=False)

            # Convert back to fleet
            attrs = self.source.attrs
            attrs.pop("fl_attrs", None)
            attrs.pop("data_keys", None)
            self.source = Fleet.from_seq(fls, broadcast_numeric=False, attrs=attrs)

        # Single flight
        else:
            self.source = _eval_aircraft_performance(ap_model, self.source)
            self.source = _eval_emissions(emissions, self.source)

        # Scale nvPM with parameter (fleet / flight)
        factor = self.params["nvpm_ei_n_enhancement_factor"]
        try:
            self.source["nvpm_ei_n"] *= factor
        except KeyError:
            self.source.attrs.update({"nvpm_ei_n": self.source.attrs["nvpm_ei_n"] * factor})

    def _find_initial_contrail_regions(self) -> None:
        """Apply Schmidt-Appleman criteria to determine regions of persistent contrail formation.

        This method:
            - Modifies :attr:`flight` in place by assigning additional columns arising from SAC
            - Creates :attr:`_sac_flight` needed for :meth:`_simulate_wave_vortex_downwash`.
        """
        # calculate the SAC for each waypoint
        sac_model = sac.SAC(
            met=None,  # self.source is already interpolated against met, so we don't need it
            copy_source=False,
        )
        self.source = sac_model.eval(source=self.source)

        # Estimate SAC threshold temperature along initial contrail
        # This variable is not involved in calculating the SAC,
        # but it is required by `contrail_properties.ice_particle_number`.
        # The three variables used below are all computed in sac_model.eval
        self.source["T_critical_sac"] = sac.T_critical_sac(
            self.source["T_sat_liquid"],
            self.source["rh"],
            self.source["G"],
        )

        # create a new Flight only at points where "sac" == 1
        filt = self.source["sac"] == 1.0
        if self.params["filter_sac"]:
            self._sac_flight = self.source.filter(filt)
            logger.debug(
                "Fraction of waypoints satisfying the SAC: %s / %s",
                len(self._sac_flight),
                len(self.source),
            )
            return

        warnings.warn("Manually overriding SAC filter")
        logger.info("Manually overriding SAC filter")
        self._sac_flight = self.source.copy()
        logger.debug(
            "Fraction of waypoints satisfying the SAC: %s / %s",
            np.sum(filt),
            self._sac_flight.size,
        )
        logger.debug("None are filtered out!")

    def _simulate_wake_vortex_downwash(self, met: MetDataset) -> None:
        """Apply wake vortex model to calculate initial contrail geometry.

        The calculation uses a parametric wake vortex transport and decay model to simulate the
        wake vortex phase and obtain the initial contrail width, depth and downward displacement.

        This method assigns additional columns "width" and "depth" to :attr:`_sac_flight`

        Parameters
        ----------
        met : MetDataset
            Meteorology data

        References
        ----------
        - :cite:`holzapfelProbabilisticTwoPhaseWake2003`
        """
        air_temperature = self._sac_flight["air_temperature"]
        u_wind = self._sac_flight["u_wind"]
        v_wind = self._sac_flight["v_wind"]
        air_pressure = self._sac_flight.air_pressure

        # flight parameters
        aircraft_mass = self._sac_flight.get_data_or_attr("aircraft_mass")
        true_airspeed = self._sac_flight["true_airspeed"]

        # In Fleet-mode, wingspan resides on `data`, and in Flight-mode,
        # wingspan resides on `attrs`.
        wingspan = self._sac_flight.get_data_or_attr("wingspan")

        # get the pressure level `dz_m` lower than element pressure
        dz_m = self.params["dz_m"]
        air_pressure_lower = thermo.pressure_dz(air_temperature, air_pressure, dz_m)
        level_lower = air_pressure_lower / 100.0

        # get full met grid or flight data interpolated to the pressure level `p_dz`
        interp_kwargs = self.interp_kwargs
        if self.params["preprocess_lowmem"]:
            interp_kwargs["lowmem"] = True
        air_temperature_lower = interpolate_met(
            met,
            self._sac_flight,
            "air_temperature",
            "air_temperature_lower",
            level=level_lower,
            **interp_kwargs,
        )
        u_wind_lower = interpolate_met(
            met,
            self._sac_flight,
            "eastward_wind",
            "u_wind_lower",
            level=level_lower,
            **interp_kwargs,
        )
        v_wind_lower = interpolate_met(
            met,
            self._sac_flight,
            "northward_wind",
            "v_wind_lower",
            level=level_lower,
            **interp_kwargs,
        )

        # Temperature gradient
        dT_dz = self._sac_flight["dT_dz"] = thermo.T_potential_gradient(
            air_temperature,
            air_pressure,
            air_temperature_lower,
            air_pressure_lower,
            dz_m,
        )

        # wind shear
        ds_dz = self._sac_flight["ds_dz"] = wind_shear.wind_shear(
            u_wind, u_wind_lower, v_wind, v_wind_lower, dz_m
        )

        # Initial contrail width, depth and downward displacement
        dz_max = self._sac_flight["dz_max"] = wake_vortex.max_downward_displacement(
            wingspan,
            true_airspeed,
            aircraft_mass,
            air_temperature,
            dT_dz,
            ds_dz,
            air_pressure,
            effective_vertical_resolution=self.params["effective_vertical_resolution"],
            wind_shear_enhancement_exponent=self.params["wind_shear_enhancement_exponent"],
        )

        # derive downwash values and save to data model
        self._sac_flight["width"] = wake_vortex.initial_contrail_width(wingspan, dz_max)
        initial_wake_vortex_depth = self.params["initial_wake_vortex_depth"]
        self._sac_flight["depth"] = wake_vortex.initial_contrail_depth(
            dz_max, initial_wake_vortex_depth
        )
        # Initially, sigma_yz is set to 0
        # See bottom left paragraph p. 552 Schumann 2012 beginning with:
        # >>> "The contrail model starts from initial values ..."
        self._sac_flight["sigma_yz"] = np.zeros_like(dz_max)

    def _find_initial_persistent_contrails(self, met: MetDataset) -> None:
        """Calculate the initial contrail properties after the wake vortex phase.

        Determine points with initial persistent contrails.

        This method first calculates ice water content (``iwc``) and number of ice particles per
        distance (``n_ice_per_m_1``).

        It then tests each contrail waypoint for initial persistence and calculates
        parameters for the first iteration of the time integration.

        Note that the subscript "_1" represents the conditions after the wake vortex phase.

        Parameters
        ----------
        met : MetDataset
            Meteorology data
        """

        # met parameters along Flight path
        air_pressure = self._sac_flight.air_pressure
        air_temperature = self._sac_flight["air_temperature"]
        specific_humidity = self._sac_flight["specific_humidity"]
        T_critical_sac = self._sac_flight["T_critical_sac"]

        # Flight performance parameters
        fuel_flow = self._sac_flight.get_data_or_attr("fuel_flow")
        true_airspeed = self._sac_flight["true_airspeed"]
        fuel_dist = fuel_flow / true_airspeed

        nvpm_ei_n = self._sac_flight.get_data_or_attr("nvpm_ei_n")
        ei_h2o = self._sac_flight.fuel.ei_h2o

        # get initial contrail parameters from wake vortex simulation
        width = self._sac_flight["width"]
        depth = self._sac_flight["depth"]

        # initial contrail altitude set to 0.5 * depth
        contrail_1 = GeoVectorDataset(
            longitude=self._sac_flight["longitude"],
            latitude=self._sac_flight["latitude"],
            altitude=self._sac_flight.altitude - 0.5 * depth,
            time=self._sac_flight["time"],
            copy=False,
        )

        # get met post wake vortex along initial contrail
        interp_kwargs = self.interp_kwargs
        if self.params["preprocess_lowmem"]:
            interp_kwargs["lowmem"] = True
        air_temperature_1 = interpolate_met(met, contrail_1, "air_temperature", **interp_kwargs)
        interpolate_met(met, contrail_1, "specific_humidity", **interp_kwargs)

        humidity_scaling = self.params["humidity_scaling"]
        if humidity_scaling is not None:
            humidity_scaling.eval(contrail_1, copy_source=False)
        else:
            contrail_1["air_pressure"] = contrail_1.air_pressure
            contrail_1["rhi"] = thermo.rhi(
                contrail_1["specific_humidity"], air_temperature_1, contrail_1["air_pressure"]
            )

        air_pressure_1 = contrail_1["air_pressure"]
        specific_humidity_1 = contrail_1["specific_humidity"]
        rhi_1 = contrail_1["rhi"]
        level_1 = contrail_1.level
        altitude_1 = contrail_1.altitude

        # calculate thermo properties post wake vortex
        q_sat_1 = thermo.q_sat_ice(air_temperature_1, air_pressure_1)
        rho_air_1 = thermo.rho_d(air_temperature_1, air_pressure_1)

        # Initialize initial contrail properties
        iwc = contrail_properties.initial_iwc(
            air_temperature, specific_humidity, air_pressure, fuel_dist, width, depth, ei_h2o
        )
        iwc_ad = contrail_properties.iwc_adiabatic_heating(
            air_temperature, air_pressure, air_pressure_1
        )
        iwc_1 = contrail_properties.iwc_post_wake_vortex(iwc, iwc_ad)

        if self.params["unterstrasser_ice_survival_fraction"]:
            wingspan = self._sac_flight.get_data_or_attr("wingspan")
            rhi_0 = thermo.rhi(specific_humidity, air_temperature, air_pressure)
            f_surv = unterstrasser_wake_vortex.ice_particle_number_survival_fraction(
                air_temperature,
                rhi_0,
                ei_h2o,
                wingspan,
                true_airspeed,
                fuel_flow,
                nvpm_ei_n,
                0.5 * depth,  # Taking the mid-point of the contrail plume
            )
        else:
            f_surv = contrail_properties.ice_particle_survival_fraction(iwc, iwc_1)

        n_ice_per_m_0 = contrail_properties.initial_ice_particle_number(
            nvpm_ei_n=nvpm_ei_n,
            fuel_dist=fuel_dist,
            air_temperature=air_temperature,
            T_crit_sac=T_critical_sac,
            min_ice_particle_number_nvpm_ei_n=self.params["min_ice_particle_number_nvpm_ei_n"],
        )
        n_ice_per_m_1 = n_ice_per_m_0 * f_surv

        # Check for persistent initial_contrails
        persistent_1 = contrail_properties.initial_persistent(iwc_1, rhi_1)

        self._sac_flight["altitude_1"] = altitude_1
        self._sac_flight["level_1"] = level_1
        self._sac_flight["air_temperature_1"] = air_temperature_1
        self._sac_flight["specific_humidity_1"] = specific_humidity_1
        self._sac_flight["q_sat_1"] = q_sat_1
        self._sac_flight["air_pressure_1"] = air_pressure_1
        self._sac_flight["rho_air_1"] = rho_air_1
        self._sac_flight["rhi_1"] = rhi_1
        self._sac_flight["iwc_1"] = iwc_1
        self._sac_flight["f_surv"] = f_surv
        self._sac_flight["n_ice_per_m_0"] = n_ice_per_m_0
        self._sac_flight["n_ice_per_m_1"] = n_ice_per_m_1
        self._sac_flight["persistent_1"] = persistent_1

        # Create new Flight only at persistent points
        if self.params["filter_initially_persistent"]:
            self._downwash_flight = self._sac_flight.filter(persistent_1.astype(bool))
            logger.debug(
                "Fraction of waypoints with initially persistent contrails: %s / %s",
                len(self._downwash_flight),
                len(self._sac_flight),
            )

        else:
            warnings.warn("Manually overriding initially persistent filter")
            logger.info("Manually overriding initially persistent filter")
            self._downwash_flight = self._sac_flight.copy()
            logger.debug(
                "Fraction of waypoints with initially persistent contrails: %s / %s",
                persistent_1.sum(),
                persistent_1.size,
            )
            logger.debug("None are filtered out!")

    def _process_downwash_flight(self) -> tuple[MetDataset | None, MetDataset | None]:
        """Create and calculate properties of contrails created by downwash vortex.

        ``_downwash_contrail`` is a contrail representation of the waypoints of
        ``_downwash_flight``, which has already been filtered for initial persistent waypoints.

        Returns MetDatasets for subsequent use if ``preprocess_lowmem=False``.
        """
        self._downwash_contrail = self._create_downwash_contrail()
        buffers = {
            f"{coord}_buffer": self.params[f"met_{coord}_buffer"]
            for coord in ("longitude", "latitude", "level")
        }
        logger.debug("Downselect met for start of Cocip evolution")
        met = self._downwash_contrail.downselect_met(self.met, **buffers)
        met = add_tau_cirrus(met)
        rad = self._downwash_contrail.downselect_met(self.rad, **buffers)

        calc_continuous(self._downwash_contrail)
        calc_timestep_geometry(self._downwash_contrail)

        interp_kwargs = self.interp_kwargs
        if self.params["preprocess_lowmem"]:
            interp_kwargs["lowmem"] = True
        calc_timestep_meteorology(self._downwash_contrail, met, self.params, **interp_kwargs)
        calc_shortwave_radiation(rad, self._downwash_contrail, **interp_kwargs)
        calc_outgoing_longwave_radiation(rad, self._downwash_contrail, **interp_kwargs)
        calc_contrail_properties(
            self._downwash_contrail,
            self.params["effective_vertical_resolution"],
            self.params["wind_shear_enhancement_exponent"],
            self.params["sedimentation_impact_factor"],
            self.params["radiative_heating_effects"],
        )

        # Intersect with rad dataset
        calc_radiative_properties(self._downwash_contrail, self.params)

        if self.params["preprocess_lowmem"]:
            return None, None
        return met, rad

    def _simulate_contrail_evolution(self) -> None:
        """Simulate contrail evolution."""

        met, rad = self._process_downwash_flight()
        interp_kwargs = self.interp_kwargs

        contrail_contrail_overlapping = self.params["contrail_contrail_overlapping"]
        if contrail_contrail_overlapping and not isinstance(self.source, Fleet):
            warnings.warn("Contrail-Contrail Overlapping is only valid for Fleet mode.")

        # Complete iteration at time_idx - 2
        for time_idx, time_end in enumerate(self.timesteps[:-1]):
            logger.debug("Start time integration step %s ending at time %s", time_idx, time_end)

            # get the last evolution step of contrail waypoints, if it exists
            latest_contrail = self.contrail_list[-1] if self.contrail_list else GeoVectorDataset()

            # load new contrail segments from downwash_flight
            contrail_2_segments = self._get_contrail_2_segments(time_idx)
            if contrail_2_segments:
                logger.debug(
                    "Discover %s new contrail waypoints formed by downwash_flight waypoints.",
                    contrail_2_segments.size,
                )
                logger.debug("Previously persistent contrail size: %s", latest_contrail.size)

                # Append new waypoints to latest contrail, or set as first contrail waypoints
                latest_contrail = latest_contrail + contrail_2_segments

                # CRITICAL: When running in "Fleet" mode, the latest_contrail is no longer
                # sorted by flight_id. Fixing that below.
                if isinstance(self.source, Fleet):
                    latest_contrail = latest_contrail.sort(["flight_id", "time"])

            # Check for an empty contrail
            if not latest_contrail:
                logger.debug("Empty latest_contrail at timestep %s", time_end)
                if np.all(time_end > self._downwash_contrail["time"]):
                    logger.debug("No remaining downwash_contrail waypoints. Break.")
                    break
                continue

            # Update met, rad slices as needed
            met, rad = self._maybe_downselect_met_rad(met, rad, latest_contrail, time_end)

            # Recalculate latest_contrail with new values
            # NOTE: We are doing a substantial amount of redundant computation here
            # At waypoints for which the continuity hasn't changed, there is nothing
            # new going on, and so we are overwriting variables in latest_contrail
            # with the same values
            # NOTE: Both latest_contrail and contrail_2_segments contain all
            # 54 variables. The only difference is the change in continuity.
            # The change in continuity impacts:
            # - sin_a, cos_a, segment_length (in calc_timestep_geometry)
            # - dsn_dz (in calc_timestep_meteorology, then enhanced in contrail_properties)
            # And there is huge room to optimize this
            calc_continuous(latest_contrail)
            calc_timestep_geometry(latest_contrail)
            calc_timestep_meteorology(latest_contrail, met, self.params, **interp_kwargs)
            calc_contrail_properties(
                latest_contrail,
                self.params["effective_vertical_resolution"],
                self.params["wind_shear_enhancement_exponent"],
                self.params["sedimentation_impact_factor"],
                self.params["radiative_heating_effects"],
            )

            final_contrail = calc_timestep_contrail_evolution(
                met=met,
                rad=rad,
                contrail_1=latest_contrail,
                time_2=time_end,
                params=self.params,
                **interp_kwargs,
            )

            if contrail_contrail_overlapping:
                final_contrail = _contrail_contrail_overlapping(final_contrail, self.params)

            self.contrail_list.append(final_contrail)

    def _maybe_downselect_met_rad(
        self,
        met: MetDataset | None,
        rad: MetDataset | None,
        latest_contrail: GeoVectorDataset,
        time_end: np.datetime64,
    ) -> tuple[MetDataset, MetDataset]:
        """Downselect ``self.met`` and ``self.rad`` if necessary to cover ``time_end``.

        If current ``met`` and ``rad`` slices to not include ``time_end``, new slices are selected
        from ``self.met`` and ``self.rad``. Downselection in space will cover
        - locations of current contrails (``latest_contrail``),
        - locations of additional contrails that will be loaded from ``self._downwash_flight``
          before the new slices expire,
        plus a user-defined buffer.
        """
        if met is None or time_end > met.indexes["time"].to_numpy()[-1]:
            logger.debug("Downselect met at time_end %s within Cocip evolution", time_end)
            met = self._definitely_downselect_met_or_rad(self.met, latest_contrail, time_end)
            met = add_tau_cirrus(met)

        if rad is None or time_end > rad.indexes["time"].to_numpy()[-1]:
            logger.debug("Downselect rad at time_end %s within Cocip evolution", time_end)
            rad = self._definitely_downselect_met_or_rad(self.rad, latest_contrail, time_end)

        return met, rad

    def _definitely_downselect_met_or_rad(
        self, met: MetDataset, latest_contrail: GeoVectorDataset, time_end: np.datetime64
    ) -> MetDataset:
        """Perform downselection when required by :meth:`_maybe_downselect_met_rad`.

        Downselects ``met`` (which should be one of ``self.met`` or ``self.rad``)
        to cover ``time_end``. Downselection in space covers
        - locations of current contrails (``latest_contrail``),
        - locations of additional contrails that will be loaded from ``self._downwash_flight``
          before the new slices expire,
        plus a user-defined buffer, as described in :meth:`_maybe_downselect_met_rad`.
        """
        # compute lookahead for future contrails from downwash_flight
        met_time = met.indexes["time"].to_numpy()
        mask = met_time >= time_end
        lookahead = np.min(met_time[mask]) if np.any(mask) else time_end

        # create vector for downselection based on current + future contrails
        future_contrails = self._downwash_flight.filter(
            (self._downwash_flight["time"] >= time_end)
            & (self._downwash_flight["time"] <= lookahead),
            copy=False,
        )
        vector = GeoVectorDataset._from_fastpath(
            {
                key: np.concatenate((latest_contrail[key], future_contrails[key]))
                for key in ("longitude", "latitude", "level", "time")
            },
        )

        # compute time buffer to ensure downselection extends to time_end
        buffers = {
            f"{coord}_buffer": self.params[f"met_{coord}_buffer"]
            for coord in ("longitude", "latitude", "level")
        }
        buffers["time_buffer"] = (
            np.timedelta64(0, "ns"),
            max(np.timedelta64(0, "ns"), time_end - vector["time"].max()),
        )

        return vector.downselect_met(met, **buffers)

    def _create_downwash_contrail(self) -> GeoVectorDataset:
        """Get Contrail representation of downwash flight."""

        downwash_contrail_data = {
            "waypoint": self._downwash_flight["waypoint"],
            "flight_id": self._downwash_flight["flight_id"],
            "time": self._downwash_flight["time"],
            "longitude": self._downwash_flight["longitude"],
            "latitude": self._downwash_flight["latitude"],
            # intentionally specify altitude and level to avoid pressure level calculations
            "altitude": self._downwash_flight["altitude_1"],
            "level": self._downwash_flight["level_1"],
            "air_pressure": self._downwash_flight["air_pressure_1"],
            "width": self._downwash_flight["width"],
            "depth": self._downwash_flight["depth"],
            "sigma_yz": self._downwash_flight["sigma_yz"],
            "air_temperature": self._downwash_flight["air_temperature_1"],
            "specific_humidity": self._downwash_flight["specific_humidity_1"],
            "q_sat": self._downwash_flight["q_sat_1"],
            "rho_air": self._downwash_flight["rho_air_1"],
            "rhi": self._downwash_flight["rhi_1"],
            "iwc": self._downwash_flight["iwc_1"],
            "n_ice_per_m": self._downwash_flight["n_ice_per_m_1"],
            "persistent": self._downwash_flight["persistent_1"],
        }

        contrail = GeoVectorDataset._from_fastpath(downwash_contrail_data).copy()
        contrail["formation_time"] = contrail["time"].copy()
        contrail["age"] = contrail["formation_time"] - contrail["time"]

        # Heating rate, differential heating rate and
        # cumulative heat energy absorbed by the contrail
        if self.params["radiative_heating_effects"]:
            contrail["heat_rate"] = np.zeros_like(contrail["n_ice_per_m"])
            contrail["d_heat_rate"] = np.zeros_like(contrail["n_ice_per_m"])

            # Increase in temperature averaged over the contrail plume
            contrail["cumul_heat"] = np.zeros_like(contrail["n_ice_per_m"])

            # Temperature difference between the upper and lower half of the contrail plume
            contrail["cumul_differential_heat"] = np.zeros_like(contrail["n_ice_per_m"])

        # Initially set energy forcing to 0 because the contrail just formed (age = 0)
        contrail["ef"] = np.zeros_like(contrail["n_ice_per_m"])
        if self.params["compute_atr20"]:
            contrail["global_yearly_mean_rf"] = np.zeros_like(contrail["n_ice_per_m"])
            contrail["atr20"] = np.zeros_like(contrail["n_ice_per_m"])

        if not self.params["filter_sac"]:
            contrail["sac"] = self._downwash_flight["sac"]
        if not self.params["filter_initially_persistent"]:
            contrail["initially_persistent"] = self._downwash_flight["persistent_1"]
        if self.params["persistent_buffer"] is not None:
            contrail["end_of_life"] = np.full(contrail.size, np.datetime64("NaT", "ns"))

        return contrail

    def _get_contrail_2_segments(self, time_idx: int) -> GeoVectorDataset:
        """Get batch of newly formed contrail segments."""
        time = self._downwash_contrail["time"]
        t_cur = self.timesteps[time_idx]
        if time_idx == 0:
            filt = time < t_cur
        else:
            t_prev = self.timesteps[time_idx - 1]
            filt = (time < t_cur) & (time >= t_prev)

        return self._downwash_contrail.filter(filt)

    @override
    def _cleanup_indices(self) -> None:
        """Cleanup interpolation artifacts."""

        if not self.params["interpolation_use_indices"]:
            return

        if hasattr(self, "contrail_list"):
            for contrail in self.contrail_list:
                contrail._invalidate_indices()

        self.source._invalidate_indices()
        self._sac_flight._invalidate_indices()
        if hasattr(self, "_downwash_flight"):
            self._downwash_flight._invalidate_indices()
        if hasattr(self, "_downwash_contrail"):
            self._downwash_contrail._invalidate_indices()

    def _bundle_results(self) -> None:
        # ---
        # Create contrail dataframe (self.contrail)
        # ---
        self.contrail = GeoVectorDataset.sum(self.contrail_list).dataframe
        self.contrail["timestep"] = np.concatenate(
            [np.full(c.size, i) for i, c in enumerate(self.contrail_list)]
        )

        # add age in hours to the contrail waypoint outputs
        age_hours = np.empty_like(self.contrail["ef"])
        np.divide(self.contrail["age"], np.timedelta64(1, "h"), out=age_hours)
        self.contrail["age_hours"] = age_hours

        verbose_outputs = self.params["verbose_outputs"]
        if verbose_outputs:
            # Compute dt_integration -- logic is somewhat complicated, but
            # we're simply addressing that the first dt_integration
            # is different from the rest

            # We call reset_index to introduces an `index` RangeIndex column,
            # Which we use in the `groupby` to identify the
            # index of the first evolution step at each waypoint.
            tmp = self.contrail.reset_index()
            cols = ["formation_time", "time", "index"]
            first_form_time = tmp.groupby("waypoint")[cols].first()
            first_dt = first_form_time["time"] - first_form_time["formation_time"]
            first_dt = first_dt.set_axis(first_form_time["index"])

            self.contrail = tmp.set_index("index")
            self.contrail["dt_integration"] = first_dt
            self.contrail.fillna({"dt_integration": self.params["dt_integration"]}, inplace=True)

            # ---
            # Create contrail xr.Dataset (self.contrail_dataset)
            # ---
            if isinstance(self.source, Fleet):
                keys = ["flight_id", "timestep", "waypoint"]
            else:
                keys = ["timestep", "waypoint"]
            self.contrail_dataset = xr.Dataset.from_dataframe(self.contrail.set_index(keys))

        # ---
        # Create output Flight / Fleet (self.source)
        # ---

        col_idx = ["flight_id", "waypoint"] if isinstance(self.source, Fleet) else ["waypoint"]
        del self.source["_met_intersection"]

        # Attach intermediate calculations from `sac_flight` and `downwash_flight` to flight
        sac_cols = [
            "width",
            "depth",
            "rhi_1",
            "air_temperature_1",
            "specific_humidity_1",
            "altitude_1",
            "persistent_1",
        ]

        # add additional columns
        if verbose_outputs:
            sac_cols += ["dT_dz", "ds_dz", "dz_max"]

        downwash_cols = [
            "rho_air_1",
            "iwc_1",
            "f_surv",
            "n_ice_per_m_0",
            "n_ice_per_m_1",
        ]
        df = pd.concat(
            [
                self.source.dataframe.set_index(col_idx),
                self._sac_flight.dataframe.set_index(col_idx)[sac_cols],
                self._downwash_flight.dataframe.set_index(col_idx)[downwash_cols],
            ],
            axis=1,
        )

        # Aggregate contrail data back to flight
        grouped = self.contrail.groupby(col_idx)

        # Perform all aggregations
        agg_dict = {"ef": ["sum"], "age": ["max"]}
        if self.params["compute_atr20"]:
            agg_dict["global_yearly_mean_rf"] = ["sum"]
            agg_dict["atr20"] = ["sum"]

        rad_keys = ["sdr", "rsr", "olr", "rf_sw", "rf_lw", "rf_net"]
        for key in rad_keys:
            if verbose_outputs:
                agg_dict[key] = ["mean", "min", "max"]
            else:
                agg_dict[key] = ["mean"]

        aggregated = grouped.agg(agg_dict)
        aggregated.columns = [f"{k1}_{k2}" for k1, k2 in aggregated.columns]
        aggregated = aggregated.rename(columns={"ef_sum": "ef", "age_max": "contrail_age"})
        if self.params["compute_atr20"]:
            aggregated = aggregated.rename(
                columns={"global_yearly_mean_rf_sum": "global_yearly_mean_rf", "atr20_sum": "atr20"}
            )

        # Join the two
        df = df.join(aggregated)

        # Fill missing values for ef and contrail_age per conventions
        # Mean, max, and min radiative values are *not* filled with 0
        df.fillna({"ef": 0.0, "contrail_age": np.timedelta64(0, "ns")}, inplace=True)

        # cocip flag for each waypoint
        # -1 if negative EF, 0 if no EF, 1 if positive EF,
        # or NaN for outside of domain of flight waypoints that don't persist
        df["cocip"] = np.sign(df["ef"])
        logger.debug("Total number of waypoints with nonzero EF: %s", df["cocip"].ne(0.0).sum())

        # reset the index
        df = df.reset_index()

        # Reassign to source
        self.source.data = VectorDataDict({k: v.to_numpy() for k, v in df.items()})

    def _fill_empty_flight_results(self, return_list_flight: bool) -> Flight | list[Flight]:
        """Fill empty results into flight / fleet and return.

        This method attaches an all nan array to each of the variables:
            - sdr
            - rsr
            - olr
            - rf_sw
            - rf_lw
            - rf_net

        This method also attaches zeros (for trajectory points contained within met grid)
        or nans (for trajectory points outside of the met grid) to the following variables.
            - ef
            - cocip
            - contrail_age
            - persistent_1

        Parameters
        ----------
        return_list_flight : bool
            If True, a list of :class:`Flight` is returned. In this case, :attr:`source`
            is assumed to be a :class:`Fleet`.

        Returns
        -------
        Flight | list[Flight]
            Flight or list of Flight objects with empty variables.
        """
        self._cleanup_indices()

        intersection = self.source.data.pop("_met_intersection")
        zeros_and_nans = np.zeros(intersection.shape, dtype=np.float32)
        zeros_and_nans[~intersection] = np.nan
        self.source["ef"] = zeros_and_nans.copy()
        self.source["persistent_1"] = zeros_and_nans.copy()
        self.source["cocip"] = np.sign(zeros_and_nans)
        self.source["contrail_age"] = zeros_and_nans.astype("timedelta64[ns]")

        if return_list_flight:
            return self.source.to_flight_list()  # type: ignore[attr-defined]

        return self.source


# ----------------------------------------
# Functions used in Cocip and CocipGrid
# ----------------------------------------


def process_met_datasets(
    met: MetDataset,
    rad: MetDataset,
    compute_tau_cirrus: bool | Literal["auto"] = "auto",
) -> tuple[MetDataset, MetDataset]:
    """Process and verify ERA5 data for :class:`Cocip` and :class:`CocipGrid`.

    The implementation uses :class:`Cocip` for the source of truth in determining
    which met variables are required.

    .. versionchanged:: 0.25.0

        This method is called in the :class:`CocipGrid` constructor regardless
        of the ``process_met`` parameter. The same approach is also taken
        in :class:`Cocip` in version 0.27.0.

    .. versionchanged:: 0.48.0

        Remove the ``shift_radiation_time`` parameter. This parameter is now
        inferred from the metadata on the `rad` instance.

    Parameters
    ----------
    met : MetDataset
        Met pressure-level data
    rad : MetDataset
        Rad single-level data
    compute_tau_cirrus : bool | Literal["auto"]
        Whether to add ``"tau_cirrus"`` variable to pressure-level met data. If set to
        ``"auto"``, ``"tau_cirrus"`` will be computed iff the met data is dask-backed.

    Returns
    -------
    met : MetDataset
        Met data, possibly with "tau_cirrus" variable attached.
    rad : MetDataset
        Rad data with time shifted to account for accumulated values.

    Raises
    ------
    If a previous version of pycontrails has already scaled the gridded humidity
    data.
    """
    # Check for remnants of previous scaling.
    if "_pycontrails_modified" in met["specific_humidity"].attrs:
        raise ValueError(
            "Specific humidity enhancement of the raw specific humidity values in "
            "the underlying met data is deprecated."
        )

    if compute_tau_cirrus == "auto":
        # If met data is dask-backed, compute tau_cirrus
        compute_tau_cirrus = met.data["air_temperature"].chunks is not None

    if "tau_cirrus" not in met.data:
        met.ensure_vars(Cocip.met_variables)
        if compute_tau_cirrus:
            met = add_tau_cirrus(met)
    else:
        met.ensure_vars(Cocip.processed_met_variables)

    rad.ensure_vars(Cocip.rad_variables)
    rad = _process_rad(rad)

    return met, rad


def add_tau_cirrus(met: MetDataset) -> MetDataset:
    """Add "tau_cirrus" variable to weather data if it does not exist already.

    Parameters
    ----------
    met : MetDataset
        Met pressure-level data.

    Returns
    -------
    met : MetDataset
        Met data with "tau_cirrus" variable attached.
    """
    if "tau_cirrus" not in met.data:
        met.data["tau_cirrus"] = tau_cirrus.tau_cirrus(met)
    return met


def _process_rad(rad: MetDataset) -> MetDataset:
    """Process radiation specific variables for model.

    These variables are used to calculate the reflected solar radiation (RSR),
    and outgoing longwave radiation (OLR).

    The time stamp is adjusted by ``shift_radiation_time`` to account for
    accumulated values being averaged over the time period.

    Parameters
    ----------
    rad : MetDataset
        Rad single-level data

    Returns
    -------
    MetDataset
        Rad data with time shifted.

    Raises
    ------
    ValueError
        If a "radiation_accumulated" field is not found on ``rad.attrs``.

    Notes
    -----
    - https://www.ecmwf.int/sites/default/files/elibrary/2015/18490-radiation-quantities-ecmwf-model-and-mars.pdf
    - https://confluence.ecmwf.int/pages/viewpage.action?pageId=155337784
    """
    # If the time coordinate has already been shifted, early return
    if "shift_radiation_time" in rad["time"].attrs:
        return rad

    provider = rad.provider_attr

    # Only shift ECMWF data -- exit for anything else
    # A warning is emitted upstream if the provider is not ECMWF or NCEP
    if provider != "ECMWF":
        return rad

    dataset = rad.dataset_attr
    product = rad.product_attr

    if dataset == "HRES":
        try:
            radiation_accumulated = rad.attrs["radiation_accumulated"]
        except KeyError as exc:
            msg = (
                "HRES data must have a boolean 'radiation_accumulated' attribute. "
                "This attribute is used to determine whether the radiation data "
                "has been accumulated over the time period. This is the case for "
                "HRES data taken from a common time of forecast with multiple "
                "forecast steps. If this is not the case, set the "
                "'radiation_accumulated' attribute to False."
            )
            raise ValueError(msg) from exc
        if radiation_accumulated:
            # Don't assume that radiation data is uniformly spaced in time
            # Instead, infer the appropriate time shift
            time_diff = rad.data["time"].diff("time", label="upper")
            time_shift = -time_diff / 2

            # Keep the original attrs -- we need these later on
            old_attrs = {k: v.attrs for k, v in rad.data.items()}

            # Also need to keep dataset-level attrs, which are lost
            # when dividing a Dataset by a DataArray
            old_rad_attrs = rad.data.attrs

            # NOTE: Taking the diff will remove the first time step
            # This is typically what we want (forecast step 0 is all zeros)
            # But, if the data has been downselected for a particular Flight / Fleet,
            # we lose the first time step of the data.
            #
            # Other portions of the code convert HRES accumulated fluxes (J/m2)
            # to averaged fluxes (W/m2) assuming that accumulations are over
            # one hour. For those conversions to work correctly, must normalize
            # accumulations by number of hours between steps
            time_diff_h = time_diff / np.timedelta64(1, "h")
            rad.data = rad.data.diff("time", label="upper") / time_diff_h
            rad.data.attrs = old_rad_attrs

            # Add back the original attrs
            for k, v in rad.data.items():
                v.attrs = old_attrs[k]

            # Short-circuit to avoid idiot check
            rad.data = rad.data.assign_coords({"time": rad.data["time"] + time_shift})
            if np.unique(time_shift).size == 1:
                rad.data["time"].attrs["shift_radiation_time"] = str(time_shift.values[0])
            else:
                rad.data["time"].attrs["shift_radiation_time"] = "variable"
            return rad

        shift_radiation_time = -np.timedelta64(30, "m")

    elif dataset == "ERA5" and product == "ensemble":
        shift_radiation_time = -np.timedelta64(90, "m")
    else:
        shift_radiation_time = -np.timedelta64(30, "m")

    # Do a final idiot check -- most likely, the time resolution of the data will
    # agree with the shift_radiation_time. If not, emit a warning. There could be
    # a false positive here if the data has been downsampled in time.
    logger.debug("Shifting rad time by %s", shift_radiation_time)
    rad_time_diff = np.diff(rad.data["time"])
    if not np.all(rad_time_diff / 2 == -shift_radiation_time):
        warnings.warn(
            f"Shifting radiation time dimension by unexpected interval {shift_radiation_time}. "
            f"The rad data has metadata indicating it is {product} ECMWF data. "
            f"This dataset should have time steps of {-2 * shift_radiation_time}."
        )

    rad.data = rad.data.assign_coords({"time": rad.data["time"] + shift_radiation_time})
    rad.data["time"].attrs["shift_radiation_time"] = str(shift_radiation_time)

    return rad


def _eval_aircraft_performance(
    aircraft_performance: AircraftPerformance | None, flight: Flight
) -> Flight:
    """Evaluate the :class:`AircraftPerformance` model.

    Parameters
    ----------
    aircraft_performance : AircraftPerformance | None
        Input aircraft performance model
    flight : Flight
        Flight to evaluate

    Returns
    -------
    Flight
        Output from aircraft performance model

    Raises
    ------
    ValueError
        If ``aircraft_performance`` is None
    """

    ap_vars = {"wingspan", "engine_efficiency", "fuel_flow", "aircraft_mass"}
    missing = ap_vars.difference(flight).difference(flight.attrs)
    if not missing:
        return flight

    if aircraft_performance is None:
        msg = (
            f"An AircraftPerformance model parameter is required if the flight does not contain "
            f"the following variables: {ap_vars}. This flight is missing: {missing}. "
            "Instantiate the Cocip model with an AircraftPerformance model. "
            "For example, 'Cocip(..., aircraft_performance=PSFlight(...))'."
        )
        raise ValueError(msg)

    return aircraft_performance.eval(source=flight, copy_source=False)


def _eval_emissions(emissions: Emissions, flight: Flight) -> Flight:
    """Evaluate the :class:`Emissions` model.

    Parameters
    ----------
    emissions : Emissions
        Emissions model
    flight : Flight
        Flight to evaluate

    Returns
    -------
    Flight
        Output from emissions model
    """

    emissions_outputs = "nvpm_ei_n"
    if flight.ensure_vars(emissions_outputs, False):
        return flight
    return emissions.eval(source=flight, copy_source=False)


def calc_continuous(contrail: GeoVectorDataset) -> None:
    """Calculate the continuous segments of this timestep.

    Mutates parameter ``contrail`` in place by setting or updating the
    "continuous" variable.

    Parameters
    ----------
    contrail : GeoVectorDataset
        GeoVectorDataset instance onto which "continuous" is set.

    Raises
    ------
    ValueError
        If ``contrail`` is empty.
    """
    if not contrail:
        raise ValueError("Cannot calculate continuous on an empty contrail")
    same_flight = contrail["flight_id"][:-1] == contrail["flight_id"][1:]
    consecutive_waypoint = np.diff(contrail["waypoint"]) == 1
    continuous = np.empty(contrail.size, dtype=bool)
    continuous[:-1] = same_flight & consecutive_waypoint
    continuous[-1] = False  # This fails if contrail is empty
    contrail.update(continuous=continuous)  # overwrite continuous


def calc_timestep_geometry(contrail: GeoVectorDataset) -> None:
    """Calculate contrail segment-specific properties.

    Mutates parameter ``contrail`` in place by setting or updating the variables
    - "sin_a"
    - "cos_a"
    - "segment_length"

    Any nan values in these variables are set to 0. This ensures any segment-based property
    derived from the segment geometry does not contribute to contrail energy forcing.

    See Also
    --------
    - :func:`wind_shear.wind_shear_normal` to see how "sin_a" and "cos_a"
    are used to compute wind shear terms.
    - :func:`calc_timestep_contrail_evolution` to see how "segment_length" is used.

    Parameters
    ----------
    contrail : GeoVectorDataset
        GeoVectorDataset instance onto which variables are set.
    """
    # get contrail waypoints
    longitude = contrail["longitude"]
    latitude = contrail["latitude"]
    altitude = contrail.altitude
    continuous = contrail["continuous"]

    # calculate segment properties
    segment_length = geo.segment_length(longitude, latitude, altitude)
    sin_a, cos_a = geo.segment_angle(longitude, latitude)

    # NOTE: Set all nan and discontinuous values to 0. With our current implementation, this
    # ensures degenerate or discontinuous segments do not contribute to contrail impact.
    # At the same time, this prevents nan values from propagating through evolving contrails.
    #
    # If we want to change this, take note of the following consequences.
    # nan values flow from one variable to another as follows:
    # segment_length
    # seg_ratio
    # sigma_yz
    # area_eff_2
    # iwc, n_ice_per_m, r_vol_ice
    # terminal_fall_speed
    # level                 [level advection]
    # u_wind, v_wind        [interp]
    # longitude, latitude   [advection]
    # Finally, at the next evolution step, the previous waypoint will accrue
    # a nan value after segment_length is recalculated

    segment_length[~continuous] = 0.0
    sin_a[~continuous] = 0.0
    cos_a[~continuous] = 0.0

    np.nan_to_num(segment_length, copy=False)
    np.nan_to_num(sin_a, copy=False)
    np.nan_to_num(cos_a, copy=False)

    # override values on model
    contrail.update(segment_length=segment_length)
    contrail.update(sin_a=sin_a)
    contrail.update(cos_a=cos_a)


def calc_timestep_meteorology(
    contrail: GeoVectorDataset,
    met: MetDataset,
    params: dict[str, Any],
    **interp_kwargs: Any,
) -> None:
    """Get and store meteorology parameters.

    Mutates parameter ``contrail`` in place by setting or updating the variables
    - "sin_a"
    - "cos_a"
    - "segment_length"

    Parameters
    ----------
    contrail : GeoVectorDataset
        GeoVectorDataset object onto which meterology variables are attached.
    met : MetDataset
        MetDataset with meteorology data variables.
    params : dict[str, Any]
        Cocip model ``params``.
    **interp_kwargs : Any
        Cocip model ``interp_kwargs``.
    """
    # get contrail geometry
    sin_a = contrail["sin_a"]
    cos_a = contrail["cos_a"]

    # get standard met parameters for timestep
    air_pressure = contrail.air_pressure
    air_temperature = contrail["air_temperature"]
    u_wind = interpolate_met(met, contrail, "eastward_wind", "u_wind", **interp_kwargs)
    v_wind = interpolate_met(met, contrail, "northward_wind", "v_wind", **interp_kwargs)
    interpolate_met(
        met,
        contrail,
        "lagrangian_tendency_of_air_pressure",
        "vertical_velocity",
        **interp_kwargs,
    )
    interpolate_met(met, contrail, "tau_cirrus", **interp_kwargs)

    # get the pressure level `dz_m` lower than element pressure
    air_pressure_lower = thermo.pressure_dz(air_temperature, air_pressure, params["dz_m"])

    # get meteorology at contrail waypoints interpolated at the pressure level `air_pressure_lower`
    level_lower = air_pressure_lower / 100.0  # Pa -> hPa

    # if met is already interpolated, this will automatically skip interpolation
    air_temperature_lower = interpolate_met(
        met,
        contrail,
        "air_temperature",
        "air_temperature_lower",
        level=level_lower,
        **interp_kwargs,
    )
    u_wind_lower = interpolate_met(
        met,
        contrail,
        "eastward_wind",
        "u_wind_lower",
        level=level_lower,
        **interp_kwargs,
    )
    v_wind_lower = interpolate_met(
        met,
        contrail,
        "northward_wind",
        "v_wind_lower",
        level=level_lower,
        **interp_kwargs,
    )

    # Temperature gradient
    dT_dz = thermo.T_potential_gradient(
        air_temperature,
        air_pressure,
        air_temperature_lower,
        air_pressure_lower,
        params["dz_m"],
    )

    # wind shear
    ds_dz = wind_shear.wind_shear(u_wind, u_wind_lower, v_wind, v_wind_lower, params["dz_m"])

    # wind shear normal
    dsn_dz = wind_shear.wind_shear_normal(
        u_wind_top=u_wind,
        u_wind_btm=u_wind_lower,
        v_wind_top=v_wind,
        v_wind_btm=v_wind_lower,
        cos_a=cos_a,
        sin_a=sin_a,
        dz=params["dz_m"],
    )

    # store values on contrail model
    contrail.update(dT_dz=dT_dz)
    contrail.update(ds_dz=ds_dz)
    contrail.update(dsn_dz=dsn_dz)


def calc_shortwave_radiation(
    rad: MetDataset,
    vector: GeoVectorDataset,
    **interp_kwargs: Any,
) -> None:
    """Calculate shortwave radiation variables.

    Calculates theoretical incident (``sdr``) and
    reflected shortwave radiation (``rsr``) from the radiation data provided.

    Mutates input ``vector`` to include ``"sdr"`` and ``"rsr"``
    keys with calculated SDR, RSR values, respectively.

    Parameters
    ----------
    rad : MetDataset
       Radiation data
    vector : GeoVectorDataset
        Flight or GeoVectorDataset instance
    **interp_kwargs : Any
        Interpolation keyword arguments

    Raises
    ------
    ValueError
        If ``rad`` does not contain ``"toa_net_downward_shortwave_flux"``,
        ``"toa_upward_shortwave_flux"`` or ``"top_net_solar_radiation"`` variable.

    Notes
    -----
    In accordance with the original CoCiP Fortran algorithm,
    the SDR is set to 0 when the cosine of the solar zenith angle is below 0.01.

    See Also
    --------
    :func:`geo.solar_direct_radiation`
    """
    if "sdr" in vector and "rsr" in vector:
        return

    try:
        sdr = vector["sdr"]
    except KeyError:
        # calculate instantaneous theoretical solar direct radiation based on geo position and time
        longitude = vector["longitude"]
        latitude = vector["latitude"]
        time = vector["time"]
        sdr = geo.solar_direct_radiation(longitude, latitude, time, threshold_cos_sza=0.01)
        vector["sdr"] = sdr

    # Generic contains net downward shortwave flux at TOA (SDR - RSR) in W/m2
    generic_key = "toa_net_downward_shortwave_flux"
    if generic_key in rad:
        tnsr = interpolate_met(rad, vector, generic_key, **interp_kwargs)
        vector["rsr"] = np.maximum(sdr - tnsr, 0.0)
        return

    # GFS contains RSR (toa_upward_shortwave_flux) variable directly
    gfs_key = "toa_upward_shortwave_flux"
    if gfs_key in rad:
        interpolate_met(rad, vector, gfs_key, "rsr", **interp_kwargs)
        return

    ecmwf_key = "top_net_solar_radiation"
    if ecmwf_key not in rad:
        msg = (
            f"'rad' data must contain either '{generic_key}' (generic), "
            f"'{gfs_key}' (GFS), or '{ecmwf_key}' (ECMWF) variable."
        )
        raise ValueError(msg)

    # ECMWF also contains net downward shortwave flux at TOA, but possibly as an accumulation
    tnsr = interpolate_met(rad, vector, ecmwf_key, **interp_kwargs)
    tnsr = _rad_accumulation_to_average_instantaneous(rad, ecmwf_key, tnsr)
    vector.update({ecmwf_key: tnsr})

    vector["rsr"] = np.maximum(sdr - tnsr, 0.0)


def calc_outgoing_longwave_radiation(
    rad: MetDataset,
    vector: GeoVectorDataset,
    **interp_kwargs: Any,
) -> None:
    """Calculate outgoing longwave radiation (``olr``) from the radiation data provided.

    Mutates input ``vector`` to include ``"olr"`` key with calculated OLR values.

    Parameters
    ----------
    rad : MetDataset
       Radiation data
    vector : GeoVectorDataset
        Flight or GeoVectorDataset instance
    **interp_kwargs : Any
        Interpolation keyword arguments

    Raises
    ------
    ValueError
        If ``rad`` does not contain a ``"toa_outgoing_longwave_flux"``,
        ``"toa_upward_longwave_flux"`` or ``"top_net_thermal_radiation"`` variable.
    """

    if "olr" in vector:
        return

    # Generic contains OLR (toa_outgoing_longwave_flux) directly
    generic_key = "toa_outgoing_longwave_flux"
    if generic_key in rad:
        interpolate_met(rad, vector, generic_key, "olr", **interp_kwargs)
        return

    # GFS contains OLR (toa_upward_longwave_flux) directly
    gfs_key = "toa_upward_longwave_flux"
    if gfs_key in rad:
        interpolate_met(rad, vector, gfs_key, "olr", **interp_kwargs)
        return

    # ECMWF contains "top_net_thermal_radiation" which is -1 * OLR
    ecmwf_key = "top_net_thermal_radiation"
    if ecmwf_key not in rad:
        msg = (
            f"'rad' data must contain either '{generic_key}' (generic), "
            f"'{gfs_key}' (GFS), or '{ecmwf_key}' (ECMWF) variable."
        )
        raise ValueError(msg)

    tntr = interpolate_met(rad, vector, ecmwf_key, **interp_kwargs)
    tntr = _rad_accumulation_to_average_instantaneous(rad, ecmwf_key, tntr)
    vector.update({ecmwf_key: tntr})

    vector["olr"] = np.maximum(-tntr, 0.0)


def calc_radiative_properties(contrail: GeoVectorDataset, params: dict[str, Any]) -> None:
    """Calculate radiative properties for contrail.

    This function is used by both :class:`Cocip` and :class`CocipGrid`.

    Mutates original `contrail` parameter with additional keys:
        - "rf_sw"
        - "rf_lw"
        - "rf_net"

    Parameters
    ----------
    contrail : GeoVectorDataset
        Grid points already interpolated against met and rad data. In particular,
        the variables
            - "air_temperature"
            - "r_ice_vol"
            - "tau_contrail"
            - "tau_cirrus"
            - "olr"
            - "rsr"
            - "sdr"

        are required on the parameter `contrail`.
    params : dict[str, Any]
        Model parameters
    """
    time = contrail["time"]
    air_temperature = contrail["air_temperature"]

    if params["radiative_heating_effects"]:
        air_temperature += contrail["cumul_heat"]

    r_ice_vol = contrail["r_ice_vol"]
    tau_contrail = contrail["tau_contrail"]
    tau_cirrus_ = contrail["tau_cirrus"]

    # calculate solar constant
    theta_rad = geo.orbital_position(time)
    sd0 = geo.solar_constant(theta_rad)

    # radiation dataset with contrail waypoints at timestep
    sdr = contrail["sdr"]
    rsr = contrail["rsr"]
    olr = contrail["olr"]

    # radiation calculations
    r_vol_um = r_ice_vol * 1e6
    habit_weights = radiative_forcing.habit_weights(
        r_vol_um, params["habit_distributions"], params["radius_threshold_um"]
    )
    rf_lw = radiative_forcing.longwave_radiative_forcing(
        r_vol_um, olr, air_temperature, tau_contrail, tau_cirrus_, habit_weights
    )
    rf_sw = radiative_forcing.shortwave_radiative_forcing(
        r_vol_um, sdr, rsr, sd0, tau_contrail, tau_cirrus_, habit_weights
    )

    # scale RF by enhancement factors
    rf_lw_scaled = rf_lw * params["rf_lw_enhancement_factor"]
    rf_sw_scaled = rf_sw * params["rf_sw_enhancement_factor"]
    rf_net = radiative_forcing.net_radiative_forcing(rf_lw_scaled, rf_sw_scaled)

    # store values on contrail
    contrail["rf_sw"] = rf_sw
    contrail["rf_lw"] = rf_lw
    contrail["rf_net"] = rf_net


def calc_contrail_properties(
    contrail: GeoVectorDataset,
    effective_vertical_resolution: float | npt.NDArray[np.floating],
    wind_shear_enhancement_exponent: float | npt.NDArray[np.floating],
    sedimentation_impact_factor: float | npt.NDArray[np.floating],
    radiative_heating_effects: bool,
) -> None:
    """Calculate geometric and ice-related properties of contrail.

    This function is used by both :class:`Cocip` and :class`CocipGrid`.

    This function modifies parameter `contrail` in place:
        - Mutates contrail data variables:
            - "ds_dz"
            - "dsn_dz"
        - Attaches additional variables:
            - "area_eff"
            - "plume_mass_per_m"
            - "r_ice_vol"
            - "terminal_fall_speed"
            - "diffuse_h"
            - "diffuse_v"
            - "n_ice_per_vol"
            - "tau_contrail"
            - "dn_dt_agg"
            - "dn_dt_turb"

    Parameters
    ----------
    contrail : GeoVectorDataset
        Grid points with many precomputed keys.
    effective_vertical_resolution : float | npt.NDArray[np.floating]
        Passed into :func:`wind_shear.wind_shear_enhancement_factor`.
    wind_shear_enhancement_exponent : float | npt.NDArray[np.floating]
        Passed into :func:`wind_shear.wind_shear_enhancement_factor`.
    sedimentation_impact_factor: float | npt.NDArray[np.floating]
        Passed into `contrail_properties.vertical_diffusivity`.
    radiative_heating_effects: bool
        Include radiative heating effects on contrail cirrus properties.
    """
    time = contrail["time"]
    iwc = contrail["iwc"]
    depth = contrail["depth"]
    width = contrail["width"]
    n_ice_per_m = contrail["n_ice_per_m"]
    t_cirrus_ = contrail["tau_cirrus"]

    # get required meteorology
    air_temperature = contrail["air_temperature"]
    air_pressure = contrail.air_pressure
    rhi = contrail["rhi"]
    rho_air = contrail["rho_air"]
    dT_dz = contrail["dT_dz"]
    ds_dz = contrail["ds_dz"]
    dsn_dz = contrail["dsn_dz"]
    sigma_yz = contrail["sigma_yz"]

    if radiative_heating_effects:
        air_temperature += contrail["cumul_heat"]

    # get required radiation
    sdr = contrail["sdr"]
    rsr = contrail["rsr"]
    olr = contrail["olr"]

    # shear enhancements
    shear_enhancement = wind_shear.wind_shear_enhancement_factor(
        contrail_depth=depth,
        effective_vertical_resolution=effective_vertical_resolution,
        wind_shear_enhancement_exponent=wind_shear_enhancement_exponent,
    )
    ds_dz = ds_dz * shear_enhancement
    dsn_dz = dsn_dz * shear_enhancement

    # effective area
    area_eff = contrail_properties.plume_effective_cross_sectional_area(width, depth, sigma_yz)
    depth_eff = contrail_properties.plume_effective_depth(width, area_eff)

    # ice particles
    n_ice_per_vol = contrail_properties.ice_particle_number_per_volume_of_plume(
        n_ice_per_m, area_eff
    )
    n_ice_per_kg_air = contrail_properties.ice_particle_number_per_mass_of_air(
        n_ice_per_vol, rho_air
    )
    plume_mass_per_m = contrail_properties.plume_mass_per_distance(area_eff, rho_air)
    r_ice_vol = contrail_properties.ice_particle_volume_mean_radius(iwc, n_ice_per_kg_air)
    tau_contrail = contrail_properties.contrail_optical_depth(r_ice_vol, n_ice_per_m, width)

    terminal_fall_speed = contrail_properties.ice_particle_terminal_fall_speed(
        air_pressure, air_temperature, r_ice_vol
    )
    diffuse_h = contrail_properties.horizontal_diffusivity(ds_dz, depth)

    if radiative_heating_effects:
        # theta_rad has float64 dtype, convert back to float32 if needed
        theta_rad = geo.orbital_position(time).astype(sdr.dtype, copy=False)
        sd0 = geo.solar_constant(theta_rad)
        heat_rate = radiative_heating.heating_rate(
            air_temperature=air_temperature,
            rhi=rhi,
            rho_air=rho_air,
            r_ice_vol=r_ice_vol,
            depth_eff=depth_eff,
            tau_contrail=tau_contrail,
            tau_cirrus=t_cirrus_,
            sd0=sd0,
            sdr=sdr,
            rsr=rsr,
            olr=olr,
        )

        cumul_differential_heat = contrail["cumul_differential_heat"]
        d_heat_rate = radiative_heating.differential_heating_rate(
            air_temperature=air_temperature,
            rhi=rhi,
            rho_air=rho_air,
            r_ice_vol=r_ice_vol,
            depth_eff=depth_eff,
            tau_contrail=tau_contrail,
            tau_cirrus=t_cirrus_,
            sd0=sd0,
            sdr=sdr,
            rsr=rsr,
            olr=olr,
        )

        eff_heat_rate = radiative_heating.effective_heating_rate(
            d_heat_rate, cumul_differential_heat, dT_dz, depth
        )
    else:
        eff_heat_rate = None

    diffuse_v = contrail_properties.vertical_diffusivity(
        air_pressure=air_pressure,
        air_temperature=air_temperature,
        dT_dz=dT_dz,
        depth_eff=depth_eff,
        terminal_fall_speed=terminal_fall_speed,
        sedimentation_impact_factor=sedimentation_impact_factor,
        eff_heat_rate=eff_heat_rate,
    )

    dn_dt_agg = contrail_properties.particle_losses_aggregation(
        r_ice_vol, terminal_fall_speed, area_eff
    )
    dn_dt_turb = contrail_properties.particle_losses_turbulence(
        width, depth, depth_eff, diffuse_h, diffuse_v
    )

    # Set properties to model
    contrail.update(ds_dz=ds_dz)
    contrail.update(dsn_dz=dsn_dz)
    contrail.update(area_eff=area_eff)
    contrail.update(plume_mass_per_m=plume_mass_per_m)
    contrail.update(r_ice_vol=r_ice_vol)
    contrail.update(terminal_fall_speed=terminal_fall_speed)
    contrail.update(diffuse_h=diffuse_h)
    contrail.update(diffuse_v=diffuse_v)
    contrail.update(n_ice_per_vol=n_ice_per_vol)
    contrail.update(tau_contrail=tau_contrail)
    contrail.update(dn_dt_agg=dn_dt_agg)
    contrail.update(dn_dt_turb=dn_dt_turb)
    if radiative_heating_effects:
        contrail.update(heat_rate=heat_rate, d_heat_rate=d_heat_rate)


def calc_timestep_contrail_evolution(
    met: MetDataset,
    rad: MetDataset,
    contrail_1: GeoVectorDataset,
    time_2: np.datetime64,
    params: dict[str, Any],
    **interp_kwargs: Any,
) -> GeoVectorDataset:
    """Calculate the contrail evolution across timestep (t1 -> t2).

    Note the variable suffix "_1" is used to reference the current time
    and the suffix "_2" is used to refer to the time at the next timestep.

    Parameters
    ----------
    met : MetDataset
       Meteorology data
    rad : MetDataset
        Radiation data
    contrail_1 : GeoVectorDataset
        Contrail waypoints at current timestep (1)
    time_2 : np.datetime64
        Time at the end of the evolution step (2)
    params : dict[str, Any]
        Model parameters
    **interp_kwargs : Any
        Interpolation keyword arguments

    Returns
    -------
    GeoVectorDataset
        The contrail evolved to ``time_2``.
    """

    # get lat/lon for current timestep (t1)
    longitude_1 = contrail_1["longitude"]
    latitude_1 = contrail_1["latitude"]
    level_1 = contrail_1.level
    time_1 = contrail_1["time"]

    # get contrail_1 geometry
    segment_length_1 = contrail_1["segment_length"]
    width_1 = contrail_1["width"]
    depth_1 = contrail_1["depth"]

    # get required met values for evolution calculations
    q_sat_1 = contrail_1["q_sat"]
    rho_air_1 = contrail_1["rho_air"]
    u_wind_1 = contrail_1["u_wind"]
    v_wind_1 = contrail_1["v_wind"]

    specific_humidity_1 = contrail_1["specific_humidity"]
    vertical_velocity_1 = contrail_1["vertical_velocity"]
    iwc_1 = contrail_1["iwc"]

    # get required contrail_1 properties
    sigma_yz_1 = contrail_1["sigma_yz"]
    dsn_dz_1 = contrail_1["dsn_dz"]
    terminal_fall_speed_1 = contrail_1["terminal_fall_speed"]
    diffuse_h_1 = contrail_1["diffuse_h"]
    diffuse_v_1 = contrail_1["diffuse_v"]
    plume_mass_per_m_1 = contrail_1["plume_mass_per_m"]
    dn_dt_agg_1 = contrail_1["dn_dt_agg"]
    dn_dt_turb_1 = contrail_1["dn_dt_turb"]
    n_ice_per_m_1 = contrail_1["n_ice_per_m"]

    # get contrail_1 radiative properties
    rf_net_1 = contrail_1["rf_net"]

    # initialize new timestep with evolved coordinates
    # assume waypoints are the same to start
    waypoint_2 = contrail_1["waypoint"]
    formation_time_2 = contrail_1["formation_time"]
    time_2_array = np.full_like(time_1, time_2)
    dt = time_2_array - time_1

    # get new contrail location & segment properties after t_step
    longitude_2, latitude_2 = geo.advect_horizontal(longitude_1, latitude_1, u_wind_1, v_wind_1, dt)
    level_2 = geo.advect_level(level_1, vertical_velocity_1, rho_air_1, terminal_fall_speed_1, dt)
    altitude_2 = units.pl_to_m(level_2)

    contrail_2 = GeoVectorDataset._from_fastpath(
        {
            "waypoint": waypoint_2,
            "flight_id": contrail_1["flight_id"],
            "formation_time": formation_time_2,
            "time": time_2_array,
            "age": time_2_array - formation_time_2,
            "longitude": longitude_2,
            "latitude": latitude_2,
            "altitude": altitude_2,
            "level": level_2,
        },
    )
    intersection = contrail_2.coords_intersect_met(met)
    if not np.any(intersection):
        warnings.warn(
            f"At time {time_2}, the contrail has no intersection with the met data. "
            "This is likely due to the contrail being advected outside the met domain."
        )

    # Update cumulative radiative heating energy absorbed by the contrail
    # This will always be zero if radiative_heating_effects is not activated in cocip_params
    if params["radiative_heating_effects"]:
        dt_sec = dt / np.timedelta64(1, "s")
        heat_rate_1 = contrail_1["heat_rate"]
        cumul_heat = contrail_1["cumul_heat"]
        cumul_heat += heat_rate_1 * dt_sec
        cumul_heat.clip(max=1.5, out=cumul_heat)  # Constrain additional heat to 1.5 K as precaution
        contrail_2["cumul_heat"] = cumul_heat

        d_heat_rate_1 = contrail_1["d_heat_rate"]
        cumul_differential_heat = contrail_1["cumul_differential_heat"]
        cumul_differential_heat += -d_heat_rate_1 * dt_sec
        contrail_2["cumul_differential_heat"] = cumul_differential_heat

    # Attach a few more artifacts for disabled filtering
    if not params["filter_sac"]:
        contrail_2["sac"] = contrail_1["sac"]
    if not params["filter_initially_persistent"]:
        contrail_2["initially_persistent"] = contrail_1["initially_persistent"]
    if params["persistent_buffer"] is not None:
        contrail_2["end_of_life"] = contrail_1["end_of_life"]

    # calculate initial contrail properties for the next timestep
    calc_continuous(contrail_2)
    calc_timestep_geometry(contrail_2)

    # get next segment lengths
    segment_length_2 = contrail_2["segment_length"]

    # new contrail dimensions
    seg_ratio_2 = contrail_properties.segment_length_ratio(segment_length_1, segment_length_2)
    sigma_yy_2, sigma_zz_2, sigma_yz_2 = contrail_properties.plume_temporal_evolution(
        width_1,
        depth_1,
        sigma_yz_1,
        dsn_dz_1,
        diffuse_h_1,
        diffuse_v_1,
        seg_ratio_2,
        dt,
        max_depth=params["max_depth"],
    )
    width_2, depth_2 = contrail_properties.new_contrail_dimensions(sigma_yy_2, sigma_zz_2)

    # store data back in model to support next time step calculations
    contrail_2["width"] = width_2
    contrail_2["depth"] = depth_2
    contrail_2["sigma_yz"] = sigma_yz_2

    # new contrail meteorology parameters
    air_temperature_2 = interpolate_met(met, contrail_2, "air_temperature", **interp_kwargs)

    if params["radiative_heating_effects"]:
        air_temperature_2 += contrail_2["cumul_heat"]

    interpolate_met(met, contrail_2, "specific_humidity", **interp_kwargs)

    humidity_scaling = params["humidity_scaling"]
    if humidity_scaling is not None:
        humidity_scaling.eval(contrail_2, copy_source=False)
    else:
        contrail_2["air_pressure"] = contrail_2.air_pressure
        contrail_2["rhi"] = thermo.rhi(
            contrail_2["specific_humidity"], air_temperature_2, contrail_2["air_pressure"]
        )

    air_pressure_2 = contrail_2["air_pressure"]
    specific_humidity_2 = contrail_2["specific_humidity"]
    rho_air_2 = thermo.rho_d(air_temperature_2, air_pressure_2)
    q_sat_2 = thermo.q_sat_ice(air_temperature_2, air_pressure_2)

    # store data back in model to support next ``calc_timestep_meteorology``
    contrail_2["rho_air"] = rho_air_2
    contrail_2["q_sat"] = q_sat_2

    # New contrail ice particle mass and number
    area_eff_2 = contrail_properties.new_effective_area_from_sigma(
        sigma_yy_2, sigma_zz_2, sigma_yz_2
    )
    plume_mass_per_m_2 = contrail_properties.plume_mass_per_distance(area_eff_2, rho_air_2)
    iwc_2 = contrail_properties.new_ice_water_content(
        iwc_1,
        specific_humidity_1,
        specific_humidity_2,
        q_sat_1,
        q_sat_2,
        plume_mass_per_m_1,
        plume_mass_per_m_2,
    )
    n_ice_per_m_2 = contrail_properties.new_ice_particle_number(
        n_ice_per_m_1, dn_dt_agg_1, dn_dt_turb_1, seg_ratio_2, dt
    )

    contrail_2["n_ice_per_m"] = n_ice_per_m_2
    contrail_2["iwc"] = iwc_2

    # calculate next timestep meteorology, contrail, and radiative properties
    calc_timestep_meteorology(contrail_2, met, params, **interp_kwargs)

    # Intersect with rad dataset
    calc_shortwave_radiation(rad, contrail_2, **interp_kwargs)
    calc_outgoing_longwave_radiation(rad, contrail_2, **interp_kwargs)
    calc_contrail_properties(
        contrail_2,
        params["effective_vertical_resolution"],
        params["wind_shear_enhancement_exponent"],
        params["sedimentation_impact_factor"],
        params["radiative_heating_effects"],
    )
    calc_radiative_properties(contrail_2, params)

    # get properties to measure persistence
    latitude_2 = contrail_2["latitude"]
    altitude_2 = contrail_2.altitude
    age_2 = contrail_2["age"]
    tau_contrail_2 = contrail_2["tau_contrail"]
    n_ice_per_vol_2 = contrail_2["n_ice_per_vol"]

    # calculate next persistent
    persistent_2 = contrail_2["persistent"] = contrail_properties.contrail_persistent(
        latitude=latitude_2,
        altitude=altitude_2,
        segment_length=segment_length_2,
        age=age_2,
        tau_contrail=tau_contrail_2,
        n_ice_per_m3=n_ice_per_vol_2,
        params=params,
    )

    # Get energy forcing by looking forward to the next time step radiative forcing
    rf_net_2 = contrail_2["rf_net"]
    energy_forcing_2 = contrail_properties.energy_forcing(
        rf_net_t1=rf_net_1,
        rf_net_t2=rf_net_2,
        width_t1=width_1,
        width_t2=width_2,
        seg_length_t2=segment_length_2,
        dt=dt,
    )

    # NOTE: Because of our geometry-continuity conventions, any waypoint without
    # continuity automatically has 0 EF. This is because calc_timestep_geometry
    # sets the segment_length entries to 0, thereby eliminating EF.
    # If we change conventions in calc_timestep_geometry, we may want to
    # explicitly set the discontinuous entries here to 0. We do this for age
    # here as well. It's somewhat of a hack, but it ensures nonzero contrail_age
    # is consistent with nonzero EF.
    contrail_2["ef"] = energy_forcing_2
    contrail_2["age"][energy_forcing_2 == 0.0] = np.timedelta64(0, "ns")

    if params["compute_atr20"]:
        contrail_2["global_yearly_mean_rf"] = (
            contrail_2["ef"] / constants.surface_area_earth / constants.seconds_per_year
        )
        contrail_2["atr20"] = (
            params["global_rf_to_atr20_factor"] * contrail_2["global_yearly_mean_rf"]
        )

    # filter contrail_2 by persistent waypoints, if any continuous segments are left
    logger.debug(
        "Fraction of waypoints surviving: %s / %s",
        persistent_2.sum(),
        persistent_2.size,
    )

    if (buff := params["persistent_buffer"]) is not None:
        # Here mask gets waypoints that are just now losing persistence
        mask = (~persistent_2) & np.isnat(contrail_2["end_of_life"])
        contrail_2["end_of_life"][mask] = time_2

        # Keep waypoints that are still persistent, which is determined by filt2
        # And waypoints within the persistent buffer, which is determined by filt1
        # So we only drop waypoints that are outside of the persistent buffer
        filt1 = contrail_2["time"] - contrail_2["end_of_life"] < buff
        filt2 = np.isnat(contrail_2["end_of_life"])
        filt = filt1 | filt2
        logger.debug(
            "Fraction of waypoints surviving with buffer %s: %s / %s",
            buff,
            filt.sum(),
            filt.size,
        )
        return contrail_2.filter(filt)

    # filter persistent contrails
    final_contrail = contrail_2.filter(persistent_2)

    # recalculate continuous contrail segments
    # and set EF, age for any newly discontinuous segments to 0
    if final_contrail:
        calc_continuous(final_contrail)
        continuous = final_contrail["continuous"]
        final_contrail["ef"][~continuous] = 0.0
        final_contrail["age"][~continuous] = np.timedelta64(0, "ns")
        if params["compute_atr20"]:
            final_contrail["global_yearly_mean_rf"][~continuous] = 0.0
            final_contrail["atr20"][~continuous] = 0.0
    return final_contrail


def _rad_accumulation_to_average_instantaneous(
    rad: MetDataset,
    name: str,
    arr: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Convert from radiation accumulation to average instantaneous values.

    .. versionadded:: 0.48.0

    Parameters
    ----------
    rad : MetDataset
        Radiation data
    name : str
        Variable name
    arr : npt.NDArray[np.floating]
        Array of values already interpolated from ``rad``

    Returns
    -------
    npt.NDArray[np.floating]
        Array of values converted from accumulation to average instantaneous values

    Raises
    ------
    KeyError
        If units are not provided on ``rad``.
    ValueError
        If unknown units are provided on ``rad``.
    """
    mda = rad[name]
    try:
        unit = mda.attrs["units"]
    except KeyError as e:
        msg = (
            f"Radiation data contains '{name}' variable "
            "but units are not specified. Provide units in the "
            f"rad['{name}'].attrs passed into Cocip."
        )
        raise KeyError(msg) from e

    # The unit is already instantaneous
    if unit == "W m**-2":
        return arr

    if unit != "J m**-2":
        msg = f"Unexpected units '{unit}' for '{name}' variable. Expected 'J m**-2' or 'W m**-2'."
        raise ValueError(msg)

    # Convert from J m**-2 to W m**-2
    if rad.dataset_attr == "ERA5" and rad.product_attr == "ensemble":
        n_seconds = 3.0 * 3600.0  # 3 hour interval
    else:
        n_seconds = 3600.0  # 1 hour interval

    return arr / n_seconds


def _emissions_variables() -> tuple[str, ...]:
    """Return variables required for emissions calculation."""
    return (
        "engine_efficiency",
        "fuel_flow",
        "aircraft_mass",
        "nvpm_ei_n",
        "wingspan",
    )


def _contrail_contrail_overlapping(
    contrail: GeoVectorDataset, params: dict[str, Any]
) -> GeoVectorDataset:
    """Mutate ``contrail`` to account for contrail-contrail overlapping effects."""

    if not contrail:
        return contrail

    contrail = radiative_forcing.contrail_contrail_overlap_radiative_effects(
        contrail,
        habit_distributions=params["habit_distributions"],
        radius_threshold_um=params["radius_threshold_um"],
        min_altitude_m=params["min_altitude_m"],
        max_altitude_m=params["max_altitude_m"],
        dz_overlap_m=params["dz_overlap_m"],
    )

    contrail.update(
        rsr=contrail.data.pop("rsr_overlap"),
        olr=contrail.data.pop("olr_overlap"),
        tau_cirrus=contrail.data.pop("tau_cirrus_overlap"),
        rf_sw=contrail.data.pop("rf_sw_overlap"),
        rf_lw=contrail.data.pop("rf_lw_overlap"),
        rf_net=contrail.data.pop("rf_net_overlap"),
    )

    return contrail
