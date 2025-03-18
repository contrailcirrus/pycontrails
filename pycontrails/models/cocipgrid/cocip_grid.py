"""Gridded CoCiP model."""

from __future__ import annotations

import itertools
import logging
import warnings
from collections.abc import Generator, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

import pycontrails
from pycontrails.core import models
from pycontrails.core.met import MetDataset, maybe_downselect_mds
from pycontrails.core.vector import GeoVectorDataset, VectorDataset
from pycontrails.models import humidity_scaling, sac
from pycontrails.models.cocip import cocip, contrail_properties, wake_vortex, wind_shear
from pycontrails.models.cocipgrid.cocip_grid_params import CocipGridParams
from pycontrails.models.emissions import Emissions
from pycontrails.physics import constants, geo, thermo, units
from pycontrails.utils import dependencies

if TYPE_CHECKING:
    import tqdm

logger = logging.getLogger(__name__)


class CocipGrid(models.Model):
    """Run CoCiP simulation on a grid.

    See :meth:`eval` for a description of model evaluation ``source`` parameters.

    Parameters
    ----------
    met, rad : MetDataset
        CoCiP-specific met data to interpolate against
    params : dict[str, Any], optional
        Override :class:`CocipGridParams` defaults. Most notably, the model is highly
        dependent on the parameter ``dt_integration``. Memory usage is also affected by
        parameters ``met_slice_dt`` and ``target_split_size``.
    param_kwargs : Any
        Override CocipGridParams defaults with arbitrary keyword arguments.

    Notes
    -----
    - If ``rad`` contains accumulated radiative fluxes, differencing to obtain
      time-averaged fluxes will reduce the time coverage of ``rad`` by half a forecast
      step. A warning will be produced during :meth:`eval` if the time coverage of
      ``rad`` (after differencing) is too short given the model evaluation parameters.
      If this occurs, provide an additional step of radiation data at the start or end
      of ``rad``.

    References
    ----------
    - :cite:`schumannPotentialReduceClimate2011`
    - :cite:`schumannContrailsVisibleAviation2012`

    See Also
    --------
    :class:`CocipGridParams`
    :class:`Cocip`
    :mod:`wake_vortex`
    :mod:`contrail_properties`
    :mod:`radiative_forcing`
    :mod:`humidity_scaling`
    :class:`Emissions`
    :mod:`sac`
    :mod:`tau_cirrus`
    """

    __slots__ = (
        "_target_dtype",
        "contrail",
        "contrail_list",
        "rad",
        "timesteps",
    )

    name = "contrail_grid"
    long_name = "Gridded Contrail Cirrus Prediction Model"
    default_params = CocipGridParams

    # Reference Cocip as the source of truth for met variables
    met_variables = cocip.Cocip.met_variables
    rad_variables = cocip.Cocip.rad_variables
    processed_met_variables = cocip.Cocip.processed_met_variables
    generic_rad_variables = cocip.Cocip.generic_rad_variables
    ecmwf_rad_variables = cocip.Cocip.ecmwf_rad_variables
    gfs_rad_variables = cocip.Cocip.gfs_rad_variables

    #: Met data is not optional
    met: MetDataset
    met_required = True
    rad: MetDataset

    #: Last evaluated input source
    source: MetDataset | GeoVectorDataset

    #: Artifacts attached when parameter ``verbose_outputs_evolution`` is True
    #: These allow for some additional information and parity with the approach
    #: taken by :class:`Cocip`.
    contrail_list: list[GeoVectorDataset]
    contrail: pd.DataFrame

    def __init__(
        self,
        met: MetDataset,
        rad: MetDataset,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ):
        super().__init__(met, params=params, **params_kwargs)

        compute_tau_cirrus = self.params["compute_tau_cirrus_in_model_init"]
        self.met, self.rad = cocip.process_met_datasets(met, rad, compute_tau_cirrus)

        # Convenience -- only used in `run_interpolators`
        self.params["_interp_kwargs"] = self.interp_kwargs

        if self.params["radiative_heating_effects"]:
            msg = "Parameter 'radiative_heating_effects' is not yet implemented in CocipGrid"
            raise NotImplementedError(msg)

        if self.params["unterstrasser_ice_survival_fraction"]:
            msg = (
                "Parameter 'unterstrasser_ice_survival_fraction' is not "
                "yet implemented in CocipGrid"
            )
            raise NotImplementedError(msg)

        self._target_dtype = np.result_type(*self.met.data.values())

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset: ...

    @overload
    def eval(self, source: MetDataset, **params: Any) -> MetDataset: ...

    @overload
    def eval(self, source: None = ..., **params: Any) -> NoReturn: ...

    def eval(
        self, source: GeoVectorDataset | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | MetDataset:
        """Run CoCiP simulation on a 4d coordinate grid or arbitrary set of 4d points.

        If the :attr:`params` ``verbose_outputs_evolution`` is True, the model holds
        :attr:`contrail_list` and :attr:`contrail` attributes for viewing intermediate
        artifacts. If ``source`` data is large, these intermediate vectors may consume
        substantial memory.

        .. versionchanged:: 0.25.0

            No longer explicitly support :class:`Flight` as a source. Any flight source
            will be viewed as a :class:`GeoVectorDataset`. In order to evaluate CoCiP
            predictions over a flight trajectory, it is best to use the :class:`Cocip`
            model. It's also possible to pre-compute segment azimuth and true airspeed
            before passing the flight trajectory in here.

        Parameters
        ----------
        source : GeoVectorDataset | MetDataset | None
            Input :class:`GeoVectorDataset` or :class:`MetDataset`. If None,
            a ``NotImplementedError`` is raised. If any subclass of :class:`GeoVectorDataset`
            is passed (e.g., :class:`Flight`), the additional structure is forgotten and
            the model is evaluated as if it were a :class:`GeoVectorDataset`.
            Additional variables may be passed as ``source`` data or attrs. These
            include:

            - ``aircraft_type``: This overrides any value in :attr:`params`. Must be included
              in the source attrs (not data).
            - ``fuel_flow``, ``engine_efficiency``, ``true_airspeed``, ``wingspan``,
              ``aircraft_mass``: These override any value in :attr:`params`.
            - ``azimuth``: This overrides any value in :attr:`params`.
            - ``segment_length``: This overrides any value in :attr:`params`.
        **params : Any
            Overwrite model parameters before eval

        Returns
        -------
        GeoVectorDataset | MetDataset
            CoCiP predictions for each point in ``source``. Output data contains variables
            ``contrail_age`` and ``ef_per_m``. Additional variables specified by the model
            :attr:`params` ``verbose_outputs_formation`` are also included.

        Raises
        ------
        NotImplementedError
            If ``source`` is None

        Notes
        -----
        At a high level, the model is broken down into the following steps:
          - Convert any :class:`MetDataset` ``source`` to :class:`GeoVectorDataset`.
          - Split the ``source`` into chunks of size ``params["target_split_size"]``.
          - For each timestep in :attr:`timesteps`:

            - Generate any new waypoints from the source data. Calculate aircraft performance
              and run the CoCiP downwash routine over the new waypoints.
            - For each "active" contrail (i.e., a contrail that has been initialized but
              has not yet reach its end of life), evolve the contrail forward one step.
              Filter any waypoint that has reached its end of life.

          - Aggregate contrail age and energy forcing predictions to a single
            output variable to return.
        """
        self.update_params(params)
        if source is None:
            # Unclear how to implement this
            # We expect met and rad to contain time slices beyond what is found
            # in the source (we need to evolve contrails forward in time).
            # Perhaps we could use the isel(time=0) slice to construct the source
            # from the met and rad data.
            msg = "CocipGrid.eval() with 'source=None' is not implemented."
            raise NotImplementedError(msg)
        self.set_source(source)

        self.met, self.rad = _downselect_met(self.source, self.met, self.rad, self.params)
        self.met = cocip.add_tau_cirrus(self.met)
        self._check_met_covers_source()

        # Save humidity scaling type to output attrs
        humidity_scaling = self.params["humidity_scaling"]
        if humidity_scaling is not None:
            for k, v in humidity_scaling.description.items():
                self.source.attrs[f"humidity_scaling_{k}"] = v

        self._parse_verbose_outputs()

        self._set_timesteps()
        pbar = self._init_pbar()

        met: MetDataset | None = None
        rad: MetDataset | None = None

        ef_summary: list[VectorDataset] = []
        verbose_dicts: list[dict[str, pd.Series]] = []
        contrail_list: list[GeoVectorDataset] = []
        existing_vectors: Iterator[GeoVectorDataset] = iter(())

        for time_idx, time_end in enumerate(self.timesteps):
            evolved_this_step = []
            ef_summary_this_step = []
            downwash_vectors_this_step = []
            for vector in self._generate_new_vectors(time_idx):
                t0 = vector["time"].min()
                met, rad = self._maybe_downselect_met_rad(met, rad, t0, time_end)
                downwash, verbose_dict = _run_downwash(vector, met, rad, self.params)

                if downwash:
                    # T_crit_sac is no longer needed. If verbose_outputs_formation is True,
                    # it's already storied in the verbose_dict data
                    downwash.data.pop("T_crit_sac", None)
                    downwash_vectors_this_step.append(downwash)
                    if self.params["verbose_outputs_evolution"]:
                        contrail_list.append(downwash)

                if self.params["verbose_outputs_formation"] and verbose_dict:
                    verbose_dicts.append(verbose_dict)

                if pbar is not None:
                    pbar.update()

            for vector in itertools.chain(existing_vectors, downwash_vectors_this_step):
                t0 = vector["time"].min()
                met, rad = self._maybe_downselect_met_rad(met, rad, t0, time_end)
                contrail, ef = _evolve_vector(
                    vector,
                    met=met,
                    rad=rad,
                    params=self.params,
                    t=time_end,
                )
                if ef:
                    evolved_this_step.append(contrail)
                    ef_summary_this_step.append(ef)
                    if self.params["verbose_outputs_evolution"]:
                        contrail_list.append(contrail)

                if pbar is not None:
                    pbar.update()

            if not evolved_this_step:
                if np.all(time_end > self.source_time):
                    break
                continue

            existing_vectors = combine_vectors(evolved_this_step, self.params["target_split_size"])

            summary = VectorDataset.sum(ef_summary_this_step)
            if summary:
                ef_summary.append(summary)

        if pbar is not None:
            logger.debug("Close progress bar")
            pbar.refresh()
            pbar.close()

        self._attach_verbose_outputs_evolution(contrail_list)
        total_ef_summary = _aggregate_ef_summary(ef_summary)
        return self._bundle_results(total_ef_summary, verbose_dicts)

    def _maybe_downselect_met_rad(
        self,
        met: MetDataset | None,
        rad: MetDataset | None,
        t0: np.datetime64,
        t1: np.datetime64,
    ) -> tuple[MetDataset, MetDataset]:
        """Downselect ``self.met`` and ``self.rad`` if necessary to cover ``[t0, t1]``.

        This implementation assumes ``t0 <= t1``, but does not enforce this.

        If the currently used ``met`` and ``rad`` slices do not include the time
        interval ``[t0, t1]``, new slices are selected from the larger ``self.met``
        and ``self.rad`` data. The slicing only occurs in the time domain.

        Existing slices from ``met`` and ``rad`` will be used when possible to avoid
        losing and re-loading already-loaded met data.

        If ``self.params["downselect_met"]`` is True, the :func:`_downselect_met` has
        already performed a spatial downselection of the met data.
        """
        met = maybe_downselect_mds(self.met, met, t0, t1)
        rad = maybe_downselect_mds(self.rad, rad, t0, t1)

        return met, rad

    def _attach_verbose_outputs_evolution(self, contrail_list: list[GeoVectorDataset]) -> None:
        """Attach intermediate artifacts to the model.

        This method attaches :attr:`contrail_list` and :attr:`contrail` when
        :attr:`params["verbose_outputs_evolution"]` is True.

        Mirrors implementation in :class:`Cocip`. We could do additional work here
        if this turns out to be useful.
        """
        if not self.params["verbose_outputs_evolution"]:
            return

        self.contrail_list = contrail_list  # attach raw data

        if contrail_list:
            # And the contrail DataFrame (pd.concat is expensive here)
            dfs = [contrail.dataframe for contrail in contrail_list]
            dfs = [df.assign(timestep=t_idx) for t_idx, df in enumerate(dfs)]
            self.contrail = pd.concat(dfs)
        else:
            self.contrail = pd.DataFrame()

    def _bundle_results(
        self,
        summary: VectorDataset | None,
        verbose_dicts: list[dict[str, pd.Series]],
    ) -> GeoVectorDataset | MetDataset:
        """Gather and massage model outputs for return."""
        max_age = self.params["max_age"]
        dt_integration = self.params["dt_integration"]
        azimuth = self.get_source_param("azimuth")
        segment_length = self.get_source_param("segment_length")
        if segment_length is None:
            segment_length = 1.0

        # Deal with verbose_dicts
        verbose_dict = _concat_verbose_dicts(
            verbose_dicts, self.source.size, self.params["verbose_outputs_formation"]
        )

        # Make metadata in attrs more readable
        if max_age.astype("timedelta64[h]") == max_age:
            max_age_str = str(max_age.astype("timedelta64[h]"))
        else:
            max_age_str = str(max_age.astype("timedelta64[m]"))
        if dt_integration.astype("timedelta64[m]") == dt_integration:
            dt_integration_str = str(dt_integration.astype("timedelta64[m]"))
        else:
            dt_integration_str = str(dt_integration.astype("timedelta64[s]"))

        self.transfer_met_source_attrs()
        attrs: dict[str, Any] = {
            "description": self.long_name,
            "max_age": max_age_str,
            "dt_integration": dt_integration_str,
            "aircraft_type": self.get_source_param("aircraft_type"),
            "pycontrails_version": pycontrails.__version__,
            **self.source.attrs,  # type: ignore[dict-item]
        }
        if ap_model := self.params["aircraft_performance"]:
            attrs["ap_model"] = type(ap_model).__name__

        if isinstance(azimuth, np.floating | np.integer):
            attrs["azimuth"] = azimuth.item()
        elif isinstance(azimuth, float | int):
            attrs["azimuth"] = azimuth

        if isinstance(self.source, MetDataset):
            self.source = result_to_metdataset(
                result=summary,
                verbose_dict=verbose_dict,
                source=self.source,
                nominal_segment_length=segment_length,
                attrs=attrs,
            )

            if self.params["compute_atr20"]:
                self.source["global_yearly_mean_rf_per_m"] = (
                    self.source["ef_per_m"].data
                    / constants.surface_area_earth
                    / constants.seconds_per_year
                )
                self.source["atr20_per_m"] = (
                    self.params["global_rf_to_atr20_factor"]
                    * self.source["global_yearly_mean_rf_per_m"].data
                )
        else:
            self.source = result_merge_source(
                result=summary,
                verbose_dict=verbose_dict,
                source=self.source,
                nominal_segment_length=segment_length,
                attrs=attrs,
            )

            if self.params["compute_atr20"]:
                self.source["global_yearly_mean_rf_per_m"] = (
                    self.source["ef_per_m"]
                    / constants.surface_area_earth
                    / constants.seconds_per_year
                )
                self.source["atr20_per_m"] = (
                    self.params["global_rf_to_atr20_factor"]
                    * self.source["global_yearly_mean_rf_per_m"]
                )

        return self.source

    # ---------------------------
    # Common Methods & Properties
    # ---------------------------

    @property
    def source_time(self) -> npt.NDArray[np.datetime64]:
        """Return the time array of the :attr:`source` data."""
        try:
            source = self.source
        except AttributeError as exc:
            msg = "No source set"
            raise AttributeError(msg) from exc

        if isinstance(source, GeoVectorDataset):
            return source["time"]
        if isinstance(source, MetDataset):
            return source.indexes["time"].values

        msg = f"Cannot calculate timesteps for {source}"
        raise TypeError(msg)

    def _set_timesteps(self) -> None:
        """Set the :attr:`timesteps` based on the ``source`` time range."""
        source_time = self.source_time
        tmin = source_time.min()
        tmax = source_time.max()

        tmin = pd.to_datetime(tmin)
        tmax = pd.to_datetime(tmax)
        dt = pd.to_timedelta(self.params["dt_integration"])

        t_start = tmin.ceil(dt)
        t_end = tmax.floor(dt) + self.params["max_age"] + dt

        # Pass in t_end (as opposed to tmax) to ensure that the met and rad data
        # cover the entire evolution period.
        _check_met_rad_time(self.met, self.rad, tmin, t_end)

        self.timesteps = np.arange(t_start, t_end, dt)

    def _init_pbar(self) -> tqdm.tqdm | None:
        """Initialize a progress bar for model evaluation.

        The total number of steps is estimated in a very crude way. Do not
        rely on the progress bar for accurate estimates of runtime.

        Returns
        -------
        tqdm.tqdm | None
            A progress bar for model evaluation. If ``show_progress`` is False, returns None.
        """

        if not self.params["show_progress"]:
            return None

        try:
            from tqdm.auto import tqdm
        except ModuleNotFoundError as exc:
            dependencies.raise_module_not_found_error(
                name="CocipGrid._init_pbar method",
                package_name="tqdm",
                module_not_found_error=exc,
                extra="Alternatively, set model parameter 'show_progress=False'.",
            )

        split_size = self.params["target_split_size"]
        if isinstance(self.source, MetDataset):
            n_splits_by_time = self._metdataset_source_n_splits()
            n_splits = len(self.source_time) * n_splits_by_time
        else:
            tmp1 = self.source_time[:, None] < self.timesteps[1:]
            tmp2 = self.source_time[:, None] >= self.timesteps[:-1]
            n_points_by_timestep = np.sum(tmp1 & tmp2, axis=0)

            init_split_size = self.params["target_split_size_pre_SAC_boost"] * split_size
            n_splits_by_time = np.ceil(n_points_by_timestep / init_split_size)
            n_splits = np.sum(n_splits_by_time)

        n_init_surv = 0.1 * n_splits  # assume 10% of points survive the downwash
        n_evo_steps = len(self.timesteps) * n_init_surv
        total = n_splits + n_evo_steps

        return tqdm(total=int(total), desc=f"{type(self).__name__} eval")

    def _metdataset_source_n_splits(self) -> int:
        """Compute the number of splits at a given time for a :class:`MetDataset` source.

        This method assumes :attr:`source` is a :class:`MetDataset`.

        Returns
        -------
        int
            The number of splits.
        """
        if not isinstance(self.source, MetDataset):
            msg = f"Expected source to be a MetDataset, found {type(self.source)}"
            raise TypeError(msg)

        indexes = self.source.indexes
        grid_size = indexes["longitude"].size * indexes["latitude"].size * indexes["level"].size

        split_size = int(
            self.params["target_split_size_pre_SAC_boost"] * self.params["target_split_size"]
        )
        return max(grid_size // split_size, 1)

    def _parse_verbose_outputs(self) -> None:
        """Confirm param "verbose_outputs" has the expected type for grid and path mode.

        This function mutates the "verbose_outputs" field on :attr:`params`.

        Currently, the list of all supported variables for verbose outputs
        is determine by :func:`_supported_verbose_outputs`.
        """
        if self.params["verbose_outputs"]:
            msg = (
                "Parameter 'verbose_outputs' is no longer supported for grid mode. "
                "Instead, use 'verbose_outputs_formation' and 'verbose_outputs_evolution'."
            )
            raise ValueError(msg)
        vo = self.params["verbose_outputs_formation"]
        supported = _supported_verbose_outputs_formation()

        # Parse to set of strings
        if isinstance(vo, str):
            vo = {vo}
        elif isinstance(vo, bool):
            vo = supported if vo else set()
        else:
            vo = set(vo)

        unknown_vars = vo - supported
        if unknown_vars:
            warnings.warn(
                f"Unknown variables in 'verbose_outputs': {unknown_vars}. "
                "Presently, CocipGrid only supports verbose outputs for "
                f"variables {supported}. The unknown variables will be ignored."
            )
        self.params["verbose_outputs_formation"] = vo & supported

    def _generate_new_vectors(self, time_idx: int) -> Generator[GeoVectorDataset, None, None]:
        """Generate :class:`GeoVectorDataset` instances from :attr:`source`.

        Parameters
        ----------
        time_idx : int
            The index of the current time slice in :attr:`timesteps`.

        Yields
        ------
        GeoVectorDataset
            Unevolved vectors arising from :attr`self.source_time` filtered by ``filt``.
            When :attr:`source` is a :class:`MetDataset`, each yielded dataset has a
            constant time value.
        """
        if "index" in self.source:
            # FIXME: We can simply change the internal variable to __index
            msg = "The variable 'index' is used internally. Found in source."
            raise RuntimeError(msg)

        source_time = self.source_time
        t_cur = self.timesteps[time_idx]
        if time_idx == 0:
            filt = source_time < t_cur
        else:
            t_prev = self.timesteps[time_idx - 1]
            filt = (source_time >= t_prev) & (source_time < t_cur)

        if not filt.any():
            return

        if isinstance(self.source, MetDataset):
            times_in_filt = source_time[filt]
            filt_start_idx = np.argmax(filt).item()  # needed to ensure globally unique indexes

            n_splits = self._metdataset_source_n_splits()
            for idx, time in enumerate(times_in_filt):
                # For now, sticking with the convention that every vector should
                # have a constant time value.
                source_slice = MetDataset._from_fastpath(self.source.data.sel(time=[time]))

                # Convert the 4D grid to a vector
                vector = source_slice.to_vector()
                vector.update(
                    longitude=vector["longitude"].astype(self._target_dtype, copy=False),
                    latitude=vector["latitude"].astype(self._target_dtype, copy=False),
                    level=vector["level"].astype(self._target_dtype, copy=False),
                )
                vector["index"] = source_time.size * np.arange(vector.size) + filt_start_idx + idx

                # Split into chunks
                for subvector in vector.generate_splits(n_splits):
                    subvector = self._build_subvector(subvector)
                    logger.debug(
                        "Yield new vector at time %s with size %s",
                        time.astype("datetime64[m]"),
                        subvector.size,
                    )
                    yield subvector

        elif isinstance(self.source, GeoVectorDataset):
            split_size = (
                self.params["target_split_size_pre_SAC_boost"] * self.params["target_split_size"]
            )
            n_splits = max(filt.sum() // split_size, 1)
            # Don't copy here ... we copy when we call `generate_splits`
            vector = self.source.filter(filt, copy=False)
            if vector:
                vector["index"] = np.flatnonzero(filt)

                # Split into chunks
                for subvector in vector.generate_splits(n_splits, copy=True):
                    subvector = self._build_subvector(subvector)
                    logger.debug("Yield new vector with size %s", subvector.size)
                    yield subvector

        else:
            msg = f"Unknown source {self.source}"
            raise TypeError(msg)

    def _build_subvector(self, vector: GeoVectorDataset) -> GeoVectorDataset:
        """Mutate `vector` by adding additional keys."""
        # Add time
        vector["formation_time"] = vector["time"]
        vector["age"] = np.full(vector.size, np.timedelta64(0, "ns"))

        # Precompute
        vector["air_pressure"] = vector.air_pressure
        vector["altitude"] = vector.altitude

        # Add nominals -- it's a little strange that segment_length
        # is also a nominal
        for key in _nominal_params():
            _setdefault_from_params(key, vector, self.params)

        segment_length = self._get_source_param_override("segment_length", vector)
        azimuth = self._get_source_param_override("azimuth", vector)

        # Experimental segment-free mode logic
        if azimuth is None and segment_length is None:
            return vector
        if azimuth is None:
            msg = "Set 'segment_length' to None for experimental segment-free model"
            raise ValueError(msg)
        if segment_length is None:
            msg = "Set 'azimuth' to None for experimental segment-free model"
            raise ValueError(msg)
        if self.params["dsn_dz_factor"]:
            msg = "'dsn_dz_factor' not supported outside of the segment-free mode"
            raise ValueError(msg)

        lons = vector["longitude"]
        lats = vector["latitude"]
        dist = segment_length / 2.0

        # These should probably not be included in model input ... so
        # we'll get a warning if they get overwritten
        lon_head, lat_head = geo.forward_azimuth(lons=lons, lats=lats, az=azimuth, dist=dist)
        vector["longitude_head"] = lon_head.astype(self._target_dtype, copy=False)
        vector["latitude_head"] = lat_head.astype(self._target_dtype, copy=False)

        lon_tail, lat_tail = geo.forward_azimuth(lons=lons, lats=lats, az=azimuth, dist=-dist)
        vector["longitude_tail"] = lon_tail.astype(self._target_dtype, copy=False)
        vector["latitude_tail"] = lat_tail.astype(self._target_dtype, copy=False)

        return vector

    def _check_met_covers_source(self) -> None:
        """Ensure that the met and rad data cover the source data.

        See also :func:`_check_met_rad_time` which checks the time coverage
        in more detail.
        """
        try:
            source = self.source
        except AttributeError as exc:
            msg = "No source set"
            raise AttributeError(msg) from exc

        if isinstance(source, MetDataset):
            indexes = source.indexes
            longitude = indexes["longitude"].to_numpy()
            latitude = indexes["latitude"].to_numpy()
            level = indexes["level"].to_numpy()
            time = indexes["time"].to_numpy()
        else:
            longitude = source["longitude"]
            latitude = source["latitude"]
            level = source.level
            time = source["time"]

        indexes = self.met.indexes
        _check_coverage(indexes["longitude"].to_numpy(), longitude, "longitude", "met")
        _check_coverage(indexes["latitude"].to_numpy(), latitude, "latitude", "met")
        _check_coverage(indexes["level"].to_numpy(), level, "level", "met")
        _check_coverage(indexes["time"].to_numpy(), time, "time", "met")

        indexes = self.rad.indexes
        _check_coverage(indexes["longitude"].to_numpy(), longitude, "longitude", "rad")
        _check_coverage(indexes["latitude"].to_numpy(), latitude, "latitude", "rad")
        _check_coverage(indexes["time"].to_numpy(), time, "time", "rad")

        _warn_not_wrap(self.met)
        _warn_not_wrap(self.rad)

    def _get_source_param_override(self, key: str, vector: GeoVectorDataset) -> Any:
        return _get_source_param_override(key, vector, self.params)

    # ------------
    # Constructors
    # ------------

    @staticmethod
    def create_source(
        level: npt.NDArray[np.floating] | list[float] | float,
        time: npt.NDArray[np.datetime64] | list[np.datetime64] | np.datetime64,
        longitude: npt.NDArray[np.floating] | list[float] | None = None,
        latitude: npt.NDArray[np.floating] | list[float] | None = None,
        lon_step: float = 1.0,
        lat_step: float = 1.0,
    ) -> MetDataset:
        """
        Shortcut to create a :class:`MetDataset` source from coordinate arrays.

        .. versionchanged:: 0.54.3
            By default, the returned latitude values now extend to the poles.

        Parameters
        ----------
        level : level: npt.NDArray[np.floating] | list[float] | float
            Pressure levels for gridded cocip.
            To avoid interpolating outside of the passed ``met`` and ``rad`` data, this
            parameter should avoid the extreme values of the ``met`` and `rad` levels.
            If ``met`` is already defined, a good choice for ``level`` is
            ``met.data['level'].values[1: -1]``.
        time: npt.NDArray[np.datetime64 | list[np.datetime64] | np.datetime64,
            One or more time values for gridded cocip.
        longitude, latitude : npt.NDArray[np.floating] | list[float], optional
            Longitude and latitude arrays, by default None. If not specified, values of
            ``lon_step`` and ``lat_step`` are used to define ``longitude`` and ``latitude``.
        lon_step, lat_step : float, optional
            Longitude and latitude resolution, by default 1.0.
            Only used if parameter ``longitude`` (respective ``latitude``) not specified.

        Returns
        -------
        MetDataset
            MetDataset that can be used as ``source`` input to :meth:`CocipGrid.eval(source=...)`

        See Also
        --------
        :meth:`MetDataset.from_coords`
        """
        if longitude is None:
            longitude = np.arange(-180, 180, lon_step, dtype=float)
        if latitude is None:
            latitude = np.arange(-90, 90.000001, lat_step, dtype=float)

        return MetDataset.from_coords(
            longitude=longitude, latitude=latitude, level=level, time=time
        )


################################
# Functions used by CocipGrid
################################


def _get_source_param_override(key: str, vector: GeoVectorDataset, params: dict[str, Any]) -> Any:
    """Mimic logic in :meth:`Models.get_source_param` replacing :attr:`source` with a ``vector``."""
    try:
        return vector[key]
    except KeyError:
        pass

    try:
        return vector.attrs[key]
    except KeyError:
        pass

    val = params[key]
    vector.attrs[key] = val
    return val


def _setdefault_from_params(key: str, vector: GeoVectorDataset, params: dict[str, Any]) -> None:
    """Set a parameter on ``vector`` if it is not already set.

    This method only sets "scalar" values.
    If ``params[key]`` is None, the parameter is not set.
    If ``params[key]`` is not a scalar, a TypeError is raised.
    """

    if key in vector:
        return
    if key in vector.attrs:
        return

    scalar = params[key]
    if scalar is None:
        return

    if not isinstance(scalar, int | float):
        msg = (
            f"Parameter {key} must be a scalar. For non-scalar values, directly "
            "set the data on the 'source'."
        )
        raise TypeError(msg)
    vector.attrs[key] = float(scalar)


def _nominal_params() -> Iterable[str]:
    """Return fields from :class:`CocipGridParams` that override values computed by the model.

    Each of the fields returned by this method is included in :class:`CocipGridParams`
    with a default value of None. When a non-None scalar value is provided for one of
    these fields and the :attr:`source` data does not provide a value, the scalar value
    is used (and broadcast over :attr:`source`) instead of running the AP or Emissions models.

    If non-scalar values are desired, they should be provided directly on
    :attr:`source` instead.

    Returns
    -------
    Iterable[str]
    """
    return (
        "wingspan",
        "aircraft_mass",
        "true_airspeed",
        "engine_efficiency",
        "fuel_flow",
    )


def run_interpolators(
    vector: GeoVectorDataset,
    met: MetDataset,
    rad: MetDataset | None = None,
    *,
    dz_m: float | None = None,
    humidity_scaling: humidity_scaling.HumidityScaling | None = None,
    keys: Sequence[str] | None = None,
    **interp_kwargs: Any,
) -> GeoVectorDataset:
    """Run interpolators over ``vector``.

    Intersect ``vector`` with DataArrays in met and rad needed for CoCiP. In addition, calculate
    three "lower level" intersections in which the level of the ``vector`` data is decreased
    according to the "dz_m" key in ``params``.

    Modifies ``vector`` in place and returns it.

    This function avoids overwriting existing variables on ``vector``.

    Aim to confine all interpolation to this function

    Parameters
    ----------
    vector : GeoVectorDataset
        Grid points.
    met, rad : MetDataset
        CoCiP met and rad slices. See :class:`CocipGrid`.
    dz_m : float | None, optional
        Difference in altitude between top and bottom layer for stratification calculations (m).
        Must be specified if ``keys`` is None.
    humidity_scaling : humidity_scaling.HumidityScaling | None, optional
        Specific humidity scaling scheme. Must be specified if ``keys`` is None.
    keys : list[str]
        Only run interpolators for select keys from ``met``
    **interp_kwargs : Any
        Interpolation keyword arguments

    Returns
    -------
    GeoVectorDataset
        Parameter ``vector`` with interpolated variables

    Raises
    ------
    TypeError
        If a required parameter is None
    ValueError
        If parameters `keys` and `rad` are both defined
    """
    # Avoid scaling specific humidity twice
    humidity_interpolated = "specific_humidity" not in vector

    if keys:
        if rad is not None:
            msg = "The 'keys' override only valid for 'met' input"
            raise ValueError(msg)

        for met_key in keys:
            # NOTE: Changed in v0.43: no longer overwrites existing variables
            models.interpolate_met(met, vector, met_key, **interp_kwargs)

        return _apply_humidity_scaling(vector, humidity_scaling, humidity_interpolated)

    if dz_m is None:
        msg = "Specify 'dz_m'."
        raise TypeError(msg)
    if rad is None:
        msg = "Specify 'rad'."
        raise TypeError(msg)

    # Interpolation at usual level
    # Excluded keys are not needed -- only used to initially compute tau_cirrus
    excluded = {
        "specific_cloud_ice_water_content",
        "ice_water_mixing_ratio",
        "geopotential",
        "geopotential_height",
    }
    for met_key in met:
        if met_key in excluded:
            continue
        models.interpolate_met(met, vector, met_key, **interp_kwargs)

    # calculate radiative properties
    cocip.calc_shortwave_radiation(rad, vector, **interp_kwargs)
    cocip.calc_outgoing_longwave_radiation(rad, vector, **interp_kwargs)

    # Interpolation at lower level
    air_temperature = vector["air_temperature"]
    air_pressure = vector.air_pressure
    air_pressure_lower = thermo.pressure_dz(air_temperature, air_pressure, dz_m)
    lower_level = air_pressure_lower / 100.0

    # Advect at lower_level
    for met_key in ("air_temperature", "eastward_wind", "northward_wind"):
        vector_key = f"{met_key}_lower"
        models.interpolate_met(
            met,
            vector,
            met_key,
            vector_key,
            **interp_kwargs,
            level=lower_level,
        )

    # Experimental segment-free model
    if _is_segment_free_mode(vector):
        return _apply_humidity_scaling(vector, humidity_scaling, humidity_interpolated)

    longitude_head = vector["longitude_head"]
    latitude_head = vector["latitude_head"]
    longitude_tail = vector["longitude_tail"]
    latitude_tail = vector["latitude_tail"]

    # Advect at head and tail
    # NOTE: Not using head_tail_dt here to offset time. We could do this for slightly
    # more accurate interpolation, but we would have to load an additional met time
    # slice at t_{-1}. After t_0, the head_tail_dt offset is not used.
    for met_key in ("eastward_wind", "northward_wind"):
        vector_key = f"{met_key}_head"
        models.interpolate_met(
            met,
            vector,
            met_key,
            vector_key,
            **interp_kwargs,
            longitude=longitude_head,
            latitude=latitude_head,
        )

        vector_key = f"{met_key}_tail"
        models.interpolate_met(
            met,
            vector,
            met_key,
            vector_key,
            **interp_kwargs,
            longitude=longitude_tail,
            latitude=latitude_tail,
        )

    return _apply_humidity_scaling(vector, humidity_scaling, humidity_interpolated)


def _apply_humidity_scaling(
    vector: GeoVectorDataset,
    humidity_scaling: humidity_scaling.HumidityScaling | None,
    humidity_interpolated: bool,
) -> GeoVectorDataset:
    """Scale specific humidity if it has been added by interpolator.

    Assumes that air_temperature and pressure are available on ``vector``.
    """
    if "specific_humidity" not in vector:
        return vector

    if humidity_scaling is not None and humidity_interpolated:
        humidity_scaling.eval(vector, copy_source=False)
        return vector

    if "rhi" in vector:
        return vector

    vector["rhi"] = thermo.rhi(
        vector["specific_humidity"], vector["air_temperature"], vector.air_pressure
    )

    return vector


def _evolve_vector(
    vector: GeoVectorDataset,
    *,
    met: MetDataset,
    rad: MetDataset,
    params: dict[str, Any],
    t: np.datetime64,
) -> tuple[GeoVectorDataset, VectorDataset]:
    """Evolve ``vector`` to time ``t``.

    Return surviving contrail at end of evolution and aggregate metrics from evolution.

    .. versionchanged:: 0.25.0

        No longer expect ``vector`` to have a constant time variable. Consequently,
        time step handling now mirrors that in :class:`Cocip`. Moreover, this method now
        handles both :class:`GeoVectorDataset` and :class:`MetDataset` vectors derived
        from :attr:`source`.

    Parameters
    ----------
    vector : GeoVectorDataset
        Contrail points that have been initialized and are ready for evolution.
    met, rad : MetDataset
        CoCiP met and rad slices. See :class:`CocipGrid`.
    params : dict[str, Any]
        CoCiP model parameters. See :class:`CocipGrid`.
    t : np.datetime64
        Time to evolve to.

    Returns
    -------
    contrail : GeoVectorDataset
        Evolved contrail at end of the evolution step.
    ef_summary : VectorDataset
        The ``contrail`` summary statistics. The result of
        ``contrail.select(("index", "age", "ef"), copy=False)``.
    """
    dt = t - vector["time"]

    if _is_segment_free_mode(vector):
        dt_head = None
        dt_tail = None
    else:
        head_tail_dt = vector["head_tail_dt"]
        half_head_tail_dt = head_tail_dt / 2
        dt_head = dt - half_head_tail_dt  # type: ignore[operator]
        dt_tail = dt + half_head_tail_dt  # type: ignore[operator]

    # After advection, out has time t
    out = advect(vector, dt, dt_head, dt_tail)  # type: ignore[arg-type]

    out = run_interpolators(
        out,
        met,
        rad,
        dz_m=params["dz_m"],
        humidity_scaling=params["humidity_scaling"],
        **params["_interp_kwargs"],
    )
    out = calc_evolve_one_step(vector, out, params)
    ef_summary = out.select(("index", "age", "ef"), copy=False)

    return out, ef_summary


def _run_downwash(
    vector: GeoVectorDataset, met: MetDataset, rad: MetDataset, params: dict[str, Any]
) -> tuple[GeoVectorDataset, dict[str, pd.Series]]:
    """Perform calculations involving downwash and initial contrail.

    .. versionchanged:: 0.25.0

        No longer return ``summary_data``. This was previously a vector of zeros,
        and does not give any useful information.

    Parameters
    ----------
    vector : GeoVectorDataset
        Grid values
    met, rad : MetDataset
        CoCiP met and rad slices. See :class:`CocipGrid`.
    params : dict[str, Any]
        CoCiP model parameters. See :class:`CocipGrid`.

    Returns
    -------
    vector : GeoVectorDataset
        Downwash vector.
    verbose_dict : dict[str, pd.Series]
        Dictionary of verbose outputs.
    """
    # All extra variables as required by verbose_outputs are computed on the
    # initial calculations involving the downwash contrail.
    verbose_dict: dict[str, pd.Series] = {}
    verbose_outputs_formation = params["verbose_outputs_formation"]

    keys = "air_temperature", "specific_humidity"
    vector = run_interpolators(
        vector,
        met,
        humidity_scaling=params["humidity_scaling"],
        keys=keys,
        **params["_interp_kwargs"],
    )
    calc_emissions(vector, params)

    # Get verbose outputs from emissions. These include fuel_flow, nvpm_ei_n, true_airspeed.
    for key in verbose_outputs_formation:
        if (val := vector.get(key)) is not None:
            verbose_dict[key] = pd.Series(data=val, index=vector["index"])

    # Get verbose outputs from SAC calculation.
    vector, sac_ = find_initial_contrail_regions(vector, params)
    if (key := "sac") in verbose_outputs_formation:
        verbose_dict[key] = sac_
    if (key := "T_crit_sac") in verbose_outputs_formation and (val := vector.get(key)) is not None:
        verbose_dict[key] = pd.Series(data=val, index=vector["index"])

    # Early exit if nothing in vector passes the SAC
    if not vector:
        logger.debug("No vector waypoints satisfy SAC")
        return vector, verbose_dict

    vector = run_interpolators(vector, met, rad, dz_m=params["dz_m"], **params["_interp_kwargs"])
    out = simulate_wake_vortex_downwash(vector, params)

    out = run_interpolators(
        out,
        met,
        rad,
        dz_m=params["dz_m"],
        humidity_scaling=params["humidity_scaling"],
        **params["_interp_kwargs"],
    )
    out, persistent = find_initial_persistent_contrails(vector, out, params)

    if (key := "persistent") in verbose_outputs_formation:
        verbose_dict[key] = persistent
    if (key := "iwc") in verbose_outputs_formation and (data := out.get(key)) is not None:
        verbose_dict[key] = pd.Series(data=data, index=out["index"])

    return out, verbose_dict


def combine_vectors(
    vectors: list[GeoVectorDataset],
    target_split_size: int,
) -> Generator[GeoVectorDataset, None, None]:
    """Combine vectors until size exceeds ``target_split_size``.

    .. versionchanged:: 0.25.0

        Ignore common end of life constraint previously imposed.

        Change function to return a generator.

    Parameters
    ----------
    vectors : list[GeoVectorDataset]
        Vectors to combine
    target_split_size : int
        Target vector size in combined vectors

    Yields
    ------
    GeoVectorDataset
        Combined vectors.
    """
    # Loop through vectors until we've accumulated more grid points than the
    # target size. Once have, concatenate and yield
    i0 = 0
    cum_size = 0
    for i1, vector in enumerate(vectors):
        cum_size += vector.size
        if cum_size >= target_split_size:
            yield GeoVectorDataset.sum(vectors[i0 : i1 + 1])
            i0 = i1 + 1
            cum_size = 0

    # If there is anything nontrivial left over, yield it
    if cum_size:
        yield GeoVectorDataset.sum(vectors[i0:])


def find_initial_contrail_regions(
    vector: GeoVectorDataset, params: dict[str, Any]
) -> tuple[GeoVectorDataset, pd.Series]:
    """Filter ``vector`` according to the SAC.

    This function also attaches the ``T_crit_sac`` variable to the returned
    GeoVectorDataset instance.

    Parameters
    ----------
    vector : GeoVectorDataset
        Data to apply SAC. Must contain variables
        - "air_temperature"
        - "specific_humidity"

    params : dict[str, Any]
        CoCiP model parameters. See :class:`CocipGrid`. Must contain keys
        - "fuel"
        - "filter_sac"

    Returns
    -------
    filtered_vector : GeoVectorDataset
        Input parameter ``vector`` filtered according to SAC (if ``param["filter_sac"]``).
    sac_series : pd.Series
        SAC values for each point in input ``vector``. The :class:`pd.Series` is
        indexed by ``vector["index"]``
    """
    air_temperature = vector["air_temperature"]
    specific_humidity = vector["specific_humidity"]
    air_pressure = vector.air_pressure
    engine_efficiency = vector["engine_efficiency"]
    fuel = vector.attrs["fuel"]
    ei_h2o = fuel.ei_h2o
    q_fuel = fuel.q_fuel

    G = sac.slope_mixing_line(specific_humidity, air_pressure, engine_efficiency, ei_h2o, q_fuel)
    t_sat_liq = sac.T_sat_liquid(G)
    rh = thermo.rh(specific_humidity, air_temperature, air_pressure)
    rh_crit = sac.rh_critical_sac(air_temperature, t_sat_liq, G)
    sac_ = sac.sac(rh, rh_crit)

    filt = sac_ == 1.0
    logger.debug(
        "Fraction of grid points satisfying the SAC: %s / %s",
        filt.sum(),
        vector.size,
    )

    if params["filter_sac"]:
        filtered_vector = vector.filter(filt)
    else:
        filt = np.ones(vector.size, dtype=bool)  # needed below in T_crit_sac
        filtered_vector = vector.copy()
        logger.debug("Not filtering SAC")

    # If filtered_vector is already empty, sac.T_critical_sac will raise an error
    # in the Newton approximation
    # So just return the empty vector here
    if not filtered_vector:
        return filtered_vector, pd.Series([], dtype=float)

    # This is only used in `calc_first_contrail`, but we compute it here in order
    # to do everything SAC related at once.
    # It is slightly more performant to compute this AFTER we filter by sac_ == 1,
    # which is why we compute it here
    T_crit_sac = sac.T_critical_sac(t_sat_liq[filt], rh[filt], G[filt])
    filtered_vector["T_crit_sac"] = T_crit_sac
    return filtered_vector, pd.Series(data=sac_, index=vector["index"])


def simulate_wake_vortex_downwash(
    vector: GeoVectorDataset, params: dict[str, Any]
) -> GeoVectorDataset:
    """Calculate regions of initial contrail formation.

    This function calculates data effective flight downwash, then constructs a
    GeoVectorDataset object consisting persistent downwash regions. No filtering
    occurs here; the length of the returned GeoVectorDataset equals the length
    of the parameter ``vector``.

    Of all steps in the gridded cocip pipeline, this one is generally the slowest
    since grid points have not yet been filtered by persistence (only SAC filtering
    has been applied in the CocipGrid pipeline). This function includes abundant
    logging.

    Parameters
    ----------
    vector : GeoVectorDataset
        Grid points from which initial contrail regions are calculated.
        Must already be interpolated against CoCiP met data.
    params : dict[str, Any]
        CoCiP model parameters. See :class:`CocipGrid`.

    Returns
    -------
    GeoVectorDataset
        Initial regions of persistent contrail.
    """
    # stored in `calc_emissions`
    true_airspeed = vector["true_airspeed"]

    # NOTE: The `calc_wind_shear` function is run on both the downwash and contrail object.
    # This is the only time it is called with the `is_downwash` flag on
    calc_wind_shear(
        vector,
        dz_m=params["dz_m"],
        is_downwash=True,
        dsn_dz_factor=params["dsn_dz_factor"],
    )

    # Stored in `calc_wind_shear`
    dT_dz = vector["dT_dz"]
    ds_dz = vector["ds_dz"]

    air_pressure = vector.air_pressure
    air_temperature = vector["air_temperature"]
    T_crit_sac = vector["T_crit_sac"]

    wsee = _get_source_param_override("wind_shear_enhancement_exponent", vector, params)
    wingspan = _get_source_param_override("wingspan", vector, params)
    aircraft_mass = _get_source_param_override("aircraft_mass", vector, params)

    dz_max = wake_vortex.max_downward_displacement(
        wingspan=wingspan,
        true_airspeed=true_airspeed,
        aircraft_mass=aircraft_mass,
        air_temperature=air_temperature,
        dT_dz=dT_dz,
        ds_dz=ds_dz,
        air_pressure=air_pressure,
        effective_vertical_resolution=params["effective_vertical_resolution"],
        wind_shear_enhancement_exponent=wsee,
    )

    width = wake_vortex.initial_contrail_width(wingspan, dz_max)
    iwvd = _get_source_param_override("initial_wake_vortex_depth", vector, params)
    depth = wake_vortex.initial_contrail_depth(dz_max, iwvd)
    # Initially, sigma_yz is set to 0
    # See bottom left paragraph p. 552 Schumann 2012 beginning with:
    # >>> "The contrail model starts from initial values ..."
    sigma_yz = np.zeros_like(width)

    index = vector["index"]
    time = vector["time"]
    longitude = vector["longitude"]
    latitude = vector["latitude"]
    altitude = vector.altitude
    formation_time = vector["formation_time"]
    age = vector["age"]

    # Initial contrail is constructed at a lower altitude
    altitude_1 = altitude - 0.5 * depth
    level_1 = units.m_to_pl(altitude_1)
    air_pressure_1 = 100.0 * level_1

    data = {
        "index": index,
        "longitude": longitude,
        "latitude": latitude,
        "level": level_1,
        "altitude": altitude_1,
        "air_pressure": air_pressure_1,
        "time": time,
        "formation_time": formation_time,
        "age": age,
        "T_crit_sac": T_crit_sac,
        "width": width,
        "depth": depth,
        "sigma_yz": sigma_yz,
        **_get_uncertainty_params(vector),
    }

    # Experimental segment-free model
    if _is_segment_free_mode(vector):
        return GeoVectorDataset._from_fastpath(data, attrs=vector.attrs).copy()

    # Stored in `_generate_new_grid_vectors`
    data["longitude_head"] = vector["longitude_head"]
    data["latitude_head"] = vector["latitude_head"]
    data["longitude_tail"] = vector["longitude_tail"]
    data["latitude_tail"] = vector["latitude_tail"]
    data["head_tail_dt"] = vector["head_tail_dt"]

    segment_length = _get_source_param_override("segment_length", vector, params)
    if isinstance(segment_length, np.ndarray):
        data["segment_length"] = segment_length
    else:
        # This should be broadcast over the source: subsequent vectors created during
        # evolution always recompute the segment length. GeoVectorDataset.sum will
        # raise an error if the wake vortex GeoVectorDataset does not contain a
        # segment_length variable.
        data["segment_length"] = np.full_like(data["longitude"], segment_length)

    return GeoVectorDataset._from_fastpath(data, attrs=vector.attrs).copy()


def find_initial_persistent_contrails(
    vector: GeoVectorDataset, contrail: GeoVectorDataset, params: dict[str, Any]
) -> tuple[GeoVectorDataset, pd.Series]:
    """Calculate first contrail immediately after downwash calculation.

    This function filters according to :func:`contrail_properties.initial_persistant`.

    The ``_1`` naming convention represents conditions are the wake vortex phase.

    Parameters
    ----------
    vector : GeoVectorDataset
        Data from original grid points after SAC filtering.
    contrail : GeoVectorDataset
        Output of :func:`simulate_wake_vortex_downwash`.
    params : dict[str, Any]
        CoCiP model parameters. See :class:`CocipGrid`.

    Returns
    -------
    tuple[GeoVectorDataset, pd.Series]
        The first entry in the tuple holds the first contrail after filtering
        by initially persistent. This GeoVectorDataset instance is equipped with
        all necessary keys to begin main contrail evolution. The second entry is
        the raw output of the :func:`contrail_properties.initial_persistant`
        computation as needed if "persistent" is in the "verbose_outputs"
        parameter. The :class:`pd.Series` is indexed by ``vector["index"]``.
    """
    # Gridpoint data
    # NOTE: In the Cocip implementation, these are variables on sac_flight
    # without the suffix "_1"
    air_pressure = vector.air_pressure
    air_temperature = vector["air_temperature"]
    specific_humidity = vector["specific_humidity"]
    fuel_dist = vector["fuel_flow"] / vector["true_airspeed"]
    nvpm_ei_n = vector["nvpm_ei_n"]
    ei_h2o = params["fuel"].ei_h2o

    # Contrail data
    T_crit_sac = contrail["T_crit_sac"]
    width = contrail["width"]
    depth = contrail["depth"]
    air_pressure_1 = contrail.air_pressure

    # Initialize initial contrail properties (ice water content, number of ice particles)
    # The logic here is fairly different from the later timesteps
    iwc = contrail_properties.initial_iwc(
        air_temperature=air_temperature,
        specific_humidity=specific_humidity,
        air_pressure=air_pressure,
        fuel_dist=fuel_dist,
        width=width,
        depth=depth,
        ei_h2o=ei_h2o,
    )
    iwc_ad = contrail_properties.iwc_adiabatic_heating(
        air_temperature=air_temperature,
        air_pressure=air_pressure,
        air_pressure_1=air_pressure_1,
    )
    iwc_1 = contrail_properties.iwc_post_wake_vortex(iwc, iwc_ad)
    f_surv = contrail_properties.ice_particle_survival_fraction(iwc, iwc_1)
    n_ice_per_m_0 = contrail_properties.initial_ice_particle_number(
        nvpm_ei_n=nvpm_ei_n,
        fuel_dist=fuel_dist,
        air_temperature=air_temperature,
        T_crit_sac=T_crit_sac,
        min_ice_particle_number_nvpm_ei_n=params["min_ice_particle_number_nvpm_ei_n"],
    )
    n_ice_per_m_1 = n_ice_per_m_0 * f_surv

    # The logic below corresponds to Cocip._create_downwash_contrail (roughly)
    contrail["iwc"] = iwc_1
    contrail["n_ice_per_m"] = n_ice_per_m_1

    # Check for persistent initial_contrails
    rhi_1 = contrail["rhi"]
    persistent_1 = contrail_properties.initial_persistent(iwc_1=iwc_1, rhi_1=rhi_1)

    logger.debug(
        "Fraction of grid points with persistent initial contrails: %s / %s",
        int(persistent_1.sum()),
        contrail.size,
    )

    # Filter by persistent
    if params["filter_initially_persistent"]:
        persistent_contrail = contrail.filter(persistent_1.astype(bool))
    else:
        persistent_contrail = contrail.copy()
        logger.debug("Not filtering initially persistent")

    # Attach a bunch of other initialization variables
    # (Previously, this was done before filtering. It's computationally more
    # efficient to do it down here)
    calc_thermal_properties(persistent_contrail)
    calc_wind_shear(
        persistent_contrail,
        is_downwash=False,
        dz_m=params["dz_m"],
        dsn_dz_factor=params["dsn_dz_factor"],
    )

    effective_vertical_resolution = persistent_contrail.get(
        "effective_vertical_resolution", params["effective_vertical_resolution"]
    )
    wind_shear_enhancement_exponent = persistent_contrail.get(
        "wind_shear_enhancement_exponent", params["wind_shear_enhancement_exponent"]
    )
    sedimentation_impact_factor = persistent_contrail.get(
        "sedimentation_impact_factor", params["sedimentation_impact_factor"]
    )
    cocip.calc_contrail_properties(
        persistent_contrail,
        effective_vertical_resolution=effective_vertical_resolution,
        wind_shear_enhancement_exponent=wind_shear_enhancement_exponent,
        sedimentation_impact_factor=sedimentation_impact_factor,
        radiative_heating_effects=False,  # Not yet supported in CocipGrid
    )

    # assumes "sdr", "rsr", and "olr" are already available on vector
    cocip.calc_radiative_properties(persistent_contrail, params)

    # no EF forcing on first contrail
    persistent_contrail["ef"] = np.zeros_like(persistent_contrail["n_ice_per_m"])

    persistent_series = pd.Series(data=persistent_1, index=contrail["index"])
    return persistent_contrail, persistent_series


def calc_evolve_one_step(
    curr_contrail: GeoVectorDataset,
    next_contrail: GeoVectorDataset,
    params: dict[str, Any],
) -> GeoVectorDataset:
    """Calculate contrail properties of ``next_contrail``.

    This function attaches additional variables to ``next_contrail``, then
    filters by :func:`contrail_properties.contrail_persistent`.

    Parameters
    ----------
    curr_contrail : GeoVectorDataset
        Existing contrail
    next_contrail : GeoVectorDataset
        Result of advecting existing contrail already interpolated against CoCiP met data
    params : dict[str, Any]
        CoCiP model parameters. See :class:`CocipGrid`.

    Returns
    -------
    GeoVectorDataset
        Parameter ``next_contrail`` filtered by persistence.
    """
    calc_wind_shear(
        next_contrail,
        is_downwash=False,
        dz_m=params["dz_m"],
        dsn_dz_factor=params["dsn_dz_factor"],
    )
    calc_thermal_properties(next_contrail)

    iwc_t1 = curr_contrail["iwc"]
    specific_humidity_t1 = curr_contrail["specific_humidity"]
    specific_humidity_t2 = next_contrail["specific_humidity"]
    q_sat_t1 = curr_contrail["q_sat"]
    q_sat_t2 = next_contrail["q_sat"]
    plume_mass_per_m_t1 = curr_contrail["plume_mass_per_m"]
    width_t1 = curr_contrail["width"]
    depth_t1 = curr_contrail["depth"]
    sigma_yz_t1 = curr_contrail["sigma_yz"]
    dsn_dz_t1 = curr_contrail["dsn_dz"]
    diffuse_h_t1 = curr_contrail["diffuse_h"]
    diffuse_v_t1 = curr_contrail["diffuse_v"]

    # Segment-free mode logic
    segment_length_t2: np.ndarray | float
    seg_ratio_t12: np.ndarray | float
    if _is_segment_free_mode(curr_contrail):
        segment_length_t2 = 1.0
        seg_ratio_t12 = 1.0
    else:
        segment_length_t1 = curr_contrail["segment_length"]
        segment_length_t2 = next_contrail["segment_length"]
        seg_ratio_t12 = contrail_properties.segment_length_ratio(
            segment_length_t1, segment_length_t2
        )

    dt = next_contrail["time"] - curr_contrail["time"]

    sigma_yy_t2, sigma_zz_t2, sigma_yz_t2 = contrail_properties.plume_temporal_evolution(
        width_t1=width_t1,
        depth_t1=depth_t1,
        sigma_yz_t1=sigma_yz_t1,
        dsn_dz_t1=dsn_dz_t1,
        diffuse_h_t1=diffuse_h_t1,
        diffuse_v_t1=diffuse_v_t1,
        seg_ratio=seg_ratio_t12,
        dt=dt,
        max_depth=params["max_depth"],
    )

    width_t2, depth_t2 = contrail_properties.new_contrail_dimensions(sigma_yy_t2, sigma_zz_t2)
    next_contrail["sigma_yz"] = sigma_yz_t2
    next_contrail["width"] = width_t2
    next_contrail["depth"] = depth_t2

    area_eff_t2 = contrail_properties.new_effective_area_from_sigma(
        sigma_yy=sigma_yy_t2,
        sigma_zz=sigma_zz_t2,
        sigma_yz=sigma_yz_t2,
    )

    rho_air_t2 = next_contrail["rho_air"]
    plume_mass_per_m_t2 = contrail_properties.plume_mass_per_distance(area_eff_t2, rho_air_t2)
    iwc_t2 = contrail_properties.new_ice_water_content(
        iwc_t1=iwc_t1,
        q_t1=specific_humidity_t1,
        q_t2=specific_humidity_t2,
        q_sat_t1=q_sat_t1,
        q_sat_t2=q_sat_t2,
        mass_plume_t1=plume_mass_per_m_t1,
        mass_plume_t2=plume_mass_per_m_t2,
    )
    next_contrail["iwc"] = iwc_t2

    n_ice_per_m_t1 = curr_contrail["n_ice_per_m"]
    dn_dt_agg = curr_contrail["dn_dt_agg"]
    dn_dt_turb = curr_contrail["dn_dt_turb"]

    n_ice_per_m_t2 = contrail_properties.new_ice_particle_number(
        n_ice_per_m_t1=n_ice_per_m_t1,
        dn_dt_agg=dn_dt_agg,
        dn_dt_turb=dn_dt_turb,
        seg_ratio=seg_ratio_t12,
        dt=dt,
    )
    next_contrail["n_ice_per_m"] = n_ice_per_m_t2

    cocip.calc_contrail_properties(
        next_contrail,
        params["effective_vertical_resolution"],
        params["wind_shear_enhancement_exponent"],
        params["sedimentation_impact_factor"],
        radiative_heating_effects=False,  # Not yet supported in CocipGrid
    )
    cocip.calc_radiative_properties(next_contrail, params)

    rf_net_t1 = curr_contrail["rf_net"]
    rf_net_t2 = next_contrail["rf_net"]
    ef = contrail_properties.energy_forcing(
        rf_net_t1=rf_net_t1,
        rf_net_t2=rf_net_t2,
        width_t1=width_t1,
        width_t2=width_t2,
        seg_length_t2=segment_length_t2,
        dt=dt,
    )
    # NOTE: This will get masked below if `persistent` is False
    # That is, we are taking a right Riemann sum of a decreasing function, so we are
    # underestimating the truth. With dt small enough, this is fine.
    next_contrail["ef"] = ef

    # NOTE: Only dealing with `next_contrail` here
    latitude = next_contrail["latitude"]
    altitude = next_contrail["altitude"]
    tau_contrail = next_contrail["tau_contrail"]
    n_ice_per_vol = next_contrail["n_ice_per_vol"]
    age = next_contrail["age"]

    # Both tau_contrail and n_ice_per_vol could have nan values
    # These are mostly due to out of bounds interpolation
    # Both are computed in cocip.calc_contrail_properties
    # Interpolation out-of-bounds nan values first appear in tau_contrail,
    # then in n_ice_per_vol at the next time step.
    # We can use something like np.nan(tau_contrail) to get values that
    # are filled with nan in interpolation.
    persistent = contrail_properties.contrail_persistent(
        latitude=latitude,
        altitude=altitude,
        segment_length=segment_length_t2,  # type: ignore[arg-type]
        age=age,
        tau_contrail=tau_contrail,
        n_ice_per_m3=n_ice_per_vol,
        params=params,
    )

    # Filter by persistent
    logger.debug(
        "Fraction of grid points surviving: %s / %s",
        np.sum(persistent),
        next_contrail.size,
    )
    if params["persistent_buffer"] is not None:
        # See Cocip implementation if we want to support this
        raise NotImplementedError
    return next_contrail.filter(persistent)


def calc_emissions(vector: GeoVectorDataset, params: dict[str, Any]) -> None:
    """Calculate aircraft performance (AP) and emissions data.

    This function mutates the ``vector`` parameter in-place by setting keys:
        - "true_airspeed": computed by the aircraft performance model
        - "engine_efficiency": computed by the aircraft performance model
        - "fuel_flow": computed by the aircraft performance model
        - "nvpm_ei_n": computed by the :class:`Emissions` model
        - "head_tail_dt"

    The ``params`` parameter is also mutated in-place by setting keys:
        - "wingspan": aircraft wingspan
        - "aircraft_mass": mass of aircraft

    Implementation note: Previously, this function computed "fuel_dist" instead of
    "fuel_flow". While "fuel_dist" is the only variabled needed in
    :func:`find_initial_persistent_contrails`, "fuel_flow" is needed for verbose
    outputs. Moreover, we are anticipating having "fuel_flow" as a preexisting
    variable on the input ``source``, whereas "fuel_dist" is less common and
    less interpretable. So, we set "fuel_flow" here and then calculate "fuel_dist"
    in :func:`find_initial_persistent_contrails`.

    Parameters
    ----------
    vector : GeoVectorDataset
        Grid points already interpolated against CoCiP met data
    params : dict[str, Any]
        CoCiP model parameters. See :class:`CocipGrid`.

    Raises
    ------
    NotImplementedError
        Aircraft type in `params` not found in EDB
    """
    logger.debug("Process emissions")

    # PART 1: Fuel flow data
    vector.attrs.setdefault("aircraft_type", params["aircraft_type"])

    # Important: If params["engine_uid"] is None (the default value), let Emissions
    # overwrite with the assumed value.
    # Otherwise, set the non-None value on vector here
    if param_engine_uid := params["engine_uid"]:
        vector.attrs.setdefault("engine_uid", param_engine_uid)
    vector.attrs.setdefault("fuel", params["fuel"])

    ap_vars = {
        "true_airspeed",
        "engine_efficiency",
        "fuel_flow",
        "aircraft_mass",
        "n_engine",
        "wingspan",
    }

    # Look across both vector.data and vector.attrs
    missing = ap_vars.difference(vector).difference(vector.attrs)

    if missing == {"true_airspeed"}:
        # If we're only missing true_airspeed but mach_number is present,
        # we can still proceed
        mach_number = vector.get_data_or_attr("mach_number", None)
        if mach_number is not None:
            air_temperature = vector["air_temperature"]
            vector["true_airspeed"] = units.mach_number_to_tas(mach_number, air_temperature)
            missing = set()

    if missing:
        ap_model = params["aircraft_performance"]
        if ap_model is None:
            msg = (
                f"Missing variables: {missing} and no aircraft_performance included in "
                "params. Instantiate 'CocipGrid' with an 'aircraft_performance' param. "
                "For example: 'CocipGrid(..., aircraft_performance=PSGrid())'"
            )
            raise ValueError(msg)
        ap_model.eval(vector, copy_source=False)

    # PART 2: True airspeed logic
    # NOTE: This doesn't exactly fit here, but it is closely related to
    # true_airspeed calculations, so it's convenient to get it done now
    # For the purpose of Cocip <> CocipGrid model parity, we attach a
    # "head_tail_dt" variable. This variable is used the first time `advect`
    # is called. It makes a small but noticeable difference in model outputs.
    true_airspeed = vector.get_data_or_attr("true_airspeed")

    if not _is_segment_free_mode(vector):
        segment_length = _get_source_param_override("segment_length", vector, params)
        head_tail_dt_s = segment_length / true_airspeed
        head_tail_dt_ns = 1_000_000_000.0 * head_tail_dt_s
        head_tail_dt = head_tail_dt_ns.astype("timedelta64[ns]")
        vector["head_tail_dt"] = head_tail_dt

    # PART 3: Emissions data
    factor = _get_source_param_override("nvpm_ei_n_enhancement_factor", vector, params)
    default_nvpm_ei_n = params["default_nvpm_ei_n"]

    # Early exit
    if not params["process_emissions"]:
        vector.attrs.setdefault("nvpm_ei_n", factor * default_nvpm_ei_n)
        return

    emissions = Emissions()
    emissions.eval(vector, copy_source=False)
    vector.update(nvpm_ei_n=factor * vector["nvpm_ei_n"])


def calc_wind_shear(
    contrail: GeoVectorDataset,
    dz_m: float,
    *,
    is_downwash: bool,
    dsn_dz_factor: float,
) -> None:
    """Calculate wind shear data.

    This function is used for both `first_contrail` calculation and `evolve_one_step`. The
    data requirements of these two functions is slightly different, and the `is_downwash` flag
    allows for this discrepancy.

    This function modifies the `contrail` parameter in-place by attaching `data` keys:
        - "dT_dz"
        - "ds_dz"
        - "dsn_dz" (attached only if `is_downwash=False`)

    NOTE: This is the only function involving interpolation at a different `level`.

    Parameters
    ----------
    contrail : GeoVectorDataset
        Grid points already interpolated against CoCiP met data
    dz_m : float
        Difference in altitude between top and bottom layer for stratification calculations (m)
    is_downwash : bool
        Only used initially in the `first_contrail` function
    dsn_dz_factor : float
        Experimental parameter for segment-free model.
    """
    air_temperature = contrail["air_temperature"]
    air_pressure = contrail.air_pressure

    u_wind = contrail["eastward_wind"]
    v_wind = contrail["northward_wind"]

    air_pressure_lower = thermo.pressure_dz(air_temperature, air_pressure, dz_m)
    air_temperature_lower = contrail["air_temperature_lower"]
    u_wind_lower = contrail["eastward_wind_lower"]
    v_wind_lower = contrail["northward_wind_lower"]

    dT_dz = thermo.T_potential_gradient(
        air_temperature, air_pressure, air_temperature_lower, air_pressure_lower, dz_m
    )
    ds_dz = wind_shear.wind_shear(u_wind, u_wind_lower, v_wind, v_wind_lower, dz_m)

    contrail["dT_dz"] = dT_dz
    contrail["ds_dz"] = ds_dz

    # Calculate wind shear normal: NOT needed for downwash step
    if is_downwash:
        return

    # Experimental segment-free mode
    # Instead of calculating dsn_dz, just multiply ds_dz with some scalar
    if _is_segment_free_mode(contrail):
        contrail["dsn_dz"] = dsn_dz_factor * ds_dz
        return

    # NOTE: This is the only function requiring cos_a and sin_a
    # Consequently, we don't store sin_a and cos_a on the contrail, but just
    # use them once here to compute dsn_dz
    sin_a, cos_a = geo.longitudinal_angle(
        lons0=contrail["longitude_tail"],
        lats0=contrail["latitude_tail"],
        lons1=contrail["longitude_head"],
        lats1=contrail["latitude_head"],
    )
    dsn_dz = wind_shear.wind_shear_normal(
        u_wind_top=u_wind,
        u_wind_btm=u_wind_lower,
        v_wind_top=v_wind,
        v_wind_btm=v_wind_lower,
        cos_a=cos_a,
        sin_a=sin_a,
        dz=dz_m,
    )
    contrail["dsn_dz"] = dsn_dz


def calc_thermal_properties(contrail: GeoVectorDataset) -> None:
    """Calculate contrail thermal properties.

    Modifies parameter `contrail` in place by attaching keys:
        - "q_sat"
        - "rho_air"

    Parameters
    ----------
    contrail : GeoVectorDataset
        Grid points already interpolated against CoCiP met data.
    """
    air_pressure = contrail.air_pressure
    air_temperature = contrail["air_temperature"]

    # calculate thermo properties
    contrail["q_sat"] = thermo.q_sat_ice(air_temperature, air_pressure)
    contrail["rho_air"] = thermo.rho_d(air_temperature, air_pressure)


def advect(
    contrail: GeoVectorDataset,
    dt: np.timedelta64 | npt.NDArray[np.timedelta64],
    dt_head: np.timedelta64 | None,
    dt_tail: np.timedelta64 | None,
) -> GeoVectorDataset:
    """Form new contrail by advecting existing contrail.

    Parameter ``contrail`` is not modified.

    .. versionchanged:: 0.25.0

        The ``dt_head`` and ``dt_tail`` parameters are no longer optional.
        Set these to ``dt`` to evolve the contrail uniformly over a constant time.
        Set to None for segment-free mode.

    Parameters
    ----------
    contrail : GeoVectorDataset
        Grid points already interpolated against wind data
    dt : np.timedelta64 | npt.NDArray[np.timedelta64]
        Time step for advection
    dt_head : np.timedelta64 | None
        Time step for segment head advection. Use None for segment-free mode.
    dt_tail : np.timedelta64 | None
        Time step for segment tail advection. Use None for segment-free mode.

    Returns
    -------
    GeoVectorDataset
        New contrail instance with keys:
            - "index"
            - "longitude"
            - "latitude"
            - "level"
            - "air_pressure"
            - "altitude",
            - "time"
            - "formation_time"
            - "age"
            - "longitude_head"  (only if `is_segment_free=False`)
            - "latitude_head"  (only if `is_segment_free=False`)
            - "longitude_tail"  (only if `is_segment_free=False`)
            - "longitude_tail"  (only if `is_segment_free=False`)
            - "segment_length"  (only if `is_segment_free=False`)
            - "head_tail_dt"  (only if `is_segment_free=False`)
    """
    longitude = contrail["longitude"]
    latitude = contrail["latitude"]
    level = contrail["level"]
    time = contrail["time"]
    formation_time = contrail["formation_time"]
    age = contrail["age"]
    u_wind = contrail["eastward_wind"]
    v_wind = contrail["northward_wind"]
    vertical_velocity = contrail["lagrangian_tendency_of_air_pressure"]
    rho_air = contrail["rho_air"]
    terminal_fall_speed = contrail["terminal_fall_speed"]

    # Using the _t2 convention for post-advection data
    index_t2 = contrail["index"]
    time_t2 = time + dt
    age_t2 = age + dt

    longitude_t2, latitude_t2 = geo.advect_horizontal(
        longitude=longitude,
        latitude=latitude,
        u_wind=u_wind,
        v_wind=v_wind,
        dt=dt,
    )
    level_t2 = geo.advect_level(level, vertical_velocity, rho_air, terminal_fall_speed, dt)
    altitude_t2 = units.pl_to_m(level_t2)

    data = {
        "index": index_t2,
        "longitude": longitude_t2,
        "latitude": latitude_t2,
        "level": level_t2,
        "air_pressure": 100.0 * level_t2,
        "altitude": altitude_t2,
        "time": time_t2,
        "formation_time": formation_time,
        "age": age_t2,
        **_get_uncertainty_params(contrail),
    }

    if dt_tail is None or dt_head is None:
        assert _is_segment_free_mode(contrail)
        assert dt_tail is None
        assert dt_head is None
        return GeoVectorDataset._from_fastpath(data, attrs=contrail.attrs).copy()

    longitude_head = contrail["longitude_head"]
    latitude_head = contrail["latitude_head"]
    longitude_tail = contrail["longitude_tail"]
    latitude_tail = contrail["latitude_tail"]
    u_wind_head = contrail["eastward_wind_head"]
    v_wind_head = contrail["northward_wind_head"]
    u_wind_tail = contrail["eastward_wind_tail"]
    v_wind_tail = contrail["northward_wind_tail"]

    longitude_head_t2, latitude_head_t2 = geo.advect_horizontal(
        longitude=longitude_head,
        latitude=latitude_head,
        u_wind=u_wind_head,
        v_wind=v_wind_head,
        dt=dt_head,
    )
    longitude_tail_t2, latitude_tail_t2 = geo.advect_horizontal(
        longitude=longitude_tail,
        latitude=latitude_tail,
        u_wind=u_wind_tail,
        v_wind=v_wind_tail,
        dt=dt_tail,
    )

    segment_length_t2 = geo.haversine(
        lons0=longitude_head_t2,
        lats0=latitude_head_t2,
        lons1=longitude_tail_t2,
        lats1=latitude_tail_t2,
    )

    head_tail_dt_t2 = np.full(contrail.size, np.timedelta64(0, "ns"))  # trivial

    data["longitude_head"] = longitude_head_t2
    data["latitude_head"] = latitude_head_t2
    data["longitude_tail"] = longitude_tail_t2
    data["latitude_tail"] = latitude_tail_t2
    data["segment_length"] = segment_length_t2
    data["head_tail_dt"] = head_tail_dt_t2

    return GeoVectorDataset._from_fastpath(data, attrs=contrail.attrs).copy()


def _aggregate_ef_summary(vector_list: list[VectorDataset]) -> VectorDataset | None:
    """Aggregate EF results after cocip simulation.

    Results are summed over each vector in ``vector_list``.

    If ``vector_list`` is empty, return None.

    Parameters
    ----------
    vector_list : list[VectorDataset]
        List of :class:`VectorDataset` objects each containing keys "index", "age", and "ef".

    Returns
    -------
    VectorDataset | None
        Dataset with keys:
            - "index": Used to join to :attr:`CocipGrid.source`
            - "ef": Sum of ef values
            - "age": Contrail age associated to each index
        Only return points with non-zero ef or age.
    """
    if not vector_list:
        return None

    i0 = min(v["index"].min() for v in vector_list)
    i1 = max(v["index"].max() for v in vector_list)
    index = np.arange(i0, i1 + 1)

    # Use the dtype of the first vector to determine the dtype of the aggregate
    v0 = vector_list[0]
    ef = np.zeros(index.shape, dtype=v0["ef"].dtype)
    age = np.zeros(index.shape, dtype=v0["age"].dtype)

    for v in vector_list:
        idx = v["index"] - i0
        ef[idx] += v["ef"]
        age[idx] = np.maximum(age[idx], v["age"])

    # Only return points with non-zero ef or age
    cond = age.astype(bool) | ef.astype(bool)
    index = index[cond].copy()
    ef = ef[cond].copy()
    age = age[cond].copy()

    data = {"index": index, "ef": ef, "age": age}
    return VectorDataset(data, copy=False)


def result_to_metdataset(
    result: VectorDataset | None,
    verbose_dict: dict[str, npt.NDArray[np.floating]],
    source: MetDataset,
    nominal_segment_length: float,
    attrs: dict[str, str],
) -> MetDataset:
    """Convert aggregated data in ``result`` to MetDataset.

    Parameters
    ----------
    result : VectorDataset | None
        Aggregated data arising from contrail evolution. Expected to contain keys:
        ``index``, ``age``, ``ef``.
    verbose_dict : dict[str, npt.NDArray[np.floating]]:
        Verbose outputs to attach to results.
    source : MetDataset
        :attr:`CocipGrid.`source` data on which to attach results.
    nominal_segment_length : float
        Used to normalize energy forcing cumulative sum.
    attrs : dict[str, str]
        Additional global attributes to attach to xr.Dataset.

    Returns
    -------
    MetDataset
        Data with variables ``contrail_age``, ``ef_per_m``, and any other keys
        in ``verbose_dicts`.
    """
    logger.debug("Desparsify grid results into 4D numpy array")

    shape = tuple(value.size for value in source.coords.values())
    size = np.prod(shape)

    dtype = result["ef"].dtype if result else np.float32
    contrail_age_1d = np.zeros(size, dtype=np.float32)
    ef_per_m_1d = np.zeros(size, dtype=dtype)

    if result:
        contrail_idx = result["index"]
        # Step 1: Contrail age. Convert from timedelta to float
        contrail_age_1d[contrail_idx] = result["age"] / np.timedelta64(1, "h")
        # Step 2: EF
        ef_per_m_1d[contrail_idx] = result["ef"] / nominal_segment_length

    contrail_age_4d = contrail_age_1d.reshape(shape)
    ef_per_m_4d = ef_per_m_1d.reshape(shape)

    # Step 3: Dataset dims and attrs
    dims = tuple(source.coords)
    local_attrs = _contrail_grid_variable_attrs()

    # Step 4: Dataset core variables
    data_vars = {
        "contrail_age": (dims, contrail_age_4d, local_attrs["contrail_age"]),
        "ef_per_m": (dims, ef_per_m_4d, local_attrs["ef_per_m"]),
    }

    # Step 5: Dataset variables from verbose_dicts
    for k, v in verbose_dict.items():
        data_vars[k] = (dims, v.reshape(shape), local_attrs[k])

    # Update source
    for k, v in data_vars.items():  # type: ignore[assignment]
        source[k] = v
    source.attrs.update(attrs)  # type: ignore[arg-type]

    # Return reference to source
    return source


def result_merge_source(
    result: VectorDataset | None,
    verbose_dict: dict[str, npt.NDArray[np.floating]],
    source: GeoVectorDataset,
    nominal_segment_length: float | npt.NDArray[np.floating],
    attrs: dict[str, str],
) -> GeoVectorDataset:
    """Merge ``results`` and ``verbose_dict`` onto ``source``."""

    # Initialize the main output arrays to all zeros
    dtype = result["age"].dtype if result else "timedelta64[ns]"
    contrail_age = np.zeros(source.size, dtype=dtype)

    dtype = result["ef"].dtype if result else np.float32
    ef_per_m = np.zeros(source.size, dtype=dtype)

    # If there are results, merge them in
    if result:
        index = result["index"]
        contrail_age[index] = result["age"]

        if isinstance(nominal_segment_length, np.ndarray):
            ef_per_m[index] = result["ef"] / nominal_segment_length[index]
        else:
            ef_per_m[index] = result["ef"] / nominal_segment_length

    # Set the output variables onto the source
    source["contrail_age"] = contrail_age
    source["ef_per_m"] = ef_per_m
    for k, v in verbose_dict.items():
        source.setdefault(k, v)
    source.attrs.update(attrs)

    return source


def _concat_verbose_dicts(
    verbose_dicts: list[dict[str, pd.Series]],
    source_size: int,
    verbose_outputs_formation: set[str],
) -> dict[str, npt.NDArray[np.floating]]:
    # Concatenate the values and return
    ret: dict[str, np.ndarray] = {}
    for key in verbose_outputs_formation:
        series_list = [v for d in verbose_dicts if d and (v := d.get(key)) is not None]
        data = np.concatenate(series_list)
        index = np.concatenate([s.index for s in series_list])

        # Reindex to source_size. Assuming all verbose_dicts have float dtype
        out = np.full(source_size, np.nan, dtype=data.dtype)
        out[index] = data
        ret[key] = out

    return ret


def _contrail_grid_variable_attrs() -> dict[str, dict[str, str]]:
    """Get attributes for each variables in :class:`CocipGrid` gridded output.

    TODO: It might be better for these to live elsewhere (ie, in some `variables.py`).
    """
    return {
        "contrail_age": {
            "long_name": "Total age in hours of persistent contrail",
            "units": "hours",
        },
        "ef_per_m": {
            "long_name": "Energy forcing per meter of flight trajectory",
            "units": "J / m",
        },
        "sac": {"long_name": "Schmidt-Appleman Criterion"},
        "persistent": {"long_name": "Contrail initially persistent state"},
        "T_crit_sac": {
            "long_name": "Schmidt-Appleman critical temperature threshold",
            "units": "K",
        },
        "engine_efficiency": {"long_name": "Engine efficiency"},
        "true_airspeed": {"long_name": "True airspeed", "units": "m / s"},
        "aircraft_mass": {"long_name": "Aircraft mass", "units": "kg"},
        "nvpm_ei_n": {
            "long_name": "Black carbon emissions index number",
            "units": "kg^{-1}",
        },
        "fuel_flow": {"long_name": "Jet engine fuel flow", "units": "kg / s"},
        "specific_humidity": {"long_name": "Specific humidity", "units": "kg / kg"},
        "air_temperature": {"long_name": "Air temperature", "units": "K"},
        "rhi": {"long_name": "Relative humidity", "units": "dimensionless"},
        "iwc": {
            "long_name": "Ice water content after the wake vortex phase",
            "units": "kg_h2o / kg_air",
        },
        "global_yearly_mean_rf_per_m": {
            "long_name": "Global yearly mean RF per meter of flight trajectory",
            "units": "W / m**2 / m",
        },
        "atr20_per_m": {
            "long_name": "Average Temperature Response over a 20 year horizon",
            "units": "K / m",
        },
    }


def _supported_verbose_outputs_formation() -> set[str]:
    """Get supported keys for verbose outputs.

    Uses output of :func:`_contrail_grid_variable_attrs` as a source of truth.
    """
    return set(_contrail_grid_variable_attrs()) - {
        "contrail_age",
        "ef_per_m",
        "global_yearly_mean_rf_per_m",
        "atr20_per_m",
    }


def _warn_not_wrap(met: MetDataset) -> None:
    """Warn user if parameter met should be wrapped.

    Parameters
    ----------
    met : MetDataset
        Met dataset
    """
    if met.is_wrapped:
        return
    lon = met.indexes["longitude"]
    if lon.min() == -180.0 and lon.max() == 179.75:
        warnings.warn(
            "The MetDataset `met` not been wrapped. The CocipGrid model may "
            "perform better if `met.wrap_longitude()` is called first."
        )


def _get_uncertainty_params(contrail: VectorDataset) -> dict[str, npt.NDArray[np.floating]]:
    """Return uncertainty parameters in ``contrail``.

    This function assumes the underlying humidity scaling model is
    :class:`ConstantHumidityScaling`. This function should get revised if other
    humidity scaling models are used for uncertainty analysis.

    For each of the keys:
        - "rhi_adj",
        - "rhi_boost_exponent",
        - "sedimentation_impact_factor",
        - "wind_shear_enhancement_exponent",

    this function checks if key is present in contrail. The data is then
    bundled and returned as a dictionary.

    Parameters
    ----------
    contrail : VectorDataset
        Data from which uncertainty parameters are extracted

    Returns
    -------
    dict[str, npt.NDArray[np.floating]]
        Dictionary of uncertainty parameters.
    """
    keys = (
        "rhi_adj",
        "rhi_boost_exponent",
        "sedimentation_impact_factor",
        "wind_shear_enhancement_exponent",
    )
    return {key: val for key in keys if (val := contrail.get(key)) is not None}


_T = TypeVar("_T", np.float64, np.datetime64)


def _check_coverage(
    met_array: npt.NDArray[_T], grid_array: npt.NDArray[_T], coord: str, name: str
) -> None:
    """Warn if the met data does not cover the entire source domain.

    Parameters
    ----------
    met_array : np.ndarray
        Coordinate on met data
    grid_array : np.ndarray
        Coordinate on grid data
    coord : {"longitude", "latitude", "level", "time"}
        Name of coordinate. Only used for warning message.
    name : str
        Name of met dataset. Only used for warning message.
    """
    if met_array.min() > grid_array.min() or met_array.max() < grid_array.max():
        warnings.warn(
            f"Met data '{name}' does not cover the source domain along the {coord} axis. "
            "This causes some interpolated values to be nan, leading to meaningless results."
        )


def _downselect_met(
    source: GeoVectorDataset | MetDataset, met: MetDataset, rad: MetDataset, params: dict[str, Any]
) -> tuple[MetDataset, MetDataset]:
    """Downselect met and rad to the bounding box of the source.

    Implementation is nearly identical to the :meth:`Model.downselect_met` method. The
    key difference is that this method uses the "max_age" and "dt_integration" parameters
    to further buffer the bounding box in the time dimension.

    .. versionchanged:: 0.25.0

        Support :class:`MetDataset` ``source`` for use in :class:`CocipGrid`.

    Parameters
    ----------
    source : GeoVectorDataset | MetDataset
        Model source
    met : MetDataset
        Model met
    rad : MetDataset
        Model rad
    params : dict[str, Any]
        Model parameters

    Returns
    -------
    met : MetDataset
        MetDataset with met data copied within the bounding box of ``source``.
    rad : MetDataset
        MetDataset with rad data copied within the bounding box of ``source``.

    See Also
    --------
    :meth:`Model.downselect_met`
    """

    if not params["downselect_met"]:
        logger.debug("Avoiding downselecting met because params['downselect_met'] is False")
        return met, rad

    logger.debug("Downselecting met domain to vector points")

    # check params
    longitude_buffer = params["met_longitude_buffer"]
    latitude_buffer = params["met_latitude_buffer"]
    level_buffer = params["met_level_buffer"]
    time_buffer = params["met_time_buffer"]

    # Down select met relative to min / max integration timesteps, not Flight
    t0 = time_buffer[0]
    t1 = time_buffer[1] + params["max_age"] + params["dt_integration"]

    met = source.downselect_met(
        met,
        latitude_buffer=latitude_buffer,
        longitude_buffer=longitude_buffer,
        level_buffer=level_buffer,
        time_buffer=(t0, t1),
    )

    rad = source.downselect_met(
        rad,
        latitude_buffer=latitude_buffer,
        longitude_buffer=longitude_buffer,
        time_buffer=(t0, t1),
    )

    return met, rad


def _is_segment_free_mode(vector: GeoVectorDataset) -> bool:
    """Determine if model is run in a segment-free mode."""
    return "longitude_head" not in vector


def _check_met_rad_time(
    met: MetDataset,
    rad: MetDataset,
    tmin: pd.Timestamp,
    tmax: pd.Timestamp,
) -> None:
    """Warn if meteorology data doesn't cover a required time range.

    Parameters
    ----------
    met : MetDataset
        Meteorology dataset
    rad : MetDataset
        Radiative flux dataset
    tmin: pd.Timestamp
        Start of required time range
    tmax:pd.Timestamp
        End of required time range
    """
    met_time = met.data["time"].values
    met_tmin = pd.to_datetime(met_time.min())
    met_tmax = pd.to_datetime(met_time.max())
    _check_start_time(met_tmin, tmin, "met")
    _check_end_time(met_tmax, tmax, "met")

    rad_time = rad.data["time"].values
    rad_tmin = pd.to_datetime(rad_time.min())
    rad_tmax = pd.to_datetime(rad_time.max())
    note = "differencing reduces time coverage when providing accumulated radiative fluxes."
    _check_start_time(rad_tmin, tmin, "rad", note=note)
    _check_end_time(rad_tmax, tmax, "rad", note=note)


def _check_start_time(
    met_start: pd.Timestamp,
    model_start: pd.Timestamp,
    name: str,
    *,
    note: str | None = None,
) -> None:
    if met_start > model_start:
        note = f" Note: {note}" if note else ""
        warnings.warn(
            f"Start time of parameter '{name}' ({met_start}) "
            f"is after model start time ({model_start}). "
            f"Include additional time at the start of '{name}'."
            f"{note}"
        )


def _check_end_time(
    met_end: pd.Timestamp,
    model_end: pd.Timestamp,
    name: str,
    *,
    note: str | None = None,
) -> None:
    if met_end < model_end:
        note = f" Note: {note}" if note else ""
        warnings.warn(
            f"End time of parameter '{name}' ({met_end}) "
            f"is before model end time ({model_end}). "
            f"Include additional time at the end of '{name}' or reduce 'max_age' parameter."
            f"{note}"
        )
