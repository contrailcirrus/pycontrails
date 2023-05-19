"""Gridded CoCiP model."""

from __future__ import annotations

import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Iterator,
    NoReturn,
    Sequence,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

import pycontrails
from pycontrails.core import models
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataset
from pycontrails.core.vector import GeoVectorDataset, VectorDataset
from pycontrails.models import humidity_scaling, sac
from pycontrails.models.cocip import cocip, contrail_properties, wake_vortex, wind_shear
from pycontrails.models.cocipgrid import cocip_time_handling
from pycontrails.models.cocipgrid.cocip_grid_params import CocipGridParams
from pycontrails.models.emissions import black_carbon, emissions
from pycontrails.physics import geo, thermo, units

try:
    from pycontrails.ext.bada import BADAGrid
except ImportError as e:
    raise ImportError(
        'CocipGrid requires BADA extension. Install with `pip install "pycontrails-bada @'
        ' git+ssh://git@github.com/contrailcirrus/pycontrails-bada.git"`'
    ) from e

if TYPE_CHECKING:
    import tqdm

logger = logging.getLogger(__name__)


class CocipGrid(models.Model, cocip_time_handling.CocipTimeHandlingMixin):
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

    References
    ----------
    - :cite:`schumannPotentialReduceClimate2011`
    - :cite:`schumannContrailsVisibleAviation2012`

    See Also
    --------
    :class:`CocipGridParams`
    :class:`Cocip`
    :class:`BADAGrid`
    :mod:`wake_vortex`
    :mod:`contrail_properties`
    :mod:`radiative_forcing`
    :mod:`humidity_scaling`
    :class:`Emissions`
    :mod:`sac`
    :mod:`tau_cirrus`
    """

    name = "contrail_grid"
    long_name = "Gridded Contrail Cirrus Prediction Model"
    default_params = CocipGridParams

    # Reference Cocip as the source of truth for met variables
    met_variables = cocip.Cocip.met_variables
    rad_variables = cocip.Cocip.rad_variables
    processed_met_variables = cocip.Cocip.processed_met_variables

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
        self.validate_time_params()

        shift_radiation_time = self.params["shift_radiation_time"]
        met, rad = cocip.process_met_datasets(met, rad, shift_radiation_time)

        self.met = met
        self.rad = rad

        # Convenience -- only used in `run_interpolators`
        self.params["_interp_kwargs"] = self.interp_kwargs

        if self.params["radiative_heating_effects"]:
            raise NotImplementedError(
                "Parameter 'radiative_heating_effects' is not yet implemented in CocipGrid"
            )

        self._target_dtype = np.result_type(*self.met.data.values())

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset:
        ...

    @overload
    def eval(self, source: MetDataset, **params: Any) -> MetDataset:
        ...

    @overload
    def eval(self, source: None = None, **params: Any) -> NoReturn:
        ...

    def eval(
        self, source: GeoVectorDataset | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | MetDataset | NoReturn:
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
            Input :class:`GeoVectorDataset` or :class:`MetDataset`. If ``source`` is
            :class:`MetDataset`, only its underlying coordinates (longitude, latitude,
            level, and time values) are used.
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
            - Convert any :class:`MetDataset` ``source`` to :class:`GeoVectorDataset`
            - Use the :attr:`params` ``met_slice_dt`` to slice :attr:`met` into
              4d sub-grids with at least two timesteps.
            - For each ``met`` time slice, initialize ``source`` points belonging to
              the slice.
            - For each "active" contrail (i.e., a contrail that has been initialized but
              has not yet reach its end of life), evolve the contrail to the terminal
              timestep of the ``met`` slice.
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
            raise NotImplementedError("CocipGrid.eval() with 'source=None' is not implemented.")
        if isinstance(source, Flight):
            if not source.ensure_vars(("true_airspeed", "azimuth"), False):
                warnings.warn(
                    "Flight source no longer supported by CocipGrid. "
                    "Any Flight source will be viewed as a GeoVectorDataset. "
                    "In particular, flight segment variable such as azimuth and true_airspeed "
                    "are not used by CocipGrid (nominal values are used instead). Attach "
                    "these to the Flight source before calling 'eval' to use them in CocipGrid."
                )
        self.set_source(source)

        self.met, self.rad = _downselect_met(self.source, self.met, self.rad, self.params)
        self._check_met_source_overlap()

        # Save humidity scaling type to output attrs
        if self.params["humidity_scaling"] is not None:
            for k, v in self.params["humidity_scaling"].description.items():
                self.source.attrs[f"humidity_scaling_{k}"] = v

        self._parse_verbose_outputs()
        self.attach_timedict()
        pbar = self.init_pbar()

        summary: VectorDataset | None
        summary_by_met_slice: list[VectorDataset] = []
        existing_vectors: Iterator[GeoVectorDataset] = iter(())
        verbose_dicts: list[dict[str, pd.Series]] = []
        contrail_list: list[GeoVectorDataset] = []
        for time, filt in self.timedict.items():
            met, rad = self._load_met_slices(time, pbar)

            new_vectors = self._generate_new_vectors(filt)
            evolved = []
            for vector in new_vectors:
                evolved.append(_evolve_vector(vector, met, rad, self.params, True, pbar))
            for vector in existing_vectors:
                evolved.append(_evolve_vector(vector, met, rad, self.params, False, pbar))
            if not evolved:
                break

            vectors, summary_data, verbose_dicts_this_step, contrail_lists = zip(*evolved)
            existing_vectors = combine_vectors(vectors, self.params["target_split_size"])
            summary = VectorDataset.sum([r for r in summary_data if r])
            summary_by_met_slice.append(summary)

            if self.params["verbose_outputs_formation"]:
                verbose_dicts.extend([d for d in verbose_dicts_this_step if d])
            if self.params["verbose_outputs_evolution"]:
                contrail_list.extend([v for cl in contrail_lists for v in cl if v])

            # TODO: Adjust pbar

        if pbar is not None:
            logger.debug("Close progress bar")
            pbar.refresh()
            pbar.close()

        self._attach_verbose_outputs_evolution(contrail_list)
        summary_by_met_slice = [s for s in summary_by_met_slice if s]
        if summary_by_met_slice:
            summary = calc_intermediate_results(summary_by_met_slice)
        else:
            summary = None
        return self._bundle_results(summary, verbose_dicts)

    def _attach_verbose_outputs_evolution(self, contrail_list: list[GeoVectorDataset]) -> None:
        """Attach intermediate artifacts to the model.

        This method attaches :attr:`contrail_list` and :attr:`contrail` when
        :attr:`params["verbose_outputs_evolution"]` is True.

        Mirrors implementation in :class:`Cocip`. We could do additional work here
        if this turns out to be useful.
        """
        if self.params["verbose_outputs_evolution"]:
            # Attach the raw data
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

        attrs = {
            "description": self.long_name,
            "max_age": max_age_str,
            "dt_integration": dt_integration_str,
            "aircraft_type": self.get_source_param("aircraft_type"),
            "bada_model": self.get_source_param("bada_model"),
            "met_source": self.met.attrs.get("met_source", "unknown"),
            "pycontrails_version": pycontrails.__version__,
            **cast(dict[str, Any], self.source.attrs),
        }

        if isinstance(azimuth, (np.floating, np.integer)):
            attrs["azimuth"] = azimuth.item()
        elif isinstance(azimuth, (float, int)):
            attrs["azimuth"] = azimuth

        if isinstance(self.source, MetDataset):
            self.source = result_to_metdataset(
                result=summary,
                verbose_dict=verbose_dict,
                source=self.source,
                nominal_segment_length=segment_length,
                attrs=attrs,
            )
        else:
            self.source = result_merge_source(
                result=summary,
                verbose_dict=verbose_dict,
                source=self.source,
                nominal_segment_length=segment_length,
                attrs=attrs,
            )
        return self.source

    # ---------------------------
    # Common Methods & Properties
    # ---------------------------

    @property
    def _nominal_vector_params(self) -> list[str]:
        """Return list of parameters that act as nominal vector values.

        Returns
        -------
        list[str]
        """
        return [
            "aircraft_mass",
            "true_airspeed",
            "engine_efficiency",
            "fuel_flow",
            "thrust",
            "segment_length",
        ]

    @property
    def _nominal_attrs_params(self) -> list[str]:
        """Return list of parameters that act as nominal vector values.

        Returns
        -------
        list[str]
        """
        return ["aircraft_type", "wingspan"]

    def _parse_verbose_outputs(self) -> None:
        """Confirm param "verbose_outputs" has the expected type for grid and path mode.

        This function mutates the "verbose_outputs" field on :attr:`params`.

        Currently, the list of all supported variables for verbose outputs
        is determine by :func:`_supported_verbose_outputs`.
        """
        if self.params["verbose_outputs"]:
            raise ValueError(
                "Parameter 'verbose_outputs' is no longer supported for grid mode. "
                "Instead, use 'verbose_outputs_formation' and 'verbose_outputs_evolution'."
            )
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

    def _generate_new_vectors(
        self, filt: npt.NDArray[np.bool_]
    ) -> Generator[GeoVectorDataset, None, None]:
        """Generate :class:`GeoVectorDataset` instances from :attr:`source`.

        Parameters
        ----------
        filt : npt.NDArray[np.bool_]
            Boolean array that can be used to filter :attr:`self.source_time`.

        Yields
        ------
        GeoVectorDataset
            Unevolved vectors arising from :attr`self.source_time` filtered by ``filt``.
            When :attr:`source` is a :class:`MetDataset`, each yielded dataset has a
            constant time value.
        """
        if "index" in self.source:
            # FIXME: We can simply change the internal variable to __index
            raise RuntimeError("The variable 'index' is used internally. Found in source.")

        if isinstance(self.source, MetDataset):
            source_time = self.source_time
            times_in_filt = self.source_time[filt]
            filt_start_idx = np.argmax(filt).item()  # needed to ensure globally unique indexes

            n_splits = self._grid_spatial_n_splits()
            for idx, time in enumerate(times_in_filt):
                # For now, sticking with the convention that every vector should
                # have a constant time value.
                source_slice = MetDataset(self.source.data.sel(time=[time]))

                # Convert the 4D grid to a vector
                vector = source_slice.to_vector()
                vector.update(longitude=vector["longitude"].astype(self._target_dtype, copy=False))
                vector.update(latitude=vector["latitude"].astype(self._target_dtype, copy=False))
                vector.update(level=vector["level"].astype(self._target_dtype, copy=False))
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
            raise TypeError("Unknown source")

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
        for key in self._nominal_vector_params:
            if key in vector:
                continue
            if (scalar := self.params[key]) is None:
                continue
            dtype = np.result_type(np.float32, np.min_scalar_type(scalar))
            vector[key] = np.full(vector.size, scalar, dtype=dtype)

        # Mirror logic of method get_source_param, but for vector
        segment_length = vector.get(
            "segment_length", vector.attrs.get("segment_length", self.params["segment_length"])
        )
        azimuth = vector.get("azimuth", vector.attrs.get("azimuth", self.params["azimuth"]))

        # Experimental segment-free mode logic
        if azimuth is None and segment_length is None:
            return vector
        if azimuth is None:
            raise ValueError("Set segment_length to None for experimental segment-free model")
        if segment_length is None:
            raise ValueError("Set azimuth to None for experimental segment-free model")
        if self.params["dsn_dz_factor"]:
            raise ValueError("dsn_dz_factor not supported outside of the segment-free mode")

        lons = vector["longitude"]
        lats = vector["latitude"]
        dist = segment_length / 2

        # These should probably not be included in model input ... so
        # we'll get a warning if they get overwritten
        vector["longitude_head"], vector["latitude_head"] = geo.forward_azimuth(
            lons=lons, lats=lats, az=azimuth, dist=dist
        )
        vector["longitude_tail"], vector["latitude_tail"] = geo.forward_azimuth(
            lons=lons, lats=lats, az=azimuth, dist=-dist
        )

        return vector

    def _check_met_source_overlap(self) -> None:
        if not hasattr(self, "source"):
            raise ValueError("No source set")

        if isinstance(self.source, MetDataset):
            longitude = self.source.data["longitude"].values
            latitude = self.source.data["latitude"].values
            level = self.source.data["level"].values
            time = self.source.data["time"].values
        else:
            longitude = self.source["longitude"]
            latitude = self.source["latitude"]
            level = self.source.level
            time = self.source["time"]

        _check_overlap(self.met.data["longitude"].values, longitude, "longitude", "met")
        _check_overlap(self.met.data["latitude"].values, latitude, "latitude", "met")
        _check_overlap(self.met.data["level"].values, level, "level", "met")
        _check_overlap(self.met.data["time"].values, time, "time", "met")
        _check_overlap(self.rad.data["longitude"].values, longitude, "longitude", "rad")
        _check_overlap(self.rad.data["latitude"].values, latitude, "latitude", "rad")
        _check_overlap(self.rad.data["time"].values, time, "time", "rad")

        _warn_not_wrap(self.met)
        _warn_not_wrap(self.rad)

    # ------------
    # Constructors
    # ------------

    @staticmethod
    def create_source(
        level: npt.NDArray[np.float_] | list[float] | float,
        time: npt.NDArray[np.datetime64] | list[np.datetime64] | np.datetime64,
        longitude: npt.NDArray[np.float_] | list[float] | None = None,
        latitude: npt.NDArray[np.float_] | list[float] | None = None,
        lon_step: float = 1.0,
        lat_step: float = 1.0,
    ) -> MetDataset:
        """
        Shortcut to create a MetDataset source from coordinate arrays.

        Parameters
        ----------
        level : level: npt.NDArray[np.float_] | list[float] | float
            Pressure levels for gridded cocip.
            To avoid interpolating outside of the passed ``met`` and ``rad`` data, this
            parameter should avoid the extreme values of the ``met`` and `rad` levels.
            If ``met`` is already defined, a good choice for ``level`` is
            ``met.data['level'].values[1: -1]``.
        time: npt.NDArray[np.datetime64 | list[np.datetime64] | np.datetime64,
            One or more time values for gridded cocip.
        longitude, latitude : npt.NDArray[np.float_] | list[float], optional
            Longitude and latitude arrays, by default None. If not specified, values of
            ``lon_step`` and ``lat_step`` are used to define ``longitude`` and ``latitude``.
            To avoid model degradation at the poles, latitude values are expected to be
            between -80 and 80 degrees.
        lon_step, lat_step : float, optional
            Longitude and latitude resolution, by default 1.0.
            Only used if parameter ``longitude`` (respective ``latitude``) not specified.

        Returns
        -------
        MetDataset
            MetDataset that can be used as ``source`` input to :meth:`CocipGrid.eval(source=...)`
        """
        if longitude is None:
            longitude = np.arange(-180, 180, lon_step, dtype=float)
        if latitude is None:
            latitude = np.arange(-80, 80.000001, lat_step, dtype=float)

        out = MetDataset.from_coords(longitude=longitude, latitude=latitude, level=level, time=time)

        if np.any(out.data.latitude > 80.0001) or np.any(out.data.latitude < -80.0001):
            raise ValueError("Model only supports latitude between -80 and 80.")

        return out


################################
# Functions used by CocipGrid
################################


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
            raise ValueError("`keys` override only valid for `met` input")

        for met_key in keys:
            # NOTE: Changed in v0.43: no longer overwrites existing variables
            models.interpolate_met(met, vector, met_key, **interp_kwargs)

        return _apply_humidity_scaling(vector, humidity_scaling, humidity_interpolated)

    if dz_m is None:
        raise TypeError("Specify `dz_m`.")
    if rad is None:
        raise TypeError("Specify `rad`")

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
    air_pressure_lower = thermo.p_dz(air_temperature, air_pressure, dz_m)
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
    met: MetDataset,
    rad: MetDataset,
    params: dict[str, Any],
    run_downwash: bool,
    pbar: "tqdm.tqdm" | None,
) -> tuple[
    GeoVectorDataset,
    VectorDataset | None,
    dict[str, pd.Series] | None,
    list[GeoVectorDataset] | None,
]:
    """Evolve ``vector`` over lifespan of parameter ``met``.

    The parameter ``met`` is used as the source of timesteps for contrail evolution.

    Return surviving contrail at end of evolution and aggregate metrics from evolution.

    .. versionchanged:: 0.25.0

        No longer expect ``vector`` to have a constant time variable. Consequently,
        time step handling now mirrors that in :class:`Cocip`. Moreover, this method now
        handles both :class:`GeoVectorDataset` and :class:`MetDataset` vectors derived
        from :attr:`source`.

    Parameters
    ----------
    vector : GeoVectorDataset
        Grid points of interest.
    met, rad : MetDataset
        CoCiP met and rad slices. See :class:`CocipGrid`.
    params : dict[str, Any]
        CoCiP model parameters. See :class:`CocipGrid`.
    run_downwash : bool
        Use to run initial downwash on waypoints satisfying SAC
    pbar : tqdm.tqmd | None
        Track ``tqdm`` progress bar over simulation.

    Returns
    -------
    vector : GeoVectorDataset
        Evolved contrail at end of evolution.
    summary_data : VectorDataset | None
        Contrail summary statistics. Includes keys "index", "age", "ef".
    verbose_dict : dict[str, pd.Series] | None
        Dictionary of verbose outputs. None if ``run_downwash`` is False.
    contrail_list : list[GeoVectorDataset] | None
        List of intermediate evolved contrails. None if
        ``params["verbose_outputs_evolution"]`` is False.
    """
    # Determine if we need to keep
    contrail_list: list[GeoVectorDataset] | None
    contrail_list = [] if params["verbose_outputs_evolution"] else None

    # Run downwash and first contrail calculation
    if run_downwash:
        vector, verbose_dict = _run_downwash(vector, met, rad, params)
        if contrail_list is not None:
            contrail_list.append(vector.copy())

        # T_crit_sac is no longer needed. If verbose_outputs_formation is True,
        # it's already storied in the verbose_dict adta
        vector.data.pop("T_crit_sac", None)
        if pbar is not None:
            pbar.update()

        # Early exit if no waypoints survive downwash
        if not vector:
            return vector, None, verbose_dict, contrail_list
    else:
        verbose_dict = None

    summary_data = []

    met_times = met.data["time"].values
    t0 = met_times[0]
    t1 = met_times[-1]
    dt_integration = params["dt_integration"]
    timesteps = np.arange(t0 + dt_integration, t1 + dt_integration, dt_integration)

    # Not strictly necessary: Avoid looping few first few timesteps if waypoints not
    # yet online. Cocip uses similar logic in _calc_timesteps.
    timesteps = timesteps[timesteps > vector["time"].min()]

    # Only used for logging below
    start_size = vector.size

    for t in timesteps:
        if not vector:
            break

        # This if-else below is not strictly necessary ... it might be slightly
        # more performant to avoid the call to vector.filter, which only occurs
        # with GeoVectorDataset sources.
        filt = vector["time"] < t
        if np.all(filt):
            v_now = vector
            v_future = None
        else:
            v_now = vector.filter(filt)
            v_future = vector.filter(~filt)

            if not v_now:
                continue

        dt = t - v_now["time"]

        # Segment-free mode
        if _is_segment_free_mode(v_now):
            dt_head = None
            dt_tail = None
        else:
            head_tail_dt = v_now["head_tail_dt"]
            half_head_tail_dt = head_tail_dt / 2
            dt_head = dt - half_head_tail_dt
            dt_tail = dt + half_head_tail_dt

        # After advection, v_next has time t
        v_next = advect(v_now, dt, dt_head, dt_tail)

        v_next = run_interpolators(
            v_next,
            met,
            rad,
            dz_m=params["dz_m"],
            humidity_scaling=params["humidity_scaling"],
            **params["_interp_kwargs"],
        )
        v_next = calc_evolve_one_step(v_now, v_next, params)
        if v_next:
            summary_data.append(v_next.select(("index", "age", "ef")))
        vector = v_next + v_future

        if contrail_list is not None:
            contrail_list.append(vector)
        if pbar is not None:
            pbar.update()

    # Bundle results, return tuple
    end_size = vector.size
    logger.debug("After evolution, contrail contains %s / %s points.", end_size, start_size)

    if summary_data:
        return vector, calc_intermediate_results(summary_data), verbose_dict, contrail_list
    return vector, None, verbose_dict, contrail_list


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
    for key in vector:
        if key in verbose_outputs_formation:
            verbose_dict[key] = pd.Series(data=vector[key], index=vector["index"])

    # Get verbose outputs from SAC calculation.
    vector, sac_ = find_initial_contrail_regions(vector, params)
    if (key := "sac") in verbose_outputs_formation:
        verbose_dict[key] = sac_
    if (key := "T_crit_sac") in verbose_outputs_formation:
        # This key isn't always present, e.g. find_initial_contrail_regions can exit early
        if (data := vector.get(key)) is not None:
            verbose_dict[key] = pd.Series(data=data, index=vector["index"])

    # Early exit if nothing in vector passes the SAC
    if not vector:
        logger.debug("No vector waypoints satisfy SAC")
        return vector, verbose_dict

    vector = run_interpolators(vector, met, rad, dz_m=params["dz_m"], **params["_interp_kwargs"])
    contrail = simulate_wake_vortex_downwash(vector, params)

    contrail = run_interpolators(
        contrail,
        met,
        rad,
        dz_m=params["dz_m"],
        humidity_scaling=params["humidity_scaling"],
        **params["_interp_kwargs"],
    )
    contrail, persistent = find_initial_persistent_contrails(vector, contrail, params)

    if (key := "persistent") in verbose_outputs_formation:
        verbose_dict[key] = persistent
    if (key := "iwc") in verbose_outputs_formation:
        if (data := contrail.get(key)) is not None:
            verbose_dict[key] = pd.Series(data=data, index=contrail["index"])

    return contrail, verbose_dict


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
    fuel = vector.attrs.get("fuel", params["fuel"])
    ei_h2o = fuel.ei_h2o
    q_fuel = fuel.q_fuel

    G = sac.slope_mixing_line(specific_humidity, air_pressure, engine_efficiency, ei_h2o, q_fuel)
    t_sat_liq = sac.T_sat_liquid(G)
    rh = thermo.rh(specific_humidity, air_temperature, air_pressure)
    rh_crit = sac.rh_critical_sac(air_temperature, t_sat_liq, G)
    sac_ = sac.sac(rh, rh_crit)

    filt = sac_ == 1
    logger.debug(
        "Fraction of grid points satisfying the SAC: %s / %s.",
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

    wsee = vector.get("wind_shear_enhancement", params["wind_shear_enhancement_exponent"])
    wingspan = vector.get("wingspan", vector.attrs.get("wingspan", params.get("wingspan")))
    aircraft_mass = vector.get(
        "aircraft_mass", vector.attrs.get("aircraft_mass", params.get("aircraft_mass"))
    )
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
    iwvd = vector.get("initial_wake_vortex_depth", params["initial_wake_vortex_depth"])
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
    air_pressure_1 = 100 * level_1

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
        return GeoVectorDataset(data, attrs=vector.attrs, copy=True)

    # Stored in `_generate_new_grid_vectors`
    data["longitude_head"] = vector["longitude_head"]
    data["latitude_head"] = vector["latitude_head"]
    data["longitude_tail"] = vector["longitude_tail"]
    data["latitude_tail"] = vector["latitude_tail"]
    data["segment_length"] = vector["segment_length"]
    data["head_tail_dt"] = vector["head_tail_dt"]

    return GeoVectorDataset(data, attrs=vector.attrs, copy=True)


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
    n_ice_per_m = contrail_properties.ice_particle_number(
        nvpm_ei_n=nvpm_ei_n,
        fuel_dist=fuel_dist,
        iwc=iwc,
        iwc_1=iwc_1,
        air_temperature=air_temperature,
        T_crit_sac=T_crit_sac,
        min_ice_particle_number_nvpm_ei_n=params["min_ice_particle_number_nvpm_ei_n"],
    )

    # The logic below corresponds to Cocip._create_downwash_contrail (roughly)
    contrail["iwc"] = iwc_1
    contrail["n_ice_per_m"] = n_ice_per_m

    # Check for persistent initial_contrails
    rhi_1 = contrail["rhi"]
    persistent_1 = contrail_properties.initial_persistent(iwc_1=iwc_1, rhi_1=rhi_1)

    logger.debug(
        "Fraction of grid points with persistent initial contrails: %s / %s",
        persistent_1.sum(),
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
    try:
        segment_length_t1 = curr_contrail["segment_length"]
        segment_length_t2 = next_contrail["segment_length"]
        seg_ratio_t12 = contrail_properties.segment_length_ratio(
            segment_length_t1, segment_length_t2
        )
    except KeyError:
        segment_length_t2 = 1.0  # type: ignore[assignment]
        seg_ratio_t12 = 1.0  # type: ignore[assignment]

    sigma_yy_t2, sigma_zz_t2, sigma_yz_t2 = contrail_properties.plume_temporal_evolution(
        width_t1=width_t1,
        depth_t1=depth_t1,
        sigma_yz_t1=sigma_yz_t1,
        dsn_dz_t1=dsn_dz_t1,
        diffuse_h_t1=diffuse_h_t1,
        diffuse_v_t1=diffuse_v_t1,
        seg_ratio=seg_ratio_t12,
        dt=params["dt_integration"],
        max_contrail_depth=params["max_contrail_depth"],
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
        dt=params["dt_integration"],
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
        dt=params["dt_integration"],
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
        segment_length=segment_length_t2,
        age=age,
        tau_contrail=tau_contrail,
        n_ice_per_m3=n_ice_per_vol,
        params=params,
    )

    # Filter by persistent
    logger.debug(
        "Fraction of gridpoints surviving: %s / %s",
        np.sum(persistent),
        next_contrail.size,
    )
    if params["persistent_buffer"] is not None:
        # See Cocip implementation if we want to support this
        raise NotImplementedError
    return next_contrail.filter(persistent)


def calc_emissions(vector: GeoVectorDataset, params: dict[str, Any]) -> None:
    """Calculate BADA-derived fuel and emissions data.

    The BADA database to use (BADA3 or BADA4) is determined by the `bada_priority`
    parameter.

    This function mutates the ``vector`` parameter in-place by setting keys:
        - "true_airspeed": nominal value from BADA database to use for TAS
        - "engine_efficiency": computed in :meth:`BADA.simulate_fuel_and_performance`
        - "fuel_flow": compute in :meth:`BADA.simulate_fuel_and_performance`
        - "nvpm_ei_n": computed in Emissions methods
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

    .. versionchanged:: 0.25.0

        No longer support :class:`BADAFlight` model for emissions.

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

    # Important: If params["engine_uid"] is None (the default value), let BADAGrid
    # overwrite with the assumed value.
    # Otherwise, set the non-None value on vector here
    if param_engine_uid := params["engine_uid"]:
        vector.attrs.setdefault("engine_uid", param_engine_uid)
    fuel = vector.attrs.setdefault("fuel", params["fuel"])

    bada_vars = ["true_airspeed", "engine_efficiency", "fuel_flow", "aircraft_mass", "n_engine"]
    if not vector.ensure_vars(bada_vars, False):
        bada_params = {
            "bada3_path": params["bada3_path"],
            "bada4_path": params["bada4_path"],
            "bada_priority": params["bada_priority"],
            "copy_source": False,
        }

        bada_model = BADAGrid(**bada_params)
        vector = bada_model.eval(vector)

    # PART 2: True airspeed logic
    # NOTE: This doesn't exactly fit here, but it is closely related to
    # true_airspeed calculations, so it's convenient to get it done now
    # For the purpose of Cocip <> CocipGrid model parity, we attach a
    # "head_tail_dt" variable. This variable is used the first time `advect`
    # is called. It makes a small but noticeable difference in model outputs.
    true_airspeed = vector["true_airspeed"]

    if not _is_segment_free_mode(vector):
        head_tail_dt_s = vector["segment_length"] / true_airspeed
        head_tail_dt = (head_tail_dt_s * 1_000_000_000.0).astype("timedelta64[ns]")
        vector["head_tail_dt"] = head_tail_dt

    params["bada_model"] = vector.attrs["bada_model"]

    # Update wingspan from BADA model, if its not already defined
    if params["wingspan"] is None:
        params["wingspan"] = vector.attrs["wingspan"]

    # PART 3: Emissions data
    factor = vector.get("nvpm_ei_n_enhancement_factor", params["nvpm_ei_n_enhancement_factor"])

    # Early exit
    if not params["process_emissions"]:
        logger.debug("Use default nvpm_ei_n value %s", params["default_nvpm_ei_n"])
        vector.setdefault("nvpm_ei_n", factor * np.full(vector.size, params["default_nvpm_ei_n"]))
        return

    # Extract engine UID attached in BADAGrid
    engine_uid = vector.attrs["engine_uid"]
    assert engine_uid is not None
    emissions_model = emissions.Emissions()

    # If engine UID not present in EDB, early exit
    if (edb_gaseous := emissions_model.edb_engine_gaseous.get(engine_uid)) is None:
        warnings.warn(
            f"Cannot find 'engine_uid' {engine_uid} in EDB. A constant nvpm_ei_n will be used."
        )
        vector.setdefault("nvpm_ei_n", factor * np.full(vector.size, params["default_nvpm_ei_n"]))
        return

    fuel_flow_per_engine = vector["fuel_flow"] / vector.attrs["n_engine"]
    air_temperature = vector["air_temperature"]
    thrust_setting = emissions.get_thrust_setting(
        edb_gaseous,
        fuel_flow_per_engine=fuel_flow_per_engine,
        air_pressure=vector.air_pressure,
        air_temperature=air_temperature,
        true_airspeed=true_airspeed,
    )

    # Only the gaseous engine is found in the EDB, early exit
    if (edb_nvpm := emissions_model.edb_engine_nvpm.get(engine_uid)) is None:
        nvpm_ei_m = emissions.nvpm_mass_emissions_index_sac(
            edb_gaseous,
            air_pressure=vector.air_pressure,
            true_airspeed=true_airspeed,
            air_temperature=air_temperature,
            thrust_setting=thrust_setting,
            fuel_flow_per_engine=fuel_flow_per_engine,
            hydrogen_content=fuel.hydrogen_content,
        )
        nvpm_gmd = emissions.nvpm_geometric_mean_diameter_sac(
            edb_gaseous,
            air_pressure=vector.air_pressure,
            true_airspeed=true_airspeed,
            air_temperature=air_temperature,
            thrust_setting=thrust_setting,
            q_fuel=fuel.q_fuel,
        )
        nvpm_ei_n = black_carbon.number_emissions_index_fractal_aggregates(nvpm_ei_m, nvpm_gmd)
        vector.setdefault("nvpm_ei_n", factor * nvpm_ei_n)
        return

    # Main branch when gaseous engine and nvPM engine data are both present
    _, nvpm_ei_n = emissions.get_nvpm_emissions_index_edb(
        edb_nvpm,
        true_airspeed=true_airspeed,
        air_temperature=air_temperature,
        air_pressure=vector.air_pressure,
        thrust_setting=thrust_setting,
        q_fuel=fuel.q_fuel,
    )
    vector.setdefault("nvpm_ei_n", factor * nvpm_ei_n)


def calc_wind_shear(
    contrail: GeoVectorDataset,
    dz_m: float,
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

    air_pressure_lower = thermo.p_dz(air_temperature, air_pressure, dz_m)
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
    dt: np.timedelta64,
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
    dt : np.timedelta64
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

    longitude_t2 = geo.advect_longitude(
        longitude=longitude, latitude=latitude, u_wind=u_wind, dt=dt
    )
    latitude_t2 = geo.advect_latitude(latitude=latitude, v_wind=v_wind, dt=dt)
    level_t2 = geo.advect_level(level, vertical_velocity, rho_air, terminal_fall_speed, dt)
    altitude_t2 = units.pl_to_m(level_t2)

    data = {
        "index": index_t2,
        "longitude": longitude_t2,
        "latitude": latitude_t2,
        "level": level_t2,
        "air_pressure": 100 * level_t2,
        "altitude": altitude_t2,
        "time": time_t2,
        "formation_time": formation_time,
        "age": age_t2,
        **_get_uncertainty_params(contrail),
    }

    if dt_tail is None or dt_head is None:
        assert _is_segment_free_mode(contrail)
        assert dt_tail is None and dt_head is None
        return GeoVectorDataset(data, attrs=contrail.attrs, copy=True)

    longitude_head = contrail["longitude_head"]
    latitude_head = contrail["latitude_head"]
    longitude_tail = contrail["longitude_tail"]
    latitude_tail = contrail["latitude_tail"]
    u_wind_head = contrail["eastward_wind_head"]
    v_wind_head = contrail["northward_wind_head"]
    u_wind_tail = contrail["eastward_wind_tail"]
    v_wind_tail = contrail["northward_wind_tail"]

    longitude_head_t2 = geo.advect_longitude(
        longitude=longitude_head, latitude=latitude_head, u_wind=u_wind_head, dt=dt_head
    )
    latitude_head_t2 = geo.advect_latitude(latitude=latitude_head, v_wind=v_wind_head, dt=dt_head)

    longitude_tail_t2 = geo.advect_longitude(
        longitude=longitude_tail, latitude=latitude_tail, u_wind=u_wind_tail, dt=dt_tail
    )
    latitude_tail_t2 = geo.advect_latitude(latitude=latitude_tail, v_wind=v_wind_tail, dt=dt_tail)

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

    return GeoVectorDataset(data, attrs=contrail.attrs, copy=True)


def calc_intermediate_results(vector_list: list[VectorDataset]) -> VectorDataset:
    """Aggregate results after cocip simulation.

    Results are summed over each vector in `vector_list`.

    Parameters
    ----------
    vector_list : list[VectorDataset]
        List of `VectorDataset` objects each containing keys "index" and "ef". List is expected
        to be nonempty.

    Returns
    -------
    VectorDataset
        Dataset with keys:
            - "index": Used to join to :attr:`CocipGrid.source`
            - "ef": Sum of ef values
            - "age": Contrail age associated to each index
    """

    def key_to_df(key: str) -> pd.DataFrame:
        """Get DataFrame of values over vector lifetime.

        Generally, we try to avoid ``pd.concat`` because of the memory explosion it
        often causes. This could be probably be reimplemented without ``pandas``
        if needed, though there doesn't seem to be an obvious numpy approach to take.
        We'd need to start by taking the union of v["index"] for v in vector_list.
        Alternatively, we could use the full source index to determine the global
        index of each vector.

        For example, something like this would work for summing ef and avoid some of the
        pd.concat overhead.

        ```
        index = source_index
        out = np.zeros(len(index))
        for v in vector_list:
            out[v["index"]] += v["ef"]
        nonzero = out != 0
        series = pd.Series(out[nonzero], index=index[nonzero])
        ```
        """
        # dtype = np.result_type(*[v[key].dtype for v in vector_list], np.float32)
        dfs = [pd.DataFrame(data=v[key], index=v["index"]) for v in vector_list]
        return pd.concat(dfs, axis=1)

    ef = key_to_df("ef").sum(axis=1)
    age = key_to_df("age").max(axis=1)

    data = {
        "index": ef.index.to_numpy(),
        "ef": ef.to_numpy(),
        "age": age.to_numpy(),
    }

    return VectorDataset(data, copy=False)


def result_to_metdataset(
    result: VectorDataset | None,
    verbose_dict: dict[str, npt.NDArray[np.float_]],
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
    verbose_dict : dict[str, npt.NDArray[np.float_]]:
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
    contrail_age = np.zeros(size, dtype=np.float32)
    ef_per_m = np.zeros(size, dtype=dtype)

    if result:
        contrail_idx = result["index"]
        # Step 1: Contrail age. Convert from timedelta to float
        contrail_age[contrail_idx] = result["age"] / np.timedelta64(1, "h")
        # Step 2: EF
        ef_per_m[contrail_idx] = result["ef"] / nominal_segment_length

    contrail_age = contrail_age.reshape(shape)
    ef_per_m = ef_per_m.reshape(shape)

    # Step 3: Dataset dims and attrs
    dims = tuple(source.coords)
    local_attrs = _contrail_grid_variable_attrs()

    # Step 4: Dataset core variables
    data_vars = {
        "contrail_age": (dims, contrail_age, local_attrs["contrail_age"]),
        "ef_per_m": (dims, ef_per_m, local_attrs["ef_per_m"]),
    }

    # Step 5: Dataset variables from verbose_dicts
    for k, v in verbose_dict.items():
        data_vars[k] = (dims, v.reshape(shape), local_attrs[k])

    # Bundle the package and return
    ds = xr.Dataset(data_vars=data_vars, coords=source.coords, attrs=attrs)
    return MetDataset(ds, copy=False)


def result_merge_source(
    result: VectorDataset | None,
    verbose_dict: dict[str, npt.NDArray[np.float_]],
    source: GeoVectorDataset,
    nominal_segment_length: float | npt.NDArray[np.float_],
    attrs: dict[str, str],
) -> GeoVectorDataset:
    """Merge ``results`` and ``verbose_dict`` onto ``source``."""

    # Initialize the main output arrays to all zeros
    dtype = result["age"].dtype if result else "timedelta64[ns]"
    contrail_age = np.full(source.size, 0, dtype=dtype)

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
) -> dict[str, npt.NDArray[np.float_]]:
    # Concatenate the values and return
    ret: dict[str, np.ndarray] = {}
    for key in verbose_outputs_formation:
        series_list = [v for d in verbose_dicts if (v := d.get(key)) is not None]
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
        "nvpm_ei_n": {
            "long_name": "Black carbon emissions index number",
            "units": "kg^{-1}",
        },
        "fuel_flow": {"long_name": "Jet engine fuel flow", "units": "kg / s"},
        "specific_humidity": {"long_name": "Specific humidity", "units": "kg / kg"},
        "rhi": {"long_name": "Relative humidity", "units": "dimensionless"},
        "iwc": {
            "long_name": "Ice water content after the wake vortex phase",
            "units": "kg_h2o / kg_air",
        },
    }


def _supported_verbose_outputs_formation() -> set[str]:
    """Get supported keys for verbose outputs.

    Uses output of :func:`_contrail_grid_variable_attrs` as a source of truth.
    """
    return set(_contrail_grid_variable_attrs()) - {"contrail_age", "ef_per_m"}


def _warn_not_wrap(met: MetDataset) -> None:
    """Warn user if parameter met should be wrapped.

    Parameters
    ----------
    met : MetDataset
        Met dataset
    """
    if not met.is_wrapped:
        lon = met.data["longitude"]
        if lon.min() == -180 and lon.max() == 179.75:
            warnings.warn(
                "The MetDataset `met` not been wrapped. The CocipGrid model may "
                "perform better if `met.wrap_longitude()` is called first."
            )


def _get_uncertainty_params(contrail: VectorDataset) -> dict[str, npt.NDArray[np.float_]]:
    """Return uncertainty parameters in `contrail.data`.

    For each of the keys:
        - "rhi_adj",
        - "rhi_boost_exponent",
        - "sedimentation_impact_factor",
        - "wind_shear_enhancement_exponent",
    this function checks if key is present in contrail. The data is then bundled and returned.

    Parameters
    ----------
    contrail : VectorDataset
        Data from which uncertainty parameters are extracted

    Returns
    -------
    dict[str, npt.NDArray[np.float_]]
        Dictionary of uncertainty parameters.
    """
    keys = [
        "rhi_adj",
        "rhi_boost_exponent",
        "sedimentation_impact_factor",
        "wind_shear_enhancement_exponent",
    ]
    return {key: contrail[key] for key in keys if key in contrail}


_T = TypeVar("_T", np.float_, np.datetime64)


def _check_overlap(
    met_array: npt.NDArray[_T], grid_array: npt.NDArray[_T], coord: str, name: str
) -> None:
    """Check if met data should be downselected.

    Warn if grid coordinate extends beyond met coordinate.

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
    if coord not in ["longitude", "latitude", "level", "time"]:
        raise ValueError(f"Unsupported coordinate: {coord}")

    if met_array.min() > grid_array.min() or met_array.max() < grid_array.max():
        warnings.warn(
            f"Met data '{name}' does not overlap the grid domain along the {coord} axis. "
            "This causes interpolated values to be nan, leading to meaningless results."
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

    # return if downselect_met is False
    if not params["downselect_met"]:
        logger.debug("Avoiding downselecting met because params['downselect_met'] is False")
        return met, rad

    logger.debug("Downselecting met domain to vector points")

    # check params
    longitude_buffer = params["met_longitude_buffer"]
    latitude_buffer = params["met_latitude_buffer"]
    level_buffer = params["met_level_buffer"]

    # Down select met relative to min / max integration timesteps, not Flight
    t0 = params["met_time_buffer"][0]
    t1 = params["met_time_buffer"][1] + params["max_age"] + params["dt_integration"]

    if isinstance(source, MetDataset):
        # MetDataset doesn't have a downselect_met method, so create a
        # GeoVectorDataset and downselect there
        # Just take extreme here for downselection
        # We may want to change min / max to nanmin / nanmax
        ds = source.data
        source = GeoVectorDataset(
            longitude=np.array([ds["longitude"].values.min(), ds["longitude"].values.max()]),
            latitude=np.array([ds["latitude"].values.min(), ds["latitude"].values.max()]),
            level=np.array([ds["level"].values.min(), ds["level"].values.max()]),
            time=np.array([ds["time"].values.min(), ds["time"].values.max()]),
        )

    # `downselect_met(met=...)` copies `met`, `rad`
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
