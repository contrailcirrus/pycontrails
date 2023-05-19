"""Utilities for :class:`Cocip` and :class:`CocipGrid`."""

from __future__ import annotations

import dataclasses
import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from pycontrails.core import coordinates
from pycontrails.core.met import MetDataset
from pycontrails.core.vector import GeoVectorDataset

if TYPE_CHECKING:
    import tqdm

logger = logging.getLogger(__name__)

# Crude constants used to estimate model runtime
# If we seek more accuracy, these should be seasonally adjusted
# The first constant is the proportion of points generating persistent contrails
# that survive to the end of the hour
# The second constants it the probability of a contrail surviving an hour of
# ongoing model evolution
HEURISTIC_INITIAL_SURVIVAL_RATE = 0.1
HEURISTIC_EVOLUTION_SURVIVAL_RATE = 0.8


class CocipTimeHandlingMixin:
    """Support :class:`Cocip` and :class:`CocipGrid` time handling."""

    params: dict[str, Any]
    source: GeoVectorDataset | MetDataset
    met: MetDataset
    rad: MetDataset

    #: Convenience container to hold time filters
    #: See :meth:`CocipGrid.eval` for usage.
    timedict: dict[np.datetime64, np.ndarray]

    def validate_time_params(self) -> None:
        """Raise a `ValueError` if `met_slice_dt` is not a multiple of `dt_integration`."""
        met_slice_dt: np.timedelta64 | None = self.params.get("met_slice_dt")

        if met_slice_dt is None:
            # TODO: Reset to something big enough to cover the entire met dataset
            raise NotImplementedError
        if met_slice_dt == np.timedelta64(0, "s"):
            raise ValueError("met_slice_dt must be a positive timedelta")

        dt_integration: np.timedelta64 = self.params["dt_integration"]
        ratio = met_slice_dt / dt_integration
        if not ratio.is_integer():
            raise ValueError(
                f"met_slice_dt ({met_slice_dt}) must be a multiple "
                f"of dt_integration ({dt_integration})"
            )

        # I'm not sure how necessary this is ...
        met_time_diff = np.diff(self.met.data["time"].values)
        ratios = met_slice_dt / met_time_diff
        if np.any(ratios != ratios.astype(int)):
            raise ValueError(
                f"met_slice_dt ({met_slice_dt}) must be a multiple "
                "of the time difference between met time steps "
                f"({met_time_diff})"
            )

        # Validate met_slice_dt
        met_time_diff = np.diff(self.met.data["time"].values).astype("timedelta64[h]")
        if np.unique(met_time_diff).size > 1:
            raise NotImplementedError("CocipGrid only supports met with constant time diff.")
        met_time_res = met_time_diff[0]

        n_hours = met_slice_dt / met_time_res
        if n_hours < 1 or not n_hours.is_integer():
            raise ValueError(
                f"Parameter `met_slice_dt` must be a positive multiple of {met_time_res}."
            )

    @property
    def source_time(self) -> npt.NDArray[np.datetime64]:
        """Return the time array of the :attr:`source` data."""
        if not hasattr(self, "source"):
            raise AttributeError("source not set")
        if isinstance(self.source, GeoVectorDataset):
            return self.source["time"]
        if isinstance(self.source, MetDataset):
            return self.source.variables["time"].values
        raise TypeError(f"Cannot calculate timesteps for {self.source}")

    def attach_timedict(self) -> None:
        """Attach or update :attr:`timedict`.

        This attribute is a dictionary of the form::

            {t: filt}

        where the key ``t`` is the time of the start of the met slice and the value
        ``filt`` is a :class:`numpy.ndarray` of the same shape as the
        the time source. Specifically, ``filt`` is a boolean array that can
        be used to filter the time source.

        Presently, keys are hard-coded with :attr:`dtype` ``datetime64[h]``.

        The keys of this dictionary can be used to slice the ``met`` dataset
        in the ``time`` dimension. However, this must be done with care in case
        the met dataset is not aligned with the time source.
        """
        met_slice_dt = self.params.get("met_slice_dt")
        if met_slice_dt is None:
            raise ValueError("met_slice_dt must be set")
        self.validate_time_params()

        # Cast to pandas to use ceil and floor methods below
        met_slice_dt = pd.to_timedelta(met_slice_dt)
        source_time = self.source_time
        tmin = pd.to_datetime(source_time.min())
        tmax = pd.to_datetime(source_time.max() + self.params["max_age"])
        self._check_met_rad_time(tmin, tmax)

        # Ideally we'd use the keys to index into the met dataset
        # with met.data.sel(time=slice(t1, t2)), but this is not even
        # possible the the rad data because of time shifting.
        # So, at the very least, we'll want to use np.searchsorted
        # to find the indices of the met dataset that correspond to
        # the time slice. See _load_met_slices
        t_start = tmin.floor(met_slice_dt)
        t_end = tmax.ceil(met_slice_dt)

        met_times = np.arange(t_start, t_end + met_slice_dt, met_slice_dt).astype("datetime64[h]")
        zipped = zip(met_times, met_times[1:])
        self.timedict = {t1: (source_time >= t1) & (source_time < t2) for t1, t2 in zipped}

    def _check_met_rad_time(self, tmin: pd.Timestamp, tmax: pd.Timestamp) -> None:
        if self.met.data["time"].min() > tmin or self.met.data["time"].max() < tmax:
            warnings.warn(
                "Parameter 'met' is too short in the time dimension. "
                "Include additional time in 'met' or reduce 'max_age' parameter. "
                f"Model start time: {tmin} Model end time: {tmax}"
            )
        if self.rad.data["time"].min() > tmin or self.rad.data["time"].max() < tmax:
            warnings.warn(
                "Parameter 'rad' is too short in the time dimension. "
                "Include additional time in 'rad' or reduce 'max_age' parameter."
                f"Model start time: {tmin} Model end time: {tmax}"
            )

    def init_pbar(self) -> "tqdm.tqdm" | None:
        """Initialize a progress bar for model evaluation."""

        if not self.params["show_progress"]:
            return None

        try:
            from tqdm.auto import tqdm
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Running model {type(self).__name__}  with parameter "
                "show_progress=True requires the 'tqdm' module, which can be "
                "installed with 'pip install tqdm'. "
                "Alternatively, set model parameter 'show_progress=False'."
            ) from e

        estimate = self._estimate_runtime()

        # We call update on contrail initialization and at each evolution step
        total = sum(e.n_steps_new_vectors + e.n_steps_old_vectors for e in estimate.values())
        # We also call update on each met loading step
        total += len(estimate)

        return tqdm(total=total, desc=f"{type(self).__name__} eval")

    def _estimate_runtime(self) -> dict[np.datetime64, CocipRuntimeStats]:
        """Calculate number of new meshes and predict number of persistent meshes by met slice.

        Returns
        -------
        dict[np.datetime64, CocipRuntimeStats]
            Estimate of the runtime for each met slice.
        """

        met_slice_dt = self.params["met_slice_dt"]
        if met_slice_dt is None:
            raise ValueError("met_slice_dt must be set")
        if not hasattr(self, "timedict"):
            raise AttributeError("First call attach_met_slice_timedict")

        dt_integration = self.params["dt_integration"]
        met_slice_dt = self.params["met_slice_dt"]
        if isinstance(self.source, MetDataset):
            n_splits = self._grid_spatial_n_splits()
        else:
            split_size = (
                self.params["target_split_size_pre_SAC_boost"] * self.params["target_split_size"]
            )

        # Adjust scaling factors when met_slice_dt is more than a single hour
        met_slice_scale = met_slice_dt / np.timedelta64(1, "h")

        # Mirror the structure of timedict to estimate the runtime
        estimate: dict[np.datetime64, CocipRuntimeStats] = {}
        for t_start, filt in self.timedict.items():
            t_next = t_start + met_slice_dt
            t_prev = t_start - met_slice_dt

            times_in_filt = self.source_time[filt]
            n_steps_by_time = np.ceil((t_next - times_in_filt) / dt_integration)

            # Tricky logic for different sources
            if isinstance(self.source, MetDataset):
                n_new_vectors = len(times_in_filt) * n_splits
                n_steps_new_vectors = int(n_steps_by_time.sum() + 1) * n_splits
            elif times_in_filt.size:
                n_new_vectors = max(int(times_in_filt.size / split_size), 1)
                n_steps_new_vectors = int(n_steps_by_time.max() + 1) * n_new_vectors
            else:
                n_new_vectors = 0
                n_steps_new_vectors = 0

            prev_stats = estimate.get(t_prev)
            if prev_stats is None:
                prev_stats = CocipRuntimeStats(t_prev, 0, 0, 0, 0)

            n_old_vectors_float = int(
                HEURISTIC_INITIAL_SURVIVAL_RATE * prev_stats.n_new_vectors
                + HEURISTIC_EVOLUTION_SURVIVAL_RATE**met_slice_scale * prev_stats.n_old_vectors
            )
            n_old_vectors = max(round(n_old_vectors_float), 1)
            n_steps_old_vectors = n_old_vectors * (met_slice_dt // dt_integration).item()
            estimate[t_start] = CocipRuntimeStats(
                t_start=t_start,
                n_new_vectors=n_new_vectors,
                n_steps_new_vectors=n_steps_new_vectors,
                n_old_vectors=n_old_vectors,
                n_steps_old_vectors=n_steps_old_vectors,
            )

        return estimate

    def _grid_spatial_n_splits(self) -> int:
        """Compute the number of vector "spatial" splits at a single time.

        Helper method used in :meth:`_estimate_runtime` and :meth:`_generate_new_grid_vectors`.

        This method assumes :attr:`source` is a :class:`MetDataset`.

        Returns
        -------
        int
            The number of spatial splits.
        """
        grid_size = (
            self.source.data["longitude"].size
            * self.source.data["latitude"].size
            * self.source.data["level"].size
        )
        split_size = int(
            self.params["target_split_size_pre_SAC_boost"] * self.params["target_split_size"]
        )
        return max(grid_size // split_size, 1)

    def _load_met_slices(
        self, start: np.datetime64, pbar: "tqdm.tqdm" | None = None
    ) -> tuple[MetDataset, MetDataset]:
        """Load met and rad slices for interpolation.

        :attr:`met` and :attr:`rad` are sliced by `slice(start, start + .met_slice_dt)`

        Parameters
        ----------
        start : np.datetime64
            Start of time domain of interest. Does not need to have hour resolution,
            but we will get runtime errors in other methods if not.
        pbar : tqdm.tqdm | None, optional
            Progress bar. The :meth:`pbar.update` method is called after both slices
            are loaded.

        Returns
        -------
        met : MetDataset
            Met data sliced to the time domain of interest.
        rad : MetDataset
            Rad data sliced to the time domain of interest.

        Raises
        ------
        NotImplementedError
            If :attr:`met` data has finer than hourly "time" resolution
        RuntimeError
            If ``met`` or ``rad`` time slices do not not contain at least two time values.
        """

        # Only support met with hourly timestamps
        time = self.met.variables["time"].values
        remainder = time - time.astype("datetime64[h]")
        if np.any(remainder.astype(float)):
            raise NotImplementedError("Only support met data with hourly time coordinates.")

        request = start, start + self.params["met_slice_dt"]
        buffer = np.timedelta64(0, "h"), np.timedelta64(0, "h")
        met_sl = coordinates.slice_domain(time, request, buffer)
        rad_sl = coordinates.slice_domain(self.rad.variables["time"].values, request, buffer)

        logger.debug("Update met slices. Start: %s, Stop: %s", met_sl.start, met_sl.stop)
        logger.debug("Update rad slices. Start: %s, Stop: %s", rad_sl.start, rad_sl.stop)
        xr_met_slice = self.met.data.isel(time=met_sl)
        xr_rad_slice = self.rad.data.isel(time=rad_sl)

        # We no longer require time two slices for linear interpolation, but for the
        # sake of contrail evolution, we expect to always use two. See method
        # _evolve_vector, where we explicitly unpack the time domain into two variables.
        if len(xr_met_slice["time"]) < 2:
            raise RuntimeError("Malformed met slice.")
        if len(xr_rad_slice["time"]) < 2:
            raise RuntimeError("Malformed rad slice.")

        # If data is already loaded into memory, calling load will not waste memory
        met_slice = MetDataset(xr_met_slice, copy=False)
        rad_slice = MetDataset(xr_rad_slice, copy=False)

        if pbar is not None:
            pbar.update()
        return met_slice, rad_slice


@dataclasses.dataclass
class CocipRuntimeStats:
    """Support for estimating runtime and progress bar."""

    t_start: np.datetime64
    n_new_vectors: int
    n_old_vectors: int
    n_steps_new_vectors: int
    n_steps_old_vectors: int
