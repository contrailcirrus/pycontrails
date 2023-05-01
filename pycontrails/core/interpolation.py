"""Interpolation utilities."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import scipy.interpolate
import xarray as xr

from pycontrails.core import rgi_cython  # type: ignore[attr-defined]

# ------------------------------------------------------------------------------
# Multidimensional interpolation
# ------------------------------------------------------------------------------


logger = logging.getLogger(__name__)


class PycontrailsRegularGridInterpolator(scipy.interpolate.RegularGridInterpolator):
    """Support for performant interpolation over a regular grid.

    This class is a thin wrapper around the
    :class:`scipy.interpolate.RegularGridInterpolator` in order to make typical
    ``pycontrails`` use-cases more efficient.

    #. Avoid ``RegularGridInterpolator`` constructor validation. In :func:`interp`,
    parameters are carefully crafted to fit into the intended form, thereby making
    validation unnecessary.
    #. Override the :meth:`_evaluate_linear` method with a faster implementation. See
    the :meth:`_evaluate_linear` docstring for more information.

    This class should not be used directly. Instead, use the :func:`interp` function.

    .. versionchanged:: 0.40.0

        The :meth:`_evaluate_linear` method now uses a Cython implementation. The dtype
        of the output is now consistent with the dtype of the underlying :attr:`values`

    Parameters
    ----------
    points : tuple[npt.NDArray[np.float64], ...]
        Coordinates of the grid points.
    values : npt.NDArray[np.float_]
        Grid values. The shape of this array must be compatible with the
        coordinates. An error is raised if the dtype is not ``np.float32``
        or ``np.float64``.
    method : str
        Passed into :class:`scipy.interpolate.RegularGridInterpolator`
    bounds_error : bool
        Passed into :class:`scipy.interpolate.RegularGridInterpolator`
    fill_value : float | np.float64 | None
        Passed into :class:`scipy.interpolate.RegularGridInterpolator`
    """

    def __init__(
        self,
        points: tuple[npt.NDArray[np.float64], ...],
        values: npt.NDArray[np.float_],
        method: str,
        bounds_error: bool,
        fill_value: float | np.float64 | None,
    ):
        if values.dtype not in (np.float32, np.float64):
            raise ValueError("values must be a float array")

        self.grid = points
        self.values = values
        self.method = method
        self.bounds_error = bounds_error
        self.fill_value = fill_value

    def _prepare_xi_simple(self, xi: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Run looser version of :meth:`_prepare_xi`.

        Parameters
        ----------
        xi : npt.NDArray[np.float64]
            Points at which to interpolate.

        Returns
        -------
        npt.NDArray[np.bool_]
            A 1-dimensional Boolean array indicating which points are out of bounds.
            If ``bounds_error`` is ``True``, this will be all ``False``.
        """

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                g0 = self.grid[i][0]
                g1 = self.grid[i][-1]
                if not (np.all(p >= g0) and np.all(p <= g1)):
                    raise ValueError(f"One of the requested xi is out of bounds in dimension {i}")

            return np.zeros(xi.shape[0], dtype=bool)

        return self._find_out_of_bounds(xi.T)

    def __call__(
        self, xi: npt.NDArray[np.float64], method: str | None = None
    ) -> npt.NDArray[np.float_]:
        """Evaluate the interpolator at the given points.

        Parameters
        ----------
        xi : npt.NDArray[np.float64]
            Points at which to interpolate. Must have shape ``(n, ndim)``, where
            ``ndim`` is the number of dimensions of the interpolator.
        method : str | None
            Override the :attr:`method` to keep parity with the base class.

        Returns
        -------
        npt.NDArray[np.float_]
            Interpolated values. Has shape ``(n,)``. When computing linear interpolation,
            the dtype is the same as the :attr:`values` array.
        """

        method = method or self.method
        if method != "linear":
            return super().__call__(xi, method)

        out_of_bounds = self._prepare_xi_simple(xi)
        xi_indices, norm_distances = self._find_indices(xi.T)

        out = self._evaluate_linear(xi_indices, norm_distances)
        return self._set_out_of_bounds(out, out_of_bounds)

    def _set_out_of_bounds(
        self,
        out: npt.NDArray[np.float_],
        out_of_bounds: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float_]:
        """Set out-of-bounds values to the fill value.

        Parameters
        ----------
        out : npt.NDArray[np.float_]
            Values from interpolation. This is modified in-place.
        out_of_bounds : npt.NDArray[np.bool_]
            A 1-dimensional Boolean array indicating which points are out of bounds.

        Returns
        -------
        out : npt.NDArray[np.float_]
            A reference to the ``out`` array.
        """
        if self.fill_value is not None and np.any(out_of_bounds):
            out[out_of_bounds] = self.fill_value

        return out

    def _evaluate_linear(
        self,
        indices: npt.NDArray[np.int64],
        norm_distances: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float_]:
        """Evaluate the interpolator using linear interpolation.

        This is a faster alternative to
        :meth:`scipy.interpolate.RegularGridInterpolator._evaluate_linear`.

        .. versionadded:: 0.24

        .. versionchanged:: 0.40.0

            Use Cython routines for evaluating the interpolation when the
            dimension is 1, 2, 3, or 4.

        Parameters
        ----------
        indices : npt.NDArray[np.int64]
            Indices of the grid points to the left of the interpolation points.
            Has shape ``(ndim, n_points)``.
        norm_distances : npt.NDArray[np.float64]
            Normalized distances between the interpolation points and the grid
            points to the left. Has shape ``(ndim, n_points)``.

        Returns
        -------
        npt.NDArray[np.float_]
            Interpolated values with shape ``(n_points,)`` and the same dtype as
            the :attr:`values` array.
        """
        # Let scipy "slow" implementation deal with high-dimensional grids
        if indices.shape[0] > 4:
            return super()._evaluate_linear(indices, norm_distances)

        # Squeeze as much as possible
        # Our cython implementation requires non-degenerate arrays
        non_degen = tuple(s > 1 for s in self.values.shape)
        values = self.values.squeeze()
        indices = indices[non_degen, :]
        norm_distances = norm_distances[non_degen, :]

        ndim, n_points = indices.shape
        out = np.empty(n_points, dtype=self.values.dtype)

        if ndim == 4:
            return rgi_cython.evaluate_linear_4d(values, indices, norm_distances, out)

        if ndim == 3:
            return rgi_cython.evaluate_linear_3d(values, indices, norm_distances, out)

        if ndim == 2:
            return rgi_cython.evaluate_linear_2d(values, indices, norm_distances, out)

        if ndim == 1:
            # np.interp could be better ... although that may also promote the dtype
            return rgi_cython.evaluate_linear_1d(values, indices, norm_distances, out)

        raise ValueError(f"Invalid number of dimensions: {ndim}")


def _floatize_time(
    time: npt.NDArray[np.datetime64], offset: np.datetime64
) -> npt.NDArray[np.float64]:
    """Convert an array of ``np.datetime64`` to an array of ``np.float64``.

    In calls to :class:`scipy.interpolate.RegularGridInterpolator`, it's critical
    that every coordinate be of same type. This creates complications: spatial
    coordinates are float-like, whereas time coordinates are datetime-like. In
    particular, it is not possible to cast an ``np.datetime64`` to a float
    without losing information. In practice, this is not problematic because
    ``np.float64`` has plenty of precision. Previously, this was more of an issue
    because we used ``np.float32``.

    This function uses a fixed time resolution (1 millisecond) to convert the time
    coordinate to a float-like coordinate. The time resolution is taken to avoid
    losing too much information for the time scales we encounter.

    Care is taken to ensure "nat" values are converted to "nan".

    Note that ``xarray`` also must confront this issue. They take a similar approach
    in :func:`xarray.core.missing._floatize_x`. See
    https://github.com/pydata/xarray/blob/d4db16699f30ad1dc3e6861601247abf4ac96567/xarray/core/missing.py#L572

    .. versionchanged:: 0.40.0

        No longer allow the option of converting to ``np.float32``. No longer
        floor the time values to the preceding millisecond.

    Parameters
    ----------
    time : npt.NDArray[np.datetime64]
        Array of ``np.datetime64`` values.
    offset : np.datetime64
        The offset to subtract from ``time``.

    Returns
    -------
    npt.NDArray[np.float64]
        The number of milliseconds since ``offset``.
    """
    delta = time - offset
    resolution = np.timedelta64(1, "ms")
    return delta / resolution


def _localize(da: xr.DataArray, coords: dict[str, np.ndarray]) -> xr.DataArray:
    """Clip ``da`` to the smallest bounding box that contains all of ``coords``.

    Roughly follows approach taken by :func:`xarray.core.missing._localize`. See
    https://github.com/pydata/xarray/blob/56f05c37924071eb4712479d47432aafd4dce38b/xarray/core/missing.py#L557

    Parameters
    ----------
    da : xr.DataArray
        DataArray to clip.
    coords : dict[str, np.ndarray]
        Coordinates to clip to.

    Returns
    -------
    xr.DataArray
        Clipped :class:`xarray.DataArray`. Has the same dimensions as the input ``da``.
        In particular, each dimension of the returned DataArray is a slice of the
        corresponding dimension of the input ``da``.
    """
    indexes: dict[str, Any] = {}
    for dim, arr in coords.items():
        dim_vals = da[dim].values

        # Skip single level
        if dim == "level" and dim_vals.size == 1 and dim_vals.item() == -1:
            continue

        # Create slice
        minval = np.nanmin(arr)
        maxval = np.nanmax(arr)
        imin = np.searchsorted(dim_vals, minval, side="right") - 1
        imin = max(0, imin)
        imax = np.searchsorted(dim_vals, maxval, side="left") + 1
        indexes[dim] = slice(imin, imax)

        # Logging
        n_in_bounds = np.sum((arr >= minval) & (arr <= maxval))
        logger.debug(
            "Interpolation in bounds along dimension %s: %d/%d",
            dim,
            n_in_bounds,
            arr.size,
        )

    return da.isel(**indexes)


@overload
def interp(
    longitude: npt.NDArray[np.float_],
    latitude: npt.NDArray[np.float_],
    level: npt.NDArray[np.float_],
    time: npt.NDArray[np.datetime64],
    da: xr.DataArray,
    method: str,
    bounds_error: bool,
    fill_value: float | np.float64 | None,
    localize: bool,
    *,
    indices: RGIArtifacts | None = ...,
    return_indices: Literal[False] = ...,
) -> npt.NDArray[np.float_]:
    ...


@overload
def interp(
    longitude: npt.NDArray[np.float_],
    latitude: npt.NDArray[np.float_],
    level: npt.NDArray[np.float_],
    time: npt.NDArray[np.datetime64],
    da: xr.DataArray,
    method: str,
    bounds_error: bool,
    fill_value: float | np.float64 | None,
    localize: bool,
    *,
    indices: RGIArtifacts | None = ...,
    return_indices: Literal[True],
) -> tuple[npt.NDArray[np.float_], RGIArtifacts]:
    ...


@overload
def interp(
    longitude: npt.NDArray[np.float_],
    latitude: npt.NDArray[np.float_],
    level: npt.NDArray[np.float_],
    time: npt.NDArray[np.datetime64],
    da: xr.DataArray,
    method: str,
    bounds_error: bool,
    fill_value: float | np.float64 | None,
    localize: bool,
    *,
    indices: RGIArtifacts | None = ...,
    return_indices: bool = ...,
) -> npt.NDArray[np.float_] | tuple[npt.NDArray[np.float_], RGIArtifacts]:
    ...


def interp(
    longitude: npt.NDArray[np.float_],
    latitude: npt.NDArray[np.float_],
    level: npt.NDArray[np.float_],
    time: npt.NDArray[np.datetime64],
    da: xr.DataArray,
    method: str,
    bounds_error: bool,
    fill_value: float | np.float64 | None,
    localize: bool,
    *,
    indices: RGIArtifacts | None = None,
    return_indices: bool = False,
) -> npt.NDArray[np.float_] | tuple[npt.NDArray[np.float_], RGIArtifacts]:
    """Interpolate over a grid with ``localize`` option.

    .. versionchanged:: 0.25.6

        Utilize scipy 1.9 upgrades to remove singleton dimensions.

    .. versionchanged:: 0.26.0

        Include ``indices`` and ``return_indices`` experimental parameters.
        Currently, nan values in ``longitude``, ``latitude``, ``level``, or ``time``
        are always propagated through to the output, regardless of ``bounds_error``.
        In other words, a ValueError for an out of bounds coordinate is only raised
        if a non-nan value is out of bounds.

    .. versionchanged:: 0.40.0

        When ``return_indices`` is True, an instance of :class:`RGIArtifacts`
        is used to store the indices artifacts.

    Parameters
    ----------
    longitude, latitude, level, time : np.ndarray
        Coordinates of points to be interpolated. These parameters have the same
        meaning as ``x`` in analogy with :func:`numpy.interp`. All four of these
        arrays must be 1 dimensional of the same size.
    da : xr.DataArray
        Gridded data interpolated over. Must adhere to ``MetDataArray`` conventions.
        In particular, the dimensions of ``da`` must be ``longitude``, ``latitude``,
        ``level``, and ``time``. The three spatial dimensions must be monotonically
        increasing with ``float64`` dtype. The ``time`` dimension must be
        monotonically increasing with ``datetime64`` dtype.
        Assumed to be cheap to load into memory (:attr:`xr.DataArray.values` is
        used without hesitation).
    method : str
        Passed into :class:`scipy.interpolate.RegularGridInterpolator`.
    bounds_error : bool
        Passed into :class:`scipy.interpolate.RegularGridInterpolator`.
    fill_value : float | np.float64 | None
        Passed into :class:`scipy.interpolate.RegularGridInterpolator`.
    localize : bool
        If True, clip ``da`` to the smallest bounding box that contains all of
        ``coords``.
    indices : tuple | None, optional
        Experimental. Provide intermediate artifacts computed by
        :meth:``scipy.interpolate.RegularGridInterpolator._find_indices`
        to avoid redundant computation. If known and provided, this can speed
        up interpolation by avoiding an unnecessary call to ``_find_indices``.
        By default, None. Must be used precisely.
    return_indices : bool, optional
        If True, return output of :meth:`scipy.interpolate.RegularGridInterpolator._find_indices`
        in addition to interpolated values.

    Returns
    -------
    npt.NDArray[np.float_] | tuple[npt.NDArray[np.float_], RGIArtifacts]
        Interpolated values with same size as ``longitude``. If ``return_indices``
        is True, return intermediate indices artifacts as well.

    See Also
    --------
    - :meth:`MetDataArray.interpolate`
    - :func:`scipy.interpolate.interpn`
    - :class:`scipy.interpolate.RegularGridInterpolator`
    """
    if localize:
        coords = {"longitude": longitude, "latitude": latitude, "level": level, "time": time}
        da = _localize(da, coords)

    # Using da.coords.variables is slightly more performant than da["longitude"].values
    x = da.coords.variables["longitude"].values
    y = da.coords.variables["latitude"].values
    z = da.coords.variables["level"].values
    if any(v.dtype != np.float64 for v in (x, y, z)):
        raise ValueError(
            "da must have float64 dtype for longitude, latitude, and level coordinates"
        )

    # Convert t and time to float64
    t = da.coords.variables["time"].values
    offset = t[0]
    t = _floatize_time(t, offset)

    single_level = z.size == 1 and z.item() == -1
    points: tuple[npt.NDArray[np.float_], ...]
    if single_level:
        values = da.values.squeeze(axis=2)
        points = x, y, t
    else:
        values = da.values
        points = x, y, z, t

    interp_ = PycontrailsRegularGridInterpolator(
        points=points,
        values=values,
        method=method,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )

    if indices is None:
        xi = _buildxi(longitude, latitude, level, time, offset, single_level)
        if return_indices:
            out, indices = _linear_interp_with_indices(interp_, xi, localize, None)
            return out, indices
        return interp_(xi)

    out, indices = _linear_interp_with_indices(interp_, None, localize, indices)
    if return_indices:
        return out, indices
    return out


def _buildxi(
    longitude: npt.NDArray[np.float_],
    latitude: npt.NDArray[np.float_],
    level: npt.NDArray[np.float_],
    time: npt.NDArray[np.datetime64],
    offset: np.datetime64,
    single_level: bool,
) -> npt.NDArray[np.float64]:
    """Build the input array for interpolation.

    The implementation below achieves the same result as the following::

        np.stack([longitude, latitude, level, time_float], axis=1])

    This implementation is slightly faster than the above.
    """

    time_float = _floatize_time(time, offset)

    if single_level:
        ndim = 3
    else:
        ndim = 4

    shape = longitude.size, ndim
    xi = np.empty(shape, dtype=np.float64)

    xi[:, 0] = longitude
    xi[:, 1] = latitude
    if not single_level:
        xi[:, 2] = level
    xi[:, -1] = time_float

    return xi


def _linear_interp_with_indices(
    interp: PycontrailsRegularGridInterpolator,
    xi: npt.NDArray[np.float64] | None,
    localize: bool,
    indices: RGIArtifacts | None,
) -> tuple[npt.NDArray[np.float_], RGIArtifacts]:
    if interp.method != "linear":
        raise ValueError("Parameter 'indices' is only supported for 'method=linear'")
    if localize:
        raise ValueError("Parameter 'indices' is only supported for 'localize=False'")

    if indices is None:
        assert xi is not None, "xi must be provided if indices is None"
        out_of_bounds = interp._prepare_xi_simple(xi)
        xi_indices, norm_distances = interp._find_indices(xi.T)
        indices = RGIArtifacts(xi_indices, norm_distances, out_of_bounds)

    out = interp._evaluate_linear(indices.xi_indices, indices.norm_distances)
    out = interp._set_out_of_bounds(out, indices.out_of_bounds)
    return out, indices


@dataclasses.dataclass
class RGIArtifacts:
    """An interface to intermediate RGI interpolation artifacts."""

    xi_indices: npt.NDArray[np.int64]
    norm_distances: npt.NDArray[np.float64]
    out_of_bounds: npt.NDArray[np.bool_]


# ------------------------------------------------------------------------------
# 1 dimensional interpolation
# ------------------------------------------------------------------------------


class EmissionsProfileInterpolator:
    """Support for interpolating a profile on a linear or logarithmic scale.

    This class simply wraps :func:`numpy.interp` with fixed values for the
    ``xp`` and ``fp`` arguments. Unlike :class:`xarray.DataArray` interpolation,
    the `numpy.interp` automatically clips values outside the range of the
    ``xp`` array.

    Parameters
    ----------
    xp : npt.NDarray[np.float_]
        Array of x-values. These must be strictly increasing and free from
        any nan values. Passed to :func:`numpy.interp`.
    fp : npt.NDarray[np.float_]
        Array of y-values. Passed to :func:`numpy.interp`.
    drop_duplicates : bool, optional
        Whether to drop duplicate values in ``xp``. Defaults to ``True``.

    Examples
    --------
    >>> xp = np.array([3, 7, 10, 30], dtype=float)
    >>> fp = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    >>> epi = EmissionsProfileInterpolator(xp, fp)
    >>> # Interpolate a single value
    >>> epi.interp(5)
    0.150000...

    >>> # Interpolate a single value on a logarithmic scale
    >>> epi.log_interp(5)
    1.105171...

    >>> # Demonstrate speed up compared with xarray.DataArray interpolation
    >>> import time, xarray as xr
    >>> da = xr.DataArray(fp, dims=["x"], coords={"x": xp})

    >>> inputs = [np.random.uniform(0, 31, size=200) for _ in range(1000)]
    >>> t0 = time.perf_counter()
    >>> xr_out = [da.interp(x=x.clip(3, 30)).values for x in inputs]
    >>> t1 = time.perf_counter()
    >>> np_out = [epi.interp(x) for x in inputs]
    >>> t2 = time.perf_counter()
    >>> assert np.allclose(xr_out, np_out)

    >>> # We see a 100 fold speed up (more like 500x faster, but we don't
    >>> # want the test to fail!)
    >>> assert t2 - t1 < (t1 - t0) / 100
    """

    def __init__(
        self, xp: npt.NDArray[np.float_], fp: npt.NDArray[np.float_], drop_duplicates: bool = True
    ) -> None:
        if drop_duplicates:
            # Using np.diff to detect duplicates ... this assumes xp is sorted.
            # If xp is not sorted, an ValueError will be raised in _validate
            mask = np.abs(np.diff(xp, prepend=np.inf)) < 1e-15  # small tolerance
            xp = xp[~mask]
            fp = fp[~mask]

        self.xp = np.asarray(xp)
        self.fp = np.asarray(fp)
        self._validate()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(xp={self.xp}, fp={self.fp})"

    def _validate(self) -> None:
        if not len(self.xp):
            raise ValueError("xp must not be empty")
        if len(self.xp) != len(self.fp):
            raise ValueError("xp and fp must have the same length")
        if not np.all(np.diff(self.xp) > 0.0):
            raise ValueError("xp must be strictly increasing")
        if np.any(np.isnan(self.xp)):
            raise ValueError("xp must not contain nan values")

    def interp(self, x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Interpolate x against xp and fp.

        Parameters
        ----------
        x : npt.NDArray[np.float_]
            Array of x-values to interpolate.

        Returns
        -------
        npt.NDArray[np.float_]
            Array of interpolated y-values arising from the x-values. The ``dtype`` of
            the output array is the same as the ``dtype`` of ``x``.
        """
        # Need to explicitly cast back to x.dtype
        # https://github.com/numpy/numpy/issues/11214
        x = np.asarray(x)
        dtype = np.result_type(x, np.float32)
        return np.interp(x, self.xp, self.fp).astype(dtype)

    def log_interp(self, x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Interpolate x against xp and fp on a logarithmic scale.

        This method composes the following three functions.
            1. :func:`numpy.log`
            2. :meth:`interp`
            3. :func:`numpy.exp`

        Parameters
        ----------
        x : npt.NDArray[np.float_]
            Array of x-values to interpolate.

        Returns
        -------
        npt.NDArray[np.float_]
            Array of interpolated y-values arising from the x-values.
        """
        return np.exp(self.interp(np.log(x)))
