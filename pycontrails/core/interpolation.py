"""Interpolation utilities."""

from __future__ import annotations

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


def _is_const_diff(a: npt.NDArray[np.float_], rtol: float = 1e-4) -> bool:
    """Determine if an array of coordinates has a constant difference.

    Parameters
    ----------
    a : npt.NDArray[np.float_]
        Array of floating point values.
    rtol : float
        Relative tolerance for agreement. Passed into :func:`np.allclose`.

    Returns
    -------
    bool
        ``True`` if the array has a constant difference, ``False`` otherwise.

    Raises
    ------
    ValueError
        If ``a`` has fewer than two elements.
    """
    try:
        return np.allclose(np.diff(a), a[1] - a[0], rtol=rtol)
    except IndexError as e:
        raise ValueError("The array must have at least two elements") from e


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

    .. versionchanged:: XXX

        The :meth:`_evaluate_linear` method now uses a Cython implementation. The dtype
        of the output is now consistent with the dtype of the underlying :attr:`values`

    Parameters
    ----------
    points : tuple[npt.NDArray[np.float_], ...]
        Coordinates of the grid points.
    values : npt.NDArray[np.float_]
        Grid values. The shape of this array must be compatible with the
        coordinates.
    method : str
        Passed into :class:`scipy.interpolate.RegularGridInterpolator`
    bounds_error : bool
        Passed into :class:`scipy.interpolate.RegularGridInterpolator`
    fill_value : float | np.float64 | None
        Passed into :class:`scipy.interpolate.RegularGridInterpolator`
    """

    def __init__(
        self,
        points: tuple[npt.NDArray[np.float_], ...],
        values: npt.NDArray[np.float_],
        method: str,
        bounds_error: bool,
        fill_value: float | np.float64 | None,
    ):
        self.grid = points
        self.values = values
        self.method = method
        self.bounds_error = bounds_error
        self.fill_value = fill_value

    def _prepare_xi_simple(
        self, xi: npt.NDArray[np.float_]
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_] | None]:
        """Run looser more efficient version of :meth:`_prepare_xi`.

        In particular, this method does not automatically promote ``xi`` to
        ``np.float64``.
        """
        if xi.ndim != 2:
            raise ValueError("xi must be 2D")

        if xi.dtype not in (np.float32, np.float64):
            raise ValueError("xi must be float32 or float64")

        if xi.shape[1] != len(self.grid):
            raise ValueError(
                f"The requested sample points xi have dimension {xi.shape[1]} but this "
                f"PycontrailsRegularGridInterpolator has dimension {len(self.grid)}"
            )

        nans = np.any(np.isnan(xi), axis=1)

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                g0 = self.grid[i][0]
                g1 = self.grid[i][-1]
                if np.any(p < g0) or np.any(p > g1):
                    raise ValueError(f"One of the requested xi is out of bounds in dimension {i}")
            out_of_bounds = None
        else:
            out_of_bounds = self._find_out_of_bounds(xi.T)

        return nans, out_of_bounds

    def __call__(
        self, xi: npt.NDArray[np.float_], method: str | None = None
    ) -> npt.NDArray[np.float_]:
        """Evaluate the interpolator at the given points."""

        method = self.method if method is None else method
        if method == "linear":
            nans, out_of_bounds = self._prepare_xi_simple(xi)
            indices, norm_distances = self._find_indices(xi.T)
            out = self._evaluate_linear(indices, norm_distances)

            if out_of_bounds is not None and self.fill_value is not None:
                out[out_of_bounds] = self.fill_value

            # f(nan) = nan, if any
            if np.any(nans):
                out[nans] = np.nan

            return out

        return super().__call__(xi, method)

    def _find_indices(
        self, xi: npt.NDArray[np.float_]
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]]:
        indices = np.empty_like(xi, dtype=np.int_, order="C")
        norm_distances = np.empty_like(xi, order="C")

        for i, (x, grid) in enumerate(zip(xi, self.grid, strict=True)):
            # The grid is degenerate
            if grid.size == 1:
                indices[i] = 0
                norm_distances[i] = 0.0
                continue

            # Here the grid is monotonically increasing with a constant step
            if _is_const_diff(grid):
                g0 = grid[0]
                g1 = grid[1]

                # One implementation -- this one is a hair slower
                np.floor_divide(x - g0, g1 - g0, out=indices[i], casting="unsafe")
                np.clip(indices[i], 0, len(grid) - 2, out=indices[i])
                norm_distances[i] = (x - g0) / (g1 - g0) - indices[i]

                # Another implementation -- this one is a hair faster, but suffers
                # from some numerical instability with roundoff error
                # out1 = indices[i]
                # out2 = norm_distances[i]
                # np.divmod(x - g0, g1 - g0, out=(out1, out2), casting="unsafe")
                # norm_distances[i] /= g1 - g0  # normalize remainder to [0, 1)

                continue

            # Otherwise, use the slower searchsorted method
            indice = np.searchsorted(grid, x) - 1
            np.clip(indice, 0, len(grid) - 2, out=indice)
            indices[i] = indice

            grid_indice = grid[indice]
            denom = grid[indice + 1] - grid_indice
            out = norm_distances[i]
            where = denom != 0
            np.divide(x - grid_indice, denom, out=out, where=where)
            norm_distances[i][~where] = 0.0

        return indices, norm_distances

    def _evaluate_linear(
        self,
        indices: npt.NDArray[np.int_],
        norm_distances: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        """Evaluate the interpolator using linear interpolation.

        This is a faster alternative to
        :meth:`scipy.interpolate.RegularGridInterpolator._evaluate_linear`.

        .. versionadded:: 0.24

        FIXME: Update this docstring

        Notes
        -----
        For an underlying 3-dimensional grid, this method computes the following
        expression ``out``::

            i0, i1, i2 = indices
            nd0, nd1, nd2 = norm_distances
            out = self.values[(i0, i1, i2)] * (1 - nd0) * (1 - nd1) * (1 - nd2) + \
                self.values[(i0, i1, i2 + 1)] * (1 - nd0) * (1 - nd1) * nd2 + \
                self.values[(i0, i1 + 1, i2)] * (1 - nd0) * nd1 * (1 - nd2) + \
                self.values[(i0, i1 + 1, i2 + 1)] * (1 - nd0) * nd1 * nd2 + \
                self.values[(i0 + 1, i1, i2)] * nd0 * (1 - nd1) * (1 - nd2) + \
                self.values[(i0 + 1, i1, i2 + 1)] * nd0 * (1 - nd1) * nd2 + \
                self.values[(i0 + 1, i1 + 1, i2)] * nd0 * nd1 * (1 - nd2) + \
                self.values[(i0 + 1, i1 + 1, i2 + 1)] * nd0 * nd1 * nd2

        The scipy implementation is somewhat slower than this method for two reasons:
            - The shifted term ``1 - norm_distances`` is computed multiple times.
            - There is an unnecessary comparison ``np.where(ei == i, 1 - yi, yi)``
            which can be inferred from the indexing itself.

        The implementation here aims to overcome these two issues. In general, for large
        vectorized input, this method is 20 - 40% faster than the scipy implementation.

        This method is almost always the bottleneck in running a large CoCiP simulation.

        This implementation will be included in the scipy 1.10 release (summer 2023?).
        Once pycontrails is updated to use scipy 1.10, this method can be removed.
        """
        # Let scipy implementation deal with high-dimensional grids
        if indices.shape[0] > 4:
            return super()._evaluate_linear(indices, norm_distances)

        # Squeeze as much as possible
        # The cython implementation requires non-degenerate arrays
        non_degen = tuple(s > 1 for s in self.values.shape)
        values = self.values.squeeze()
        indices = indices[non_degen, :]
        norm_distances = norm_distances[non_degen, :]

        ndim, n_points = indices.shape
        out = np.empty(n_points, dtype=self.values.dtype)
        # FIXME! Typeguard dtype for fused type specialization

        if ndim == 4:
            return rgi_cython.evaluate_linear_4d(
                values,
                indices,
                norm_distances,
                out,
            )

        if ndim == 3:
            return rgi_cython.evaluate_linear_3d(
                values,
                indices,
                norm_distances,
                out,
            )

        if ndim == 2:
            return rgi_cython.evaluate_linear_2d(
                values,
                indices,
                norm_distances,
                out,
            )

        if ndim == 1:
            # np.interp could be better ... although that may also promote the dtype
            return rgi_cython.evaluate_linear_1d(
                values,
                indices,
                norm_distances,
                out,
            )

        raise ValueError(f"Invalid number of dimensions: {ndim}")


def _floatize_time(
    time: npt.NDArray[np.datetime64], offset: np.datetime64, dtype: np.dtype
) -> npt.NDArray[np.float_]:
    """Convert an array of ``np.datetime64`` to an array of ``float``.

    In calls to :class:`scipy.interpolate.RegularGridInterpolator`, it's critical
    that every coordinate be of same type. This creates complications: spatial
    coordinates are float-like, whereas time coordinates are datetime-like. In
    particular, it is not possible to cast an ``np.datetime64`` to a ``np.float64``
    without losing information.

    Note that ``xarray`` also must confront this issue. They take a similar approach
    in :func:`xarray.core.missing._floatize_x`. See
    https://github.com/pydata/xarray/blob/56f05c37924071eb4712479d47432aafd4dce38b/xarray/core/missing.py#L573

    The approach taken here is somewhat more intricate than the ``xarray`` approach. We
    allow conversion to either ``np.float32`` or ``np.float64`` and so we need to
    anticipate even more precision loss. This is accomplished by:

    #. Convert from ``np.datetime64`` to ``np.timedelta64`` by subtracting
       the ``offset``. This mimics the ``xarray`` methodology. No information is lost
       in this step.
    #. Flooring all datetime values to the nearest millisecond. Information is lost here.
       The ensuing ``dtype`` is ``np.int64``.
    #. Convert to the target ``dtype``. For typical time scales we encounter, this step
       is lossless. For example, with ``dtype="float32"``, the millisecond resolution
       is preserved for time scales up to decades in size.

    Care is taken to ensure "nat" values are converted to "nan".

    Parameters
    ----------
    time : npt.NDArray[np.datetime64]
        Array of ``np.datetime64`` values.
    offset : np.datetime64
        The offset to subtract from ``time``.
    dtype : type | str
        The target ``dtype``. One of ``np.float32`` or ``np.float64``.

    Returns
    -------
    npt.NDArray[np.float_]
        Time values converted to ``dtype``.

    Raises
    ------
    ValueError
        If ``dtype`` is not one of ``np.float32`` or ``np.float64``.
    """
    if dtype not in ("float32", "float64", np.float32, np.float64):
        raise ValueError(f"dtype {dtype} not supported")

    delta = time - offset

    # Note: using a pattern like delta // np.timedelta64(1, "ms") does not maintain nans
    quotient = delta / np.timedelta64(1, "ms")
    floored = np.floor(quotient)
    return floored.astype(dtype)


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
    indices: tuple | None = ...,
    return_indices: Literal[False] = ...,  # mypy wants the default here
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
    indices: tuple | None = ...,
    return_indices: Literal[True],  # and not here!
) -> tuple[npt.NDArray[np.float_], tuple]:
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
    indices: tuple | None = ...,
    return_indices: bool = ...,
) -> npt.NDArray[np.float_] | tuple[npt.NDArray[np.float_], tuple]:
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
    indices: tuple | None = None,
    return_indices: bool = False,
) -> npt.NDArray[np.float_] | tuple[npt.NDArray[np.float_], tuple]:
    """Interpolate over a grid with ``localize`` option.

    .. versionchanged:: 0.25.6

        Utilize scipy 1.9 upgrades to remove singleton dimensions.

    .. versionchanged:: 0.26.0

        Include ``indices`` and ``return_indices`` experimental parameters.
        Currently, nan values in ``longitude``, ``latitude``, ``level``, or ``time``
        are always propagated through to the output, regardless of ``bounds_error``.
        In other words, a ValueError for an out of bounds coordinate is only raised
        if a non-nan value is out of bounds.

    Parameters
    ----------
    longitude, latitude, level, time : np.ndarray
        Coordinates of points to be interpolated. These parameters have the same
        meaning as ``x`` in analogy with :func:`numpy.interp`. All four of these
        arrays must be 1 dimensional of the same size.
    da : xr.DataArray
        Gridded data interpolated over. Must adhere to ``MetDataArray`` conventions.
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
    npt.NDArray[np.float_] | tuple[npt.NDArray[np.float_], tuple]
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
    t = da.coords.variables["time"].values

    # Convert t to float
    offset = t[0]
    # Use dtype of spatial coordinates to determine dtype of time
    dtype = np.result_type(x, y, z, longitude, latitude, level)
    t = _floatize_time(t, offset, dtype)
    time_float = _floatize_time(time, offset, dtype)

    # Remove any "single level" (level = -1) dimension
    # We're making all sort of assumptions about the ordering of the da dimensions
    # here. We could be more rigorous if needed.
    points: tuple[np.ndarray, ...]  # help out mypy
    xi_tup: tuple[np.ndarray, ...]
    if z.size == 1 and z.item() == -1:
        # NOTE: It's much faster to call da.values.squeeze() than da.squeeze().values
        # NOTE: If the level axis is unknown, use: da.get_axis_num("level")
        values = da.values.squeeze(axis=2)
        points = x, y, t
        xi_tup = longitude, latitude, time_float
    else:
        values = da.values
        points = x, y, z, t
        xi_tup = longitude, latitude, level, time_float

    xi = np.column_stack(xi_tup)
    nans = np.any(np.isnan(xi), axis=1)

    interp_ = PycontrailsRegularGridInterpolator(
        points=points,
        values=values,
        method=method,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )

    if indices or return_indices:
        out, out_indices = _interpolation_with_indices(interp_, xi, nans, localize, indices)
    else:
        result = interp_(xi[~nans])
        out = np.empty(longitude.size, dtype=result.dtype)
        out[nans] = np.nan
        out[~nans] = result

    if return_indices:
        return out, out_indices
    return out


def _interpolation_with_indices(
    interp: scipy.interpolate.RegularGridInterpolator,
    xi: npt.NDArray[np.float_],
    nans: npt.NDArray[np.bool_],
    localize: bool,
    indices: tuple | None,
) -> tuple[npt.NDArray[np.float_], tuple]:
    if interp.method != "linear":
        raise ValueError("Parameter indices only supported for 'method=linear'")
    if localize:
        raise ValueError("Parameter indices only supported for 'localize=False'")

    # All this copied from RegularGridInterpolator.__call__
    # This implementation is not as careful to do reshaping

    if interp.bounds_error:
        for i, p in enumerate(xi.T):
            if not np.logical_and(np.all(interp.grid[i][0] <= p), np.all(p <= interp.grid[i][-1])):
                raise ValueError("One of the requested xi is out of bounds in dimension %d" % i)

    if indices is None:
        indices = interp._find_indices(xi.T)

    # DEBUG: Uncomment the lines below if use_indices gives problems
    # else:
    #     indices_ = interp._find_indices(xi.T)
    #     for idx1, idx2 in zip(indices, indices_):
    #         np.testing.assert_array_equal(idx1, idx2)

    result = interp._evaluate_linear(*indices)

    if not interp.bounds_error and interp.fill_value is not None:
        result[indices[2]] = interp.fill_value

    # f(nan) = nan, if any
    if np.any(nans):
        result[nans] = np.nan

    return result, indices


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
        if not np.all(np.diff(self.xp) > 0):
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
