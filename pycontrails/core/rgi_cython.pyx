# cython: language_level=3
"""Cythonized routines for the PycontrailsRegularGridInterpolator.

This module expands upon the cython interpolation routines introduced in
scipy 1.10 in several ways:

- Includes "fast path" linear interpolation routines for 3D and 4D arrays. The
  scipy implementation is only for 2D arrays.
- The fast path routines are specialized for float and double arrays. The scipy
  implementation is only for double arrays.
"""

from libc.math cimport NAN
import numpy as np

cimport cython
cimport numpy as np

np.import_array()


# See https://cython.readthedocs.io/en/latest/src/userguide/fusedtypes.html
ctypedef fused floating:
    float
    double


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
def evaluate_linear_4d(
    const floating[:, :, :, :] values,
    const np.intp_t[:, :] indices,
    const double[:, :] norm_distances,
    floating[:] out,
) -> np.ndarray:
    cdef:
        long n_points = indices.shape[1]
        long i0, i1, i2, i3, p
        double y0, y1, y2, y3

    for p in range(n_points):
        i0 = indices[0, p]
        i1 = indices[1, p]
        i2 = indices[2, p]
        i3 = indices[3, p]

        if i0 < 0 or i1 < 0 or i2 < 0 or i3 < 0:
            out[p] = NAN
            continue

        y0 = norm_distances[0, p]
        y1 = norm_distances[1, p]
        y2 = norm_distances[2, p]
        y3 = norm_distances[3, p]

        out[p] = (
            values[i0, i1, i2, i3] * (1 - y0) * (1 - y1) * (1 - y2) * (1 - y3) +
            values[i0, i1, i2, i3+1] * (1 - y0) * (1 - y1) * (1 - y2) * y3 +
            values[i0, i1, i2+1, i3] * (1 - y0) * (1 - y1) * y2 * (1 - y3) +
            values[i0, i1, i2+1, i3+1] * (1 - y0) * (1 - y1) * y2 * y3 +
            values[i0, i1+1, i2, i3] * (1 - y0) * y1 * (1 - y2) * (1 - y3) +
            values[i0, i1+1, i2, i3+1] * (1 - y0) * y1 * (1 - y2) * y3 +
            values[i0, i1+1, i2+1, i3] * (1 - y0) * y1 * y2 * (1 - y3) +
            values[i0, i1+1, i2+1, i3+1] * (1 - y0) * y1 * y2 * y3 +
            values[i0+1, i1, i2, i3] * y0 * (1 - y1) * (1 - y2) * (1 - y3) +
            values[i0+1, i1, i2, i3+1] * y0 * (1 - y1) * (1 - y2) * y3 +
            values[i0+1, i1, i2+1, i3] * y0 * (1 - y1) * y2 * (1 - y3) +
            values[i0+1, i1, i2+1, i3+1] * y0 * (1 - y1) * y2 * y3 +
            values[i0+1, i1+1, i2, i3] * y0 * y1 * (1 - y2) * (1 - y3) +
            values[i0+1, i1+1, i2, i3+1] * y0 * y1 * (1 - y2) * y3 +
            values[i0+1, i1+1, i2+1, i3] * y0 * y1 * y2 * (1 - y3) +
            values[i0+1, i1+1, i2+1, i3+1] * y0 * y1 * y2 * y3
        )

    return np.asarray(out)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
def evaluate_linear_3d(
    const floating[:, :, :] values,
    const np.intp_t[:, :] indices,
    const double[:, :] norm_distances,
    floating[:] out,
) -> np.ndarray:
    cdef:
        long n_points = indices.shape[1]
        long i0, i1, i2, p
        double y0, y1, y2

    for p in range(n_points):
        i0 = indices[0, p]
        i1 = indices[1, p]
        i2 = indices[2, p]

        if i0 < 0 or i1 < 0 or i2 < 0:
            out[p] = NAN
            continue

        y0 = norm_distances[0, p]
        y1 = norm_distances[1, p]
        y2 = norm_distances[2, p]

        out[p] = (
            values[i0, i1, i2] * (1 - y0) * (1 - y1) * (1 - y2) +
            values[i0, i1, i2+1] * (1 - y0) * (1 - y1) * y2 +
            values[i0, i1+1, i2] * (1 - y0) * y1 * (1 - y2) +
            values[i0, i1+1, i2+1] * (1 - y0) * y1 * y2 +
            values[i0+1, i1, i2] * y0 * (1 - y1) * (1 - y2) +
            values[i0+1, i1, i2+1] * y0 * (1 - y1) * y2 +
            values[i0+1, i1+1, i2] * y0 * y1 * (1 - y2) +
            values[i0+1, i1+1, i2+1] * y0 * y1 * y2
        )

    return np.asarray(out)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
def evaluate_linear_2d(
    const floating[:, :] values,
    const np.intp_t[:, :] indices,
    const double[:, :] norm_distances,
    floating[:] out,
) -> np.ndarray:
    cdef:
        long n_points = indices.shape[1]
        long i0, i1, p
        double y0, y1

    for p in range(n_points):
        i0 = indices[0, p]
        i1 = indices[1, p]

        if i0 < 0 or i1 < 0:
            out[p] = NAN
            continue

        y0 = norm_distances[0, p]
        y1 = norm_distances[1, p]

        out[p] = (
            values[i0, i1] * (1 - y0) * (1 - y1) +
            values[i0, i1+1] * (1 - y0) * y1 +
            values[i0+1, i1] * y0 * (1 - y1) +
            values[i0+1, i1+1] * y0 * y1
        )

    return np.asarray(out
)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
def evaluate_linear_1d(
    const floating[:] values,
    const np.intp_t[:] indices,
    const double[:] norm_distances,
    floating[:] out,
) -> np.ndarray:
    cdef:
        long n_points = indices.shape[0]
        long i0, p
        double y0

    for p in range(n_points):
        i0 = indices[p]
        if i0 < 0:
            out[p] = NAN
            continue

        y0 = norm_distances[p]
        out[p] = values[i0] * (1 - y0) + values[i0+1] * y0

    return np.asarray(out)



# -----------------------------------------------------------------------------
# The following two functions are copied directly from the scipy source code
# Once pycontrails requires scipy >= 1.12, these can be removed and the scipy
# source code can be used directly.
# This is a workaround for the following commit:
# https://github.com/scipy/scipy/commit/619e68f21769a19dd25e35506d25ddf1f960d9e3
# -----------------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int find_interval_ascending(const double *x,
                                 size_t nx,
                                 double xval,
                                 int prev_interval=0,
                                 bint extrapolate=1) noexcept nogil:
    """
    Find an interval such that x[interval] <= xval < x[interval+1]. Assuming
    that x is sorted in the ascending order.
    If xval < x[0], then interval = 0, if xval > x[-1] then interval = n - 2.

    Parameters
    ----------
    x : array of double, shape (m,)
        Piecewise polynomial breakpoints sorted in ascending order.
    xval : double
        Point to find.
    prev_interval : int, optional
        Interval where a previous point was found.
    extrapolate : bint, optional
        Whether to return the last of the first interval if the
        point is out-of-bounds.

    Returns
    -------
    interval : int
        Suitable interval or -1 if nan.

    """
    cdef:
        int high, low, mid
        int interval = prev_interval
        double a = x[0]
        double b = x[nx - 1]
    if interval < 0 or interval >= nx:
        interval = 0

    if not (a <= xval <= b):
        # Out-of-bounds (or nan)
        if xval < a and extrapolate:
            # below
            interval = 0
        elif xval > b and extrapolate:
            # above
            interval = nx - 2
        else:
            # nan or no extrapolation
            interval = -1
    elif xval == b:
        # Make the interval closed from the right
        interval = nx - 2
    else:
        # Find the interval the coordinate is in
        # (binary search with locality)
        if xval >= x[interval]:
            low = interval
            high = nx - 2
        else:
            low = 0
            high = interval

        if xval < x[low+1]:
            high = low

        while low < high:
            mid = (high + low)//2
            if xval < x[mid]:
                # mid < high
                high = mid
            elif xval >= x[mid + 1]:
                low = mid + 1
            else:
                # x[mid] <= xval < x[mid+1]
                low = mid
                break

        interval = low

    return interval


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def find_indices(tuple grid not None, const double[:, :] xi):
    # const is required for xi above in case xi is read-only
    cdef:
        long i, j, grid_i_size
        double denom, value
        # const is required in case grid is read-only
        const double[::1] grid_i
        # Axes to iterate over
        long I = xi.shape[0]
        long J = xi.shape[1]
        int index = 0
        # Indices of relevant edges between which xi are situated
        np.intp_t[:,::1] indices = np.empty_like(xi, dtype=np.intp)
        # Distances to lower edge in unity units
        double[:,::1] norm_distances = np.zeros_like(xi, dtype=float)
    # iterate through dimensions
    for i in range(I):
        grid_i = grid[i]
        grid_i_size = grid_i.shape[0]
        if grid_i_size == 1:
            # special case length-one dimensions
            for j in range(J):
                # Should equal 0. Setting it to -1 is a hack: evaluate_linear 
                # looks at indices [i, i+1] which both end up =0 with wraparound. 
                # Conclusion: change -1 to 0 here together with refactoring
                # evaluate_linear, which will also need to special-case
                # length-one axes
                indices[i, j] = -1
                # norm_distances[i, j] is already zero
        else:
            for j in range(J):
                value = xi[i, j]
                index = find_interval_ascending(&grid_i[0],
                                                grid_i_size,
                                                value,
                                                prev_interval=index,
                                                extrapolate=1)
                indices[i, j] = index
                if value == value:
                    denom = grid_i[index + 1] - grid_i[index]
                    norm_distances[i, j] = (value - grid_i[index]) / denom
                else:
                    # xi[i, j] is nan
                    norm_distances[i, j] = NAN
    return np.asarray(indices), np.asarray(norm_distances)
