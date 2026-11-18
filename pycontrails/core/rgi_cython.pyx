# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, freethreading_compatible=True
"""Cythonized routines for the PycontrailsRegularGridInterpolator.

This module expands upon the cython interpolation routines introduced in
scipy 1.10 in several ways:

- Includes "fast path" linear interpolation routines for 3D and 4D arrays. The
  scipy implementation is only for 2D arrays.
- The fast path routines are specialized for float and double arrays. The scipy
  implementation is only for double arrays.
- Includes a ``find_indices_uniform`` routine that computes interpolation indices from
  the grid spacing on equally spaced, ascending grids. The scipy implementation runs a
  binary search (O(log n)) on every grid, starting from the interval of the previous point.
"""

from libc.math cimport NAN, fabs


# See https://cython.readthedocs.io/en/latest/src/userguide/fusedtypes.html
ctypedef fused floating:
    float
    double


def evaluate_linear_4d(
    const floating[:, :, :, :] values,
    const Py_ssize_t[:, :] indices,
    const double[:, :] norm_distances,
    floating[:] out,
) -> None:
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


def evaluate_linear_3d(
    const floating[:, :, :] values,
    const Py_ssize_t[:, :] indices,
    const double[:, :] norm_distances,
    floating[:] out,
) -> None:
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


def evaluate_linear_2d(
    const floating[:, :] values,
    const Py_ssize_t[:, :] indices,
    const double[:, :] norm_distances,
    floating[:] out,
) -> None:
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


def evaluate_linear_1d(
    const floating[:] values,
    const Py_ssize_t[:] indices,
    const double[:] norm_distances,
    floating[:] out,
) -> None:
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


def is_uniform(const double[:] g) -> bool:
    """Return True if ``g`` is ascending and equally spaced within a relative tolerance of 1e-9."""
    cdef:
        Py_ssize_t n = g.shape[0]
        Py_ssize_t k
        double dx = (g[n - 1] - g[0]) / (n - 1)

    if not dx > 0.0:
        return False
    for k in range(n):
        if fabs(g[k] - (g[0] + k * dx)) > 1e-9 * dx:
            return False
    return True


def find_indices_uniform(
    const double[:] g,
    const double[:] x,
    Py_ssize_t[:] indices,
    double[:] norm_distances,
) -> None:
    """Fill ``indices`` and ``norm_distances`` for the points ``x`` on the equally spaced ``g``."""
    cdef:
        Py_ssize_t n = g.shape[0]
        Py_ssize_t i, p
        double guess
        double inv_dx = (n - 1) / (g[n - 1] - g[0])

    for p in range(x.shape[0]):
        # check if x[p] is nan
        if x[p] != x[p]:
            indices[p] = -1
            norm_distances[p] = NAN
            continue

        guess = (x[p] - g[0]) * inv_dx
        if guess < 0.0:
            i = 0
        elif guess > n - 2:
            i = n - 2
        else:
            i = <Py_ssize_t>guess

        # Near a boundary, rounding can put the guess one cell off; step left or right to correct
        # In practice, this is very rare
        while i > 0 and x[p] < g[i]:
            i -= 1
        while i < n - 2 and x[p] >= g[i + 1]:
            i += 1

        indices[p] = i
        norm_distances[p] = (x[p] - g[i]) / (g[i + 1] - g[i])
