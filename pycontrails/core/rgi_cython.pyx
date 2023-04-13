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
    const long[:, :] indices,
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
    const long[:, :] indices,
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
    const long[:, :] indices,
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
    const long[:] indices,
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
