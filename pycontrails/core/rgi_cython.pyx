# cython: language_level=3
"""Cythonized routines for the PycontrailsRegularGridInterpolator.

This module expands upon the cython interpolation routines introduced in
scipy 1.10 in several ways:

- Includes "fast path" linear interpolation routines for 3D and 4D arrays. The
  scipy implementation is only for 2D arrays.
- The fast path routines are specialized for float and double arrays. The scipy
  implementation is only for double arrays.
"""

import numpy as np
cimport numpy as np
cimport cython


np.import_array()


# Create several type aliases
# Each of the evaluate_linear_... functions will be specialized for both float and double
# In other words, we'll get four specialized functions.
# See https://cython.readthedocs.io/en/latest/src/userguide/fusedtypes.html
ctypedef fused floating_distances:
    float
    double

ctypedef fused floating_values:
    float
    double


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
def evaluate_linear_4d(
    const floating_values[:, :, :, :] values,
    const long[:, :] indices,
    const floating_distances[:, :] norm_distances,
    floating_values[:] out,
) -> np.ndarray:
    cdef:
        long n_points = indices.shape[1]
        long i0, i1, i2, i3, p
        floating_distances y0, y1, y2, y3

    for p in range(n_points):
        i0 = indices[0, p]
        i1 = indices[1, p]
        i2 = indices[2, p]
        i3 = indices[3, p]

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
    const floating_values[:, :, :] values,
    const long[:, :] indices,
    const floating_distances[:, :] norm_distances,
    floating_values[:] out,
) -> np.ndarray:
    cdef:
        long n_points = indices.shape[1]
        long i0, i1, i2, p
        floating_distances y0, y1, y2

    for p in range(n_points):
        i0 = indices[0, p]
        i1 = indices[1, p]
        i2 = indices[2, p]

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
    const floating_values[:, :] values,
    const long[:, :] indices,
    const floating_distances[:, :] norm_distances,
    floating_values[:] out,
) -> np.ndarray:
    cdef:
        long n_points = indices.shape[1]
        long i0, i1, p
        floating_distances y0, y1

    for p in range(n_points):
        i0 = indices[0, p]
        i1 = indices[1, p]

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
    const floating_values[:] values,
    const long[:] indices,
    const floating_distances[:] norm_distances,
    floating_values[:] out,
) -> np.ndarray:
    cdef:
        long n_points = indices.shape[0]
        long i0, p
        floating_distances y0

    for p in range(n_points):
        i0 = indices[p]
        y0 = norm_distances[p]
        out[p] = values[i0] * (1 - y0) + values[i0+1] * y0

    return np.asarray(out)
