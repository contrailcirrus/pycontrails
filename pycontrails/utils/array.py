"""Array utilities."""

import numpy as np
import numpy.typing as npt


def searchsorted2d(
    a: npt.NDArray[np.floating],
    v: npt.NDArray[np.floating],
) -> npt.NDArray[np.int64]:
    """Return the indices where elements in ``v`` would be inserted in ``a`` along its second axis.

    An index of 0 is returned for any rows of ``a`` that contain nan values.

    Implementation based on a `StackOverflow answer <https://stackoverflow.com/a/40588862>`_.

    Parameters
    ----------
    a : npt.NDArray[np.floating]
        2D array of shape ``(m, n)`` that is sorted along its second axis. This is not checked.
    v : npt.NDArray[np.floating]
        1D array of values of shape ``(k,)`` to insert into the second axis of ``a``.
        The current implementation could be extended to handle 2D arrays as well.

    Returns
    -------
    npt.NDArray[np.int64]
        2D array of indices where elements in ``v`` would be inserted in ``a`` along its
        second axis to keep the second axis of ``a`` sorted. The shape of the output is ``(m, k)``.

    Examples
    --------
    >>> a = np.array([
    ...  [ 1.,  8., 11., 12.],
    ...  [ 5.,  8.,  9., 14.],
    ...  [ 4.,  5.,  6., 17.],
    ...  ])
    >>> v = np.array([3., 7., 10., 13., 15.])
    >>> searchsorted2d(a, v)
    array([[1, 1, 2, 4, 4],
           [0, 1, 3, 3, 4],
           [0, 3, 3, 3, 3]])

    >>> a = np.array([
    ...  [ 1.,  8., np.nan, 12.],
    ...  [ 5.,  8.,     9., 14.],
    ...  [ 4.,  5.,     6., 17.],
    ...  ])
    >>> v = np.array([3., 7., 10., 13., 15.])
    >>> searchsorted2d(a, v)
    array([[0, 0, 0, 0, 0],
           [0, 1, 3, 3, 4],
           [0, 3, 3, 3, 3]])
    """
    if a.ndim != 2:
        msg = "The parameter 'a' must be a 2D array"
        raise ValueError(msg)
    if v.ndim != 1:
        msg = "The parameter 'v' must be a 1D array"
        raise ValueError(msg)
    if np.isnan(v).any():
        msg = "The parameter 'v' must not contain NaNs"
        raise ValueError(msg)

    m, n = a.shape

    # Record rows that contain nan values,
    # then convert nan values to numbers without
    # increasing range.
    #
    # The value chosen to replace nans doesn't matter,
    # since indices are set to 0 for rows that contain
    # any nans before returning.
    #
    # Ideally we'd like to find a non-nan value using
    # a function that runs (best case) in O(1) rather
    # than O(n) time... but I think this is probably
    # faster than manually looping over a.
    na_mask = np.isnan(a).any(axis=1, keepdims=True)
    contains_na = na_mask.any()
    if contains_na:
        a = np.nan_to_num(a, nan=np.nanmax(a))

    offset_scalar = max(np.ptp(a).item(), np.ptp(v).item()) + 1.0

    # IMPORTANT: Keep the dtype as float64 to avoid round-off error
    # when computing a_scaled and v_scaled
    # If we used float32 here, the searchsorted output below can be off by 1
    # or 2 if offset_scalar is large and m is large
    steps = np.arange(m, dtype=np.float64).reshape(-1, 1)
    offset = steps * offset_scalar
    a_scaled = a + offset  # float32 + float64 = float64
    v_scaled = v + offset  # float32 + float64 = float64

    # Return 0 for rows that contain any nan values.
    idx_scaled = np.searchsorted(a_scaled.reshape(-1), v_scaled.reshape(-1)).reshape(v_scaled.shape)
    if contains_na:
        return np.where(na_mask, 0, idx_scaled - n * steps.astype(np.int64))
    return idx_scaled - n * steps.astype(np.int64)
