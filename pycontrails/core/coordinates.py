"""Coordinates utilities."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def slice_domain(
    domain: np.ndarray,
    request: np.ndarray | tuple,
    buffer: tuple[float | np.timedelta64, float | np.timedelta64] = (0.0, 0.0),
) -> slice:
    """Return slice of ``domain`` containing coordinates overlapping ``request``.

    Computes index-based slice (to be used with :meth:`xarray.Dataset.isel` method) as
    opposed to value-based slice.

    Returns ``slice(None, None)`` when ``domain`` has a length <= 2 or the ``request``
    has all ``nan`` values.

    .. versionchanged:: 0.24.1

        The returned slice is the minimum index-based slice that contains the
        requested coordinates. In other words, it's now possible for
        ``domain[sl][0]`` to equal ``min(request)`` where ``sl`` is the output
        of this function. Previously, we were guaranteed that ``domain[sl][0]``
        would be less than ``min(request)``.


    Parameters
    ----------
    domain : np.ndarray
        Full set of domain values
    request : np.ndarray | tuple
        Requested values. Only the min and max values are considered. Either pass in
        a full array-like object or a tuple of ``(min, max)``.
    buffer : tuple[float | np.timedelta64, float | np.timedelta64], optional
        Extend the domain past the requested coordinates by ``buffer[0]`` on the low side
        and ``buffer[1]`` on the high side.
        Units of ``buffer`` must be the same as ``domain``.

    Returns
    -------
    slice
        Slice object for slicing out encompassing, or nearest, domain values

    Raises
    ------
    ValueError
        Raises a ValueError when ``domain`` has all ``nan`` values.

    Examples
    --------
    >>> domain = np.arange(-180, 180, 0.25)

    >>> # Call with request as np.array
    >>> request = np.linspace(-20, 20, 100)
    >>> slice_domain(domain, request)
    slice(640, 801, None)

    >>> # Call with request as tuple
    >>> request = -20, 20
    >>> slice_domain(domain, request)
    slice(640, 801, None)

    >>> # Call with a buffer
    >>> request = -16, 13
    >>> buffer = 4, 7
    >>> slice_domain(domain, request, buffer)
    slice(640, 801, None)

    >>> # Call with request as a single number
    >>> request = -20
    >>> slice_domain(domain, request)
    slice(640, 641, None)

    >>> request = -19.9
    >>> slice_domain(domain, request)
    slice(640, 642, None)

    """
    # if the length of domain coordinates is <= 2, return the whole domain
    if len(domain) <= 2:
        return slice(None, None)

    if buffer == (None, None):
        return slice(None, None)

    # if the request is nan, then there is nothing to slice
    if np.isnan(request).all():
        return slice(None, None)

    # if the whole domain or request is nan, then there is nothing to slice
    if np.isnan(domain).all():
        raise ValueError("Domain is all nan on request")

    # ensure domain is sorted
    zero: float | np.timedelta64 = 0.0
    if pd.api.types.is_datetime64_dtype(domain.dtype):
        zero = np.timedelta64(0)

    if not np.all(np.diff(domain) >= zero):
        raise ValueError("Domain must be sorted in ascending order")

    if np.any(np.asarray(buffer) < zero):
        warnings.warn(
            "Found buffer with negative value. This is unexpected "
            "and will reduce the size of the requested domain instead of "
            "extending it. Both the left and right buffer values should be "
            "nonnegative."
        )

    # get the index of the closest value to request min and max
    # side left returns `i`: domain[i-1] < request <= domain[i]
    # side right returns `i`: domain[i-1] <= request < domain[i]
    idx_min = np.searchsorted(domain, np.nanmin(request) - buffer[0], side="right") - 1
    idx_max = np.searchsorted(domain, np.nanmax(request) + buffer[1], side="left") + 1

    # clip idx_min between [0, len(domain) - 2]
    idx_min = min(len(domain) - 2, max(idx_min, 0))

    # clip idx_max between [2, len(domain)]
    idx_max = min(len(domain), max(idx_max, 2))

    return slice(idx_min, idx_max)


def intersect_domain(
    domain: np.ndarray,
    request: np.ndarray,
) -> np.ndarray:
    """Return boolean mask of ``request`` that are within the bounds of ``domain``.

    Parameters
    ----------
    domain : np.ndarray
        Full set of domain values
    request : np.ndarray
        Full set of requested values

    Returns
    -------
    np.ndarray
        Boolean array of ``request`` values within the bounds of ``domain``

    Raises
    ------
    ValueError
        Raises a ValueError when ``domain`` has all ``nan`` values.
    """
    # if the whole domain or request is nan, then there is nothing to slice
    if np.isnan(domain).all():
        raise ValueError("Domain is all nan on request")

    # if not np.all(np.diff(domain) >= 0):
    #     raise ValueError("Domain must be sorted in ascending order")

    return (request >= np.nanmin(domain)) & (request <= np.nanmax(domain))
