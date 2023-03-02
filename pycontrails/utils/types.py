"""Convienence types."""

from __future__ import annotations

import functools
from datetime import datetime
from typing import Any, Callable, Type, TypeVar, Union

import numpy as np
import pandas as pd
import xarray as xr

#: Array like (np.ndarray, xr.DataArray)
ArrayLike = TypeVar("ArrayLike", np.ndarray, xr.DataArray, Union[xr.DataArray, np.ndarray])

#: Array or Float (np.ndarray, float)
ArrayOrFloat = TypeVar("ArrayOrFloat", np.ndarray, float)

#: Array like input (np.ndarray, xr.DataArray, np.float64, float)
ArrayScalarLike = TypeVar(
    "ArrayScalarLike",
    np.ndarray,
    xr.DataArray,
    np.float64,
    float,
    Union[np.ndarray, float],
    Union[xr.DataArray, np.ndarray],
)

#: Datetime like input (datetime, pd.Timestamp, np.datetime64)
DatetimeLike = TypeVar("DatetimeLike", datetime, pd.Timestamp, np.datetime64)


def support_arraylike(
    func: Callable[[np.ndarray], np.ndarray]
) -> Callable[[ArrayScalarLike], ArrayScalarLike]:
    """Extend a numpy universal function operating on arrays of floats.

    This decorator allows `func` to support any ArrayScalarLike parameter and
    keeps the return type consistent with the parameter.

    Parameters
    ----------
    func : Callable[[ArrayScalarLike], np.ndarray]
        A numpy `ufunc` taking in a single array with `float`-like dtype.
        This decorator assumes `func` returns a numpy array.

    Returns
    -------
    Callable[[ArrayScalarLike], ArrayScalarLike]
        Extended function.

    See Also
    --------
    - `numpy ufuncs <https://numpy.org/doc/stable/reference/ufuncs.html>`_
    """

    @functools.wraps(func)
    def wrapped(arr: ArrayScalarLike) -> ArrayScalarLike:
        x = np.asarray(arr)

        # Convert to float if not already
        if x.dtype not in (np.float32, np.float64):
            x = x.astype(np.float64)
        ret = func(x)

        # Numpy in, numpy out
        if isinstance(arr, np.ndarray):
            return ret

        # Keep python native numeric types native
        if isinstance(arr, (float, int, np.float64)):
            return ret.item()

        # Recreate pd.Series
        if isinstance(arr, pd.Series):
            return pd.Series(data=ret, index=arr.index)

        # Recreate xr.DataArray
        if isinstance(arr, xr.DataArray):
            return arr.copy(data=ret)  # See documentation for xr.copy!

        # Pass numpy `ret` through for anything else
        return ret

    return wrapped


def apply_nan_mask_to_arraylike(arr: ArrayLike, nan_mask: np.ndarray) -> ArrayLike:
    """Apply ``nan_mask`` to ``arr`` while maintaining the type.

    The parameter ``arr`` should have a ``float`` ``dtype``.

    This function is tested against :class:`xr.DataArray`, :class:`pd.Series`, and
    :class:`np.ndarray` types.

    Parameters
    ----------
    arr : ArrayLike
        Array with ``np.float64`` entries
    nan_mask : np.ndarray
        Boolean array of the same shape as ``arr``

    Returns
    -------
    ArrayLike
        Array ``arr`` with values in ``nan_mask`` set to ``np.nan``. The ``arr`` is
        mutated in place if it is a :class:`np.ndarray`. For :class:`xr.DataArray`,
        a copy is returned.

    Notes
    -----
    When ``arr`` is a :class:`xr.DataArray`, this function keeps any ``attrs``
    from ``arr`` in the returned instance.
    """
    if isinstance(arr, xr.DataArray):
        # The previous implementation uses xr.where instead of arr.where
        # There was some change in xarray 2022.6.0 that broke the former implementation
        # Instead, use arr.where
        return arr.where(~nan_mask, np.nan)

    # If we want to avoid copying, use np.where(~nan_mask, arr, np.nan)
    arr[nan_mask] = np.nan
    return arr


_Object = TypeVar("_Object")


def type_guard(
    obj: Any,
    type_: Type[_Object] | tuple[Type[_Object], ...],
    error_message: str | None = None,
) -> _Object:
    """Shortcut utility to type guard a variable with custom error message.

    Parameters
    ----------
    obj : Any
        Any variable object
    type_ : Type[_Object]
        Type of variable.
        Can be a tuple of types
    error_message : str, optional
        Custom error message

    Returns
    -------
    _Object
        Returns the input object ensured to be ``type_``

    Raises
    ------
    ValueError
        Raises ValueError if ``obj`` is not ``type_``
    """
    if not isinstance(obj, type_):
        raise ValueError(error_message or f"Object must be of type {type_}")

    return obj
