"""Convienence types."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import xarray as xr

#: Array like (np.ndarray, xr.DataArray)
ArrayLike = TypeVar("ArrayLike", np.ndarray, xr.DataArray)

#: Array or Float (np.ndarray, float)
ArrayOrFloat = TypeVar("ArrayOrFloat", np.ndarray, float)

#: Array like input (np.ndarray, xr.DataArray, float)
ArrayScalarLike = TypeVar("ArrayScalarLike", np.ndarray, xr.DataArray, float)

#: Datetime like input (datetime, pd.Timestamp, np.datetime64)
DatetimeLike = TypeVar("DatetimeLike", datetime, pd.Timestamp, np.datetime64, str)

# Crude fix for autodoc issue calling TypeVar.__dict__ on Python 3.13
if "sphinx" in sys.modules and sys.version_info >= (3, 13):
    ArrayLike.__dict__ = {}
    ArrayOrFloat.__dict__ = {}
    ArrayScalarLike.__dict__ = {}
    DatetimeLike.__dict__ = {}


def apply_nan_mask_to_arraylike[T: (np.ndarray, xr.DataArray)](arr: T, nan_mask: np.ndarray) -> T:
    """Apply ``nan_mask`` to ``arr`` while maintaining the type.

    The parameter ``arr`` should have a ``float`` ``dtype``.

    This function is tested against :class:`xr.DataArray`, :class:`pd.Series`, and
    :class:`np.ndarray` types.

    Parameters
    ----------
    arr : T
        A :class:`np.ndarray` or :class:`xr.DataArray` with ``np.float64`` entries
    nan_mask : np.ndarray
        Boolean array of the same shape as ``arr``

    Returns
    -------
    T
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


def type_guard[T](
    obj: Any,
    type_: type[T],
    error_message: str | None = None,
) -> T:
    """Shortcut utility to type guard a variable with custom error message.

    Parameters
    ----------
    obj : Any
        Any variable object
    type_ : Type[T]
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
        raise TypeError(error_message or f"Object must be of type {type_}")

    return obj
