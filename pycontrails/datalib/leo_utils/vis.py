"""Utilities for visualization of low-Earth orbit satellite imagery."""

from typing import Any

import numpy as np

from pycontrails.utils import dependencies


def normalize(channel: np.ndarray) -> np.ndarray:
    """Normalize channel values to range [0, 1], preserving ``np.nan`` in output.

    Parameters
    ----------
    channel: np.ndarray
        Array of channel values for normalization.

    Returns
    -------
    np.ndarray
        Equalized channel values. ``np.nan`` will be preserved wherever present in input.
    """
    return (channel - np.nanmin(channel)) / (np.nanmax(channel) - np.nanmin(channel))


def equalize(channel: np.ndarray, **equalize_kwargs: Any) -> np.ndarray:
    """Apply :py:func:`ski.exposure.equalize_adapthist`, preserving ``np.nan`` in output.

    Parameters
    ----------
    channel: np.ndarray
        Array of channel values for equalization.
    **equalize_kwargs : Any
        Keyword arguments passed to :py:func:`ski.exposure.equalize_adapthist`.

    Returns
    -------
    np.ndarray
        Equalized channel values. ``np.nan`` will be preserved wherever present in input.

    Notes
    -----
    NaN values are converted to 0 before passing to :py:func:`ski.exposure.equalize_adapthist`
    and may affect equalized values in the neighborhood where they occur.
    """
    try:
        import skimage.exposure
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            name="landsat module",
            package_name="scikit-image",
            module_not_found_error=exc,
            pycontrails_optional_package="sat",
        )
    return np.where(
        np.isnan(channel),
        np.nan,
        skimage.exposure.equalize_adapthist(np.nan_to_num(channel, nan=0.0), **equalize_kwargs),
    )
