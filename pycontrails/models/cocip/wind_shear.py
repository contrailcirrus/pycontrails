"""Wind shear functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import xarray as xr


def wind_shear_enhancement_factor(
    contrail_depth: npt.NDArray[np.floating],
    effective_vertical_resolution: float | npt.NDArray[np.floating],
    wind_shear_enhancement_exponent: float | npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Calculate the multiplication factor to enhance the wind shear based on contrail depth.

    This factor accounts for any subgrid-scale that is not captured by the resolution
    of the meteorological datasets.

    Parameters
    ----------
    contrail_depth : npt.NDArray[np.floating]
        Contrail depth , [:math:`m`]. Expected to be positive and bounded away from 0.
    effective_vertical_resolution : float | npt.NDArray[np.floating]
        Vertical resolution of met data , [:math:`m`]
    wind_shear_enhancement_exponent : float | npt.NDArray[np.floating]
        Exponent used in calculation. Expected to be nonnegative.
        Discussed in paragraphs following eq. (39) in Schumann 2012 and referenced as `n`.
        When this parameter is 0, no enhancement occurs.

    Returns
    -------
    npt.NDArray[np.floating]
        Wind shear enhancement factor

    Notes
    -----
    Implementation based on eq (39) in :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    ratio = effective_vertical_resolution / contrail_depth
    return 0.5 * (1.0 + ratio**wind_shear_enhancement_exponent)


def wind_shear_normal[A: (np.ndarray, xr.DataArray, float)](
    u_wind_top: A,
    u_wind_btm: A,
    v_wind_top: A,
    v_wind_btm: A,
    cos_a: A,
    sin_a: A,
    dz: float,
) -> A:
    r"""Calculate the total wind shear normal to an axis.

    The total wind shear is the vertical gradient of the horizontal velocity.

    Parameters
    ----------
    u_wind_top : A
        u wind speed in the top layer, [:math:`m \ s^{-1}`]
    u_wind_btm : A
        u wind speed in the bottom layer, [:math:`m \ s^{-1}`]
    v_wind_top : A
        v wind speed in the top layer, [:math:`m \ s^{-1}`]
    v_wind_btm : A
        v wind speed in the bottom layer, [:math:`m \ s^{-1}`]
    cos_a : A
        Cosine component of segment
    sin_a : A
        Sine component of segment
    dz : float
        Difference in altitude between measurements, [:math:`m`]

    Returns
    -------
    A
       Wind shear normal to axis, [:math:`s^{-1}`]
    """
    du_dz = (u_wind_top - u_wind_btm) / dz
    dv_dz = (v_wind_top - v_wind_btm) / dz
    return dv_dz * cos_a - du_dz * sin_a


def wind_shear[A: (np.ndarray, xr.DataArray, float)](
    u_wind_top: A,
    u_wind_btm: A,
    v_wind_top: A,
    v_wind_btm: A,
    dz: float,
) -> A:
    r"""Calculate the total wind shear.

    The total wind shear is the vertical gradient of the horizontal velocity.

    Parameters
    ----------
    u_wind_top : A
        u wind speed in the top layer, [:math:`m \ s^{-1}`]
    u_wind_btm : A
        u wind speed in the bottom layer, [:math:`m \ s^{-1}`]
    v_wind_top : A
        v wind speed in the top layer, [:math:`m \ s^{-1}`]
    v_wind_btm : A
        v wind speed in the bottom layer, [:math:`m \ s^{-1}`]
    dz : float
        Difference in altitude between measurements, [:math:`m`]

    Returns
    -------
    A
       Total wind shear, [:math:`s^{-1}`]
    """
    du_dz = (u_wind_top - u_wind_btm) / dz
    dv_dz = (v_wind_top - v_wind_btm) / dz
    return (du_dz**2 + dv_dz**2) ** 0.5
