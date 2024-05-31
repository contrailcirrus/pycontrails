"""APCEMM interface utility functions."""

import numpy as np

from pycontrails.physics import units
from pycontrails.utils.types import ArrayScalarLike


def normal_wind_shear(
    u_hi: ArrayScalarLike,
    u_lo: ArrayScalarLike,
    v_hi: ArrayScalarLike,
    v_lo: ArrayScalarLike,
    azimuth: ArrayScalarLike,
    dz: float,
) -> ArrayScalarLike:
    r"""Compute segment-normal wind shear from wind speeds at lower and upper levels.

    Parameters
    ----------
    u_hi : ArrayScalarLike
        Eastward wind at upper level [:math:`m/s`]
    u_lo : ArrayScalarLike
        Eastward wind at lower level [:math:`m/s`]
    v_hi : ArrayScalarLike
        Northward wind at upper level [:math:`m/s`]
    v_lo : ArrayScalarLike
        Northward wind at lower level [:math:`m/s`]
    azimuth : ArrayScalarLike
        Segment azimuth [:math:`\deg`]
    dz : float
        Distance between upper and lower level [:math:`m`]

    Returns
    -------
    ArrayScalarLike
        Segment-normal wind shear [:math:`1/s`]
    """
    du_dz = (u_hi - u_lo) / dz
    dv_dz = (v_hi - v_lo) / dz
    az_radians = units.degrees_to_radians(azimuth)
    sin_az = np.sin(az_radians)
    cos_az = np.cos(az_radians)
    return sin_az * dv_dz - cos_az * du_dz


def soot_radius(
    nvpm_ei_m: ArrayScalarLike, nvpm_ei_n: ArrayScalarLike, rho_bc: float = 1770.0
) -> ArrayScalarLike:
    """Calculate mean soot radius from mass and number emissions indices.

    Parameters
    ----------
    nvpm_ei_m : ArrayScalarLike
        Soot mass emissions index [:math:`kg/kg`]
    nvpm_ei_n : ArrayScalarLike
        Soot number emissions index [:math:`1/kg`]
    rho_bc : float, optional
        Density of black carbon [:math:`kg/m^3`]. By default, 1770.
    """
    return ((3.0 * nvpm_ei_m) / (4.0 * np.pi * rho_bc * nvpm_ei_n)) ** (1.0 / 3.0)
