"""CoCiP output formats.

This module includes functions to produce additional output formats,
including grid and time-slice outputs, and flight summary outputs.
"""
import pandas as pd
import numpy as np
import numpy.typing as npt


# --------------
# Output formats
# --------------
def grid_and_time_slices_hourly(
        flight_waypoints: pd.DataFrame,
        contrails: pd.DataFrame
):
    # TODO: Documentations
    return


# -------------------
# Gridded outputs
# -------------------


# -------------------
# Time-slice outputs
# -------------------
def hourly_time_slice_statistics():
    return


def area_mean_ice_particle_radius(
        r_ice_vol: npt.NDArray[np.float_],
        n_ice_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_]
) -> float:
    """
    Calculate the area-mean contrail ice particle radius.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.float_]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Area-mean contrail ice particle radius `r_area`, [:math:`m`]

    Notes
    -----
    - Re-arranged from `tot_ice_cross_sec_area` = `tot_n_ice_particles` * (np.pi * `r_ice_vol`**2)
    - Assumes that the contrail ice crystals are spherical.
    """
    tot_ice_cross_sec_area = _total_ice_particle_cross_sectional_area(
        r_ice_vol, n_ice_per_m, segment_length
    )
    tot_n_ice_particles = _total_ice_particle_number(n_ice_per_m, segment_length)
    return (tot_ice_cross_sec_area / (np.pi * tot_n_ice_particles)) ** (1 / 2)


def volume_mean_ice_particle_radius(
        r_ice_vol: npt.NDArray[np.float_],
        n_ice_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_]
) -> float:
    """
    Calculate the volume-mean contrail ice particle radius.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.float_]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Volume-mean contrail ice particle radius `r_vol`, [:math:`m`]

    Notes
    -----
    - Re-arranged from `tot_ice_vol` = `tot_n_ice_particles` * (4 / 3 * np.pi * `r_ice_vol`**3)
    - Assumes that the contrail ice crystals are spherical.
    """
    tot_ice_vol = _total_ice_particle_volume(r_ice_vol, n_ice_per_m, segment_length)
    tot_n_ice_particles = _total_ice_particle_number(n_ice_per_m, segment_length)
    return (tot_ice_vol / ((4 / 3) * np.pi * tot_n_ice_particles)) ** (1 / 3)


def mean_ice_particle_effective_radius(
        r_ice_vol: npt.NDArray[np.float_],
        n_ice_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_]
) -> float:
    """
    Calculate the mean contrail ice particle effective radius.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.float_]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Mean contrail ice particle effective radius `r_eff`, [:math:`m`]

    Notes
    -----
    - `r_eff` is the ratio of the particle volume to particle projected area. 
    - `r_eff` = (3 / 4) * (`tot_ice_vol` / `tot_ice_cross_sec_area`)
    - See Eq. (62) of :cite:`schumannContrailCirrusPrediction2012`.
    """
    tot_ice_vol = _total_ice_particle_volume(r_ice_vol, n_ice_per_m, segment_length)
    tot_ice_cross_sec_area = _total_ice_particle_cross_sectional_area(
        r_ice_vol, n_ice_per_m, segment_length
    )
    return (3 / 4) * (tot_ice_vol / tot_ice_cross_sec_area)


def _total_ice_particle_cross_sectional_area(
        r_ice_vol: npt.NDArray[np.float_],
        n_ice_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_]
) -> float:
    """
    Calculate total contrail ice particle cross-sectional area.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.float_]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Total ice particle cross-sectional area from all contrail waypoints, [:math:`m^{2}`]
    """
    ice_cross_sec_area = 0.9 * np.pi * r_ice_vol**2
    return np.nansum(ice_cross_sec_area * n_ice_per_m * segment_length)


def _total_ice_particle_volume(
        r_ice_vol: npt.NDArray[np.float_],
        n_ice_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_]
) -> float:
    """
    Calculate total contrail ice particle volume.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.float_]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Total ice particle volume from all contrail waypoints, [:math:`m^{2}`]
    """
    ice_vol = (4 / 3) * np.pi * r_ice_vol**3
    return np.nansum(ice_vol * n_ice_per_m * segment_length)


def _total_ice_particle_number(
        n_ice_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_]
) -> float:
    """
    Calculate total number of contrail ice particles.

    Parameters
    ----------
    n_ice_per_m : npt.NDArray[np.float_]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Total number of ice particles from all contrail waypoints.
    """
    return np.nansum(n_ice_per_m * segment_length)
