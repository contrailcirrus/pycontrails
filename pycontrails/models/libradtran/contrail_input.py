"""Contrail inputs to libRadtran interface."""

from __future__ import annotations

import itertools
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import erf

from pycontrails.core import GeoVectorDataset
from pycontrails.models.cocip import CocipParams
from pycontrails.models.cocip import radiative_forcing as rf
from pycontrails.physics import geo, units

logger = logging.getLogger(__name__)


class ContrailInput(ABC):
    """Base class for contrail inputs to libRadtran interface.

    Implementing classes must implement the :meth:`get_profiles` method
    """

    @abstractmethod
    def get_profiles(
        self,
        source: GeoVectorDataset,
    ) -> list[list[dict[str, Any]]]:
        """Abstract method to handle calculation of profiles for libRadtran input.

        Parameters
        ----------
        source : GeoVectorDataset
            Dataset defining coordinates where profiles are required.

        Returns
        -------
        list[list[dict[str, Any]]]
            Nested list of dict with cloud properties. The first dimension corresponds
            to points defined by each element of ``source``, and the second dimension
            provides profiles required for the radiative transfer calculation at each
            point.
        """


class CocipContrailInput(ContrailInput):
    """Cocip contrail inputs to LibRadtran interface.

    TODO: document inputs.

    Notes
    -----
        - unable to reproduce habit mixture used by cocip,
        so instead going to use baum_v36 habit mixture for
        ice clouds and scale 550 nm optical depth to match
        cocip value.
        - fixing iwc within contrail and varying depth to
        reproduce iwp.
    """

    __slots__ = ("radius", "footprint", "cocip_params", "_segments")

    #: Threshold radius [:math:`m`] for generating profiles.
    #: Contrail segments outside this radius are ignored.
    radius: float

    #: Footprint of profile [:math:`m`].
    #: If larger than 0, contrail properties are averaged over
    #: ``[y - footprint, y + footprint]``, where y is the distance
    #: from the profile to the contrail segment centerline.
    footprint: float

    #: Cocip parameters.
    #: Can be passed explicitly for consistent treatment of
    #: contrail habit distributions.
    cocip_params: CocipParams

    def __init__(
        self,
        segments: pd.DataFrame,
        *,
        radius: float = 100e3,
        footprint: float = 0.0,
        cocip_params: CocipParams | None = None,
    ):
        self._segments = segments
        self.radius = radius
        self.footprint = footprint
        self.cocip_params = cocip_params or CocipParams()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, **params: Any) -> CocipContrailInput:
        """Initialize contrail inputs from Cocip ``contrail`` output.

        Parameters
        ----------
        df : pd.DataFrame
            ``contrail`` output from Cocip simulation
        """
        segments = (
            _as_segments(df)
            .groupby(["flight_id", "waypoint"])
            .apply(lambda df: df.reset_index(drop=True).sort_values("time"), include_groups=False)
        )
        return cls(segments, **params)

    def get_profiles(self, source: GeoVectorDataset) -> list[list[dict[str, Any]]]:
        """Compute libRadtran input profiles.

        Parameters
        ----------
        source : GeoVectorDataset
            Dataset defining coordinates where profiles should be computed
        radius : float
            Maximum distances between contrail and target point. Profiles are
            generated only for contrails within this radius.

        Returns
        -------
        list[list[libRadtranProfile]]
            Nested list of input profiles. The first dimension corresponds to points
            defined by each element of ``source``, and the second dimension provides
            profiles required for the radiative transfer calculation at each point.
        """
        profiles = []

        for _, point in source.dataframe.iterrows():
            t, lat, lon = point["time"], point["latitude"], point["longitude"]

            # Interpolate to target time
            candidates = self._segments.groupby(["flight_id", "waypoint"]).filter(
                lambda df, t=t: (df["time"].min() <= t) & (df["time"].max() >= t)
            )
            segments = candidates.groupby(["flight_id", "waypoint"]).apply(
                lambda df, t=t: _at_time(t, df)
            )

            # Filter by distance
            dist0 = geo.haversine(lon, lat, segments["lon0"], segments["lat0"])
            dist1 = geo.haversine(lon, lat, segments["lon1"], segments["lat1"])
            segments = segments[(dist0 <= self.radius) & (dist1 <= self.radius)]

            # Compute weights
            dx = units.longitude_distance_to_m(
                segments["lon1"] - segments["lon0"], 0.5 * (segments["lat1"] + segments["lat0"])
            )
            dy = units.latitude_distance_to_m(segments["lat1"] - segments["lat0"])
            det = dx**2 + dy**2
            dx_pt = units.longitude_distance_to_m(
                lon - segments["lon0"], 0.5 * (lat + segments["lat0"])
            )
            dy_pt = units.latitude_distance_to_m(lat - segments["lat0"])
            weights = (dx * dx_pt + dy * dy_pt) / det

            # Filter based on weights
            segments = segments[weights.between(0, 1)]
            weights = weights[weights.between(0, 1)]

            # Profiles are required for all remaining segments
            logger.debug(
                f"Profiles at {t}, {lat}N, {lon}E includes contributions "
                f"from {len(segments)} contrail segment(s) "
                f"(threshold distance {self.radius/1e3} km)"
            )
            segments["wt"] = weights
            local_profiles = [
                _generate_profile(lon, lat, segment, self.footprint, self.cocip_params)
                for _, segment in segments.iterrows()
            ]
            profiles.append([p for p in local_profiles if p is not None])

        return profiles


def _as_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Cocip ``contrail`` output to list of segments.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of Cocip ``contrail`` output at a single time step.

    Returns
    -------
    pd.DataFrame
        List of contrail segments. Each segment is labeled by the flight_id
        and waypoint number at the older of its two endpoints.
    """
    segments = []
    for (_, w0), (_, w1) in itertools.pairwise(df.iterrows()):
        if not w0["continuous"]:
            continue
        segments.append(_make_segment(w0, w1))
    return pd.DataFrame(segments)


def _make_segment(w0: pd.Series, w1: pd.Series) -> pd.Series:
    """Compute segment properties."""

    return pd.Series(
        {
            "time": w0["time"],
            "flight_id": w0["flight_id"],
            "waypoint": w0["waypoint"],
            "lon0": w0["longitude"],
            "lon1": w1["longitude"],
            "lat0": w0["latitude"],
            "lat1": w1["latitude"],
            "z0": w0["altitude"],
            "z1": w1["altitude"],
            "Syy0": 0.125 * w0["width"] ** 2,
            "Syy1": 0.125 * w1["width"] ** 2,
            "Syz0": w0["sigma_yz"],
            "Syz1": w1["sigma_yz"],
            "Szz0": 0.125 * w0["depth"] ** 2,
            "Szz1": 0.125 * w1["depth"] ** 2,
            "iwc0": w0["rho_air"] * w0["iwc"],
            "iwc1": w1["rho_air"] * w1["iwc"],
            "r0": w0["r_ice_vol"],
            "r1": w1["r_ice_vol"],
        }
    )


def _at_time(time: pd.Timestamp, df: pd.DataFrame) -> pd.Series:
    """Interpolate columns of dataframe to target time."""
    t = time.timestamp()
    df_t = df["time"].apply(pd.Timestamp.timestamp).to_numpy()
    return pd.Series(
        {
            key: np.interp(t, df_t, df[key].to_numpy(), left=np.nan, right=np.nan)
            if key != "time"
            else time
            for key in df.columns
        }
    )


def _sign_y(lon: float, lat: float, lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    """Get sign of Cocip y coordinate."""
    v_traj = geo.longitudinal_angle(lon0, lat0, lon1, lat1)
    v = geo.longitudinal_angle(lon0, lat0, lon, lat)
    return np.sign(v[0] * v_traj[1] - v_traj[0] * v[1])


def _effective_radius(r_vol: float, params: CocipParams) -> float:
    """Compute effective radius based on weighted combination of Cocip habits.

    Note that the correct average is the harmonic mean of individual habits.
    """
    r_vol_um = np.atleast_1d(r_vol) * 1e6
    G = rf.habit_weights(r_vol_um, params.habit_distributions, params.radius_threshold_um)
    idx0, idx1 = np.nonzero(G)
    r_eff_um = rf.effective_radius_by_habit(r_vol_um[idx0], idx1)
    return 1e-6 / np.sum(G[idx0, idx1] / r_eff_um)


def _generate_profile(
    lon: float, lat: float, segment: pd.Series, footprint: float, params: CocipParams
) -> dict[str, Any] | None:
    """Generate libRadtran cloud profile for segment."""

    # Compute Cocip y coordinate
    w = segment["wt"]
    slon = w * segment["lon1"] + (1 - w) * segment["lon0"]
    slat = w * segment["lat1"] + (1 - w) * segment["lat0"]
    sgn = _sign_y(lon, lat, segment["lon0"], segment["lat0"], segment["lon1"], segment["lat1"])
    y = sgn * geo.haversine(lon, lat, slon, slat)

    # Compute plume properties
    z = w * segment["z1"] + (1 - w) * segment["z0"]
    Syy = w * segment["Syy1"] + (1 - w) * segment["Syy0"]
    Syz = w * segment["Syz1"] + (1 - w) * segment["Syz0"]
    Szz = w * segment["Szz1"] + (1 - w) * segment["Szz0"]
    iwc = w * segment["iwc1"] + (1 - w) * segment["iwc0"]
    r = w * segment["r1"] + (1 - w) * segment["r0"]

    # Compute plume depth based on local ice water path
    # and plume ice water content.
    detS = Syy * Szz - Syz**2
    A = 2 * np.pi * max(0, detS) ** 0.5
    B = 8 * np.sqrt(Syy)
    prefix = (4.0 / np.pi) ** 0.5 * A / B
    if footprint > 0:
        dz = (
            prefix
            * np.sqrt(np.pi * Syy / 2)
            * (erf((y + footprint) / np.sqrt(2 * Syy)) - erf((y - footprint) / np.sqrt(2 * Syy)))
            / (2 * footprint)
        )
    else:
        dz = prefix * np.exp(-0.5 * y**2 / Syy)

    # If local plume depth is less than 1 mm, assume it can be ignored.
    if dz < 1e-3:
        return None

    # Compute altitude of local plume center as
    # concentration-weighted altitude at y.
    z0 = z + Syz * y / Szz

    # Compute effective radius based on Cocip habit parameterization
    # TODO: generate separate cloud profiles for each habit
    re = _effective_radius(r, params)

    # TODO: run Mie scattering calculations to allow re < 5
    re = np.clip(re, 5e-6, 60e-6)

    # Return profile
    return {
        "options": ["profile_properties baum_v36 interpolate", "profile_habit ghm"],
        "z": np.array([z0 + dz, z0 - dz]) / 1e3,
        "cwc": np.array([0, iwc]) * 1e3,
        "re": np.array([0, re]) * 1e6,
    }
