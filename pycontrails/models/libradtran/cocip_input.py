"""Contrail inputs to libRadtran interface."""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import erf

from pycontrails.models.cocip import CocipParams
from pycontrails.models.cocip import radiative_forcing as rf
from pycontrails.models.libradtran import utils
from pycontrails.physics import geo, units

logger = logging.getLogger(__name__)


class CocipInput:
    """Cocip input to libRadtran."""

    __slots__ = ("segments", "radius", "footprint", "params")

    #: Cocip contrails segments
    segments: pd.DataFrame

    #: Threshold radius
    radius: float

    #: Pixel footprint
    footprint: float

    #: Cocip parameters
    params: CocipParams

    def __init__(
        self,
        contrail: pd.DataFrame,
        *,
        radius: float = 100e3,
        footprint: float = 0,
        params: CocipParams | None = None,
    ):
        self.segments = contrail_to_segments(contrail)
        self.radius = radius
        self.footprint = footprint
        self.params = params or CocipParams()

    def get_profiles(self, lon: float, lat: float, time: np.datetime64) -> dict[str, Any]:
        """Compute libRadtran input profiles."""

        # Early exit if dataframe of segments is empty
        if self.segments.empty:
            return {}

        # Interpolate to target time
        candidates = self.segments.groupby(["flight_id", "waypoint"]).filter(
            lambda df: (df["time"].min() <= time) & (df["time"].max() >= time)
        )
        segments = candidates.groupby(["flight_id", "waypoint"]).apply(
            lambda df: _at_time(time, df)
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
        if len(segments) == 0:
            return {}
        weights = weights[weights.between(0, 1)]

        # Extra filter needed when exactly at a segment endpoint
        def discard(row: pd.Series, other: pd.Series) -> bool:
            flight_id, waypoint = row.name
            if (flight_id, waypoint + 1) not in other.index:
                return False
            return (row["wt"] == 1.0) and (other.loc[flight_id, waypoint + 1] == 0.0)

        mask = pd.DataFrame({"wt": weights}).apply(
            lambda row: discard(row, weights), axis="columns"
        )
        segments = segments[~mask]
        if len(segments) == 0:
            return {}
        weights = weights[~mask]

        # Profiles are required for all remaining segments
        logger.debug(
            f"Profiles at {time}, {lat}N, {lon}E includes contributions "
            f"from {len(segments)} contrail segment(s) "
            f"(threshold distance {self.radius/1e3} km)"
        )
        segments["wt"] = weights

        profiles: dict[str, Any] = {}
        for _, seg in segments.iterrows():
            profiles = utils.check_merge(profiles, self._get_segment_profiles(seg, lon, lat))
        return profiles

    def _get_segment_profiles(self, segment: pd.Series, lon: float, lat: float) -> dict[str, Any]:
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
        if self.footprint > 0:
            dz = (
                prefix
                * np.sqrt(np.pi * Syy / 2)
                * (
                    erf((y + self.footprint) / np.sqrt(2 * Syy))
                    - erf((y - self.footprint) / np.sqrt(2 * Syy))
                )
                / (2 * self.footprint)
            )
        else:
            dz = prefix * np.exp(-0.5 * y**2 / Syy)

        # If local plume depth is less than 1 mm, assume it can be ignored.
        if dz < 1e-3:
            return {}

        # Compute altitude of local plume center as
        # concentration-weighted altitude at y.
        z0 = z + Syz * y / Syy

        # Generate profile for each habit
        out = {}
        name = "-".join(str(n) for n in segment.name)
        for habit, weight, re in zip(*_cocip_habits(r, self.params), strict=True):
            out[f"{name}-{habit.lower().replace(' ', '-')}"] = _create_profile(
                habit, z0 - dz, z0 + dz, weight * iwc, re
            )

        return out


def contrail_to_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Cocip contrail output to per-segment data."""
    segments = _as_segments(df)
    if segments.empty:
        return pd.DataFrame()
    return segments.groupby(["flight_id", "waypoint"]).apply(
        lambda df: df.reset_index(drop=True).sort_values("time"), include_groups=False
    )


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
    epoch = np.datetime64("1970-01-01 00:00:00")
    tick = np.timedelta64(1, "s")
    t = (time - epoch) / tick
    df_t = (df["time"].to_numpy() - epoch) / tick
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
    if lon == lon0 and lat == lat0:
        return 0.0
    v_traj = geo.longitudinal_angle(lon0, lat0, lon1, lat1)
    v = geo.longitudinal_angle(lon0, lat0, lon, lat)
    return np.sign(v[0] * v_traj[1] - v_traj[0] * v[1])


def _cocip_habits(
    r_vol: float, params: CocipParams
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute habit weights and effective radii."""
    r_vol_um = np.atleast_1d(r_vol) * 1e6
    G = rf.habit_weights(r_vol_um, params.habit_distributions, params.radius_threshold_um)
    idx0, idx1 = np.nonzero(G)
    r_eff_um = rf.effective_radius_by_habit(r_vol_um[idx0], idx1)
    return params.habits[idx1], G[idx0, idx1], r_eff_um * 1e-6


def _create_profile(habit: str, z0: float, z1: float, iwc: float, re: float) -> dict[str, Any]:
    """Generate profile for specific habit."""

    habit = habit.lower()

    if habit == "sphere":
        msg = "Mie scattering calculations required for spheric ice particles."
        raise ValueError(msg)

    if habit == "solid column":
        return {
            "options": [
                "profile_properties yang2013 interpolate",
                "profile_habit_yang2013 solid_column severe",
            ],
            "z": np.array([z1, z0]),
            "cwc": np.array([0, iwc]),
            "re": np.array([0, np.clip(re, 5e-6, 90 - 6)]),
        }

    if habit == "hollow column":
        return {
            "options": [
                "profile_properties yang2013 interpolate",
                "profile_habit_yang2013 hollow_column severe",
            ],
            "z": np.array([z1, z0]),
            "cwc": np.array([0, iwc]),
            "re": np.array([0, np.clip(re, 5e-6, 90e-6)]),
        }

    if habit == "rough aggregate":
        return {
            "options": ["profile_properties baum_v36 interpolate", "profile_habit aggregate"],
            "z": np.array([z1, z0]),
            "cwc": np.array([0, iwc]),
            "re": np.array([0, np.clip(re, 5e-6, 60e-6)]),
        }

    if habit == "rosette-6":
        return {
            "options": [
                "profile_properties yang2013 interpolate",
                "profile_habit_yang2013 solid_bullet_rosette severe",
            ],
            "z": np.array([z1, z0]),
            "cwc": np.array([0, iwc]),
            "re": np.array([0, np.clip(re, 5e-6, 90e-6)]),
        }

    if habit == "plate":
        return {
            "options": [
                "profile_properties yang2013 interpolate",
                "profile_habit_yang2013 plate severe",
            ],
            "z": np.array([z1, z0]),
            "cwc": np.array([0, iwc]),
            "re": np.array([0, np.clip(re, 5e-6, 90e-6)]),
        }

    if habit == "droxtal":
        return {
            "options": [
                "profile_properties yang2013 interpolate",
                "profile_habit_yang2013 droxtal severe",
            ],
            "z": np.array([z1, z0]),
            "cwc": np.array([0, iwc]),
            "re": np.array([0, np.clip(re, 5e-6, 90e-6)]),
        }

    if habit == "myhre":
        return {
            "options": [
                "profile_properties yang2013 interpolate",
                "profile_habit_yang2013 solid_column severe",
            ],
            "z": np.array([z1, z0]),
            "cwc": np.array([0, iwc]),
            "re": np.array([0, 32e-6]),
        }

    msg = f"Unrecognized contrail habit {habit}"
    raise ValueError(msg)
