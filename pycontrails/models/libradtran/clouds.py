"""Contrail inputs to libRadtran interface."""

from __future__ import annotations

import itertools
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import erf, gamma

from pycontrails.core import GeoVectorDataset, MetDataset, met_var
from pycontrails.core.models import interpolate_met
from pycontrails.datalib.ecmwf import variables as ecmwf
from pycontrails.models.cocip import CocipParams
from pycontrails.models.cocip import radiative_forcing as rf
from pycontrails.models.libradtran import utils
from pycontrails.physics import constants, geo, units
from pycontrails.utils.types import ArrayScalarLike

logger = logging.getLogger(__name__)


class LRTClouds(ABC):
    """Base class for cloud inputs to libRadtran interface.

    Implementing classes must implement the :meth:`get_profiles` method
    """

    @abstractmethod
    def get_profiles(
        self,
        source: GeoVectorDataset,
    ) -> pd.DataFrame:
        """Abstract method to handle calculation of profiles for libRadtran input.

        Parameters
        ----------
        source : GeoVectorDataset
            Dataset defining coordinates where profiles are required.

        Returns
        -------
        pd.DataFrame
            TODO
        """


class MetDatasetClouds(LRTClouds):
    """MetDataset cloud inputs to LibRadtran interface.

    TODO: document inputs
    """

    __slots__ = ("met", "sfc", "interp_kwargs")

    #: Meteorology data
    met: MetDataset

    #: Surface data
    sfc: MetDataset

    #: Interpolation keyword arguments
    interp_kwargs: dict[str, Any]

    #: Required meteorology variables
    met_variables = (
        ecmwf.CloudAreaFractionInLayer,
        ecmwf.SpecificCloudLiquidWaterContent,
        ecmwf.SpecificCloudIceWaterContent,
        met_var.AirTemperature,
        met_var.Geopotential,
    )

    #: Required surface variables
    sfc_variables = (ecmwf.SurfaceGeopotential,)

    #: Cloud droplet size distribution intercept parameter [:math:`m^{-3 - \mu}`]
    n0 = 1.2e66

    #: Cloud droplet size distribution shape parameter
    mu = 11

    def __init__(
        self, met: MetDataset, sfc: MetDataset, *, interp_kwargs: dict[str, Any] | None = None
    ):
        met.ensure_vars(self.met_variables)
        sfc.ensure_vars(self.sfc_variables)
        self.met = met
        self.sfc = sfc
        self.interp_kwargs = interp_kwargs or {}

    def get_profiles(self, source: GeoVectorDataset) -> np.ndarray:
        """Compute libRadtran input profiles.

        Parameters
        ----------
        source : GeoVectorDataset
            Dataset defining coordinates where profiles should be computed

        Returns
        -------
        np.ndarray
            TODO
        """

        # Downselect meteorology
        met = source.downselect_met(self.met, level_buffer=(np.inf, np.inf), copy=False)
        logger.debug(f"Loading {met.data.nbytes/1e6} MB of met data")
        met.data.load()

        sfc = source.downselect_met(self.sfc, copy=False)
        logger.debug(f"Loading {sfc.data.nbytes/1e6} MB of surface data")
        sfc.data.load()

        interp_kwargs = self.interp_kwargs

        def _process(point: pd.Series) -> pd.Series:
            # Initialze output dataframe
            out = pd.DataFrame(columns=["options", "z", "cwc", "re"])

            # Get target points on profile
            level = met.data["level"].to_numpy()
            target = GeoVectorDataset(
                time=np.full(level.shape, point["time"]),
                level=level,
                latitude=np.full(level.shape, point["latitude"]),
                longitude=np.full(level.shape, point["longitude"]),
            )
            sfc_target = GeoVectorDataset(
                time=np.atleast_1d(point["time"]),
                level=[-1],
                latitude=np.atleast_1d(point["latitude"]),
                longitude=np.atleast_1d(point["longitude"]),
            )

            # Interpolate
            z = interpolate_met(met, target, "geopotential", **interp_kwargs) / constants.g
            zs = (
                interpolate_met(sfc, sfc_target, "geopotential_at_surface", **interp_kwargs)
                / constants.g
            )
            p = met.data["air_pressure"].to_numpy()
            t = interpolate_met(met, target, "air_temperature", **interp_kwargs)
            rho = p / (constants.R_d * t)
            lwc = (
                interpolate_met(met, target, "specific_cloud_liquid_water_content", **interp_kwargs)
                / rho
            )
            iwc = (
                interpolate_met(met, target, "specific_cloud_ice_water_content", **interp_kwargs)
                / rho
            )

            # Mask missing values at lowest levels
            mask = np.isnan(z)
            for v in [t, lwc, iwc]:
                mask = mask | np.isnan(v)
            end = np.flatnonzero(~mask).max() + 1
            z = z[:end]
            t = t[:end]
            lwc = lwc[:end]
            iwc = iwc[:end]

            # Exclude top layers with no cloud
            mask = (iwc > 0.0) | (lwc > 0.0)
            if mask.sum() == 0:
                return out
            start = max(0, np.flatnonzero(mask).min() - 1)

            # Compute altitude at layer base
            zb = 0.5 * (zs + z[-1])
            ze = np.concatenate((0.5 * (z[1:] + z[:-1]), zb))
            ze = ze[start:]
            t = t[start:]
            lwc = lwc[start:]
            iwc = iwc[start:]
            if iwc[0] != 0 or lwc[0] != 0:
                msg = "Cloud water content is not zero in highest layer. Consider extending domain."
                warnings.warn(msg)

            # Compute effective radius
            rel = np.where(lwc > 0, _reff_liquid(lwc, self.n0, self.mu), 0.0)
            rei = np.where(iwc > 0, _reff_ice(iwc, t), 0.0)

            if np.any(lwc > 0):
                out.loc["liquid"] = {
                    "options": ["profile_properties mie interpolate"],
                    "z": ze / 1e3,
                    "cwc": np.maximum(0, lwc) * 1e3,
                    "re": rel * 1e6,
                }
            if np.any(iwc > 0):
                out.loc["ice"] = {
                    "options": ["profile_properties baum_v36 interpolate", "profile_habit ghm"],
                    "z": ze / 1e3,
                    "cwc": np.maximum(0, iwc) * 1e3,
                    "re": rei * 1e6,
                }

            return out

        # create multiindex
        return utils.apply_expand_index(source.dataframe, _process)


def _reff_liquid(lwc: ArrayScalarLike, n0: ArrayScalarLike, mu: ArrayScalarLike) -> ArrayScalarLike:
    """Compute effective radius of liquid cloud.

    TODO: complete docstring.
    """
    pow = 1.0 / (mu + 3.0)
    lam = (np.pi * constants.rho_liq * n0 * gamma(mu + 3) / lwc) ** pow
    return (mu + 2.0) / (2.0 * lam)


def _reff_ice(iwc: ArrayScalarLike, t: ArrayScalarLike) -> ArrayScalarLike:
    """Compute effective radius of ice cloud.

    Reference: tau_cirrus.cirrus_effective_extinction_coefficient
    """
    riwc = iwc * 1e3
    tiwc = t + constants.absolute_zero + 190.0
    deff = 45.8966 * riwc**0.2214 + 0.7957 * tiwc * riwc**0.2535
    return np.clip(0.5e-6 * deff, 5e-6, 60e-6)


class CocipContrails(LRTClouds):
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
    def from_dataframe(cls, df: pd.DataFrame, **params: Any) -> CocipContrails:
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

    def get_profiles(self, source: GeoVectorDataset) -> pd.DataFrame:
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
        pd.DataFrame
            TODO
        """

        def _process(point: pd.Series) -> pd.Series:
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
            if len(segments) == 0:
                return pd.DataFrame(columns=["options", "z", "cwc", "re"])
            weights = weights[weights.between(0, 1)]

            # Profiles are required for all remaining segments
            logger.debug(
                f"Profiles at {t}, {lat}N, {lon}E includes contributions "
                f"from {len(segments)} contrail segment(s) "
                f"(threshold distance {self.radius/1e3} km)"
            )
            segments["wt"] = weights

            return utils.apply_expand_index(
                segments,
                lambda seg: _generate_profiles(seg, lon, lat, self.footprint, self.cocip_params),
                by=["flight_id", "waypoint"],
            )

        return utils.apply_expand_index(source.dataframe, _process)


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


def _cocip_habits(r_vol: float, params: CocipParams) -> pd.DataFrame:
    """Compute habit weights and effective radii."""
    r_vol_um = np.atleast_1d(r_vol) * 1e6
    G = rf.habit_weights(r_vol_um, params.habit_distributions, params.radius_threshold_um)
    idx0, idx1 = np.nonzero(G)
    r_eff_um = rf.effective_radius_by_habit(r_vol_um[idx0], idx1)
    return pd.DataFrame(
        data={"weight": G[idx0, idx1], "re": r_eff_um * 1e-6},
        index=[h.lower() for h in params.habits[idx1]],
    )


def _generate_profile(habits: pd.DataFrame, z0: float, z1: float, iwc: float) -> dict[str, Any]:
    """Generate profile for specific habit."""

    habit = habits.name
    cwc = np.maximum(habits["weight"] * iwc * 1e3, 0)
    re = habits["re"] * 1e6

    if habit == "sphere":
        msg = "Mie scattering calculations required for spheric ice particles."
        raise ValueError(msg)
    if habit == "solid column":
        return pd.Series(
            {
                "options": [
                    "profile_properties yang2013 interpolate",
                    "profile_habit_yang2013 solid_column severe",
                ],
                "z": np.array([z1, z0]) / 1e3,
                "cwc": np.array([0, cwc]),
                "re": np.array([0, np.clip(re, 5.0, 90.0)]),
            }
        )
    if habit == "hollow column":
        return pd.Series(
            {
                "options": [
                    "profile_properties yang2013 interpolate",
                    "profile_habit_yang2013 hollow_column severe",
                ],
                "z": np.array([z1, z0]) / 1e3,
                "cwc": np.array([0, cwc]),
                "re": np.array([0, np.clip(re, 5.0, 90.0)]),
            }
        )
    if habit == "rough aggregate":
        return pd.Series(
            {
                "options": ["profile_properties baum_v36 interpolate", "profile_habit aggregate"],
                "z": np.array([z1, z0]) / 1e3,
                "cwc": np.array([0, cwc]),
                "re": np.array([0, np.clip(re, 5.0, 60.0)]),
            }
        )
    if habit == "rosette-6":
        return pd.Series(
            {
                "options": [
                    "profile_properties yang2013 interpolate",
                    "profile_habit_yang2013 solid_bullet_rosette severe",
                ],
                "z": np.array([z1, z0]) / 1e3,
                "cwc": np.array([0, cwc]),
                "re": np.array([0, np.clip(re, 5.0, 90.0)]),
            }
        )
    if habit == "plate":
        return pd.Series(
            {
                "options": [
                    "profile_properties yang2013 interpolate",
                    "profile_habit_yang2013 plate severe",
                ],
                "z": np.array([z1, z0]) / 1e3,
                "cwc": np.array([0, cwc]),
                "re": np.array([0, np.clip(re, 5.0, 90.0)]),
            }
        )
    if habit == "droxtal":
        return pd.Series(
            {
                "options": [
                    "profile_properties yang2013 interpolate",
                    "profile_habit_yang2013 droxtal severe",
                ],
                "z": np.array([z1, z0]) / 1e3,
                "cwc": np.array([0, cwc]),
                "re": np.array([0, np.clip(re, 5.0, 90.0)]),
            }
        )
    if habit == "myhre":
        return pd.Series(
            {
                "options": [
                    "profile_properties yang2013 interpolate",
                    "profile_habit_yang2013 solid_column severe",
                ],
                "z": np.array([z1, z0]) / 1e3,
                "cwc": np.array([0, cwc]),
                "re": np.array([0, 32.0]),
            }
        )

    msg = f"Unrecognized contrail habit {habit}"
    raise ValueError(msg)


def _generate_profiles(
    segment: pd.Series, lon: float, lat: float, footprint: float, params: CocipParams
) -> pd.DataFrame:
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
        return pd.DataFrame(columns=["options", "z", "cwc", "re"])

    # Compute altitude of local plume center as
    # concentration-weighted altitude at y.
    z0 = z + Syz * y / Szz

    # Generate profile for each habit
    habits = _cocip_habits(r, params)
    return habits.apply(lambda row: _generate_profile(row, z0 - dz, z0 + dz, iwc), axis="columns")
