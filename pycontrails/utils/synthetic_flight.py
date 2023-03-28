"""Tools for creating synthetic data."""

from __future__ import annotations

import logging
import pathlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
from pyproj.geod import Geod

from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataArray
from pycontrails.physics import constants, geo, units
from pycontrails.utils.types import ArrayOrFloat

try:
    from pycontrails.ext.bada import bada_model
except ImportError as e:
    raise ImportError(
        'SyntheticFlight requires BADA extension. Install with `pip install "pycontrails-bada @'
        ' git+ssh://git@github.com/contrailcirrus/pycontrails-bada.git"`'
    ) from e

logger = logging.getLogger(__name__)

SAMPLE_AIRCRAFT_TYPES = [
    "A20N",
    "A319",
    "A320",
    "A321",
    "A332",
    "A333",
    "A359",
    "A388",
    "B38M",
    "B737",
    "B738",
    "B739",
    "B744",
    "B752",
    "B763",
    "B772",
    "B77W",
    "B788",
    "B789",
    "CRJ2",
    "CRJ7",
    "CRJ9",
    "E145",
    "E190",
    "E195",
    "E75L",
    "E75S",
]


class SyntheticFlight:
    """Create a synthetic flight."""

    # Random number generator
    rng = np.random.default_rng(None)

    # Maximum queue size. When generating many thousands of flights, performance is slightly
    # improved if this parameter is increased.
    max_queue_size = 100

    # Minimum number of waypoints in flight
    min_n_waypoints = 10

    # Type hint a few instance variables
    bada: bada_model.BADA | None
    aircraft_type: str

    longitude_min: float
    longitude_max: float
    latitude_min: float
    latitude_max: float
    level_min: float
    level_max: float
    time_min: np.datetime64
    time_max: np.datetime64

    def __init__(
        self,
        bounds: dict[str, np.ndarray],
        aircraft_type: str | None = None,
        bada3_path: str | pathlib.Path | None = None,
        bada4_path: str | pathlib.Path | None = None,
        speed_m_per_s: float | None = None,
        seed: int | None = None,
        u_wind: MetDataArray | None = None,
        v_wind: MetDataArray | None = None,
    ) -> None:
        """Create a synthetic flight generator.

        Parameters
        ----------
        bounds : dict[str, np.ndarray]
            A dictionary with keys "longitude", "latitude", "level", and "time". All synthetic
            flights will have coordinates bounded by the extreme values in this dictionary.
        aircraft_type : str
            A flight type, assumed to exist in the BADA3 or BADA4 dataset.
            If None provided, a random aircraft type will be chosen from
            ``SAMPLE_AIRCRAFT`` on every call.
        bada3_path : str | pathlib.Path, optional
            A path to a local BADA3 data source.
            Defaults to None.
        bada4_path : str | pathlib.Path, optional
            A path to a local BADA4 data source.
            Defaults to None.
        speed_m_per_s : float, optional
            Directly define cruising air speed. Only used if `bada3_path` and `bada4_path` are
            not specified. By default None.
        seed : int, optional
            Reseed the random generator. By default None.
        u_wind, v_wind : MetDataArray, optional
            Eastward and northward wind data. If provided, flight true airspeed is computed
            with respect to wind.
        max_queue_size : int, optional
            Maximum queue size. When generating many thousands of flights, performance is slightly
            improved if this parameter is increased.
        min_n_waypoints : int, optional
            Minimum number of waypoints in flight
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.bounds = bounds
        self.constant_aircraft_type = aircraft_type
        self.bada3_path = bada3_path
        self.bada4_path = bada4_path
        self.speed_m_per_s = speed_m_per_s
        self.geod = Geod(a=constants.radius_earth)

        def extremes(arr: np.ndarray) -> tuple[Any, Any]:
            arr = np.asarray(arr)
            return arr.min(), arr.max()

        self.longitude_min, self.longitude_max = extremes(bounds["longitude"])
        self.latitude_min, self.latitude_max = extremes(bounds["latitude"])
        self.level_min, self.level_max = extremes(bounds["level"])
        self.time_min, self.time_max = extremes(bounds["time"])

        self.u_wind = u_wind
        self.v_wind = v_wind

        self._queue: list[Flight] = []
        self._depth = 0

    def __repr__(self) -> str:
        """Get string representation."""
        msg = "Synthetic flight generator with parameters:"
        for key in ["longitude", "latitude", "level", "time"]:
            msg += "\n"
            for bound in ["min", "max"]:
                attr = f"{key}_{bound}"
                val = getattr(self, attr)
                msg += f"{attr}: {val} "
        msg += f"\nmax_queue_size: {self.max_queue_size}  min_n_waypoints: {self.min_n_waypoints}"
        return msg

    def __call__(self, timestep: np.timedelta64 = np.timedelta64(1, "m")) -> Flight:
        """Create random flight within `bounds` at a constant altitude.

        BADA4 data is used to determine flight speed at a randomly chosen altitude within
        the level constraints defined by `bounds`. The flight trajectory follows a great circle
        from a uniformly random chosen source point to a uniformly randomly chosen destination.

        For generating a large number of random flights, it's best to call this within a
        generator expression, ie,
            ``fls = (syn() for _ in range(100_000))``

        Parameters
        ----------
        timestep : np.timedelta64, optional
            Time interval between waypoints. By default, 1 minute

        Returns
        -------
        Flight
            Random `Flight` instance constrained by bounds.
        """
        # Building flights with `u_wind` and `v_wind` involved in the true airspeed calculation is
        # slow. BUT, we can do it in a vectorized way. So we maintain a short queue that gets
        # repeatedly replenished.
        if self.u_wind is not None and self.v_wind is not None:
            while not self._queue:
                # Need to do some significant refactor to use generators seemlessly
                # The parameter timestep can change between calls
                new_batch = self._generate_with_wind(self.max_queue_size, timestep)
                self._queue.extend(new_batch)
            fl = self._queue.pop()
            while np.any(np.diff(fl["time"]) != timestep):
                logger.debug("Found flight in queue with bad timestep.")
                fl = self._queue.pop()
            return fl

        self._depth = 0
        self._define_aircraft()
        return self._generate_single_flight_no_wind(timestep)

    def _id(self) -> int:
        """Get random flight ID."""
        return self.rng.integers(100_000, 999_999)

    def _define_aircraft(self) -> None:
        """Define or update instance variables pertaining to flight aircrafts.

        Specify
            - aircraft_type
            - bada_enabled
            - fl_all
            - cruise_tas

        Raises
        ------
        FileNotFoundError
            BADA files not found despite non-default BADA3 or BADA4 paths
        ValueError
            BADA files not found under default paths AND speed_m_per_s not defined
        """
        if self.constant_aircraft_type is None:
            self.aircraft_type = self.rng.choice(SAMPLE_AIRCRAFT_TYPES)
        else:
            self.aircraft_type = self.constant_aircraft_type

        try:
            self.bada = bada_model.get_bada(
                self.aircraft_type,
                bada3_path=self.bada3_path,
                bada4_path=self.bada4_path,
                bada_priority=4,
            )

        except FileNotFoundError as err:
            logger.warning("BADA files not found")

            # If non-default bada paths were passed into __init__, we should raise an error
            if self.bada3_path is not None or self.bada4_path is not None:
                raise FileNotFoundError from err

            # If  bada paths were not passed into __init__, we expect to know speed_m_per_s
            if self.speed_m_per_s is None:
                raise ValueError("Either specify `bada3_path`, `bada4_path`, or `speed_m_per_s`.")
            self.bada = None

    def _calc_speed_m_per_s(self, level: ArrayOrFloat) -> ArrayOrFloat:
        if self.bada is not None:
            alt_ft = units.pl_to_ft(level)
            return self.bada.nominal_cruising_speed(self.aircraft_type, alt_ft)

        if self.speed_m_per_s is None:
            raise ValueError("Either specify `bada3_path`, `bada4_path` or `speed_m_per_s`.")

        if isinstance(level, np.ndarray):
            return np.full_like(level, self.speed_m_per_s)
        return self.speed_m_per_s

    def _generate_single_flight_no_wind(self, timestep: np.timedelta64) -> Flight:
        src_lon = self.rng.uniform(self.longitude_min, self.longitude_max)
        src_lat = self.rng.uniform(self.latitude_min, self.latitude_max)
        dest_lon = self.rng.uniform(self.longitude_min, self.longitude_max)
        dest_lat = self.rng.uniform(self.latitude_min, self.latitude_max)
        src = src_lon, src_lat
        dest = dest_lon, dest_lat
        az, _, dist = self.geod.inv(*src, *dest)

        level = self.rng.uniform(self.level_min, self.level_max)
        speed_m_per_s = self._calc_speed_m_per_s(level)

        m_per_timestep = speed_m_per_s * (timestep / np.timedelta64(1, "s"))
        npts = int(dist // m_per_timestep)  # need to cast: dist is np.float64

        # Dealing with situations of npts too small or too big
        if npts > (self.time_max - self.time_min) / timestep:
            msg = (
                "Not enough available time in `bounds` to create good flight between "
                f"source {src} and destination {dest}. Try enlarging the time dimension, "
                "or reducing the longitude and latitude dimensions."
            )

            new_npts = int((self.time_max - self.time_min) / timestep)
            logger.debug("Override npts from %s to %s", npts, new_npts)
            npts = new_npts

            if npts < self.min_n_waypoints:
                raise ValueError(msg)
            warnings.warn(msg)

        if npts < self.min_n_waypoints:
            # Try 10 times, then give up.
            self._depth += 1
            if self._depth > 10:
                raise ValueError("Cannot create flight. Increase dimensions in `bounds`.")
            return self._generate_single_flight_no_wind(timestep)  # recursive

        result = self.geod.fwd_intermediate(
            *src, az, npts, m_per_timestep, return_back_azimuth=False  # type: ignore
        )
        longitude = np.asarray(result.lons)
        latitude = np.asarray(result.lats)
        if geo.haversine(longitude[-1], latitude[-1], *dest) > m_per_timestep:
            logger.debug(
                "Synthetic flight did not arrive at destination. "
                "This is likely due to overriding npts."
            )

        rand_range = int((self.time_max - self.time_min - npts * timestep) / timestep) + 1
        time0 = self.time_min + self.rng.integers(rand_range) * timestep
        time: np.ndarray = np.arange(time0, time0 + npts * timestep, timestep)

        df = {"longitude": longitude, "latitude": latitude, "level": level, "time": time}
        return Flight(pd.DataFrame(df), flight_id=self._id(), aircraft_type=self.aircraft_type)

    def _generate_with_wind(self, n: int, timestep: np.timedelta64) -> list[Flight]:
        logger.debug("Generate %s new flights with wind", n)

        self._define_aircraft()

        # Step 1: Randomly select longitude, latitude, level
        src_lon = self.rng.uniform(self.longitude_min, self.longitude_max, n)
        src_lat = self.rng.uniform(self.latitude_min, self.latitude_max, n)
        dest_lon = self.rng.uniform(self.longitude_min, self.longitude_max, n)
        dest_lat = self.rng.uniform(self.latitude_min, self.latitude_max, n)
        level = self.rng.uniform(self.level_min, self.level_max, n)
        src = src_lon, src_lat
        dest = dest_lon, dest_lat
        az: np.ndarray
        dist: np.ndarray
        az, _, dist = self.geod.inv(*src, *dest)

        # Step 2: Compute approximate flight times according to nominal (no wind) TAS
        nom_speed_m_per_s = self._calc_speed_m_per_s(level)
        # NOTE: Because of casting, the multiplication below is NOT associative!
        # In other words, the parentheses are needed on the right-most term
        nom_m_per_timestep = nom_speed_m_per_s * (timestep / np.timedelta64(1, "s"))
        approx_flight_duration_s = dist / nom_speed_m_per_s * np.timedelta64(1, "s")

        # Step 3: Randomly select start time -- use 0.9 for a small buffer
        rand_float = 0.9 * self.rng.random(n)
        time_windows = self.time_max - self.time_min - approx_flight_duration_s

        # Here `time_windows` can have negative timedeltas, which is not good.
        n_negative = np.sum(time_windows < np.timedelta64(0, "s"))
        logger.debug(
            "Found %s / %s src-dist pairs not fitting into the time dimension", n_negative, n
        )
        if n_negative >= 0.1 * n:
            warnings.warn(
                "Not enough available time in `bounds` to create reasonable random flights. Try "
                "enlarging the time dimension, or reducing the longitude and latitude dimensions."
            )
        # Manually clip at 0
        time_windows = np.maximum(time_windows, np.timedelta64(0, "s"))

        rand_start = rand_float * time_windows
        start_time = self.time_min + rand_start

        # Step 4: Iterate and collect
        lons = []
        lats = []
        times = []

        def take_step(
            cur_lon: np.ndarray,
            cur_lat: np.ndarray,
            cur_time: np.typing.NDArray[np.datetime64],
            cur_az: np.ndarray,
            cur_active: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            sin, cos = geo.azimuth_to_direction(cur_az, cur_lat)

            if self.u_wind is None or self.v_wind is None:
                raise TypeError("Define attributes `u_wind` and `v_wind`.")
            u_wind = self.u_wind.interpolate(cur_lon, cur_lat, level, cur_time)
            v_wind = self.v_wind.interpolate(cur_lon, cur_lat, level, cur_time)

            nom_x = nom_m_per_timestep * cos
            nom_y = nom_m_per_timestep * sin
            # NOTE: Because of casting, the multiplication below is NOT associative.
            # In other words, the parentheses are needed on the right-most term
            tas_x = nom_x + u_wind * (timestep / np.timedelta64(1, "s"))
            tas_y = nom_y + v_wind * (timestep / np.timedelta64(1, "s"))
            tas = (tas_x**2 + tas_y**2) ** 0.5

            next_lon, next_lat, _ = self.geod.fwd(cur_lon, cur_lat, cur_az, tas)
            next_az: np.ndarray
            next_dist: np.ndarray
            next_az, _, next_dist = self.geod.inv(next_lon, next_lat, *dest)
            next_time = cur_time + timestep
            next_active = (
                cur_active & (next_dist > nom_m_per_timestep) & (next_time < self.time_max)
            )

            return next_lon, next_lat, next_time, next_az, next_active

        lon = src_lon
        lat = src_lat
        time = start_time
        az = az
        active = np.ones_like(lon).astype(bool)

        while active.sum():
            lons.append(np.where(active, lon, np.nan))
            lats.append(np.where(active, lat, np.nan))
            times.append(time)
            lon, lat, time, az, active = take_step(lon, lat, time, az, active)

        # Step 5: Assemble and return
        lons_arr = np.asarray(lons).T
        lats_arr = np.asarray(lats).T
        times_arr = np.asarray(times).T
        data = [
            {"longitude": lon, "latitude": lat, "level": level, "time": time}
            for lon, lat, level, time in zip(lons_arr, lats_arr, level, times_arr)
        ]
        dfs = [pd.DataFrame(d).dropna() for d in data]
        dfs = [df for df in dfs if len(df) >= self.min_n_waypoints]
        dfs = [df.assign(altitude=units.pl_to_m(df["level"])).drop(columns="level") for df in dfs]

        return [Flight(df, flight_id=self._id(), aircraft_type=self.aircraft_type) for df in dfs]
