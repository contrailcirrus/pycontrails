"""Flight Data Handling."""

from __future__ import annotations

import dataclasses
import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import scipy.signal
from overrides import overrides

from pycontrails.core.fuel import Fuel, JetA
from pycontrails.core.vector import AttrDict, GeoVectorDataset, VectorDataDict, VectorDataset
from pycontrails.physics import constants, geo, units

logger = logging.getLogger(__name__)

# optional imports
if TYPE_CHECKING:
    import matplotlib
    import traffic


@dataclasses.dataclass
class Aircraft:
    """Base class for the physical parameters of the aircraft."""

    #: Aircraft type ICAO
    aircraft_type: str | None = None

    #: Aircraft type name
    aircraft_name: str | None = None

    #: Wingspan, [:math:`m`]
    wingspan: float | None = None

    #: Engine name
    engine_name: str | None = None

    #: Number of engines
    n_engine: int | None = None

    #: Maximum Mach number
    max_mach: float | None = None

    #: Maximum altitude, [:math:`m`]
    max_altitude: float | None = None


@dataclasses.dataclass
class FlightPhase:
    """Container for boolean arrays describing phase of the flight.

    Each array is expected to have the same shape. Arrays should be pairwise disjoint and cover
    each waypoint (for each waypoint, exactly one of the four arrays should be True.)

    .. todo::

        Refactor to include this data as property in :class:`Flight` with enum for FlightPhase
    """

    cruise: np.ndarray
    climb: np.ndarray
    descent: np.ndarray
    nan: np.ndarray

    @classmethod
    def all_cruise(cls, size: int) -> FlightPhase:
        """Generate `FlightPhase` instance in which all waypoints are at cruise.

        Parameters
        ----------
        size : int
            Number of waypoints

        Returns
        -------
        FlightPhase
            All waypoints are given a cruise state..
        """
        return cls(
            cruise=np.ones(size).astype(bool),
            climb=np.zeros(size).astype(bool),
            descent=np.zeros(size).astype(bool),
            nan=np.zeros(size).astype(bool),
        )


class Flight(GeoVectorDataset):
    """A single flight trajectory.

    Expect latitude-longitude CRS in WGS 84.
    Expect altitude in [:math:`m`].
    Expect level in [:math:`hPa`].

    Use the attribute :attr:`attrs["crs"]` to specify coordinate reference system
    using `PROJ <https://proj.org/>`_ or `EPSG <https://epsg.org/home.html>`_ syntax.

    Parameters
    ----------
    data : dict[str, np.ndarray] | pd.DataFrame | VectorDataDict | VectorDataset | None
        Flight trajectory waypoints as data dictionary or :class:`pandas.DataFrame`.
        Must include columns ``time``, ``latitude``, ``longitude``, ``altitude`` or ``level``.
        Keyword arguments for ``time``, ``latitude``, ``longitude``, ``altitude`` or ``level``
        will override ``data`` inputs. Expects ``altitude`` in meters and ``time`` as a
        DatetimeLike (or array that can processed with :func:`pd.to_datetime`).
        Additional waypoint-specific data can be included as additional keys/columns.
    longitude : np.ndarray, optional
        Flight trajectory waypoint longitude.
        Defaults to None.
    latitude : np.ndarray, optional
        Flight trajectory waypoint latitude.
        Defaults to None.
    altitude : np.ndarray, optional
        Flight trajectory waypoint altitude, [:math:`m`].
        Defaults to None.
    level : np.ndarray, optional
        Flight trajectory waypoint pressure level, [:math:`hPa`].
        Defaults to None.
    time : np.ndarray, optional
        Flight trajectory waypoint time.
        Defaults to None.
    attrs : dict[str, Any], optional
        Additional flight properties as a dictionary.
        Defaults to {}.
    copy : bool, optional
        Copy data on Flight creation.
        Defaults to True.
    aircraft : Aircraft, optional
        Aircraft flying this flight trajectory.
        Properties from the Aircraft will be added to :attr:`attrs`.
        Defaults to None.
    fuel : Fuel, optional
        Fuel used in flight trajectory.
        Defaults to JetA if None.
    drop_duplicated_times : bool, optional
        Drop duplicate times in flight trajectory. Defaults to False.
    **attrs_kwargs : Any
        Additional flight properties passed as keyword arguments.

    Raises
    ------
    KeyError
        Raises if ``data`` input does not contain at least ``time``, ``latitude``, ``longitude``,
        (``altitude`` or ``level``).


    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pycontrails import Flight

    >>> # Create `Flight` from a DataFrame.
    >>> df = pd.DataFrame({
    ...     "longitude": np.linspace(20, 30, 500),
    ...     "latitude": np.linspace(40, 10, 500),
    ...     "altitude": 10500,
    ...     "time": pd.date_range('2021-01-01T10', '2021-01-01T15', periods=500),
    ... })
    >>> fl = Flight(data=df, flight_id=123)  # specify a flight_id by keyword
    >>> fl
    Flight [4 keys x 500 length, 2 attributes]
    Keys: longitude, latitude, altitude, time
    Attributes:
    time                [2021-01-01 10:00:00, 2021-01-01 15:00:00]
    longitude           [20.0, 30.0]
    latitude            [10.0, 40.0]
    altitude            [10500.0, 10500.0]
    flight_id           123
    crs                 EPSG:4326

    >>> # Create `Flight` from keywords
    >>> fl = Flight(
    ...     longitude=np.linspace(20, 30, 200),
    ...     latitude=np.linspace(40, 30, 200),
    ...     altitude=11000 * np.ones(200),
    ...     time=pd.date_range('2021-01-01T12', '2021-01-01T14', periods=200),
    ... )
    >>> fl
    Flight [4 keys x 200 length, 1 attributes]
        Keys: longitude, latitude, time, altitude
        Attributes:
        time                [2021-01-01 12:00:00, 2021-01-01 14:00:00]
        longitude           [20.0, 30.0]
        latitude            [30.0, 40.0]
        altitude            [11000.0, 11000.0]
        crs                 EPSG:4326

    >>> # Access the underlying data as DataFrame
    >>> fl.dataframe.head()
       longitude   latitude                          time  altitude
    0  20.000000  40.000000 2021-01-01 12:00:00.000000000   11000.0
    1  20.050251  39.949749 2021-01-01 12:00:36.180904522   11000.0
    2  20.100503  39.899497 2021-01-01 12:01:12.361809045   11000.0
    3  20.150754  39.849246 2021-01-01 12:01:48.542713567   11000.0
    4  20.201005  39.798995 2021-01-01 12:02:24.723618090   11000.0
    """

    #: Fuel used in flight trajectory
    fuel: Fuel

    def __init__(
        self,
        data: dict[str, np.ndarray] | pd.DataFrame | VectorDataDict | VectorDataset | None = None,
        longitude: np.ndarray | None = None,
        latitude: np.ndarray | None = None,
        altitude: np.ndarray | None = None,
        time: np.ndarray | None = None,
        attrs: dict[str, Any] | AttrDict | None = None,
        copy: bool = True,
        aircraft: Aircraft | None = None,
        fuel: Fuel | None = None,
        drop_duplicated_times: bool = False,
        **attrs_kwargs: Any,
    ) -> None:
        # set aircraft properties to attrs
        # Note input attrs will override aircraft properties
        # TODO: refactor aircraft to handle the same way as Fuel?
        if aircraft is not None:
            aircraft_attrs = dataclasses.asdict(aircraft)
            if attrs:
                aircraft_attrs.update(attrs)
                attrs = aircraft_attrs
            else:
                attrs = aircraft_attrs

        super().__init__(
            data=data,
            longitude=longitude,
            latitude=latitude,
            altitude=altitude,
            time=time,
            attrs=attrs,
            copy=copy,
            **attrs_kwargs,
        )

        # Set fuel - fuel instance is NOT copied
        self.fuel = fuel or JetA()

        # Check flight data for possible errors
        if np.any(self.altitude > 16000):
            flight_id = self.attrs.get("flight_id", "")
            flight_id = flight_id and f" for flight {flight_id}"
            warnings.warn(
                f"Flight altitude is high{flight_id}. Expected altitude unit is meters. "
                f"Found waypoint with altitude {self.altitude.max():.0f} m."
            )

        # Get time differences between waypoints
        if np.isnat(self["time"]).any():
            warnings.warn(
                "Flight trajectory contains NaT times. This will cause errors "
                "with segment-based methods (e.g. 'segment_true_airspeed')."
            )

        diff_ = np.diff(self["time"])

        # Ensure that time is sorted
        if self and np.any(diff_ < np.timedelta64(0)):
            if not copy:
                raise ValueError(
                    "`time` data must be sorted if `copy` is False on creation. "
                    "Set copy=False, or sort data before creating Flight."
                )
            warnings.warn("Sorting Flight data by `time`.")

            sorted_flight = self.sort("time")
            self.data = sorted_flight.data
            # Update diff_ ... we use it again below
            diff_ = np.diff(self["time"])

        # Check for duplicate times. If dropping duplicates,
        # keep the *first* occurrence of each time. This is achieved with
        # the np.insert (rather than np.append) function.
        filt = np.insert(diff_ > np.timedelta64(0), 0, True)
        if not np.all(filt):
            if drop_duplicated_times:
                filtered_flight = self.filter(filt, copy=False)
                self.data = filtered_flight.data
            else:
                warnings.warn(
                    "Flight contains duplicate times. This will cause errors "
                    "with segment-based methods (e.g. 'segment_true_airspeed'). Set "
                    "'drop_duplicated=True' or call the 'resample_and_fill' method."
                )

    @overrides
    def copy(self) -> Flight:
        return Flight(data=self.data, attrs=self.attrs, fuel=self.fuel, copy=True)

    @property
    def time_start(self) -> pd.Timestamp:
        """First waypoint time.

        Returns
        -------
        pd.Timestamp
            First waypoint time
        """
        return pd.Timestamp(np.nanmin(self["time"]))

    @property
    def time_end(self) -> pd.Timestamp:
        """Last waypoint time.

        Returns
        -------
        pd.Timestamp
            Last waypoint time
        """
        return pd.Timestamp(np.nanmax(self["time"]))

    @property
    def duration(self) -> pd.Timedelta:
        """Determine flight duration.

        Returns
        -------
        pd.Timedelta
            Difference between terminal and initial time
        """
        return pd.Timedelta(self.time_end - self.time_start)

    @property
    def max_time_gap(self) -> pd.Timedelta:
        """Return maximum time gap between waypoints along flight trajectory.

        Returns
        -------
        pd.Timedelta
            Gap size

        Examples
        --------
        >>> import numpy as np
        >>> fl = Flight(
        ...     longitude=np.linspace(20, 30, 200),
        ...     latitude=np.linspace(40, 30, 200),
        ...     altitude=11000 * np.ones(200),
        ...     time=pd.date_range('2021-01-01T12', '2021-01-01T14', periods=200),
        ... )
        >>> fl.max_time_gap
        Timedelta('0 days 00:00:36.180...')
        """
        return pd.Timedelta(np.nanmax(np.diff(self["time"])))

    @property
    def max_distance_gap(self) -> float:
        """Return maximum distance gap between waypoints along flight trajectory.

        Distance is calculated based on WGS84 geodesic.

        Returns
        -------
        float
            Maximum distance between waypoints, [:math:`m`]

        Raises
        ------
        NotImplementedError
            Raises when attr:`attrs["crs"]` is not EPSG:4326

        Examples
        --------
        >>> import numpy as np
        >>> fl = Flight(
        ...     longitude=np.linspace(20, 30, 200),
        ...     latitude=np.linspace(40, 30, 200),
        ...     altitude=11000 * np.ones(200),
        ...     time=pd.date_range('2021-01-01T12', '2021-01-01T14', periods=200),
        ... )
        >>> fl.max_distance_gap
        7391.27...
        """
        if self.attrs["crs"] != "EPSG:4326":
            raise NotImplementedError("Only implemented for EPSG:4326 CRS.")

        return self.segment_length()[:-1].max()

    @property
    def length(self) -> float:
        """Return flight length based on WGS84 geodesic.

        Returns
        -------
        float
            Total flight length, [:math:`m`]

        Raises
        ------
        NotImplementedError
            Raises when attr:`attrs["crs"]` is not EPSG:4326

        Examples
        --------
        >>> import numpy as np
        >>> fl = Flight(
        ...     longitude=np.linspace(20, 30, 200),
        ...     latitude=np.linspace(40, 30, 200),
        ...     altitude=11000 * np.ones(200),
        ...     time=pd.date_range('2021-01-01T12', '2021-01-01T14', periods=200),
        ... )
        >>> fl.length
        1436924.67...
        """
        if self.attrs["crs"] != "EPSG:4326":
            raise NotImplementedError("Only implemented for EPSG:4326 CRS.")

        # drop off the nan
        return np.nansum(self.segment_length()[:-1])

    # ------------
    # Segment Properties
    # ------------

    def segment_duration(self, dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
        r"""Compute time elapsed between waypoints in seconds.

        ``np.nan`` appended so the length of the output is the same as number of waypoints.

        Parameters
        ----------
        dtype : np.dtype
            Numpy dtype for time difference.
            Defaults to ``np.float64``

        Returns
        -------
        np.ndarray
            Time difference between waypoints, [:math:`s`].
            Returns an array with dtype specified by``dtype``
        """

        return _dt_waypoints(self.data["time"], dtype=dtype)

    def segment_length(self) -> np.ndarray:
        """Compute spherical distance between flight waypoints.

        Helper function used in :meth:`length` and :meth:`length_met`.
        `np.nan` appended so the length of the output is the same as number of waypoints.

        Returns
        -------
        np.ndarray
            Array of distances in [:math:`m`] between waypoints

        Raises
        ------
        NotImplementedError
            Raises when attr:`attrs["crs"]` is not EPSG:4326

        Examples
        --------
        >>> from pycontrails import Flight
        >>> fl = Flight(
        ... longitude=np.array([1, 2, 3, 5, 8]),
        ... latitude=np.arange(5),
        ... altitude=np.full(shape=(5,), fill_value=11000),
        ... time=pd.date_range('2021-01-01T12', '2021-01-01T14', periods=5),
        ... )
        >>> fl.segment_length()
        array([157255.03346286, 157231.08336815, 248456.48781503, 351047.44358851,
                           nan])

        See Also
        --------
        :func:`segment_length`
        """
        if self.attrs["crs"] != "EPSG:4326":
            raise NotImplementedError("Only implemented for EPSG:4326 CRS.")

        return geo.segment_length(self["longitude"], self["latitude"], self.altitude)

    def segment_angle(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate sin/cos for the angle between each segment and the longitudinal axis.

        This is different from the usual navigational angle between two points known as *bearing*.

        *Bearing* in 3D spherical coordinates is referred to as *azimuth*.
        ::

                    (lon_2, lat_2)  X
                                   /|
                                  / |
                                 /  |
                                /   |
                               /    |
                              /     |
                             /      |
            (lon_1, lat_1)  X -------> longitude (x-axis)

        Returns
        -------
        np.ndarray, np.ndarray
            sin(a), cos(a), where ``a`` is the angle between the segment and the longitudinal axis
            The final values are of both arrays are ``np.nan``.

        See Also
        --------
        :func:`geo.segment_angle`
        :func:`units.heading_to_longitudinal_angle`
        :meth:`segment_azimuth`
        :func:`geo.forward_azimuth`

        Examples
        --------
        >>> from pycontrails import Flight
        >>> fl = Flight(
        ... longitude=np.array([1, 2, 3, 5, 8]),
        ... latitude=np.arange(5),
        ... altitude=np.full(shape=(5,), fill_value=11000),
        ... time=pd.date_range('2021-01-01T12', '2021-01-01T14', periods=5),
        ... )
        >>> sin, cos = fl.segment_angle()
        >>> sin
        array([0.70716063, 0.70737598, 0.44819424, 0.31820671,        nan])

        >>> cos
        array([0.70705293, 0.70683748, 0.8939362 , 0.94802136,        nan])

        """
        return geo.segment_angle(self["longitude"], self["latitude"])

    def segment_azimuth(self) -> np.ndarray:
        """Calculate (forward) azimuth at each waypoint.

        Method calls `pyproj.Geod.inv`, which is slow. See `geo.forward_azimuth`
        for an outline of a faster implementation.

        .. versionchanged:: 0.33.7

            The dtype of the output now matches the dtype of ``self["longitude"]``.

        Returns
        -------
        np.ndarray
            Array of azimuths.
        """
        lon = self["longitude"]
        lat = self["latitude"]

        lons1 = lon[:-1]
        lats1 = lat[:-1]
        lons2 = lon[1:]
        lats2 = lat[1:]

        geod = pyproj.Geod(a=constants.radius_earth)
        az, *_ = geod.inv(lons1, lats1, lons2, lats2)

        # NOTE: geod.inv automatically promotes to float64. We match the dtype of lon.
        out = np.empty_like(lon)
        out[:-1] = az
        # Convention: append nan
        out[-1] = np.nan

        return out

    def segment_groundspeed(
        self, smooth: bool = False, window_length: int = 7, polyorder: int = 1
    ) -> np.ndarray:
        """Return groundspeed across segments.

        Calculate by dividing the horizontal segment length by the difference in waypoint times.

        Parameters
        ----------
        smooth : bool, optional
            Smooth airspeed with Savitzky-Golay filter.
            Defaults to False.
        window_length : int, optional
            Passed directly to :func:`scipy.signal.savgol_filter`, by default 7.
        polyorder : int, optional
            Passed directly to :func:`scipy.signal.savgol_filter`, by default 1.

        Returns
        -------
        np.ndarray
            Groundspeed of the segment, [:math:`m s^{-1}`]
        """
        # get horizontal distance - set altitude to 0
        horizontal_segment_length = geo.segment_haversine(self["longitude"], self["latitude"])

        # time between waypoints, in seconds
        dt_sec = np.empty_like(horizontal_segment_length)
        dt_sec[:-1] = np.diff(self["time"]) / np.timedelta64(1, "s")
        dt_sec[-1] = np.nan

        # calculate groundspeed
        groundspeed = horizontal_segment_length / dt_sec

        # Savitzky-Golay filter
        if smooth:
            # omit final nan value, then reattach it afterwards
            groundspeed[:-1] = _sg_filter(groundspeed[:-1], window_length, polyorder)
            groundspeed[-1] = np.nan

        return groundspeed

    def segment_true_airspeed(
        self,
        u_wind: npt.NDArray[np.float_] | None = None,
        v_wind: npt.NDArray[np.float_] | None = None,
        smooth: bool = True,
        window_length: int = 7,
        polyorder: int = 1,
    ) -> npt.NDArray[np.float_]:
        r"""Calculate the true airspeed [:math:`m/s`] from the ground speed and horizontal winds.

        The calculated ground speed will first be smoothed with a Savitzky-Golay filter if enabled.

        Parameters
        ----------
        u_wind : npt.NDArray[np.float_], optional
            U wind speed, [:math:`m \ s^{-1}`].
            Defaults to 0 for all waypoints.
        v_wind : npt.NDArray[np.float_], optional
            V wind speed, [:math:`m \ s^{-1}`].
            Defaults to 0 for all waypoints.
        smooth : bool, optional
            Smooth airspeed with Savitzky-Golay filter.
            Defaults to True.
        window_length : int, optional
            Passed directly to :func:`scipy.signal.savgol_filter`, by default 7.
        polyorder : int, optional
            Passed directly to :func:`scipy.signal.savgol_filter`, by default 1.

        Returns
        -------
        npt.NDArray[np.float_]
            True wind speed of each segment, [:math:`m \ s^{-1}`]
        """
        groundspeed = self.segment_groundspeed(smooth, window_length, polyorder)

        if u_wind is None:
            u_wind = np.zeros_like(groundspeed)

        if v_wind is None:
            v_wind = np.zeros_like(groundspeed)

        sin_a, cos_a = self.segment_angle()
        gs_x = groundspeed * cos_a
        gs_y = groundspeed * sin_a
        tas_x = gs_x - u_wind
        tas_y = gs_y - v_wind

        return np.sqrt(tas_x * tas_x + tas_y * tas_y)

    def segment_mach_number(
        self, true_airspeed: np.ndarray, air_temperature: np.ndarray
    ) -> np.ndarray:
        r"""Calculate the mach number of each segment.

        Parameters
        ----------
        true_airspeed : np.ndarray
            True airspeed of the segment, [:math:`m \ s^{-1}`].
            See :meth:`segment_true_airspeed`.
        air_temperature : np.ndarray
            Average air temperature of each segment, [:math:`K`]

        Returns
        -------
        np.ndarray
            Mach number of each segment
        """
        return units.tas_to_mach_number(true_airspeed, air_temperature)

    # ------------
    # Filter/Resample
    # ------------

    def filter_by_first(self) -> Flight:
        """Keep first row of group of waypoints with identical coordinates.

        Chaining this method with `resample_and_fill` often gives a cleaner trajectory
        when using noisy flight waypoints.

        Returns
        -------
        Flight
            Filtered Flight instance

        Examples
        --------
        >>> from datetime import datetime
        >>> import pandas as pd

        >>> df = pd.DataFrame()
        >>> df['longitude'] = [0, 0, 50]
        >>> df['latitude'] = 0
        >>> df['altitude'] = 0
        >>> df['time'] = [datetime(2020, 1, 1, h) for h in range(3)]

        >>> fl = Flight(df)

        >>> fl.filter_by_first().dataframe
           longitude  latitude  altitude                time
        0        0.0       0.0       0.0 2020-01-01 00:00:00
        1       50.0       0.0       0.0 2020-01-01 02:00:00
        """
        df = self.dataframe.groupby(["longitude", "latitude"], sort=False).first().reset_index()
        return Flight(data=df, attrs=self.attrs)

    def resample_and_fill(
        self,
        freq: str = "1T",
        fill_method: str = "geodesic",
        geodesic_threshold: float = 100e3,
        nominal_rocd: float = constants.nominal_rocd,
        drop: bool = True,
    ) -> Flight:
        """Resample and fill flight trajectory with geodesics and linear interpolation.

        Waypoints are resampled according to the frequency `freq`. Values for :attr:`data` columns
        `longitude`, `latitude`, and `altitude` are interpolated.

        Parameters
        ----------
        freq : str, optional
            Resampling frequency, by default "1T"
        fill_method : {"geodesic", "linear"}, optional
            Choose between ``"geodesic"`` and ``"linear"``, by default ``"geodesic"``.
            In geodesic mode, large gaps between waypoints are filled with geodesic
            interpolation and small gaps are filled with linear interpolation. In linear
            mode, all gaps are filled with linear interpolation.
        geodesic_threshold : float, optional
            Threshold for geodesic interpolation, [:math:`m`].
            If the distance between consecutive waypoints is under this threshold,
            values are interpolated linearly.
        nominal_rocd : float | None, optional
            Nominal rate of climb / descent for aircraft type.
            Defaults to :attr:`constants.nominal_rocd`.
        drop : bool, optional
            Drop any columns that are not resampled and filled.
            Defaults to ``True``, dropping all keys outside of "time", "latitude",
            "longitude" and "altitude". If set to False, the extra keys will be
            kept but filled with ``nan`` or ``None`` values, depending on the data type.

        Returns
        -------
        Flight
            Filled Flight

        Raises
        ------
        ValueError
            Unknown `fill_method`

        Examples
        --------
        >>> from datetime import datetime
        >>> import pandas as pd

        >>> df = pd.DataFrame()
        >>> df['longitude'] = [0, 0, 50]
        >>> df['latitude'] = 0
        >>> df['altitude'] = 0
        >>> df['time'] = [datetime(2020, 1, 1, h) for h in range(3)]

        >>> fl = Flight(df)
        >>> fl.dataframe
           longitude  latitude  altitude                time
                   0        0.0       0.0       0.0 2020-01-01 00:00:00
                   1        0.0       0.0       0.0 2020-01-01 01:00:00
                   2       50.0       0.0       0.0 2020-01-01 02:00:00

        >>> fl.resample_and_fill('10T').dataframe  # resample with 10 minute frequency
                          time  longitude  latitude  altitude
        0  2020-01-01 00:00:00   0.000000       0.0       0.0
        1  2020-01-01 00:10:00   0.000000       0.0       0.0
        2  2020-01-01 00:20:00   0.000000       0.0       0.0
        3  2020-01-01 00:30:00   0.000000       0.0       0.0
        4  2020-01-01 00:40:00   0.000000       0.0       0.0
        5  2020-01-01 00:50:00   0.000000       0.0       0.0
        6  2020-01-01 01:00:00   0.000000       0.0       0.0
        7  2020-01-01 01:10:00   8.928571       0.0       0.0
        8  2020-01-01 01:20:00  16.964286       0.0       0.0
        9  2020-01-01 01:30:00  25.892857       0.0       0.0
        10 2020-01-01 01:40:00  33.928571       0.0       0.0
        11 2020-01-01 01:50:00  41.964286       0.0       0.0
        12 2020-01-01 02:00:00  50.000000       0.0       0.0
        """
        methods = "geodesic", "linear"
        if fill_method not in methods:
            raise ValueError(f'Unknown `fill_method`. Supported  methods: {", ".join(methods)}')

        # STEP 1: Prepare DataFrame on which we'll perform resampling
        df = self.to_dataframe()

        # put altitude on dataframe if its not already there
        if "altitude" not in df:
            df["altitude"] = self.altitude

        # always drop level
        if "level" in df:
            df.drop(columns="level", inplace=True)

        # drop all cols except time/lon/lat/alt
        if drop:
            df = df.loc[:, ["time", "longitude", "latitude", "altitude"]]

        # STEP 2: Fill large horizontal gaps with interpolated geodesics
        if fill_method == "geodesic":
            filled = self._geodesic_interpolation(geodesic_threshold)
            if filled is not None:
                df = pd.concat([df, filled])

        # STEP 3: Set the time index, and sort it
        df = df.set_index("time").sort_index()

        # STEP 4: Some adhoc code for dealing with antimeridian.
        # Idea: A flight likel crosses the antimeridian if
        #   `min_pos > 90` and `max_neg < -90`
        # This is not foolproof: it assumes the full trajectory will not
        # span more than 180 longitude degrees. There could be flights that
        # violate this near the poles (but this would be very rare -- flights
        # would instead wrap the other way). For this flights spanning the
        # antimeridian, we translate them to a common "chart" away from the
        # antimeridian (see variable `shift`), then apply the interpolation,
        # then shift back to their original position.
        lon = df["longitude"].to_numpy()
        sign_ = np.sign(lon)
        min_pos = np.min(lon[sign_ == 1], initial=np.inf)
        max_neg = np.max(lon[sign_ == -1], initial=-np.inf)
        if (180 - min_pos) + (180 + max_neg) < 180 and min_pos < np.inf and max_neg > -np.inf:
            # In this case, we believe the flight crosses the antimeridian
            shift = min_pos
            # So we shift the longitude "chart"
            df["longitude"] = (df["longitude"] - shift) % 360
        else:
            shift = None

        # STEP 5: Resample flight to freq
        df = df.resample(freq).first()

        # STEP 6: Linearly interprolate small horizontal gaps and account
        # for previous longitude shift.
        keys = ["latitude", "longitude"]
        df.loc[:, keys] = df.loc[:, keys].interpolate(method="linear")
        if shift is not None:
            # We need to translate back to the original chart here
            df["longitude"] += shift
            df["longitude"] = ((df["longitude"] + 180) % 360) - 180

        # STEP 7: Interpolate nan values in altitude
        if df["altitude"].isna().any():
            df_freq = df.index.freq.delta.to_numpy()
            new_alt = _altitude_interpolation(df["altitude"].to_numpy(), nominal_rocd, df_freq)
            _verify_altitude(new_alt, nominal_rocd, df_freq)
            df["altitude"] = new_alt

        # finally reset index
        df = df.reset_index()
        return Flight(data=df, attrs=self.attrs)

    def _geodesic_interpolation(self, geodesic_threshold: float) -> pd.DataFrame | None:
        """Geodesic interpolate between large gaps between waypoints.

        Parameters
        ----------
        geodesic_threshold : float
            The threshold for large gap, [:math:`m`].

        Returns
        -------
        pd.DataFrame | None
            Generated waypoints to be merged into underlying :attr:`data`.
            Return `None` if no new waypoints are created.

        Raises
        ------
        NotImplementedError
            Raises when attr:`attrs["crs"]` is not EPSG:4326
        """
        if self.attrs["crs"] != "EPSG:4326":
            raise NotImplementedError("Only implemented for EPSG:4326 CRS.")

        # Omit the final nan and ensure index + 1 (below) is well defined
        segs = self.segment_length()[:-1]

        # For default geodesic_threshold, we expect gap_indices to be very
        # sparse (so the for loop below is cheap)
        gap_indices = np.nonzero(segs > geodesic_threshold)[0]
        if not np.any(gap_indices):
            # For most flights, gap_indices is empty. It's more performant
            # to exit now rather than build an empty DataFrame below.
            return None

        geod = pyproj.Geod(ellps="WGS84")
        longitudes: list[float] = []
        latitudes: list[float] = []
        times: list[np.ndarray] = []

        for index in gap_indices:
            lon0, lat0, t0 = (
                self["longitude"][index],
                self["latitude"][index],
                self["time"][index],
            )
            lon1, lat1, t1 = (
                self["longitude"][index + 1],
                self["latitude"][index + 1],
                self["time"][index + 1],
            )
            distance = segs[index]
            n_steps = distance // geodesic_threshold  # number of new waypoints to generate

            # This is the expensive call within the for-loop
            # NOTE: geod.npts does not return the initial or terminal points
            lonlats: list[tuple[float, float]] = geod.npts(lon0, lat0, lon1, lat1, n_steps)

            lons, lats = zip(*lonlats)
            longitudes.extend(lons)
            latitudes.extend(lats)

            # + 1 to denominator to stay consistent with geod.npts (only interior points)
            t_step = (t1 - t0) / (n_steps + 1)
            # substract 0.5 * t_step to ensure round-off error doesn't put final arange point
            # very close to t1
            t_range = np.arange(t0 + t_step, t1 - 0.5 * t_step, t_step)
            times.append(t_range)

        times_ = np.concatenate(times)
        return pd.DataFrame({"longitude": longitudes, "latitude": latitudes, "time": times_})

    # ------------
    # I / O
    # ------------

    def to_geojson_linestring(self) -> dict[str, Any]:
        """Return trajectory as geojson FeatureCollection containing single LineString.

        Returns
        -------
        dict[str, Any]
            Python representation of geojson FeatureCollection
        """
        points = _return_linestring(
            {
                "longitude": self["longitude"],
                "latitude": self["latitude"],
                "altitude": self.altitude,
            }
        )
        geometry = {"type": "LineString", "coordinates": points}
        properties = {
            "start_time": self.time_start.isoformat(),
            "end_time": self.time_end.isoformat(),
        }
        properties.update(self.constants)
        linestring = {"type": "Feature", "geometry": geometry, "properties": properties}

        return {"type": "FeatureCollection", "features": [linestring]}

    def to_geojson_multilinestring(
        self, key: str, split_antimeridian: bool = True
    ) -> dict[str, Any]:
        """Return trajectory as GeoJSON FeatureCollection of MultiLineStrings.

        Flight :attr:`data` is grouped according to values of ``key``. Each group gives rise to a
        Feature containing a MultiLineString geometry. LineStrings can be split over the
        antimeridian.

        Parameters
        ----------
        key : str
            Name of :attr:`data` column to group by
        split_antimeridian : bool, optional
            Split linestrings that cross the antimeridian.
            Defaults to True

        Returns
        -------
        dict[str, Any]
            Python representation of GeoJSON FeatureCollection of MultiLinestring Features

        Raises
        ------
        KeyError
            :attr:`data` does not contain column ``key``
        """
        if key not in self.dataframe.columns:
            raise KeyError(f"Column {key} does not exist in data.")

        jump_index = _antimeridian_index(pd.Series(self["longitude"]), self.attrs["crs"])

        def _group_to_feature(group: pd.DataFrame) -> dict[str, str | dict[str, Any]]:
            subgrouping = group.index.to_series().diff().ne(1).cumsum()
            # additional splitting at antimeridian
            if jump_index in subgrouping and split_antimeridian:
                subgrouping.loc[jump_index:] += 1
            multi_ls = [_return_linestring(g) for _, g in group.groupby(subgrouping)]
            geometry = {"type": "MultiLineString", "coordinates": multi_ls}

            # adding in static properties
            properties: dict[str, Any] = {key: group.name}
            properties.update(self.constants)
            return {"type": "Feature", "geometry": geometry, "properties": properties}

        features = self.dataframe.groupby(key).apply(_group_to_feature).values.tolist()
        return {"type": "FeatureCollection", "features": features}

    def to_traffic(self) -> "traffic.core.Flight":
        """Convert Flight instance to :class:`traffic.core.Flight` instance.

        See https://traffic-viz.github.io/traffic.core.flight.html#traffic.core.Flight

        Returns
        -------
        :class:`traffic.core.Flight`
            `traffic.core.Flight` instance

        Raises
        ------
        ModuleNotFoundError
            `traffic` package not installed
        """
        try:
            import traffic.core
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "This requires the 'traffic' module, which can be installed using "
                "'pip install traffic'. See the installation documentation at "
                "https://traffic-viz.github.io/installation.html for more information."
            ) from e

        return traffic.core.Flight(
            self.to_dataframe(copy=True).rename(columns={"time": "timestamp"})
        )

    # ------------
    # MET
    # ------------

    def length_met(self, key: str, threshold: float = 1.0) -> float:
        """Calculate total horizontal distance where column ``key`` exceeds``threshold``.

        Parameters
        ----------
        key : str
            Column key in :attr:`data`
        threshold : float
            Consider trajectory waypoints whose associated ``key`` value exceeds ``threshold``,
            by default 1.0

        Returns
        -------
        float
            Total distance, [:math:`m`]

        Raises
        ------
        KeyError
            :attr:`data` does not contain column ``key``
        NotImplementedError
            Raises when attr:`attrs["crs"]` is not EPSG:4326

        Examples
        --------
        >>> from datetime import datetime
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pycontrails.datalib.ecmwf import ERA5
        >>> from pycontrails import Flight

        >>> # Get met data
        >>> times = (datetime(2022, 3, 1, 0),  datetime(2022, 3, 1, 3))
        >>> variables = ["air_temperature", "specific_humidity"]
        >>> levels = [300, 250, 200]
        >>> era5 = ERA5(time=times, variables=variables, pressure_levels=levels)
        >>> met = era5.open_metdataset(xr_kwargs=dict(parallel=False))

        >>> # Build flight
        >>> df = pd.DataFrame()
        >>> df['time'] = pd.date_range('2022-03-01T00', '2022-03-01T03', periods=11)
        >>> df['longitude'] = np.linspace(-20, 20, 11)
        >>> df['latitude'] = np.linspace(-20, 20, 11)
        >>> df['altitude'] = np.linspace(9500, 10000, 11)
        >>> fl = Flight(df).resample_and_fill('10S')

        >>> # Intersect and attach
        >>> fl["air_temperature"] = fl.intersect_met(met['air_temperature'])
        >>> fl["air_temperature"]
        array([235.94658, 235.55774, 235.56766, ..., 234.59956, 234.60406,
               234.60846], dtype=float32)

        >>> # Length (in meters) of waypoints whose temperature exceeds 236K
        >>> fl.length_met("air_temperature", threshold=236)
        3587431.887...

        >>> # Proportion (with respect to distance) of waypoints whose temperature exceeds 236K
        >>> fl.proportion_met("air_temperature", threshold=236)
        0.576076...
        """
        if key not in self.data:
            raise KeyError(f"Column {key} does not exist in data.")
        if self.attrs["crs"] != "EPSG:4326":
            raise NotImplementedError("Only implemented for EPSG:4326 CRS.")

        # The column of interest may contain floating point values less than 1.
        # In this case, if the default threshold is not changed, warn the user that the behavior
        # might not be what is expected.

        # Checking if column of interest contains floating point values below 1
        if threshold == 1.0 and ((self[key] > 0) & (self[key] < 1)).any():
            warnings.warn(
                f"Column {key} contains real numbers between 0 and 1. "
                "To include these values in this calculation, change the `threshold` parameter "
                "or modify the underlying DataFrame in place."
            )

        segs = self.segment_length()[:-1]  # lengths between waypoints, dropping off the nan

        # giving each waypoint the average of the segments on either side side
        segs = np.concatenate([segs[:1], (segs[1:] + segs[:-1]) / 2, segs[-1:]])

        # filter by region of interest
        indices = np.where(self[key] >= threshold)[0]

        return np.sum(segs[indices])

    def proportion_met(self, key: str, threshold: float = 1.0) -> float:
        """Calculate proportion of flight with certain meteorological constraint.

        Parameters
        ----------
        key : str
            Column key in :attr:`data`
        threshold : float
            Consider trajectory waypoints whose associated ``key`` value exceeds ``threshold``,
            Defaults to 1.0

        Returns
        -------
        float
            Ratio
        """
        try:
            return self.length_met(key, threshold) / self.length
        except ZeroDivisionError:
            return 0.0

    # ------------
    # Visualization
    # ------------

    def plot(self, **kwargs: Any) -> "matplotlib.axes.Axes":
        """Plot flight trajectory longitude-latitude values.

        Parameters
        ----------
        **kwargs : Any
            Additional plot properties to passed to `pd.DataFrame.plot`

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Plot
        """
        ax = self.dataframe.plot(x="longitude", y="latitude", legend=False, **kwargs)
        ax.set(xlabel="longitude", ylabel="latitude")
        return ax


def _return_linestring(data: dict[str, np.ndarray]) -> list[list[float]]:
    """Return list of coordinates for geojson constructions.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        :attr:`data` containing `longitude`, `latitude`, and `altitude` keys

    Returns
    -------
    list[list[float]]
        The list of coordinates
    """
    # rounding to reduce the size of resultant json arrays
    points = zip(  # pylint: disable=zip-builtin-not-iterating
        np.around(data["longitude"], decimals=4),
        np.around(data["latitude"], decimals=4),
        np.around(data["altitude"], decimals=4),
    )
    return [list(p) for p in points]


def _antimeridian_index(longitude: pd.Series, crs: str = "EPSG:4326") -> int:
    """Return index after flight crosses antimeridian, or -1 if flight does not cross.

    Parameters
    ----------
    longitude : pd.Series
        longitude values with an integer index
    crs : str, optional
        Coordinate Reference system for longitude specified in EPSG format.
        Currently only supports "EPSG:4326" and "EPSG:3857".

    Returns
    -------
    int
        Index after jump or -1

    Raises
    ------
    ValueError
        CRS is not supported.
        Flight crosses antimeridian several times.
    """
    # FIXME: This logic here is somewhat outdated - the _interpolate_altitude
    # method handles this somewhat more reliably
    # This function should get updated to follow the logic there.
    # WGS84
    if crs in ["EPSG:4326"]:
        l1 = (-180.0, -90.0)
        l2 = (90.0, 180.0)

    # pseudo mercator
    elif crs in ["EPSG:3857"]:
        # values calculated through pyproj.Transformer
        l1 = (-20037508.342789244, -10018754.171394622)
        l2 = (10018754.171394622, 20037508.342789244)

    else:
        raise ValueError("CRS must be one of EPSG:4326 or EPSG:3857")

    # TODO: When nans exist, this method *may* not find the meridian
    if np.any(np.isnan(longitude)):
        warnings.warn("Anti-meridian index can't be found accurately with nan values in longitude")

    s1 = longitude.between(*l1)
    s2 = longitude.between(*l2)
    jump12 = longitude[s1 & s2.shift()]
    jump21 = longitude[s1.shift() & s2]
    jump_index = pd.concat([jump12, jump21]).index.to_list()

    if len(jump_index) > 1:
        raise ValueError("Only implemented for trajectories jumping the antimeridian at most once.")
    if len(jump_index) == 1:
        return jump_index[0]

    return -1


def _sg_filter(vals: np.ndarray, window_length: int = 7, polyorder: int = 1) -> np.ndarray:
    """Apply Savitzky-Golay filter to smooth out noise in the time-series data.

    Used to smooth true airspeed & fuel flow.

    TODO: move to a more centralized location in the codebase if used again

    Parameters
    ----------
    vals : np.ndarray
        Input array
    window_length : int, optional
        Parameter for :func:`scipy.signal.savgol_filter`
    polyorder : int, optional
        Parameter for :func:`scipy.signal.savgol_filter`

    Returns
    -------
    np.ndarray
        Smoothed values

    Raises
    ------
    ArithmeticError
        Raised if NaN values input to SG filter
    """
    # The window_length must be less than or equal to the number of data points available.
    window_length = min(window_length, vals.size)

    # The time window_length must be odd.
    if (window_length % 2) == 0:
        window_length -= 1

    # If there is not enough data points to perform smoothing, return the mean ground speed
    if window_length <= polyorder:
        return np.full_like(vals, np.nanmean(vals))

    if np.isnan(vals).any():
        raise ArithmeticError("NaN values not supported by SG filter.")

    return scipy.signal.savgol_filter(vals, window_length, polyorder)


def _altitude_interpolation(
    altitude: np.ndarray, nominal_rocd: float, freq: np.timedelta64
) -> np.ndarray:
    """Interpolate nan values in `altitude` array.

    Suppose each group of consecutive nan values is enclosed by `a0` and `a1` with
    corresponding time values `t0` and `t1` respectively. This function immediately
    climbs or descends starting at `t0` (depending on the sign of `a1 - a0`). Once
    the filled altitude values reach the terminal value `a1`, the remaining nans
    are filled with `a1.

    Parameters
    ----------
    altitude : np.ndarray
        Array of altitude values containing nan values. This function will raise
        an error if `altitude` does not contain nan values. Moreover, this function
        assumes the initial and final entries in `altitude` are not nan.
    nominal_rocd : float
        Nominal rate of climb/descent, in m/s
    freq : np.timedelta64
        Frequency of time index associated to `altitude`.

    Returns
    -------
    np.ndarray
        Altitude after nan values have been filled
    """
    # Determine nan state of altitude
    isna = np.isnan(altitude)
    start_na = ~isna[:-1] & isna[1:]
    start_na = np.append(start_na, False)
    end_na = isna[:-1] & ~isna[1:]
    end_na = np.insert(end_na, 0, False)

    # And get the size of each group of consecutive nan values
    start_na_idxs = start_na.nonzero()[0]
    end_na_idxs = end_na.nonzero()[0]
    na_group_size = end_na_idxs - start_na_idxs

    # Form array of cumulative altitude values if the flight were to climb
    # at nominal_rocd over each group of nan
    cumalt_list = [np.arange(1, size, dtype=float) for size in na_group_size]
    cumalt = np.concatenate(cumalt_list)
    cumalt = cumalt * nominal_rocd * freq / np.timedelta64(1, "s")

    # Expand cumalt to the full size of altitude
    nominal_fill = np.zeros_like(altitude)
    nominal_fill[isna] = cumalt

    # Use pandas to forward and backfill altitude values
    s = pd.Series(altitude)
    s_ff = s.fillna(method="ffill")
    s_bf = s.fillna(method="backfill")

    # Construct altitude values if the flight were to climb / descent throughout
    # group of consecutive nan values. The call to np.minimum / np.maximum cuts
    # the climb / descent off at the terminal altitude of the nan group
    fill_climb = np.minimum(s_ff + nominal_fill, s_bf)
    fill_descent = np.maximum(s_ff - nominal_fill, s_bf)

    # Explicitly determine if the flight is in a climb or descent state
    sign = np.full_like(altitude, np.nan)
    sign[~isna] = np.sign(np.diff(altitude[~isna], append=np.nan))
    sign = pd.Series(sign).fillna(method="ffill")

    # And return the mess
    return np.where(sign == 1, fill_climb, fill_descent)


def _verify_altitude(altitude: np.ndarray, nominal_rocd: float, freq: np.timedelta64) -> None:
    """Confirm that the time derivative of `altitude` does not exceed twice `nominal_rocd`.

    Parameters
    ----------
    altitude : np.ndarray
        Array of filled altitude values containing nan values.
    nominal_rocd : float
        Nominal rate of climb/descent, in m/s
    freq : np.timedelta64
        Frequency of time index associated to `altitude`.
    """
    dalt = np.diff(altitude)
    dt = freq / np.timedelta64(1, "s")
    rocd = np.abs(dalt / dt)
    if np.any(rocd > 2 * nominal_rocd):
        warnings.warn(
            "Rate of climb/descent values greater than nominal "
            f"({nominal_rocd} m/s) after altitude interpolation"
        )
    if np.any(np.isnan(altitude)):
        warnings.warn(
            f"Found nan values altitude after ({nominal_rocd} m/s) after altitude interpolation"
        )


def _dt_waypoints(
    time: npt.NDArray[np.datetime64], dtype: np.dtype = np.dtype(np.float32)
) -> npt.NDArray[np.float_]:
    """Calculate the time difference between waypoints.

    Parameters
    ----------
    time : np.ndarray
        Waypoint time in ``np.datetime64`` format.
    dtype : np.dtype
        Numpy dtype for time difference.
        Defaults to ``np.float64``

    Returns
    -------
    np.ndarray
        Time difference between waypoints, [:math:`s`].
        This returns an array with dtype specified by``dtype``
    """
    out = np.empty_like(time, dtype=dtype)
    out[-1] = np.nan
    out[:-1] = np.diff(time) / np.timedelta64(1, "s")
    return out
