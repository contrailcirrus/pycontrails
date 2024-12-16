"""Flight Data Handling."""

from __future__ import annotations

import enum
import logging
import sys
import warnings
from typing import TYPE_CHECKING, Any, NoReturn

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.signal

from pycontrails.core.fuel import Fuel, JetA
from pycontrails.core.vector import AttrDict, GeoVectorDataset, VectorDataDict, VectorDataset
from pycontrails.physics import constants, geo, units
from pycontrails.utils import dependencies
from pycontrails.utils.types import ArrayOrFloat

logger = logging.getLogger(__name__)

# optional imports
if TYPE_CHECKING:
    import matplotlib.axes
    import traffic.core


class FlightPhase(enum.IntEnum):
    """Flight phase enumeration.

    Use :func:`segment_phase` or :meth:`Flight.segment_phase` to determine flight phase.
    """

    #: Waypoints at which the flight is in a climb phase
    CLIMB = enum.auto()

    #: Waypoints at which the flight is in a cruise phase
    CRUISE = enum.auto()

    #: Waypoints at which the flight is in a descent phase
    DESCENT = enum.auto()

    #: Waypoints at which the flight is not in a climb, cruise, or descent phase.
    #: In practice, this category is used for waypoints at which the ROCD resembles
    #: that of a cruise phase, but the altitude is below the minimum cruise altitude.
    LEVEL_FLIGHT = enum.auto()

    #: Waypoints at which the ROCD is not defined.
    NAN = enum.auto()


#: Max airport elevation, [:math:`ft`]
#: See `Daocheng_Yading_Airport <https://en.wikipedia.org/wiki/Daocheng_Yading_Airport>`_
MAX_AIRPORT_ELEVATION = 15_000.0

#: Min estimated cruise altitude, [:math:`ft`]
MIN_CRUISE_ALTITUDE = 20_000.0

#: Short haul duration cutoff, [:math:`s`]
SHORT_HAUL_DURATION = 3600.0

#: Set maximum speed compatible with "on_ground" indicator, [:math:`mph`]
#: Thresholds assessed based on scatter plot (150 knots = 278 km/h)
MAX_ON_GROUND_SPEED = 150.0


class Flight(GeoVectorDataset):
    """A single flight trajectory.

    Expect latitude-longitude coordinates in WGS 84.
    Expect altitude in [:math:`m`].
    Expect pressure level (`level`) in [:math:`hPa`].

    Parameters
    ----------
    data : dict[str, np.ndarray] | pd.DataFrame | VectorDataDict | VectorDataset | None
        Flight trajectory waypoints as data dictionary or :class:`pandas.DataFrame`.
        Must include columns ``time``, ``latitude``, ``longitude``, ``altitude`` or ``level``.
        Keyword arguments for ``time``, ``latitude``, ``longitude``, ``altitude`` or ``level``
        will override ``data`` inputs. Expects ``altitude`` in meters and ``time`` as a
        DatetimeLike (or array that can processed with :func:`pd.to_datetime`).
        Additional waypoint-specific data can be included as additional keys/columns.
    longitude : npt.ArrayLike, optional
        Flight trajectory waypoint longitude.
        Defaults to None.
    latitude : npt.ArrayLike, optional
        Flight trajectory waypoint latitude.
        Defaults to None.
    altitude : npt.ArrayLike, optional
        Flight trajectory waypoint altitude, [:math:`m`].
        Defaults to None.
    altitude_ft : npt.ArrayLike, optional
        Flight trajectory waypoint altitude, [:math:`ft`].
    level : npt.ArrayLike, optional
        Flight trajectory waypoint pressure level, [:math:`hPa`].
        Defaults to None.
    time : npt.ArrayLike, optional
        Flight trajectory waypoint time.
        Defaults to None.
    attrs : dict[str, Any], optional
        Additional flight properties as a dictionary.
        While different models may utilize Flight attributes differently,
        pycontrails applies the following conventions:

        - ``flight_id``: An internal flight identifier. Used internally
          for :class:`Fleet` interoperability.
        - ``aircraft_type``: Aircraft type ICAO, e.g. ``"A320"``.
        - ``wingspan``: Aircraft wingspan, [:math:`m`].
        - ``n_engine``: Number of aircraft engines.
        - ``engine_uid``: Aircraft engine unique identifier. Used for emissions
          calculations with the ICAO Aircraft Emissions Databank (EDB).
        - ``max_mach_number``: Maximum Mach number at cruise altitude. Used by
          some aircraft performance models to clip true airspeed.
        - ``load_factor``: The load factor used in determining the aircraft's
          take-off weight. Used by some aircraft performance models.

        Numeric quantities that are constant over the entire flight trajectory
        should be included as attributes.
    copy : bool, optional
        Copy data on Flight creation.
        Defaults to True.
    fuel : Fuel, optional
        Fuel used in flight trajectory. Defaults to :class:`JetA`.
    drop_duplicated_times : bool, optional
        Drop duplicate times in flight trajectory. Defaults to False.
    **attrs_kwargs : Any
        Additional flight properties passed as keyword arguments.

    Raises
    ------
    KeyError
        Raises if ``data`` input does not contain at least ``time``, ``latitude``, ``longitude``,
        (``altitude`` or ``level``).

    Notes
    -----
    The `Traffic <https://traffic-viz.github.io/index.html>`_ library has many helpful
    flight processing utilities.

    See :class:`traffic.core.Flight` for more information.

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
    Flight [4 keys x 500 length, 1 attributes]
    Keys: longitude, latitude, altitude, time
    Attributes:
    time                [2021-01-01 10:00:00, 2021-01-01 15:00:00]
    longitude           [20.0, 30.0]
    latitude            [10.0, 40.0]
    altitude            [10500.0, 10500.0]
    flight_id           123

    >>> # Create `Flight` from keywords
    >>> fl = Flight(
    ...     longitude=np.linspace(20, 30, 200),
    ...     latitude=np.linspace(40, 30, 200),
    ...     altitude=11000 * np.ones(200),
    ...     time=pd.date_range('2021-01-01T12', '2021-01-01T14', periods=200),
    ... )
    >>> fl
    Flight [4 keys x 200 length, 0 attributes]
        Keys: longitude, latitude, time, altitude
        Attributes:
        time                [2021-01-01 12:00:00, 2021-01-01 14:00:00]
        longitude           [20.0, 30.0]
        latitude            [30.0, 40.0]
        altitude            [11000.0, 11000.0]

    >>> # Access the underlying data as DataFrame
    >>> fl.dataframe.head()
       longitude   latitude                          time  altitude
    0  20.000000  40.000000 2021-01-01 12:00:00.000000000   11000.0
    1  20.050251  39.949749 2021-01-01 12:00:36.180904522   11000.0
    2  20.100503  39.899497 2021-01-01 12:01:12.361809045   11000.0
    3  20.150754  39.849246 2021-01-01 12:01:48.542713567   11000.0
    4  20.201005  39.798995 2021-01-01 12:02:24.723618090   11000.0
    """

    __slots__ = ("fuel",)

    #: Fuel used in flight trajectory
    fuel: Fuel

    def __init__(
        self,
        data: (
            dict[str, npt.ArrayLike] | pd.DataFrame | VectorDataDict | VectorDataset | None
        ) = None,
        *,
        longitude: npt.ArrayLike | None = None,
        latitude: npt.ArrayLike | None = None,
        altitude: npt.ArrayLike | None = None,
        altitude_ft: npt.ArrayLike | None = None,
        level: npt.ArrayLike | None = None,
        time: npt.ArrayLike | None = None,
        attrs: dict[str, Any] | AttrDict | None = None,
        copy: bool = True,
        fuel: Fuel | None = None,
        drop_duplicated_times: bool = False,
        **attrs_kwargs: Any,
    ) -> None:
        super().__init__(
            data=data,
            longitude=longitude,
            latitude=latitude,
            altitude=altitude,
            altitude_ft=altitude_ft,
            level=level,
            time=time,
            attrs=attrs,
            copy=copy,
            **attrs_kwargs,
        )

        # Set fuel - fuel instance is NOT copied
        self.fuel = fuel or JetA()

        # Check flight data for possible errors
        if np.any(self.altitude > 16000.0):
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

        time_diff = np.diff(self["time"])

        # Ensure that time is sorted
        if self and np.any(time_diff < np.timedelta64(0)):
            if not copy:
                raise ValueError(
                    "The 'time' array must be sorted if 'copy=False' on creation. "
                    "Set copy=False, or sort data before creating Flight."
                )
            warnings.warn("Sorting Flight data by time.")
            self.data = GeoVectorDataset(self, copy=False).sort("time").data

            # Update time_diff ... we use it again below
            time_diff = np.diff(self["time"])

        # Check for duplicate times. If dropping duplicates,
        # keep the *first* occurrence of each time.
        duplicated_times = time_diff == np.timedelta64(0)
        if self and np.any(duplicated_times):
            if drop_duplicated_times:
                mask = np.insert(duplicated_times, 0, False)
                filtered_flight = self.filter(~mask, copy=False)
                self.data = filtered_flight.data
            else:
                warnings.warn(
                    f"Flight contains {duplicated_times.sum()} duplicate times. "
                    "This will cause errors with segment-based methods. Set "
                    "'drop_duplicated_times=True' or call the 'resample_and_fill' method."
                )

    @override
    def copy(self, **kwargs: Any) -> Self:
        kwargs.setdefault("fuel", self.fuel)
        return super().copy(**kwargs)

    @override
    def filter(self, mask: npt.NDArray[np.bool_], copy: bool = True, **kwargs: Any) -> Self:
        kwargs.setdefault("fuel", self.fuel)
        return super().filter(mask, copy=copy, **kwargs)

    @override
    def sort(self, by: str | list[str]) -> NoReturn:
        msg = (
            "Flight.sort is not implemented. A Flight instance is automatically sorted "
            "by 'time' on creation. To force sorting, create a GeoVectorDataset instance "
            "and call the 'sort' method."
        )
        raise ValueError(msg)

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
        np.float64(7391.27...)
        """
        return self.segment_length()[:-1].max()

    @property
    def length(self) -> float:
        """Return flight length based on WGS84 geodesic.

        Returns
        -------
        float
            Total flight length, [:math:`m`]

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
        np.float64(1436924.67...)
        """
        # drop off the nan
        return np.nansum(self.segment_length()[:-1])

    # ------------
    # Segment Properties
    # ------------

    def segment_duration(self, dtype: npt.DTypeLike = np.float32) -> npt.NDArray[np.floating]:
        r"""Compute time elapsed between waypoints in seconds.

        ``np.nan`` appended so the length of the output is the same as number of waypoints.

        Parameters
        ----------
        dtype : np.dtype
            Numpy dtype for time difference.
            Defaults to ``np.float64``

        Returns
        -------
        npt.NDArray[np.floating]
            Time difference between waypoints, [:math:`s`].
            Returns an array with dtype specified by``dtype``
        """

        return segment_duration(self.data["time"], dtype=dtype)

    def segment_haversine(self) -> npt.NDArray[np.floating]:
        """Compute Haversine (great circle) distance between flight waypoints.

        Helper function used in :meth:`resample_and_fill`.
        `np.nan` appended so the length of the output is the same as number of waypoints.

        To account for vertical displacements when computing segment lengths,
        use :meth:`segment_length`.

        Returns
        -------
        npt.NDArray[np.floating]
            Array of great circle distances in [:math:`m`] between waypoints

        Examples
        --------
        >>> from pycontrails import Flight
        >>> fl = Flight(
        ... longitude=np.array([1, 2, 3, 5, 8]),
        ... latitude=np.arange(5),
        ... altitude=np.full(shape=(5,), fill_value=11000),
        ... time=pd.date_range('2021-01-01T12', '2021-01-01T14', periods=5),
        ... )
        >>> fl.segment_haversine()
        array([157255.03346286, 157231.08336815, 248456.48781503, 351047.44358851,
                           nan])

        See Also
        --------
        :func:`segment_haversine`
        :meth:`segment_length`
        """
        return geo.segment_haversine(self["longitude"], self["latitude"])

    def segment_length(self) -> npt.NDArray[np.floating]:
        """Compute spherical distance between flight waypoints.

        Helper function used in :meth:`length` and :meth:`length_met`.
        `np.nan` appended so the length of the output is the same as number of waypoints.

        Returns
        -------
        npt.NDArray[np.floating]
            Array of distances in [:math:`m`] between waypoints

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
        return geo.segment_length(self["longitude"], self["latitude"], self.altitude)

    def segment_angle(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Calculate sine and cosine for the angle between each segment and the longitudinal axis.

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
        npt.NDArray[np.floating], npt.NDArray[np.floating]
            Returns ``sin(a), cos(a)``, where ``a`` is the angle between the segment and the
            longitudinal axis. The final values are of both arrays are ``np.nan``.

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

    def segment_azimuth(self) -> npt.NDArray[np.floating]:
        """Calculate (forward) azimuth at each waypoint.

        Method calls `pyproj.Geod.inv`, which is slow. See `geo.forward_azimuth`
        for an outline of a faster implementation.

        .. versionchanged:: 0.33.7

            The dtype of the output now matches the dtype of ``self["longitude"]``.

        Returns
        -------
        npt.NDArray[np.floating]
            Array of azimuths.

        See Also
        --------
        :meth:`segment_angle`
        :func:`geo.forward_azimuth`
        """
        lon = self["longitude"]
        lat = self["latitude"]

        lons1 = lon[:-1]
        lats1 = lat[:-1]
        lons2 = lon[1:]
        lats2 = lat[1:]

        try:
            import pyproj
        except ModuleNotFoundError as exc:
            dependencies.raise_module_not_found_error(
                name="Flight.segment_azimuth method",
                package_name="pyproj",
                module_not_found_error=exc,
                pycontrails_optional_package="pyproj",
            )

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
    ) -> npt.NDArray[np.floating]:
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
        npt.NDArray[np.floating]
            Groundspeed of the segment, [:math:`m s^{-1}`]
        """
        # get horizontal distance (altitude is ignored)
        horizontal_segment_length = geo.segment_haversine(self["longitude"], self["latitude"])

        # time between waypoints, in seconds
        dt_sec = self.segment_duration(dtype=horizontal_segment_length.dtype)

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
        u_wind: npt.NDArray[np.floating] | float = 0.0,
        v_wind: npt.NDArray[np.floating] | float = 0.0,
        smooth: bool = True,
        window_length: int = 7,
        polyorder: int = 1,
    ) -> npt.NDArray[np.floating]:
        r"""Calculate the true airspeed [:math:`m/s`] from the ground speed and horizontal winds.

        The calculated ground speed will first be smoothed with a Savitzky-Golay filter if enabled.

        Parameters
        ----------
        u_wind : npt.NDArray[np.floating] | float
            U wind speed, [:math:`m \ s^{-1}`].
            Defaults to 0 for all waypoints.
        v_wind : npt.NDArray[np.floating] | float
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
        npt.NDArray[np.floating]
            True wind speed of each segment, [:math:`m \ s^{-1}`]
        """
        groundspeed = self.segment_groundspeed(smooth, window_length, polyorder)

        sin_a, cos_a = self.segment_angle()
        gs_x = groundspeed * cos_a
        gs_y = groundspeed * sin_a
        tas_x = gs_x - u_wind
        tas_y = gs_y - v_wind

        return np.sqrt(tas_x * tas_x + tas_y * tas_y)

    def segment_mach_number(
        self, true_airspeed: npt.NDArray[np.floating], air_temperature: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        r"""Calculate the mach number of each segment.

        Parameters
        ----------
        true_airspeed : npt.NDArray[np.floating]
            True airspeed of the segment, [:math:`m \ s^{-1}`].
            See :meth:`segment_true_airspeed`.
        air_temperature : npt.NDArray[np.floating]
            Average air temperature of each segment, [:math:`K`]

        Returns
        -------
        npt.NDArray[np.floating]
            Mach number of each segment
        """
        return units.tas_to_mach_number(true_airspeed, air_temperature)

    def segment_rocd(
        self,
        air_temperature: None | npt.NDArray[np.floating] = None,
    ) -> npt.NDArray[np.floating]:
        """Calculate the rate of climb and descent (ROCD).

        Parameters
        ----------
        air_temperature: None | npt.NDArray[np.floating]
            Air temperature of each flight waypoint, [:math:`K`]

        Returns
        -------
        npt.NDArray[np.floating]
            Rate of climb and descent over segment, [:math:`ft min^{-1}`]

        See Also
        --------
        :func:`segment_rocd`
        """
        return segment_rocd(self.segment_duration(), self.altitude_ft, air_temperature)

    def segment_phase(
        self,
        threshold_rocd: float = 250.0,
        min_cruise_altitude_ft: float = 20000.0,
        air_temperature: None | npt.NDArray[np.floating] = None,
    ) -> npt.NDArray[np.uint8]:
        """Identify the phase of flight (climb, cruise, descent) for each segment.

        Parameters
        ----------
        threshold_rocd : float, optional
            ROCD threshold to identify climb and descent, [:math:`ft min^{-1}`].
            Currently set to 250 ft/min.
        min_cruise_altitude_ft : float, optional
            Minimum altitude for cruise, [:math:`ft`]
            This is specific for each aircraft type,
            and can be approximated as 50% of the altitude ceiling.
            Defaults to 20000 ft.
        air_temperature: None | npt.NDArray[np.floating]
            Air temperature of each flight waypoint, [:math:`K`]

        Returns
        -------
        npt.NDArray[np.uint8]
            Array of values enumerating the flight phase.
            See :attr:`flight.FlightPhase` for enumeration.

        See Also
        --------
        :attr:`FlightPhase`
        :func:`segment_phase`
        :func:`segment_rocd`
        """
        return segment_phase(
            self.segment_rocd(air_temperature),
            self.altitude_ft,
            threshold_rocd=threshold_rocd,
            min_cruise_altitude_ft=min_cruise_altitude_ft,
        )

    # ------------
    # Filter/Resample
    # ------------

    def filter_by_first(self) -> Self:
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
        return type(self)(data=df, attrs=self.attrs, fuel=self.fuel)

    def resample_and_fill(
        self,
        freq: str = "1min",
        fill_method: str = "geodesic",
        geodesic_threshold: float = 100e3,
        nominal_rocd: float = constants.nominal_rocd,
        drop: bool = True,
        keep_original_index: bool = False,
    ) -> Self:
        """Resample and fill flight trajectory with geodesics and linear interpolation.

        Waypoints are resampled according to the frequency ``freq``. Values for :attr:`data`
        columns ``longitude``, ``latitude``, and ``altitude`` are interpolated.

        Resampled waypoints will include all multiples of ``freq`` between the flight
        start and end time. For example, when resampling to a frequency of 1 minute,
        a flight that starts at 2020/1/1 00:00:59 and ends at 2020/1/1 00:01:01
        will return a single waypoint at 2020/1/1 00:01:00, whereas a flight that
        starts at 2020/1/1 00:01:01 and ends at 2020/1/1 00:01:59 will return an empty
        flight.

        Parameters
        ----------
        freq : str, optional
            Resampling frequency, by default "1min"
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
        keep_original_index : bool, optional
            Keep the original index of the :class:`Flight` in addition to the new
            resampled index. Defaults to ``False``.
            .. versionadded:: 0.45.2

        Returns
        -------
        Flight
            Filled Flight

        Raises
        ------
        ValueError
            Unknown ``fill_method``

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

        >>> fl.resample_and_fill('10min').dataframe  # resample with 10 minute frequency
            longitude  latitude  altitude                time
        0    0.000000       0.0       0.0 2020-01-01 00:00:00
        1    0.000000       0.0       0.0 2020-01-01 00:10:00
        2    0.000000       0.0       0.0 2020-01-01 00:20:00
        3    0.000000       0.0       0.0 2020-01-01 00:30:00
        4    0.000000       0.0       0.0 2020-01-01 00:40:00
        5    0.000000       0.0       0.0 2020-01-01 00:50:00
        6    0.000000       0.0       0.0 2020-01-01 01:00:00
        7    8.333333       0.0       0.0 2020-01-01 01:10:00
        8   16.666667       0.0       0.0 2020-01-01 01:20:00
        9   25.000000       0.0       0.0 2020-01-01 01:30:00
        10  33.333333       0.0       0.0 2020-01-01 01:40:00
        11  41.666667       0.0       0.0 2020-01-01 01:50:00
        12  50.000000       0.0       0.0 2020-01-01 02:00:00
        """
        methods = "geodesic", "linear"
        if fill_method not in methods:
            raise ValueError(f"Unknown `fill_method`. Supported  methods: {', '.join(methods)}")

        # STEP 0: If self is empty, return an empty flight
        if not self:
            warnings.warn("Flight instance is empty.")
            return self.copy()

        # STEP 1: Prepare DataFrame on which we'll perform resampling
        df = self.dataframe

        # put altitude on dataframe if its not already there
        if "altitude" not in df:
            df["altitude"] = self.altitude

        # always drop level
        if "level" in df:
            df = df.drop(columns="level")

        # drop all cols except time/lon/lat/alt
        if drop:
            df = df.loc[:, ["time", "longitude", "latitude", "altitude"]]

        # STEP 2: Fill large horizontal gaps with interpolated geodesics
        if fill_method == "geodesic":
            filled = self._geodesic_interpolation(geodesic_threshold)
            if filled is not None:
                df = pd.concat([df, filled])

        # STEP 3: Set the time index, and sort it
        df = df.set_index("time", verify_integrity=True).sort_index()

        # STEP 4: handle antimeridian crossings
        # For flights spanning the antimeridian, we translate them to a
        # common "chart" away from the antimeridian (see variable `shift`),
        # then apply the interpolation, then shift back to their original position.
        shift = self._antimeridian_shift()
        if shift is not None:
            df["longitude"] = (df["longitude"] - shift) % 360.0

        # STEP 5: Resample flight to freq
        # Save altitudes to copy over - these just get rounded down in time.
        # Also get target sample indices
        df, t = _resample_to_freq(df, freq)

        if shift is not None:
            # We need to translate back to the original chart here
            df["longitude"] = ((df["longitude"] + shift + 180.0) % 360.0) - 180.0

        # STEP 6: Interpolate nan values in altitude
        altitude = df["altitude"].to_numpy()
        time = df.index.to_numpy()
        if np.any(np.isnan(altitude)):
            df["altitude"] = _altitude_interpolation(altitude, time, nominal_rocd)

        # Remove original index if requested
        if not keep_original_index:
            df = df.loc[t]

        # finally reset index
        df = df.reset_index()
        if df.empty:
            msg = "Method 'resample_and_fill' returns in an empty Flight."
            if not keep_original_index:
                msg = f"{msg} Pass 'keep_original_index=True' to keep the original index."
            warnings.warn(msg)

        # Reorder columns (this is unimportant but makes the output more canonical)
        coord_names = ("longitude", "latitude", "altitude", "time")
        df = df[[*coord_names, *[c for c in df.columns if c not in set(coord_names)]]]

        data = {k: v.to_numpy() for k, v in df.items()}
        return type(self)._from_fastpath(data, attrs=self.attrs, fuel=self.fuel)

    def clean_and_resample(
        self,
        freq: str = "1min",
        fill_method: str = "geodesic",
        geodesic_threshold: float = 100e3,
        nominal_rocd: float = constants.nominal_rocd,
        kernel_size: int = 17,
        cruise_threshold: float = 120.0,
        force_filter: bool = False,
        drop: bool = True,
        keep_original_index: bool = False,
    ) -> Self:
        """Resample and (possibly) filter a flight trajectory.

        Waypoints are resampled according to the frequency ``freq``. If the original
        flight data has a short sampling period, `filter_altitude` will also be called
        to clean the data.  Large gaps in trajectories may be interpolated as step climbs
        through `_altitude_interpolation`.

        Parameters
        ----------
        freq : str, optional
            Resampling frequency, by default "1min"
        fill_method : {"geodesic", "linear"}, optional
            Choose between ``"geodesic"`` and ``"linear"``, by default ``"geodesic"``.
            In geodesic mode, large gaps between waypoints are filled with geodesic
            interpolation and small gaps are filled with linear interpolation. In linear
            mode, all gaps are filled with linear interpolation.
        geodesic_threshold : float, optional
            Threshold for geodesic interpolation, [:math:`m`].
            If the distance between consecutive waypoints is under this threshold,
            values are interpolated linearly.
        nominal_rocd : float, optional
            Nominal rate of climb / descent for aircraft type.
            Defaults to :attr:`constants.nominal_rocd`.
        kernel_size : int, optional
            Passed directly to :func:`scipy.signal.medfilt`, by default 11.
            Passed also to :func:`scipy.signal.medfilt`
        cruise_theshold : float, optional
            Minimal length of time, in seconds, for a flight to be in cruise to apply median filter
        force_filter: bool, optional
            If set to true, meth:`filter_altitude` will always be called. otherwise, it will only
            be called if the flight has a median sample period under 10 seconds
        drop : bool, optional
            Drop any columns that are not resampled and filled.
            Defaults to ``True``, dropping all keys outside of "time", "latitude",
            "longitude" and "altitude". If set to False, the extra keys will be
            kept but filled with ``nan`` or ``None`` values, depending on the data type.
        keep_original_index : bool, optional
            Keep the original index of the :class:`Flight` in addition to the new
            resampled index. Defaults to ``False``.
            .. versionadded:: 0.45.2

        Returns
        -------
        Flight
            Filled Flight
        """
        clean_flight: Flight
        # If the flight has a large sampling period, don't try to smooth it unless requested
        seg_duration = self.segment_duration()
        median_gap = np.nanmedian(seg_duration)
        if (median_gap > 10.0) and (not force_filter):
            return self.resample_and_fill(
                freq,
                fill_method,
                geodesic_threshold,
                nominal_rocd,
                drop,
                keep_original_index,
            )

        # If the flight has large gap(s), then call resample and fill, then filter altitude
        max_gap = np.max(seg_duration)
        if max_gap > 300.0:
            # Ignore warning in intermediate resample
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="^.*greater than nominal.*$")
                clean_flight = self.resample_and_fill(
                    "1s",
                    fill_method,
                    geodesic_threshold,
                    nominal_rocd,
                    drop,
                    keep_original_index,
                )
            clean_flight = clean_flight.filter_altitude(kernel_size, cruise_threshold)
        else:
            clean_flight = self.filter_altitude(kernel_size, cruise_threshold)

        # Resample to requested rate and return
        return clean_flight.resample_and_fill(
            freq,
            fill_method,
            geodesic_threshold,
            nominal_rocd,
            drop,
            keep_original_index,
        )

    def filter_altitude(
        self,
        kernel_size: int = 17,
        cruise_threshold: float = 120.0,
    ) -> Self:
        """
        Filter noisy altitude on a single flight.

        Currently runs altitude through a median filter using :func:`scipy.signal.medfilt`
        with ``kernel_size``, then a Savitzky-Golay filter to filter noise. The median filter
        is only applied during cruise segments that are longer than ``cruise_threshold``.

        Parameters
        ----------
        kernel_size : int, optional
            Passed directly to :func:`scipy.signal.medfilt`, by default 11.
            Passed also to :func:`scipy.signal.medfilt`
        cruise_theshold : float, optional
            Minimal length of time, in seconds, for a flight to be in cruise to apply median filter

        Returns
        -------
        Flight
            Filtered Flight

        Notes
        -----
        Algorithm is derived from :meth:`traffic.core.Flight.filter`.

        The `traffic
        <https://traffic-viz.github.io/api_reference/traffic.core.flight.html#traffic.core.Flight.filter>`_
        algorithm also computes thresholds on sliding windows
        and replaces unacceptable values with NaNs.

        Errors may raised if the ``kernel_size`` is too large.

        See Also
        --------
        :meth:`traffic.core.flight.Flight.filter`
        :func:`scipy.signal.medfilt`
        """
        out = self.copy()
        altitude_ft_filtered = filter_altitude(
            self["time"], self.altitude_ft, kernel_size, cruise_threshold
        )
        out.update(altitude_ft=altitude_ft_filtered)
        out.data.pop("altitude", None)  # avoid any ambiguity
        out.data.pop("level", None)  # avoid any ambiguity
        return out

    def distance_to_coords(
        self: Flight, distance: ArrayOrFloat
    ) -> tuple[
        ArrayOrFloat,
        ArrayOrFloat,
        np.intp | npt.NDArray[np.intp],
    ]:
        """
        Convert distance along flight path to geodesic coordinates.

        Will return a tuple containing `(lat, lon, index)`, where index indicates which flight
        segment contains the returned coordinate.

        Parameters
        ----------
        distance : ArrayOrFloat
            Distance along flight path, [:math:`m`]

        Returns
        -------
        (ArrayOrFloat, ArrayOrFloat, int | npt.NDArray[int])
            latitude, longitude, and segment index cooresponding to distance.
        """

        # Check if flight crosses antimeridian line
        # If it does, shift longitude chart to remove jump
        lon_ = self["longitude"]
        lat_ = self["latitude"]
        shift = self._antimeridian_shift()
        if shift is not None:
            lon_ = (lon_ - shift) % 360.0

        # Make a fake flight that flies at constant height so distance is just
        # distance traveled across groud
        flat_dataset = Flight(
            longitude=self.coords["longitude"],
            latitude=self.coords["latitude"],
            time=self.coords["time"],
            level=[self.coords["level"][0] for _ in range(self.size)],
        )

        lengths = flat_dataset.segment_length()
        cumulative_lengths = np.nancumsum(lengths)
        cumulative_lengths = np.insert(cumulative_lengths[:-1], 0, 0)
        seg_idx: np.intp | npt.NDArray[np.intp]

        if isinstance(distance, float):
            seg_idx = np.argmax(cumulative_lengths > distance)
        else:
            seg_idx = np.argmax(cumulative_lengths > distance.reshape((distance.size, 1)), axis=1)

        # If in the last segment (which has length 0), then just return the last waypoint
        seg_idx -= 1

        # linear interpolation in lat/lon - assuming the way points are within 100-200km so this
        # should be accurate enough without needed to reproject or use spherical distance
        lat1: ArrayOrFloat = lat_[seg_idx]
        lon1: ArrayOrFloat = lon_[seg_idx]
        lat2: ArrayOrFloat = lat_[seg_idx + 1]
        lon2: ArrayOrFloat = lon_[seg_idx + 1]

        dx = distance - cumulative_lengths[seg_idx]
        fx = dx / lengths[seg_idx]
        lat = (1 - fx) * lat1 + fx * lat2
        lon = (1 - fx) * lon1 + fx * lon2

        if isinstance(distance, float):
            if distance < 0:
                lat = np.nan
                lon = np.nan
                seg_idx = np.intp(0)
            elif distance >= cumulative_lengths[-1]:
                lat = lat_[-1]
                lon = lon_[-1]
                seg_idx = np.intp(self.size - 1)
        else:
            lat[distance < 0] = np.nan
            lon[distance < 0] = np.nan
            seg_idx[distance < 0] = 0  # type: ignore

            lat[distance >= cumulative_lengths[-1]] = lat_[-1]
            lon[distance >= cumulative_lengths[-1]] = lon_[-1]
            seg_idx[distance >= cumulative_lengths[-1]] = self.size - 1  # type: ignore

        if shift is not None:
            # We need to translate back to the original chart here
            lon += shift
            lon = ((lon + 180.0) % 360.0) - 180.0

        return lat, lon, seg_idx

    def _antimeridian_shift(self) -> float | None:
        """Determine shift required for resampling trajectories that cross antimeridian.

        Because flights sometimes span more than 180 degree longitude (for example,
        when flight-level winds favor travel in a specific direction, typically eastward),
        antimeridian crossings cannot reliably be detected by looking only at minimum
        and maximum longitudes.

        Instead, this function checks each flight segment for an antimeridian crossing,
        and if it finds one returns the coordinate of a meridian that is not crossed by
        the flight.

        Returns
        -------
        float | None
            Longitude shift for handling antimeridian crossings, or None if the
            flight does not cross the antimeridian.
        """

        # logic for detecting crossings is consistent with _antimeridian_crossing,
        # but implementation is separate to keep performance costs as low as possible
        lon = self["longitude"]
        if np.any(np.isnan(lon)):
            warnings.warn("Anti-meridian crossings can't be reliably detected with nan longitudes")

        s1 = (lon >= -180) & (lon <= -90)
        s2 = (lon <= 180) & (lon >= 90)
        jump12 = s1[:-1] & s2[1:]  # westward
        jump21 = s2[:-1] & s1[1:]  # eastward
        if not np.any(jump12 | jump21):
            return None

        # separate flight into segments that are east and west of crossings
        net_westward = np.insert(np.cumsum(jump12.astype(int) - jump21.astype(int)), 0, 0)
        max_westward = net_westward.max()
        if max_westward - net_westward.min() > 1:
            msg = "Cannot handle consecutive antimeridian crossings in the same direction"
            raise ValueError(msg)
        east = (net_westward == 0) if max_westward == 1 else (net_westward == -1)

        # shift must be between maximum longitude east of crossings
        # and minimum longitude west of crossings
        shift_min = np.nanmax(lon[east])
        shift_max = np.nanmin(lon[~east])
        if shift_min >= shift_max:
            msg = "Cannot handle flight that spans more than 360 degrees longitude"
            raise ValueError(msg)
        return (shift_min + shift_max) / 2

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
        """
        # Omit the final nan and ensure index + 1 (below) is well defined
        segs = self.segment_haversine()[:-1]

        # For default geodesic_threshold, we expect gap_indices to be very
        # sparse (so the for loop below is cheap)
        gap_indices = np.flatnonzero(segs > geodesic_threshold)
        if gap_indices.size == 0:
            # For most flights, gap_indices is empty. It's more performant
            # to exit now rather than build an empty DataFrame below.
            return None

        try:
            import pyproj
        except ModuleNotFoundError as exc:
            dependencies.raise_module_not_found_error(
                name="Flight._geodesic_interpolation method",
                package_name="pyproj",
                module_not_found_error=exc,
                pycontrails_optional_package="pyproj",
            )

        geod = pyproj.Geod(ellps="WGS84")
        longitudes: list[float] = []
        latitudes: list[float] = []
        times: list[np.ndarray] = []

        longitude = self["longitude"]
        latitude = self["latitude"]
        time = self["time"]

        for index in gap_indices:
            lon0 = longitude[index]
            lat0 = latitude[index]
            t0 = time[index]
            lon1 = longitude[index + 1]
            lat1 = latitude[index + 1]
            t1 = time[index + 1]

            distance = segs[index]
            n_steps = distance // geodesic_threshold  # number of new waypoints to generate

            # This is the expensive call within the for-loop
            # NOTE: geod.npts does not return the initial or terminal points
            lonlats: list[tuple[float, float]] = geod.npts(lon0, lat0, lon1, lat1, n_steps)

            lons, lats = zip(*lonlats, strict=True)
            longitudes.extend(lons)
            latitudes.extend(lats)

            # + 1 to denominator to stay consistent with geod.npts (only interior points)
            t_step = (t1 - t0) / (n_steps + 1.0)

            # subtract 0.5 * t_step to ensure round-off error doesn't put final arange point
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
        self, key: str | None = None, split_antimeridian: bool = True
    ) -> dict[str, Any]:
        """Return trajectory as GeoJSON FeatureCollection of MultiLineStrings.

        If `key` is provided, Flight :attr:`data` is grouped according to values of ``key``.
        Each group gives rise to a Feature containing a MultiLineString geometry.
        Each MultiLineString can optionally be split over the antimeridian.

        Parameters
        ----------
        key : str, optional
            If provided, name of :attr:`data` column to group by.
        split_antimeridian : bool, optional
            Split linestrings that cross the antimeridian. Defaults to True.

        Returns
        -------
        dict[str, Any]
            Python representation of GeoJSON FeatureCollection of MultiLinestring Features

        Raises
        ------
        KeyError
            ``key`` is provided but :attr:`data` does not contain column ``key``
        """
        if key is not None and key not in self.dataframe.columns:
            raise KeyError(f"Column {key} does not exist in data.")

        jump_indices = _antimeridian_index(pd.Series(self["longitude"]))

        def _group_to_feature(name: str, group: pd.DataFrame) -> dict[str, str | dict[str, Any]]:
            # assigns a different value to each group of consecutive indices
            subgrouping = group.index.to_series().diff().ne(1).cumsum()

            # increments values after antimeridian crossings
            if split_antimeridian:
                for jump_index in jump_indices:
                    if jump_index in subgrouping:
                        subgrouping.loc[jump_index:] += 1

            # creates separate linestrings for sets of points
            # - with non-consecutive indices
            # - before and after antimeridian crossings
            multi_ls = [_return_linestring(g) for _, g in group.groupby(subgrouping)]
            geometry = {"type": "MultiLineString", "coordinates": multi_ls}

            # adding in static properties
            properties: dict[str, Any] = {key: name} if key is not None else {}
            properties.update(self.constants)
            return {"type": "Feature", "geometry": geometry, "properties": properties}

        if key is not None:
            groups = self.dataframe.groupby(key)
        else:
            # create a single group containing all rows of dataframe
            groups = self.dataframe.groupby(lambda _: 0)

        features = [_group_to_feature(*name_group) for name_group in groups]
        return {"type": "FeatureCollection", "features": features}

    def to_traffic(self) -> traffic.core.Flight:
        """Convert to :class:`traffic.core.Flight` instance.

        Returns
        -------
        traffic.core.Flight
            traffic flight instance

        Raises
        ------
        ModuleNotFoundError
            `traffic` package not installed

        See Also
        --------
        :class:`traffic.core.Flight`
        """
        try:
            import traffic.core
        except ModuleNotFoundError as e:
            dependencies.raise_module_not_found_error(
                name="Flight.to_traffic method",
                package_name="traffic",
                module_not_found_error=e,
            )

        return traffic.core.Flight(
            self.to_dataframe(copy=True).rename(columns={"time": "timestamp"})
        )

    # ------------
    # MET
    # ------------

    def length_met(self, key: str, threshold: float = 1.0) -> float:
        """Calculate total horizontal distance where column ``key`` exceeds ``threshold``.

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
        >>> met = era5.open_metdataset()

        >>> # Build flight
        >>> df = pd.DataFrame()
        >>> df["time"] = pd.date_range("2022-03-01T00", "2022-03-01T03", periods=11)
        >>> df["longitude"] = np.linspace(-20, 20, 11)
        >>> df["latitude"] = np.linspace(-20, 20, 11)
        >>> df["altitude"] = np.linspace(9500, 10000, 11)
        >>> fl = Flight(df).resample_and_fill("10s")

        >>> # Intersect and attach
        >>> fl["air_temperature"] = fl.intersect_met(met["air_temperature"])
        >>> fl["air_temperature"]
        array([235.94657007, 235.55745645, 235.56709768, ..., 234.59917962,
               234.60387402, 234.60845312], shape=(1081,))

        >>> # Length (in meters) of waypoints whose temperature exceeds 236K
        >>> fl.length_met("air_temperature", threshold=236)
        np.float64(3589705.998...)

        >>> # Proportion (with respect to distance) of waypoints whose temperature exceeds 236K
        >>> fl.proportion_met("air_temperature", threshold=236)
        np.float64(0.576...)
        """
        if key not in self.data:
            raise KeyError(f"Column {key} does not exist in data.")

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

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
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
        kwargs.setdefault("legend", False)
        ax = self.dataframe.plot(x="longitude", y="latitude", **kwargs)
        ax.set(xlabel="longitude", ylabel="latitude")
        return ax

    def plot_profile(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """Plot flight trajectory time-altitude values.

        Parameters
        ----------
        **kwargs : Any
            Additional plot properties to passed to `pd.DataFrame.plot`

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Plot
        """
        kwargs.setdefault("legend", False)
        df = self.dataframe.assign(altitude_ft=self.altitude_ft)
        ax = df.plot(x="time", y="altitude_ft", **kwargs)
        ax.set(xlabel="time", ylabel="altitude_ft")
        return ax


def _return_linestring(data: dict[str, npt.NDArray[np.floating]]) -> list[list[float]]:
    """Return list of coordinates for geojson constructions.

    Parameters
    ----------
    data : dict[str, npt.NDArray[np.floating]]
        :attr:`data` containing `longitude`, `latitude`, and `altitude` keys

    Returns
    -------
    list[list[float]]
        The list of coordinates
    """
    # rounding to reduce the size of resultant json arrays
    points = zip(
        np.round(data["longitude"], decimals=4),
        np.round(data["latitude"], decimals=4),
        np.round(data["altitude"], decimals=4),
        strict=True,
    )
    return [list(p) for p in points]


def _antimeridian_index(longitude: pd.Series) -> list[int]:
    """Return indices after flight crosses antimeridian, or an empty list if flight does not cross.

    This function assumes EPSG:4326 coordinates.

    Parameters
    ----------
    longitude : pd.Series
        longitude values with an integer index

    Returns
    -------
    list[int]
        Indices after jump, or empty list of flight does not cross antimeridian.
    """
    l1 = (-180.0, -90.0)
    l2 = (90.0, 180.0)

    # TODO: When nans exist, this method *may* not find the meridian
    if np.any(np.isnan(longitude)):
        warnings.warn("Anti-meridian index can't be found accurately with nan values in longitude")

    s1 = longitude.between(*l1)
    s2 = longitude.between(*l2)
    jump12 = longitude[s1 & s2.shift()]
    jump21 = longitude[s1.shift() & s2]
    return pd.concat([jump12, jump21]).index.to_list()


def _sg_filter(
    vals: npt.NDArray[np.floating], window_length: int = 7, polyorder: int = 1
) -> npt.NDArray[np.floating]:
    """Apply Savitzky-Golay filter to smooth out noise in the time-series data.

    Used to smooth true airspeed, fuel flow, and altitude.

    Parameters
    ----------
    vals : npt.NDArray[np.floating]
        Input array
    window_length : int, optional
        Parameter for :func:`scipy.signal.savgol_filter`
    polyorder : int, optional
        Parameter for :func:`scipy.signal.savgol_filter`

    Returns
    -------
    npt.NDArray[np.floating]
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
    altitude: npt.NDArray[np.floating],
    time: npt.NDArray[np.datetime64],
    nominal_rocd: float,
    minimum_cruise_altitude_ft: float = 20000.0,
    assumed_cruise_altitude_ft: float = 30000.0,
) -> npt.NDArray[np.floating]:
    """Interpolate nan values in ``altitude`` array.

    Suppose each group of consecutive nan values is enclosed by ``a0`` and ``a1`` with
    corresponding time values ``t0`` and ``t1`` respectively. For segments under two hours,
    this function immediately climbs  starting at ``t0``, or for descents, descents at the end
    the segment so ``a1`` is met at ``t1``. For segments greater than two hours, a descent will
    still occur at the end of the segment, but climbs will start halfway between ``t0`` and
    ``t1``.

    Parameters
    ----------
    altitude : npt.NDArray[np.floating]
        Array of altitude values containing nan values. This function will raise
        an error if ``altitude`` does not contain nan values. Moreover, this function
        assumes the initial and final entries in ``altitude`` are not nan, [:math:`m`]
    time : npt.NDArray[np.datetime64]
        Timestamp at each waypoint. Must be monotonically increasing.
    nominal_rocd : float
        Nominal rate of climb/descent, [:math:`m s^{-1}`]
    minimum_cruise_altitude_ft : float
        Minimium cruising altitude for a given aircraft type, [:math:`ft`].
        By default, this is 20000.0 ft.
    assumed_cruise_altitude_ft : float
        Assumed cruising altitude for a given aircraft type, [:math:`ft`].
        By default, this is 30000.0 ft.

    Returns
    -------
    npt.NDArray[np.floating]
        Altitude after nan values have been filled, [:math:`m`]

    Notes
    -----
    Default values for ``minimum_cruise_altitude_ft`` and ``assumed_cruise_altitude_ft`` should be
    provided if aircraft-specific parameters are available to improve the output quality.

    We can assume ``minimum_cruise_altitude_ft`` as 0.5 times the aircraft service ceiling, and
    ``assumed_cruise_altitude_ft`` as 0.8 times the aircraft service ceiling.

    Assume that aircraft will generally prefer to climb to a higher altitude as early as possible,
    and descent to a lower altitude as late as possible, because a higher altitude can reduce
    drag and fuel consumption.
    """
    # Work in units of feet
    alt_ft = units.m_to_ft(altitude)
    nominal_rocd_ft_min = units.m_to_ft(nominal_rocd) * 60.0

    # Determine nan state of altitude
    isna = np.isnan(alt_ft)

    start_na = np.empty(alt_ft.size, dtype=bool)
    start_na[:-1] = ~isna[:-1] & isna[1:]
    start_na[-1] = False

    end_na = np.empty(alt_ft.size, dtype=bool)
    end_na[0] = False
    end_na[1:] = isna[:-1] & ~isna[1:]

    # And get the size of each group of consecutive nan values
    start_na_idxs = np.flatnonzero(start_na)
    end_na_idxs = np.flatnonzero(end_na)
    na_group_size = end_na_idxs - start_na_idxs

    # NOTE: Only fill altitude gaps that require special attention
    # At the end of this for loop, those with NaN altitudes will be filled with pd.interpolate
    for i in range(len(na_group_size)):
        alt_ft_start = alt_ft[start_na_idxs[i]]
        alt_ft_end = alt_ft[end_na_idxs[i]]
        time_start = time[start_na_idxs[i]]
        time_end = time[end_na_idxs[i]]

        # Calculate parameters to determine how to interpolate altitude
        # Time to next waypoint
        dt_next = (time_end - time_start) / np.timedelta64(1, "m")

        # (1): Unrealistic scenario: first and next known waypoints are at very
        # low altitudes with a large time gap.
        is_unrealistic = (
            dt_next > 60.0
            and alt_ft_start < minimum_cruise_altitude_ft
            and alt_ft_end < minimum_cruise_altitude_ft
        )

        # If unrealistic, assume flight will climb to cruise altitudes (0.8 * max_altitude_ft),
        # stay there, and then descent to the next known waypoint
        if is_unrealistic:
            # Add altitude at top of climb
            alt_ft_cruise = assumed_cruise_altitude_ft
            d_alt_start = alt_ft_cruise - alt_ft_start
            dt_climb = int(np.ceil(d_alt_start / nominal_rocd_ft_min))
            t_cruise_start = time_start + np.timedelta64(dt_climb, "m")
            idx_cruise_start = np.searchsorted(time, t_cruise_start) + 1
            alt_ft[idx_cruise_start] = alt_ft_cruise

            # Add altitude at top of descent
            d_alt_end = alt_ft_cruise - alt_ft_end
            dt_descent = int(np.ceil(d_alt_end / nominal_rocd_ft_min))
            t_cruise_end = time_end - np.timedelta64(dt_descent, "m")
            idx_cruise_end = np.searchsorted(time, t_cruise_end) - 1
            alt_ft[idx_cruise_end] = alt_ft_cruise
            continue

        # (2): If both altitudes are the same, then skip entire operations below
        if alt_ft_start == alt_ft_end:
            continue

        # Rate of climb and descent to next waypoint, in ft/min
        rocd_next = (alt_ft_end - alt_ft_start) / dt_next

        # (3): If cruise over 2 h with small altitude change, set change to mid-point
        is_long_segment_small_altitude_change = (
            dt_next > 120.0
            and rocd_next < 500.0
            and rocd_next > -500.0
            and alt_ft_start > minimum_cruise_altitude_ft
            and alt_ft_end > minimum_cruise_altitude_ft
        )

        if is_long_segment_small_altitude_change:
            mid_na_idx = int(0.5 * (start_na_idxs[i] + end_na_idxs[i]))
            alt_ft[mid_na_idx] = alt_ft_start
            alt_ft[mid_na_idx + 1] = alt_ft_end
            continue

        # (4): Climb at the start until target altitude and level off if:
        #:  (i) large time gap (`dt_next` > 20 minutes) and positive `rocd`, or
        #:  (ii) shallow climb (0 < `rocd_next` < 500 ft/min) between current and next waypoint

        #: For (i), we only perform this for large time gaps, because we do not want the aircraft to
        #: constantly climb, level off, and repeat, while the ADS-B waypoints show that it is
        #: continuously climbing
        if (dt_next > 20.0 and rocd_next > 0.0) or (0.0 < rocd_next < 500.0):
            dt_climb = int(np.ceil((alt_ft_end - alt_ft_start) / nominal_rocd_ft_min * 60))
            t_climb_complete = time_start + np.timedelta64(dt_climb, "s")
            idx_climb_complete = np.searchsorted(time, t_climb_complete)

            #: [Safeguard for very small `dt_next`] Ensure climb can be performed within the
            #: interpolated time step. If False, then aircraft will climb between waypoints instead
            #: of levelling off.
            if start_na_idxs[i] < idx_climb_complete < end_na_idxs[i]:
                alt_ft[idx_climb_complete] = alt_ft_end

            continue

        # (5):  Descent towards the end until target altitude and level off if:
        #:  (i) large time gap (`dt_next` > 20 minutes) and negative `rocd`, or
        #:  (ii) shallow descent (-250 < `rocd_next` < 0 ft/min) between current and next waypoint
        if (dt_next > 20.0 and rocd_next < 0.0) or (-250.0 < rocd_next < 0.0):
            dt_descent = int(np.ceil((alt_ft_start - alt_ft_end) / nominal_rocd_ft_min * 60))
            t_descent_start = time_end - np.timedelta64(dt_descent, "s")
            idx_descent_start = np.where(
                -250.0 < rocd_next < 0.0,
                np.searchsorted(time, t_descent_start) - 1,
                np.searchsorted(time, t_descent_start),
            )

            #: [Safeguard for very small `dt_next`] Ensure descent can be performed within the
            #: interpolated time step. If False, then aircraft will descent between waypoints
            #: instead of levelling off.
            if start_na_idxs[i] < idx_descent_start < end_na_idxs[i]:
                alt_ft[idx_descent_start] = alt_ft_start

            continue

    # Linearly interpolate between remaining nan values
    out_alt_ft = pd.Series(alt_ft, index=time).interpolate(method="index")
    return units.ft_to_m(out_alt_ft.to_numpy())


def filter_altitude(
    time: npt.NDArray[np.datetime64],
    altitude_ft: npt.NDArray[np.floating],
    kernel_size: int = 17,
    cruise_threshold: float = 120,
    air_temperature: None | npt.NDArray[np.floating] = None,
) -> npt.NDArray[np.floating]:
    """
    Filter noisy altitude on a single flight.

    Currently runs altitude through a median filter using :func:`scipy.signal.medfilt`
    with ``kernel_size``, then a Savitzky-Golay filter to filter noise. The median filter
    is only applied during cruise segments that are longer than ``cruise_threshold``.

    Parameters
    ----------
    time : npt.NDArray[np.datetime64]
        Waypoint time in ``np.datetime64`` format.
    altitude_ft : npt.NDArray[np.floating]
        Altitude signal in feet
    kernel_size : int, optional
        Passed directly to :func:`scipy.signal.medfilt`, by default 11.
        Passed also to :func:`scipy.signal.medfilt`
    cruise_theshold : int, optional
        Minimal length of time, in seconds, for a flight to be in cruise to apply median filter
    air_temperature: None | npt.NDArray[np.floating]
        Air temperature of each flight waypoint, [:math:`K`]

    Returns
    -------
    npt.NDArray[np.floating]
        Filtered altitude

    Notes
    -----
    Algorithm is derived from :meth:`traffic.core.Flight.filter`.

    The `traffic filter algorithm
    <https://traffic-viz.github.io/api_reference/traffic.core.flight.html#traffic.core.Flight.filter>`_
    also computes thresholds on sliding windows
    and replaces unacceptable values with NaNs.

    Errors may raised if the ``kernel_size`` is too large.

    See Also
    --------
    :meth:`traffic.core.Flight.filter`
    :func:`scipy.signal.medfilt`
    """
    if not len(altitude_ft):
        raise ValueError("Altitude must have non-zero length to filter")

    # The kernel_size must be less than or equal to the number of data points available.
    kernel_size = min(kernel_size, altitude_ft.size)

    # The kernel_size must be odd.
    if (kernel_size % 2) == 0:
        kernel_size -= 1

    # Apply a median filter above a certain threshold
    altitude_filt = scipy.signal.medfilt(altitude_ft, kernel_size=kernel_size)

    # Apply Savitzky-Golay filter
    altitude_filt = _sg_filter(altitude_filt, window_length=kernel_size)

    # Remove noise manually
    # only remove above max airport elevation
    d_alt_ft = np.diff(altitude_filt, append=np.nan)
    is_noise = (np.abs(d_alt_ft) <= 25.0) & (altitude_filt > MAX_AIRPORT_ELEVATION)
    altitude_filt[is_noise] = np.round(altitude_filt[is_noise], -3)

    # Find cruise phase in filtered profile
    seg_duration = segment_duration(time)
    seg_rocd = segment_rocd(seg_duration, altitude_filt, air_temperature)
    seg_phase = segment_phase(seg_rocd, altitude_filt)
    is_cruise = seg_phase == FlightPhase.CRUISE

    # Compute cumulative segment time in cruise segments
    v = np.nan_to_num(seg_duration)
    v[~is_cruise] = 0.0
    n = v == 0.0
    c = np.cumsum(v)
    d = np.diff(c[n], prepend=0.0)
    v[n] = -d
    cruise_duration = np.cumsum(v)

    # Find cruise segment start and end indices
    not_cruise = cruise_duration == 0.0

    end_cruise = np.empty(cruise_duration.size, dtype=bool)
    end_cruise[:-1] = ~not_cruise[:-1] & not_cruise[1:]
    # if last sample is in cruise, last sample of end_cruise marks the end of a cruise segment
    end_cruise[-1] = ~not_cruise[-1]

    start_cruise = np.empty(cruise_duration.size, dtype=bool)
    # if first sample is in cruise, first sample of start_cruise marks start of a segment
    start_cruise[0] = ~not_cruise[0]
    start_cruise[1:] = not_cruise[:-1] & ~not_cruise[1:]

    start_idxs = np.flatnonzero(start_cruise)
    end_idxs = np.flatnonzero(end_cruise)

    # Threshold for min cruise segment
    long_mask = cruise_duration[end_idxs] > cruise_threshold
    start_idxs = start_idxs[long_mask]
    end_idxs = end_idxs[long_mask]

    result = np.copy(altitude_ft)
    if np.any(start_idxs):
        for i0, i1 in zip(start_idxs, end_idxs, strict=True):
            result[i0:i1] = altitude_filt[i0:i1]

    # reapply Savitzky-Golay filter to smooth climb and descent
    return _sg_filter(result, window_length=kernel_size)


def segment_duration(
    time: npt.NDArray[np.datetime64], dtype: npt.DTypeLike = np.float32
) -> npt.NDArray[np.floating]:
    """Calculate the time difference between waypoints.

    ``np.nan`` appended so the length of the output is the same as number of waypoints.

    Parameters
    ----------
    time : npt.NDArray[np.datetime64]
        Waypoint time in ``np.datetime64`` format.
    dtype : np.dtype
        Numpy dtype for time difference.
        Defaults to ``np.float32``

    Returns
    -------
    npt.NDArray[np.floating]
        Time difference between waypoints, [:math:`s`].
        This returns an array with dtype specified by ``dtype``.
    """
    out = np.empty_like(time, dtype=dtype)
    out[-1] = np.nan
    out[:-1] = np.diff(time) / np.timedelta64(1, "s")
    return out


def segment_phase(
    rocd: npt.NDArray[np.floating],
    altitude_ft: npt.NDArray[np.floating],
    *,
    threshold_rocd: float = 250.0,
    min_cruise_altitude_ft: float = MIN_CRUISE_ALTITUDE,
) -> npt.NDArray[np.uint8]:
    """Identify the phase of flight (climb, cruise, descent) for each segment.

    Parameters
    ----------
    rocd: pt.NDArray[np.float64]
        Rate of climb and descent across segment, [:math:`ft min^{-1}`].
        See output from :func:`segment_rocd`.
    altitude_ft: npt.NDArray[np.floating]
        Altitude, [:math:`ft`]
    threshold_rocd: float, optional
        ROCD threshold to identify climb and descent, [:math:`ft min^{-1}`].
        Defaults to 250 ft/min.
    min_cruise_altitude_ft: float, optional
        Minimum threshold altitude for cruise, [:math:`ft`]
        This is specific for each aircraft type,
        and can be approximated as 50% of the altitude ceiling.
        Defaults to :attr:`MIN_CRUISE_ALTITUDE`.

    Returns
    -------
    npt.NDArray[np.uint8]
        Array of values enumerating the flight phase.
        See :attr:`flight.FlightPhase` for enumeration.

    Notes
    -----
    Flight data derived from ADS-B and radar sources could contain noise leading
    to small changes in altitude and ROCD. Hence, an arbitrary ``threshold_rocd``
    is specified to identify the different phases of flight.

    The flight phase "level-flight" is when an aircraft is holding at lower altitudes.
    The cruise phase of flight only occurs above a certain threshold altitude.

    See Also
    --------
    :attr:`FlightPhase`
    :func:`segment_rocd`
    """
    nan = np.isnan(rocd)
    cruise = (
        (rocd < threshold_rocd) & (rocd > -threshold_rocd) & (altitude_ft > min_cruise_altitude_ft)
    )
    climb = ~cruise & (rocd > 0.0)
    descent = ~cruise & (rocd < 0.0)
    level_flight = ~(nan | cruise | climb | descent)

    phase = np.empty(rocd.shape, dtype=np.uint8)
    phase[cruise] = FlightPhase.CRUISE
    phase[climb] = FlightPhase.CLIMB
    phase[descent] = FlightPhase.DESCENT
    phase[level_flight] = FlightPhase.LEVEL_FLIGHT
    phase[nan] = FlightPhase.NAN

    return phase


def segment_rocd(
    segment_duration: npt.NDArray[np.floating],
    altitude_ft: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """Calculate the rate of climb and descent (ROCD).

    Parameters
    ----------
    segment_duration: npt.NDArray[np.floating]
        Time difference between waypoints, [:math:`s`].
        Expected to have numeric ``dtype``, not ``np.timedelta64``.
        See output from :func:`segment_duration`.
    altitude_ft: npt.NDArray[np.floating]
        Altitude of each waypoint, [:math:`ft`]
    air_temperature: npt.NDArray[np.floating] | None
        Air temperature of each flight waypoint, [:math:`K`]

    Returns
    -------
    npt.NDArray[np.floating]
        Rate of climb and descent over segment, [:math:`ft min^{-1}`]

    Notes
    -----
    The hydrostatic equation will be used to estimate the ROCD if ``air_temperature`` is provided.
    This will improve the accuracy of the estimated ROCD with a temperature correction. The
    estimated ROCD with the temperature correction are expected to differ by up to +-5% compared to
    those without the correction. These differences are important when the ROCD estimates are used
    as inputs to aircraft performance models.

    See Also
    --------
    segment_duration
    """
    dt_min = segment_duration / 60.0

    out = np.empty_like(altitude_ft)
    out[:-1] = np.diff(altitude_ft) / dt_min[:-1]
    out[-1] = np.nan

    if air_temperature is None:
        return out

    altitude_m = units.ft_to_m(altitude_ft)
    T_isa = units.m_to_T_isa(altitude_m)

    T_correction = np.empty_like(altitude_ft)
    T_correction[:-1] = (air_temperature[:-1] + air_temperature[1:]) / (T_isa[:-1] + T_isa[1:])
    T_correction[-1] = np.nan

    return T_correction * out  # type: ignore[return-value]


def _resample_to_freq(df: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Resample a DataFrame to a given frequency.

    This function is used to resample a DataFrame to a given frequency. The new
    index will include all the original index values and the new resampled-to-freq
    index values. The "longitude" and "latitude" columns will be linearly interpolated
    to the new index values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to resample. Assumed to have a :class:`pd.DatetimeIndex`
        and "longitude" and "latitude" columns.
    freq : str
        Frequency to resample to. See :func:`pd.DataFrame.resample` for
        valid frequency strings.

    Returns
    -------
    tuple[pd.DataFrame, pd.DatetimeIndex]
        Resampled DataFrame and the new index.
    """

    # Manually create a new index that includes all the original index values
    # and the resampled-to-freq index values.
    t0 = df.index[0].ceil(freq)
    t1 = df.index[-1]
    t = pd.date_range(t0, t1, freq=freq, name="time")

    concat_arr = np.concatenate([df.index, t])
    concat_arr = np.unique(concat_arr)
    concat_index = pd.DatetimeIndex(concat_arr, name="time", copy=False)

    out = df.reindex(concat_index)

    # Linearly interpolate small horizontal gap
    coords = ["longitude", "latitude"]
    out.loc[:, coords] = out.loc[:, coords].interpolate(method="index")

    return out, t
