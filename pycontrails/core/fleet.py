"""A single data structure encompassing a sequence of :class:`Flight` instances."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd
from overrides import overrides

from pycontrails.core.flight import Flight
from pycontrails.core.fuel import Fuel, JetA
from pycontrails.core.vector import GeoVectorDataset


class Fleet(Flight):
    """Data structure for holding a sequence of `Flight` instances.

    Flight waypoints are merged into a single `Flight`-like object.
    """

    def __init__(
        self,
        data: dict[str, npt.ArrayLike] | None = None,
        longitude: npt.ArrayLike | None = None,
        latitude: npt.ArrayLike | None = None,
        altitude: npt.ArrayLike | None = None,
        level: npt.ArrayLike | None = None,
        time: npt.ArrayLike | None = None,
        attrs: dict[str, Any] | None = None,
        copy: bool = True,
        fuel: Fuel | None = None,
        **attrs_kwargs: Any,
    ) -> None:
        # Do not want to call Flight.__init__
        # The Flight constructor assumes a sorted time column
        GeoVectorDataset.__init__(
            self,
            data=data,
            longitude=longitude,
            latitude=latitude,
            altitude=altitude,
            level=level,
            time=time,
            attrs=attrs,
            copy=copy,
            **attrs_kwargs,
        )

        if "fl_attrs" not in self.attrs:
            raise ValueError("Require key 'fl_attrs' in Fleet attrs.")

        # set fuel for ALL FLIGHTS
        # NOTE: We could also set fuel for each flight via the `fl_attrs` attribute
        if fuel is None:
            self.fuel = JetA()
        else:
            self.fuel = fuel

        self.final_waypoints = self.calc_final_waypoints()

    def calc_final_waypoints(self) -> npt.NDArray[np.bool_]:
        """Validate data and calculate the final waypoint of each flight.

        Returns
        -------
        npt.NDArray[np.bool_]
            A boolean array in which True values correspond to final waypoint of each flight.

        Raises
        ------
        KeyError, ValueError
            Fleet :attr:`data` does not take the expected form.
        """
        try:
            flight_id = self["flight_id"]
        except KeyError:
            raise KeyError("Expect `flight_id` in Fleet data.") from None

        # Some pandas groupby magic to ensure flights are arranged in blocks
        df = pd.DataFrame({"flight_id": flight_id}).reset_index()
        grouped = df.groupby("flight_id", sort=False)
        groups = grouped.agg({"flight_id": "size", "index": ["first", "last"]})
        expected_size = groups[("index", "last")] - groups[("index", "first")] + 1
        actual_size = groups[("flight_id", "size")]
        if not (expected_size == actual_size).all():
            raise ValueError("Fleet must have contiguous waypoints blocks with constant flight_id")

        # Return boolean array of final waypoints by flight
        final_waypoints = np.zeros(self.size, dtype=bool)
        final_waypoint_indices = groups[("index", "last")].to_numpy()
        final_waypoints[final_waypoint_indices] = True
        return final_waypoints

    def fit_altitude(
        self,
        max_segments: int = 30,
        pop: int = 3,
        r2_target: float = 0.999,
        max_cruise_rocd: float = 10,
        sg_window: int = 7,
        sg_polyorder: int = 1,
    ) -> Fleet:
        """Use piecewise linear fitting to smooth a flight profile.

        Fit a flight profile to a series of line segments. Segments that have a
        small rocd will be set to have a slope of zero and snapped to the
        nearest thousand foot level.  A Savitzky-Golay filter will then be
        applied to the profile to smooth the climbs and descents.  This filter
        works best for high frequency flight data, sampled at a 1-3 second
        sampling period.

        Parameters
        ----------
        max_segments : int, optional
            The maximum number of line segements to fit to the flight profile.
        pop: int, optional
            Population parameter used for the stocastic optimization routine
            used to fit the flight profile.
        r2_target: float, optional
            Target r^2 value for solver. Solver will continue to add line
            segments until the resulting r^2 value is greater than this.
        max_cruise_rocd: float, optional
            The maximum ROCD for a segment that will be forced to a slope of
            zero, [:math:`ft s^{-1}`]
        sg_window: int, optional
            Parameter for :func:`scipy.signal.savgol_filter`
        sg_polyorder: int, optional
            Parameter for :func:`scipy.signal.savgol_filter`

        Returns
        -------
        Fleet
            Smoothed flight
        """
        raise NotImplementedError("Only implemented for Flight instances")

    @classmethod
    def from_seq(
        cls,
        seq: Iterable[Flight],
        broadcast_numeric: bool = True,
        copy: bool = True,
        attrs: dict[str, Any] | None = None,
    ) -> Fleet:
        """Instantiate a `Fleet` instance from an iterable of Flight.

        The entire sequence ``seq`` is cast to a list, and so it's preferable for
        the parameter ``seq`` to come in as a list.

        Parameters
        ----------
        seq : Iterable[Flight]
            An iterable of `Flight` instances.
        broadcast_numeric : bool, optional
            If True, broadcast numeric attributes to data variables.
        copy : bool, optional
            If True, make copy of each flight instance in ``seq``.
        attrs : dict[str, Any] | None, optional
            Global attribute to attach to instance. Must NOT contain the "fl_attrs"
            key (this is automatically deduced when iterating over ``seq``).

        Returns
        -------
        Fleet
            A `Fleet` instance made from concatenating the :class:`Flight`
            instances in ``seq``.
        """
        if attrs is None:
            attrs = {}
        if "fl_attrs" in attrs:
            raise ValueError("Parameter 'attrs' cannot contain 'fl_attrs' key.")
        attrs["fl_attrs"] = {}
        attrs.setdefault("data_keys", None)
        attrs.setdefault("crs", None)

        if copy:
            seq = [fl.copy() for fl in seq]
        else:
            seq = list(seq)

        for fl in seq:
            cls._validate_fl(fl, seq[0].fuel, attrs, broadcast_numeric)

        # Surprisingly, it's faster to call np.concatenate on each variable
        # separately then it is to call pd.concat on the sequence of fl.dataframe.
        # We do this concatenation below.
        data = {var: np.concatenate([fl[var] for fl in seq]) for var in attrs["data_keys"]}
        return cls(data=data, attrs=attrs, copy=False)

    @staticmethod
    def _validate_fl(
        fl: Flight, fuel: Fuel, attrs: dict[str, Any], broadcast_numeric: bool
    ) -> None:
        """Attach "flight_id" and "waypoint" columns to flight :attr:`data`.

        Mutates parameters `fl` and `attrs` in place.

        Parameters
        ----------
        fl : Flight
            Flight instance to process.
        fuel : Fuel
            Fuel used all flights
        attrs : dict[str, Any]
            Dictionary of `Fleet` attributes. Attributes belonging to `fl` are attached
            to `attrs`.
        broadcast_numeric : bool
            If True, broadcast numeric attributes to data variables.

        Raises
        ------
        KeyError
            `fl` does not have a `flight_id` in :attr:`attrs`.
        ValueError
            If `flight_id` is duplicated or incompatible CRS found.
        AttributeError
            If `fuel` is incompatible with other flights.
        """
        # Validate and cache attrs
        if "flight_id" not in fl.attrs:
            raise KeyError("Each flight must have `flight_id` in its `attrs`.")
        flight_id = fl.attrs["flight_id"]

        if flight_id in attrs["fl_attrs"]:
            raise ValueError(f"Duplicate `flight_id` {flight_id} found.")
        attrs["fl_attrs"][flight_id] = fl.attrs

        # Verify fuel type is consistent across flights
        if fl.fuel != fuel:
            raise AttributeError(
                f"Fuel type on Flight {flight_id} ({fl.fuel.fuel_name}) "
                f"is not inconsistent with previous flights ({fuel.fuel_name}). "
                "The `fuel` attributes must be consistent between flights in a Fleet."
            )

        # Verify CRS
        crs = fl.attrs["crs"]
        if attrs["crs"] is None:
            attrs["crs"] = crs
        elif attrs["crs"] != crs:
            raise ValueError(f"Incompatible CRS {attrs['crs']} and {crs} found among flights.")

        # Expand data
        if broadcast_numeric:
            fl.broadcast_numeric_attrs(ignore_keys=["load_factor"])
        if "waypoint" not in fl:
            fl["waypoint"] = np.arange(fl.size)
        if "flight_id" not in fl:
            fl["flight_id"] = np.full(fl.size, flight_id)

        # Verify consistent data keys
        # When attaching data_keys to attrs, use set(data_keys)
        # But, we don't need to create this set separately for each flight
        # Simply use fl.data.keys() to check for consistency.
        data_keys = fl.data.keys()
        if attrs["data_keys"] is None:
            attrs["data_keys"] = set(data_keys)
        elif attrs["data_keys"] != data_keys:
            raise ValueError(
                f"Inconsistent data keys {attrs['data_keys']}  and {data_keys} found among flights."
            )

    @property
    def n_flights(self) -> int:
        """Return number of distinct flights.

        Returns
        -------
        int
            Number of flights
        """
        return len(self.attrs["fl_attrs"])

    def to_flight_list(self, copy: bool = True) -> list[Flight]:
        """De-concatenate merged waypoints into a list of Flight instances.

        Any global :attr:`attrs` are lost.

        Parameters
        ----------
        copy : bool, optional
            If True, make copy of each flight instance in `seq`.

        Returns
        -------
        list[Flight]
            List of Flights in the same order as was passed into the `Fleet` instance.
        """
        grouped = self.dataframe.groupby("flight_id", sort=False)
        return [
            Flight(df, attrs=self.attrs["fl_attrs"][flight_id], copy=copy)
            for flight_id, df in grouped
        ]

    ###################################
    # Flight methods involving segments
    ###################################

    def segment_true_airspeed(
        self,
        u_wind: npt.NDArray[np.float_] | float = 0.0,
        v_wind: npt.NDArray[np.float_] | float = 0.0,
        smooth: bool = True,
        window_length: int = 7,
        polyorder: int = 1,
    ) -> npt.NDArray[np.float_]:
        """Calculate the true airspeed [:math:`m / s`] from the ground speed and horizontal winds.

        Because Flight.segment_true_airspeed uses a smoothing pattern, waypoints in :attr:`data`
        are not independent. Moreover, we expect the final waypoint of each flight to have a nan
        value associated to any segment property. Consequently, we need to define a custom method
        here to deal with these issues when applying this method on a fleet of flights.

        See docstring for :meth:`Flight.segment_true_airspeed`.

        Raises
        ------
        RuntimeError
            Unexpected key `__u_wind` or `__v_wind` found in :attr:`data`.
        """
        if isinstance(u_wind, np.ndarray):
            # Choosing a key we don't think exists
            key = "__u_wind"
            if key in self:
                raise RuntimeError(f"Unexpected key {key} found")
            self[key] = u_wind

        if isinstance(v_wind, np.ndarray):
            # Choosing a key we don't think exists
            key = "__v_wind"
            if key in self:
                raise RuntimeError(f"Unexpected key {key} found")
            self[key] = v_wind

        # Calculate TAS on each flight individually
        def calc_tas(fl: Flight) -> npt.NDArray[np.float_]:
            u = fl.get("__u_wind", u_wind)
            v = fl.get("__v_wind", v_wind)

            return fl.segment_true_airspeed(
                u, v, smooth=smooth, window_length=window_length, polyorder=polyorder
            )

        fls = self.to_flight_list(copy=False)
        tas = [calc_tas(fl) for fl in fls]

        # Cleanup
        self.data.pop("__u_wind", None)
        self.data.pop("__v_wind", None)

        # Making an assumption here that Fleet was instantiated by `from_seq`
        # method. If this is not the case, the order may be off when to_flight_list
        # is called.
        # Currently, we expect to only use Fleet "internally", so this more general
        # use case isn't seen.
        return np.concatenate(tas)

    @overrides
    def segment_groundspeed(self, *args: Any, **kwargs: Any) -> npt.NDArray[np.float_]:
        # Implement if we have a usecase for this.
        # Because the super() method uses a smoothing pattern, it will not reliably
        # work on Fleet.
        raise NotImplementedError

    @overrides
    def resample_and_fill(self, *args: Any, **kwargs: Any) -> Fleet:
        # Definitely do not try to call this on a Fleet! Lots of trajectory
        # specific logic is employed on the Fligth method.
        raise NotImplementedError

    @overrides
    def segment_length(self) -> npt.NDArray[np.float_]:
        return np.where(self.final_waypoints, np.nan, super().segment_length())

    @property
    @overrides
    def max_distance_gap(self) -> float:
        if self.attrs["crs"] != "EPSG:4326":
            raise NotImplementedError("Only implemented for EPSG:4326 CRS.")

        return np.nanmax(self.segment_length()).item()

    @overrides
    def segment_azimuth(self) -> npt.NDArray[np.float_]:
        return np.where(self.final_waypoints, np.nan, super().segment_azimuth())

    @overrides
    def segment_angle(self) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        sin_a, cos_a = super().segment_angle()
        sin_a[self.final_waypoints] = np.nan
        cos_a[self.final_waypoints] = np.nan
        return sin_a, cos_a
