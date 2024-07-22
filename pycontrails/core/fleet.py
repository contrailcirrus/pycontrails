"""A single data structure encompassing a sequence of :class:`Flight` instances."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt
import pandas as pd
from overrides import overrides

from pycontrails.core.flight import Flight
from pycontrails.core.fuel import Fuel, JetA
from pycontrails.core.vector import GeoVectorDataset, VectorDataDict, VectorDataset


class Fleet(Flight):
    """Data structure for holding a sequence of :class:`Flight` instances.

    Flight waypoints are merged into a single :class:`Flight`-like object.
    """

    __slots__ = ("fl_attrs", "final_waypoints")

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
        attrs: dict[str, Any] | None = None,
        copy: bool = True,
        fuel: Fuel | None = None,
        fl_attrs: dict[str, Any] | None = None,
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
            altitude_ft=altitude_ft,
            level=level,
            time=time,
            attrs=attrs,
            copy=copy,
            **attrs_kwargs,
        )

        self.fuel = fuel or JetA()
        self.final_waypoints, self.fl_attrs = self._validate(fl_attrs)

    def _validate(
        self, fl_attrs: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.bool_], dict[str, Any]]:
        """Validate data, update fl_attrs and calculate the final waypoint of each flight.

        Parameters
        ----------
        fl_attrs : dict[str, Any] | None, optional
            Dictionary of individual :class:`Flight` attributes.

        Returns
        -------
        final_waypoints : npt.NDArray[np.bool_]
            A boolean array in which True values correspond to final waypoint of each flight.
        fl_attrs : dict[str, Any]
            Updated dictionary of individual :class:`Flight` attributes.

        Raises
        ------
        KeyError, ValueError
            Fleet :attr:`data` does not take the expected form.
        """
        try:
            flight_id = self["flight_id"]
        except KeyError as exc:
            msg = "Fleet must have a 'flight_id' key in its 'data'."
            raise KeyError(msg) from exc

        # Some pandas groupby magic to ensure flights are arranged in blocks
        df = pd.DataFrame({"flight_id": flight_id, "index": np.arange(self.size)})
        grouped = df.groupby("flight_id", sort=False)
        groups = grouped.agg({"flight_id": "size", "index": ["first", "last"]})

        expected_size = groups[("index", "last")] - groups[("index", "first")] + 1
        actual_size = groups[("flight_id", "size")]
        if not np.array_equal(expected_size, actual_size):
            msg = (
                "Fleet must have contiguous waypoint blocks with constant flight_id. "
                "If instantiating from a DataFrame, call df.sort_values(by=['flight_id', 'time']) "
                "before passing to Fleet."
            )
            raise ValueError(msg)

        # Calculate boolean array of final waypoints by flight
        final_waypoints = np.zeros(self.size, dtype=bool)
        final_waypoint_indices = groups[("index", "last")].to_numpy()
        final_waypoints[final_waypoint_indices] = True

        # Set default fl_attrs if not provided
        fl_attrs = fl_attrs or {}
        for flight_id in groups.index:
            fl_attrs.setdefault(flight_id, {})  # type: ignore[call-overload]

        extra = fl_attrs.keys() - groups.index
        if extra:
            msg = f"Unexpected flight_id(s) {extra} in fl_attrs."
            raise ValueError(msg)

        return final_waypoints, fl_attrs

    @overrides
    def copy(self, **kwargs: Any) -> Fleet:
        kwargs.setdefault("fuel", self.fuel)
        kwargs.setdefault("fl_attrs", self.fl_attrs)
        return super().copy(**kwargs)

    @overrides
    def filter(self, mask: npt.NDArray[np.bool_], copy: bool = True, **kwargs: Any) -> Fleet:
        kwargs.setdefault("fuel", self.fuel)

        flight_ids = set(np.unique(self["flight_id"][mask]))
        fl_attrs = {k: v for k, v in self.fl_attrs.items() if k in flight_ids}
        kwargs.setdefault("fl_attrs", fl_attrs)

        return super().filter(mask, copy=copy, **kwargs)

    @overrides
    def sort(self, by: str | list[str]) -> NoReturn:
        msg = (
            "Fleet.sort is not implemented. A Fleet instance must be sorted "
            "by ['flight_id', 'time'] (this is enforced in Fleet._validate). "
            "To force sorting, create a GeoVectorDataset instance "
            "and call the 'sort' method."
        )
        raise ValueError(msg)

    @classmethod
    def from_seq(
        cls,
        seq: Iterable[Flight],
        broadcast_numeric: bool = True,
        copy: bool = True,
        attrs: dict[str, Any] | None = None,
    ) -> Fleet:
        """Instantiate a :class:`Fleet` instance from an iterable of :class:`Flight`.

        .. versionchanged:: 0.49.3

            Empty flights are now filtered out before concatenation.

        Parameters
        ----------
        seq : Iterable[Flight]
            An iterable of :class:`Flight` instances.
        broadcast_numeric : bool, optional
            If True, broadcast numeric attributes to data variables.
        copy : bool, optional
            If True, make copy of each flight instance in ``seq``.
        attrs : dict[str, Any] | None, optional
            Global attribute to attach to instance.

        Returns
        -------
        Fleet
            A `Fleet` instance made from concatenating the :class:`Flight`
            instances in ``seq``. The fuel type is taken from the first :class:`Flight`
            in ``seq``.
        """

        def _maybe_copy(fl: Flight) -> Flight:
            return fl.copy() if copy else fl

        def _maybe_warn(fl: Flight) -> Flight:
            if not fl:
                warnings.warn("Empty flight found in sequence. It will be filtered out.")
            return fl

        seq = tuple(_maybe_copy(fl) for fl in seq if _maybe_warn(fl))

        if not seq:
            msg = "Cannot create Fleet from empty sequence."
            raise ValueError(msg)

        fl_attrs: dict[str, Any] = {}

        # Pluck from the first flight to get fuel, data_keys, and crs
        fuel = seq[0].fuel
        data_keys = set(seq[0])  # convert to a new instance to because we mutate seq[0]
        crs = seq[0].attrs["crs"]

        for fl in seq:
            _validate_fl(
                fl,
                fl_attrs=fl_attrs,
                data_keys=data_keys,
                crs=crs,
                fuel=fuel,
                broadcast_numeric=broadcast_numeric,
            )

        data = {var: np.concatenate([fl[var] for fl in seq]) for var in seq[0]}
        return cls(data=data, attrs=attrs, copy=False, fuel=fuel, fl_attrs=fl_attrs)

    @property
    def n_flights(self) -> int:
        """Return number of distinct flights.

        Returns
        -------
        int
            Number of flights
        """
        return len(self.fl_attrs)

    def to_flight_list(self, copy: bool = True) -> list[Flight]:
        """De-concatenate merged waypoints into a list of :class:`Flight` instances.

        Any global :attr:`attrs` are lost.

        Parameters
        ----------
        copy : bool, optional
            If True, make copy of each :class:`Flight` instance.

        Returns
        -------
        list[Flight]
            List of Flights in the same order as was passed into the ``Fleet`` instance.
        """
        indices = self.dataframe.groupby("flight_id", sort=False).indices
        return [
            Flight(
                data=VectorDataDict({k: v[idx] for k, v in self.data.items()}),
                attrs=self.fl_attrs[flight_id],
                copy=copy,
                fuel=self.fuel,
            )
            for flight_id, idx in indices.items()
        ]

    ###################################
    # Flight methods involving segments
    ###################################

    def segment_true_airspeed(
        self,
        u_wind: npt.NDArray[np.float64] | float = 0.0,
        v_wind: npt.NDArray[np.float64] | float = 0.0,
        smooth: bool = True,
        window_length: int = 7,
        polyorder: int = 1,
    ) -> npt.NDArray[np.float64]:
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
                msg = f"Unexpected key {key} found"
                raise RuntimeError(msg)
            self[key] = u_wind

        if isinstance(v_wind, np.ndarray):
            # Choosing a key we don't think exists
            key = "__v_wind"
            if key in self:
                msg = f"Unexpected key {key} found"
                raise RuntimeError(msg)
            self[key] = v_wind

        # Calculate TAS on each flight individually
        def calc_tas(fl: Flight) -> npt.NDArray[np.float64]:
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
    def segment_groundspeed(self, *args: Any, **kwargs: Any) -> npt.NDArray[np.float64]:
        # Implement if we have a usecase for this.
        # Because the super() method uses a smoothing pattern, it will not reliably
        # work on Fleet.
        raise NotImplementedError

    @overrides
    def resample_and_fill(self, *args: Any, **kwargs: Any) -> Fleet:
        flights = self.to_flight_list(copy=False)
        flights = [fl.resample_and_fill(*args, **kwargs) for fl in flights]
        return type(self).from_seq(flights, copy=False, broadcast_numeric=False, attrs=self.attrs)

    @overrides
    def segment_length(self) -> npt.NDArray[np.float64]:
        return np.where(self.final_waypoints, np.nan, super().segment_length())

    @property
    @overrides
    def max_distance_gap(self) -> float:
        if self.attrs["crs"] != "EPSG:4326":
            msg = "Only implemented for EPSG:4326 CRS."
            raise NotImplementedError(msg)

        return np.nanmax(self.segment_length()).item()

    @overrides
    def segment_azimuth(self) -> npt.NDArray[np.float64]:
        return np.where(self.final_waypoints, np.nan, super().segment_azimuth())

    @overrides
    def segment_angle(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        sin_a, cos_a = super().segment_angle()
        sin_a[self.final_waypoints] = np.nan
        cos_a[self.final_waypoints] = np.nan
        return sin_a, cos_a

    @overrides
    def clean_and_resample(
        self,
        freq: str = "1min",
        fill_method: str = "geodesic",
        geodesic_threshold: float = 100e3,
        nominal_rocd: float = 0.0,
        kernel_size: int = 17,
        cruise_threshold: float = 120,
        force_filter: bool = False,
        drop: bool = True,
        keep_original_index: bool = False,
        climb_descend_at_end: bool = False,
    ) -> Flight:
        msg = "Only implemented for Flight instances"
        raise NotImplementedError(msg)


def _extract_flight_id(fl: Flight) -> str:
    """Extract flight_id from Flight instance."""

    try:
        return fl.attrs["flight_id"]
    except KeyError:
        pass

    try:
        flight_ids = fl["flight_id"]
    except KeyError as exc:
        msg = "Each flight must have a 'flight_id' key in its 'attrs'."
        raise KeyError(msg) from exc

    tmp = np.unique(flight_ids)
    if len(tmp) > 1:
        msg = f"Multiple flight_ids {tmp} found in Flight."
        raise ValueError(msg)
    if len(tmp) == 0:
        msg = "Flight has no flight_id."
        raise ValueError(msg)
    return tmp[0]


def _validate_fl(
    fl: Flight,
    *,
    fl_attrs: dict[str, Any],
    data_keys: set[str],
    crs: str,
    fuel: Fuel,
    broadcast_numeric: bool,
) -> None:
    """Attach "flight_id" and "waypoint" columns to flight :attr:`data`.

    Mutates parameter ``fl`` and ``fl_attrs`` in place.

    Parameters
    ----------
    fl : Flight
        Flight instance to process.
    fl_attrs : dict[str, Any]
        Dictionary of `Flight` attributes. Attributes belonging to `fl` are attached
        to `fl_attrs` under the "flight_id" key.
    data_keys : set[str]
        Set of data keys expected in each flight.
    fuel : Fuel
        Fuel used all flights
    crs : str
        CRS to use all flights
    broadcast_numeric : bool
        If True, broadcast numeric attributes to data variables.

    Raises
    ------
    KeyError
        ``fl`` does not have a ``flight_id`` key in :attr:`attrs`.
    ValueError
        If ``flight_id`` is duplicated or incompatible CRS found.
    """
    flight_id = _extract_flight_id(fl)

    if flight_id in fl_attrs:
        msg = f"Duplicate 'flight_id' {flight_id} found."
        raise ValueError(msg)
    fl_attrs[flight_id] = fl.attrs

    # Verify consistency across flights
    if fl.fuel != fuel:
        msg = (
            f"Fuel type on Flight {flight_id} ({fl.fuel.fuel_name}) "
            f"is not inconsistent with previous flights ({fuel.fuel_name}). "
            "The 'fuel' attributes must be consistent between flights in a Fleet."
        )
        raise ValueError(msg)
    if fl.attrs["crs"] != crs:
        msg = (
            f"CRS on Flight {flight_id} ({fl.attrs['crs']}) "
            f"is not inconsistent with previous flights ({crs}). "
            "The 'crs' attributes must be consistent between flights in a Fleet."
        )
        raise ValueError(msg)
    if fl.data.keys() != data_keys:
        msg = (
            f"Data keys on Flight {flight_id} ({fl.data.keys()}) "
            f"is not inconsistent with previous flights ({data_keys}). "
            "The 'data_keys' attributes must be consistent between flights in a Fleet."
        )
        raise ValueError(msg)

    # Expand data
    if broadcast_numeric:
        fl.broadcast_numeric_attrs()
    if "waypoint" not in fl:
        fl["waypoint"] = np.arange(fl.size)
    if "flight_id" not in fl:
        fl["flight_id"] = np.full(fl.size, flight_id)
