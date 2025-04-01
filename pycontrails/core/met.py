"""Meteorology data models."""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import sys
import typing
import warnings
from abc import ABC, abstractmethod
from collections.abc import (
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import ExitStack
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    overload,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pycontrails.core import interpolation
from pycontrails.core import vector as vector_module
from pycontrails.core.cache import CacheStore, DiskCacheStore
from pycontrails.core.met_var import AirPressure, Altitude, MetVariable
from pycontrails.physics import units
from pycontrails.utils import dependencies
from pycontrails.utils import temp as temp_module

logger = logging.getLogger(__name__)

# optional imports
if TYPE_CHECKING:
    import open3d as o3d

XArrayType = TypeVar("XArrayType", xr.Dataset, xr.DataArray)
MetDataType = TypeVar("MetDataType", "MetDataset", "MetDataArray")
DatasetType = TypeVar("DatasetType", xr.Dataset, "MetDataset")

COORD_DTYPE = np.float64


class MetBase(ABC, Generic[XArrayType]):
    """Abstract class for building Meteorology Data handling classes.

    All support here should be generic to work on xr.DataArray
    and xr.Dataset.
    """

    __slots__ = ("cachestore", "data")

    #: DataArray or Dataset
    data: XArrayType

    #: Cache datastore to use for :meth:`save` or :meth:`load`
    cachestore: CacheStore | None

    #: Default dimension order for DataArray or Dataset (x, y, z, t)
    dim_order = (
        "longitude",
        "latitude",
        "level",
        "time",
    )

    @classmethod
    def _from_fastpath(cls, data: XArrayType, cachestore: CacheStore | None = None) -> Self:
        """Create new instance from consistent data.

        This is a low-level method that bypasses the standard constructor in certain
        special cases. It is intended for internal use only.

        In essence, this method skips any validation from __init__ and directly sets
        ``data`` and ``attrs``. This is useful when creating a new instance from an existing
        instance the data has already been validated.
        """
        obj = cls.__new__(cls)
        obj.data = data
        obj.cachestore = cachestore
        return obj

    def __repr__(self) -> str:
        data = getattr(self, "data", None)
        return (
            f"{self.__class__.__name__} with data:\n\n{data.__repr__() if data is not None else ''}"
        )

    def _repr_html_(self) -> str:
        try:
            return f"<b>{type(self).__name__}</b> with data:<br/ ><br/> {self.data._repr_html_()}"
        except AttributeError:
            return f"<b>{type(self).__name__}</b> without data"

    def _validate_dim_contains_coords(self) -> None:
        """Check that data contains four temporal-spatial coordinates.

        Raises
        ------
        ValueError
            If data does not contain all four coordinates (longitude, latitude, level, time).
        """
        missing = set(self.dim_order).difference(self.data.dims)
        if not missing:
            return

        dim = sorted(missing)
        msg = f"Meteorology data must contain dimension(s): {dim}."
        if "level" in dim:
            msg += (
                " For single level data, set 'level' coordinate to constant -1 "
                "using `ds = ds.expand_dims({'level': [-1]})`"
            )
        raise ValueError(msg)

    def _validate_longitude(self) -> None:
        """Check longitude bounds.

        Assumes ``longitude`` dimension is already sorted.

        Raises
        ------
        ValueError
            If longitude values are not contained in the interval [-180, 180].
        """
        longitude = self.indexes["longitude"].to_numpy()
        if longitude.dtype != COORD_DTYPE:
            msg = f"Longitude values must have dtype {COORD_DTYPE}. Instantiate with 'copy=True'."
            raise ValueError(msg)

        if self.is_wrapped:
            # Relax verification if the longitude has already been processed and wrapped
            if longitude[-1] > 360.0:
                raise ValueError(
                    "Longitude contains values > 360. Shift to WGS84 with "
                    "'data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180))'"
                )
            if longitude[0] < -360.0:
                raise ValueError(
                    "Longitude contains values < -360. Shift to WGS84 with "
                    "'data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180))'"
                )
            return

        # Strict!
        if longitude[-1] > 180.0:
            raise ValueError(
                "Longitude contains values > 180. Shift to WGS84 with "
                "'data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180))'"
            )
        if longitude[0] < -180.0:
            raise ValueError(
                "Longitude contains values < -180. Shift to WGS84 with "
                "'data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180))'"
            )

    def _validate_latitude(self) -> None:
        """Check latitude bounds.

        Assumes ``latitude`` dimension is already sorted.

        Raises
        ------
        ValueError
            If latitude values are not contained in the interval [-90, 90].
        """
        latitude = self.indexes["latitude"].to_numpy()
        if latitude.dtype != COORD_DTYPE:
            msg = f"Latitude values must have dtype {COORD_DTYPE}. Instantiate with 'copy=True'."
            raise ValueError(msg)

        if latitude[0] < -90.0:
            raise ValueError(
                "Latitude contains values < -90 . "
                "Latitude values must be contained in the interval [-90, 90]."
            )
        if latitude[-1] > 90.0:
            raise ValueError(
                "Latitude contains values > 90 . "
                "Latitude values must be contained in the interval [-90, 90]."
            )

    def _validate_sorting(self) -> None:
        """Check that all coordinates are sorted.

        Raises
        ------
        ValueError
            If one of the coordinates is not sorted.
        """
        indexes = self.indexes
        if not np.all(np.diff(indexes["time"]) > np.timedelta64(0, "ns")):
            raise ValueError("Coordinate 'time' not sorted. Instantiate with 'copy=True'.")
        for coord in self.dim_order[:3]:  # exclude time, the 4th dimension
            if not np.all(np.diff(indexes[coord]) > 0.0):
                raise ValueError(f"Coordinate '{coord}' not sorted. Instantiate with 'copy=True'.")

    def _validate_transpose(self) -> None:
        """Check that data is transposed according to :attr:`dim_order`."""

        def _check_da(da: xr.DataArray, key: Hashable | None = None) -> None:
            if da.dims != self.dim_order:
                if key is not None:
                    msg = (
                        f"Data dimension not transposed on variable '{key}'. "
                        "Instantiate with 'copy=True'."
                    )
                else:
                    msg = "Data dimension not transposed. Instantiate with 'copy=True'."
                raise ValueError(msg)

        data = self.data
        if isinstance(data, xr.DataArray):
            _check_da(data)
            return

        for key, da in self.data.items():
            _check_da(da, key)

    def _validate_dims(self) -> None:
        """Apply all validators."""
        self._validate_dim_contains_coords()

        # Apply this one first: validate_longitude and validate_latitude assume sorted
        self._validate_sorting()
        self._validate_longitude()
        self._validate_latitude()
        self._validate_transpose()
        if self.data["level"].dtype != COORD_DTYPE:
            msg = f"Level values must have dtype {COORD_DTYPE}. Instantiate with 'copy=True'."
            raise ValueError(msg)

    def _preprocess_dims(self, wrap_longitude: bool) -> None:
        """Confirm DataArray or Dataset include required dimension in a consistent format.

        Expects DataArray or Dataset to contain dimensions ``latitude`, ``longitude``, ``time``,
        and ``level`` (in hPa/mbar).
        Adds additional coordinate variables ``air_pressure`` and ``altitude`` coordinates
        mapped to "level" dimension if "level" > 0.

        Set ``level`` to -1 to signify single level.

        .. versionchanged:: 0.40.0

            All coordinate data (longitude, latitude, level) are promoted to ``float64``.
            Auxiliary coordinates (altitude and air_pressure) are now cast to the same
            dtype as the underlying grid data.


        Parameters
        ----------
        wrap_longitude : bool
            If True, ensure longitude values cover the interval ``[-180, 180]``.

        Raises
        ------
        ValueError
            Raises if required dimension names are not found
        """
        self._validate_dim_contains_coords()

        # Ensure spatial coordinates all have dtype COORD_DTYPE
        indexes = self.indexes
        for coord in ("longitude", "latitude", "level"):
            arr = indexes[coord].to_numpy()
            if arr.dtype != COORD_DTYPE:
                self.data[coord] = arr.astype(COORD_DTYPE)

        # Ensure time is np.datetime64[ns]
        self.data["time"] = self.data["time"].astype("datetime64[ns]", copy=False)

        # sortby to ensure each coordinate has ascending order
        self.data = self.data.sortby(list(self.dim_order), ascending=True)

        if not self.is_wrapped:
            # Ensure longitude is contained in interval [-180, 180)
            # If longitude has value at 180, we might not want to shift it?
            lon = self.indexes["longitude"].to_numpy()

            # This longitude shifting can give rise to precision errors with float32
            # Only shift if necessary
            if np.any(lon >= 180.0) or np.any(lon < -180.0):
                self.data = shift_longitude(self.data)
            else:
                self.data = self.data.sortby("longitude", ascending=True)

            # wrap longitude, if requested
            if wrap_longitude:
                self.data = _wrap_longitude(self.data)

        self._validate_longitude()
        self._validate_latitude()

        # transpose to have ordering (x, y, z, t, ...)
        dim_order = [*self.dim_order, *(d for d in self.data.dims if d not in self.dim_order)]
        self.data = self.data.transpose(*dim_order)

        # single level data
        if self.is_single_level:
            # add level attributes to reflect surface level
            level_attrs = self.data["level"].attrs
            if not level_attrs:
                level_attrs.update(units="", long_name="Single Level")
            return

        self.data = _add_vertical_coords(self.data)

    @property
    def hash(self) -> str:
        """Generate a unique hash for this met instance.

        Note this is not as robust as it could be since `repr`
        cuts off.

        Returns
        -------
        str
            Unique hash for met instance (sha1)
        """
        _hash = repr(self.data)
        return hashlib.sha1(bytes(_hash, "utf-8")).hexdigest()

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the size of (each) array in underlying :attr:`data`.

        Returns
        -------
        int
            Total number of grid points in underlying data
        """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int, int]:
        """Return the shape of the dimensions.

        Returns
        -------
        tuple[int, int, int, int]
            Shape of underlying data
        """

    @property
    def coords(self) -> dict[str, np.ndarray]:
        """Get coordinates of underlying :attr:`data` coordinates.

        Only return non-dimension coordinates.

        See:
        http://xarray.pydata.org/en/stable/user-guide/data-structures.html#coordinates

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of coordinates
        """
        variables = self.indexes
        return {
            "longitude": variables["longitude"].to_numpy(),
            "latitude": variables["latitude"].to_numpy(),
            "level": variables["level"].to_numpy(),
            "time": variables["time"].to_numpy(),
        }

    @property
    def indexes(self) -> dict[Hashable, pd.Index]:
        """Low level access to underlying :attr:`data` indexes.

        This method is typically is faster for accessing coordinate indexes.

        .. versionadded:: 0.25.2

        Returns
        -------
        dict[Hashable, pd.Index]
            Dictionary of indexes.

        Examples
        --------
        >>> from pycontrails.datalib.ecmwf import ERA5
        >>> times = (datetime(2022, 3, 1, 12),  datetime(2022, 3, 1, 13))
        >>> variables = "air_temperature", "specific_humidity"
        >>> levels = [200, 300]
        >>> era5 = ERA5(times, variables, levels)
        >>> mds = era5.open_metdataset()
        >>> mds.indexes["level"].to_numpy()
        array([200., 300.])

        >>> mda = mds["air_temperature"]
        >>> mda.indexes["level"].to_numpy()
        array([200., 300.])
        """
        return {k: v.index for k, v in self.data._indexes.items()}  # type: ignore[attr-defined]

    @property
    def is_wrapped(self) -> bool:
        """Check if the longitude dimension covers the closed interval ``[-180, 180]``.

        Assumes the longitude dimension is sorted (this is established by the
        :class:`MetDataset` or :class:`MetDataArray` constructor).

        .. versionchanged:: 0.26.0

            The previous implementation checked for the minimum and maximum longitude
            dimension values to be duplicated. The current implementation only checks for
            that the interval ``[-180, 180]`` is covered by the longitude dimension. The
            :func:`pycontrails.physics.geo.advect_longitude` is designed for compatibility
            with this convention.

        Returns
        -------
        bool
            True if longitude coordinates cover ``[-180, 180]``

        See Also
        --------
        :func:`pycontrails.physics.geo.advect_longitude`
        """
        longitude = self.indexes["longitude"].to_numpy()
        return _is_wrapped(longitude)

    @property
    def is_single_level(self) -> bool:
        """Check if instance contains "single level" or "surface level" data.

        This method checks if ``level`` dimension contains a single value equal
        to -1, the pycontrails convention for surface only data.

        Returns
        -------
        bool
            If instance contains single level data.
        """
        level = self.indexes["level"].to_numpy()
        return len(level) == 1 and level[0] == -1

    @abstractmethod
    def broadcast_coords(self, name: str) -> xr.DataArray:
        """Broadcast coordinates along other dimensions.

        Parameters
        ----------
        name : str
            Coordinate/dimension name to broadcast.
            Can be a dimension or non-dimension coordinates.

        Returns
        -------
        xr.DataArray
            DataArray of the coordinate broadcasted along all other dimensions.
            The DataArray will have the same shape as the gridded data.
        """

    def _save(self, dataset: xr.Dataset, **kwargs: Any) -> list[str]:
        """Save dataset to netcdf files named for the met hash and hour.

        Does not yet save in parallel.

        .. versionchanged:: 0.34.1

            If :attr:`cachestore` is None, this method assigns it
            to new :class:`DiskCacheStore`.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to save
        **kwargs
            Keyword args passed directly on to :func:`xarray.save_mfdataset`

        Returns
        -------
        list[str]
            List of filenames saved
        """
        self.cachestore = self.cachestore or DiskCacheStore()

        # group by hour and save one dataset for each hour to temp file
        times, datasets = zip(*dataset.groupby("time", squeeze=False), strict=True)

        # Open ExitStack to control temp_file context manager
        with ExitStack() as stack:
            xarray_temp_filenames = [stack.enter_context(temp_module.temp_file()) for _ in times]
            xr.save_mfdataset(datasets, xarray_temp_filenames, **kwargs)

            # set filenames by hash
            filenames = [f"{self.hash}-{t_idx}.nc" for t_idx, _ in enumerate(times)]

            # put each hourly file into cache
            self.cachestore.put_multiple(xarray_temp_filenames, filenames)

        return filenames

    def __len__(self) -> int:
        return self.data.__len__()

    @property
    def attrs(self) -> dict[Hashable, Any]:
        """Pass through to :attr:`self.data.attrs`."""
        return self.data.attrs

    def downselect(self, bbox: tuple[float, ...]) -> Self:
        """Downselect met data within spatial bounding box.

        Parameters
        ----------
        bbox : list[float]
            List of coordinates defining a spatial bounding box in WGS84 coordinates.
            For 2D queries, list is [west, south, east, north].
            For 3D queries, list is [west, south, min-level, east, north, max-level]
            with level defined in [:math:`hPa`].

        Returns
        -------
        Self
            Return downselected data
        """
        data = downselect(self.data, bbox)
        return type(self)._from_fastpath(data, cachestore=self.cachestore)

    @property
    def is_zarr(self) -> bool:
        """Check if underlying :attr:`data` is sourced from a Zarr group.

        Implementation is very brittle, and may break as external libraries change.

        Some ``dask`` intermediate artifact is cached when this is called. Typically,
        subsequent calls to this method are much faster than the initial call.

        .. versionadded:: 0.26.0

        Returns
        -------
        bool
            If ``data`` is based on a Zarr group.
        """
        return _is_zarr(self.data)

    def downselect_met(
        self,
        met: MetDataType,
        *,
        longitude_buffer: tuple[float, float] = (0.0, 0.0),
        latitude_buffer: tuple[float, float] = (0.0, 0.0),
        level_buffer: tuple[float, float] = (0.0, 0.0),
        time_buffer: tuple[np.timedelta64, np.timedelta64] = (
            np.timedelta64(0, "h"),
            np.timedelta64(0, "h"),
        ),
    ) -> MetDataType:
        """Downselect ``met`` to encompass a spatiotemporal region of the data.

        .. warning::

            This method is analogous to :meth:`GeoVectorDataset.downselect_met`.
            It does not change the instance data, but instead operates on the
            ``met`` input. This method is different from :meth:`downselect` which
            operates on the instance data.

        .. versionchanged:: 0.54.5

            Data is no longer copied when downselecting.

        Parameters
        ----------
        met : MetDataset | MetDataArray
            MetDataset or MetDataArray to downselect.
        longitude_buffer : tuple[float, float], optional
            Extend longitude domain past by ``longitude_buffer[0]`` on the low side
            and ``longitude_buffer[1]`` on the high side.
            Units must be the same as class coordinates.
            Defaults to ``(0, 0)`` degrees.
        latitude_buffer : tuple[float, float], optional
            Extend latitude domain past by ``latitude_buffer[0]`` on the low side
            and ``latitude_buffer[1]`` on the high side.
            Units must be the same as class coordinates.
            Defaults to ``(0, 0)`` degrees.
        level_buffer : tuple[float, float], optional
            Extend level domain past by ``level_buffer[0]`` on the low side
            and ``level_buffer[1]`` on the high side.
            Units must be the same as class coordinates.
            Defaults to ``(0, 0)`` [:math:`hPa`].
        time_buffer : tuple[np.timedelta64, np.timedelta64], optional
            Extend time domain past by ``time_buffer[0]`` on the low side
            and ``time_buffer[1]`` on the high side.
            Units must be the same as class coordinates.
            Defaults to ``(np.timedelta64(0, "h"), np.timedelta64(0, "h"))``.

        Returns
        -------
        MetDataset | MetDataArray
            Copy of downselected MetDataset or MetDataArray.
        """
        indexes = self.indexes
        lon = indexes["longitude"].to_numpy()
        lat = indexes["latitude"].to_numpy()
        level = indexes["level"].to_numpy()
        time = indexes["time"].to_numpy()

        vector = vector_module.GeoVectorDataset(
            longitude=[lon.min(), lon.max()],
            latitude=[lat.min(), lat.max()],
            level=[level.min(), level.max()],
            time=[time.min(), time.max()],
        )

        return vector.downselect_met(
            met,
            longitude_buffer=longitude_buffer,
            latitude_buffer=latitude_buffer,
            level_buffer=level_buffer,
            time_buffer=time_buffer,
        )

    def wrap_longitude(self) -> Self:
        """Wrap longitude coordinates.

        Returns
        -------
        Self
            Copy of instance with wrapped longitude values.
            Returns copy of data when longitude values are already wrapped
        """
        return type(self)._from_fastpath(_wrap_longitude(self.data), cachestore=self.cachestore)

    def copy(self) -> Self:
        """Create a shallow copy of the current class.

        See :meth:`xarray.Dataset.copy` for reference.

        Returns
        -------
        Self
            Copy of the current class
        """
        return type(self)._from_fastpath(self.data.copy(), cachestore=self.cachestore)


class MetDataset(MetBase):
    """Meteorological dataset with multiple variables.

    Composition around xr.Dataset to enforce certain
    variables and dimensions for internal usage

    Parameters
    ----------
    data : xr.Dataset
        :class:`xarray.Dataset` containing meteorological variables and coordinates
    cachestore : :class:`CacheStore`, optional
        Cache datastore for staging intermediates with :meth:`save`.
        Defaults to None.
    wrap_longitude : bool, optional
        Wrap data along the longitude dimension. If True, duplicate and shift longitude
        values (ie, -180 -> 180) to ensure that the longitude dimension covers the entire
        interval ``[-180, 180]``. Defaults to False.
    copy : bool, optional
        Copy data on construction. Defaults to True.
    attrs : dict[str, Any], optional
        Attributes to add to :attr:`data.attrs`. Defaults to None.
        Generally, pycontrails :class:`Models` may use the following attributes:

        - ``provider``: Name of the data provider (e.g. "ECMWF").
        - ``dataset``: Name of the dataset (e.g. "ERA5").
        - ``product``: Name of the product type (e.g. "reanalysis").

    **attrs_kwargs : Any
        Keyword arguments to add to :attr:`data.attrs`. Defaults to None.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import xarray as xr
    >>> from pycontrails.datalib.ecmwf import ERA5

    >>> time = ("2022-03-01T00", "2022-03-01T02")
    >>> variables = ["air_temperature", "specific_humidity"]
    >>> pressure_levels = [200, 250, 300]
    >>> era5 = ERA5(time, variables, pressure_levels)

    >>> # Open directly as `MetDataset`
    >>> met = era5.open_metdataset()
    >>> # Use `data` attribute to access `xarray` object
    >>> assert isinstance(met.data, xr.Dataset)

    >>> # Alternatively, open with `xarray` and cast to `MetDataset`
    >>> ds = xr.open_mfdataset(era5._cachepaths)
    >>> met = MetDataset(ds)

    >>> # Access sub-`DataArrays`
    >>> mda = met["t"]  # `MetDataArray` instance, needed for interpolation operations
    >>> da = mda.data  # Underlying `xarray` object

    >>> # Check out a few values
    >>> da[5:8, 5:8, 1, 1].values
    array([[224.08959005, 224.41374427, 224.75945349],
           [224.09456429, 224.42037658, 224.76525676],
           [224.10036756, 224.42617985, 224.77106004]])

    >>> # Mean temperature over entire array
    >>> da.mean().load().item()
    223.5083
    """

    __slots__ = ()

    data: xr.Dataset

    def __init__(
        self,
        data: xr.Dataset,
        cachestore: CacheStore | None = None,
        wrap_longitude: bool = False,
        copy: bool = True,
        attrs: dict[str, Any] | None = None,
        **attrs_kwargs: Any,
    ) -> None:
        self.cachestore = cachestore

        data.attrs.update(attrs or {}, **attrs_kwargs)

        # if input is already a Dataset, copy into data
        if not isinstance(data, xr.Dataset):
            raise TypeError("Input 'data' must be an xarray Dataset")

        # copy Dataset into data
        if copy:
            self.data = data.copy()
            self._preprocess_dims(wrap_longitude)

        else:
            if wrap_longitude:
                raise ValueError("Set 'copy=True' when using 'wrap_longitude=True'.")
            self.data = data
            self._validate_dims()
            if not self.is_single_level:
                self.data = _add_vertical_coords(self.data)

    def __getitem__(self, key: Hashable) -> MetDataArray:
        """Return DataArray of variable ``key`` cast to a :class:`MetDataArray` object.

        Parameters
        ----------
        key : Hashable
            Variable name

        Returns
        -------
        MetDataArray
            MetDataArray instance associated to :attr:`data` variable `key`

        Raises
        ------
        KeyError
            If ``key`` not found in :attr:`data`
        """
        try:
            da = self.data[key]
        except KeyError as e:
            raise KeyError(
                f"Variable {key} not found. Available variables: {', '.join(self.data.data_vars)}. "
                "To get items (e.g. 'time' or 'level') from underlying xr.Dataset object, "
                "use the 'data' attribute."
            ) from e
        return MetDataArray._from_fastpath(da)

    def get(self, key: str, default_value: Any = None) -> Any:
        """Shortcut to :meth:`data.get(k, v)` method.

        Parameters
        ----------
        key : str
            Key to get from :attr:`data`
        default_value : Any, optional
            Return `default_value` if `key` not in :attr:`data`, by default `None`

        Returns
        -------
        Any
            Values returned from  :attr:`data.get(key, default_value)`
        """
        return self.data.get(key, default_value)

    def __setitem__(
        self,
        key: Hashable | list[Hashable] | Mapping,
        value: Any,
    ) -> None:
        """Shortcut to set data variable on :attr:`data`.

        Warns if ``key`` is already present in dataset.

        Parameters
        ----------
        key : Hashable | list[Hashable] | Mapping
            Variable name
        value : Any
            Value to set to variable names

        See Also
        --------
        - :class:`xarray.Dataset.__setitem__`
        """

        # pull data of MetDataArray value
        if isinstance(value, MetDataArray):
            value = value.data

        # warn if key is already in Dataset
        override_keys: list[Hashable] = []

        if isinstance(key, Hashable):
            if key in self:
                override_keys = [key]

        # xarray.core.utils.is_dict_like
        # https://github.com/pydata/xarray/blob/4cae8d0ec04195291b2315b1f21d846c2bad61ff/xarray/core/utils.py#L244
        elif xr.core.utils.is_dict_like(key) or isinstance(key, list):
            override_keys = [k for k in key if k in self.data]

        if override_keys:
            warnings.warn(
                f"Overwriting data in keys `{override_keys}`. "
                "Use `.update(...)` to suppress warning."
            )

        self.data.__setitem__(key, value)

    def update(self, other: MutableMapping | None = None, **kwargs: Any) -> None:
        """Shortcut to :meth:`data.update`.

        See :meth:`xarray.Dataset.update` for reference.

        Parameters
        ----------
        other : MutableMapping
            Variables with which to update this dataset
        **kwargs : Any
            Variables defined by keyword arguments. If a variable exists both in
            ``other`` and as a keyword argument, the keyword argument takes
            precedence.

        See Also
        --------
        - :meth:`xarray.Dataset.update`
        """
        other = other or {}
        other.update(kwargs)

        # pull data of MetDataArray value
        for k, v in other.items():
            if isinstance(v, MetDataArray):
                other[k] = v.data

        self.data.update(other)

    def __iter__(self) -> Iterator[str]:
        """Allow for the use as "key" in self.met, where "key" is a data variable."""
        # From the typing perspective, `iter(self.data)`` returns Hashables (not
        # necessarily strs). In everything we do, we use str variables.
        # If we decide to extend this to support Hashable, we'll also want to
        # change VectorDataset -- the underlying :attr:`data` should then be
        # changed to `dict[Hashable, np.ndarray]`.
        for key in self.data:
            yield str(key)

    def __contains__(self, key: Hashable) -> bool:
        """Check if key ``key`` is in :attr:`data`.

        Parameters
        ----------
        key : Hashable
            Key to check

        Returns
        -------
        bool
            True if ``key`` is in :attr:`data`, False otherwise
        """
        return key in self.data

    @property
    @override
    def shape(self) -> tuple[int, int, int, int]:
        sizes = self.data.sizes
        return sizes["longitude"], sizes["latitude"], sizes["level"], sizes["time"]

    @property
    @override
    def size(self) -> int:
        return np.prod(self.shape).item()

    def ensure_vars(
        self,
        vars: MetVariable | str | Sequence[MetVariable | str | Sequence[MetVariable]],
        raise_error: bool = True,
    ) -> list[str]:
        """Ensure variables exist in xr.Dataset.

        Parameters
        ----------
        vars : MetVariable | str | Sequence[MetVariable | str | list[MetVariable]]
            List of MetVariable (or string key), or individual MetVariable (or string key).
            If ``vars`` contains an element with a list[MetVariable], then
            only one variable in the list must be present in dataset.
        raise_error : bool, optional
            Raise KeyError if data does not contain variables.
            Defaults to True.

        Returns
        -------
        list[str]
            List of met keys verified in MetDataset.
            Returns an empty list if any MetVariable is missing.

        Raises
        ------
        KeyError
            Raises when dataset does not contain variable in ``vars``
        """
        if isinstance(vars, MetVariable | str):
            vars = (vars,)

        met_keys: list[str] = []
        for variable in vars:
            met_key: str | None = None

            # input is a MetVariable or str
            if isinstance(variable, MetVariable):
                if (key := variable.standard_name) in self:
                    met_key = key
            elif isinstance(variable, str):
                if variable in self:
                    met_key = variable

            # otherwise, assume input is an sequence
            # Sequence[MetVariable] means that any variable in list will work
            else:
                for v in variable:
                    if (key := v.standard_name) in self:
                        met_key = key
                        break

            if met_key is None:
                if not raise_error:
                    return []

                if isinstance(variable, MetVariable):
                    raise KeyError(f"Dataset does not contain variable `{variable.standard_name}`")
                if isinstance(variable, list):
                    raise KeyError(
                        "Dataset does not contain one of variables "
                        f"`{[v.standard_name for v in variable]}`"
                    )
                raise KeyError(f"Dataset does not contain variable `{variable}`")

            met_keys.append(met_key)

        return met_keys

    def save(self, **kwargs: Any) -> list[str]:
        """Save intermediate to :attr:`cachestore` as netcdf.

        Load and restore using :meth:`load`.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed directly to :meth:`xarray.Dataset.to_netcdf`

        Returns
        -------
        list[str]
            Returns filenames saved
        """
        return self._save(self.data, **kwargs)

    @classmethod
    def load(
        cls,
        hash: str,
        cachestore: CacheStore | None = None,
        chunks: dict[str, int] | None = None,
    ) -> Self:
        """Load saved intermediate from :attr:`cachestore`.

        Parameters
        ----------
        hash : str
            Saved hash to load.
        cachestore : :class:`CacheStore`, optional
            Cache datastore to use for sourcing files.
            Defaults to DiskCacheStore.
        chunks : dict[str: int], optional
            Chunks kwarg passed to :func:`xarray.open_mfdataset()` when opening files.

        Returns
        -------
        Self
            New MetDataArray with loaded data.
        """
        cachestore = cachestore or DiskCacheStore()
        chunks = chunks or {}
        data = _load(hash, cachestore, chunks)
        return cls(data)

    @override
    def broadcast_coords(self, name: str) -> xr.DataArray:
        da = xr.ones_like(self.data[next(iter(self.data.keys()))]) * self.data[name]
        da.name = name

        return da

    def to_vector(self, transfer_attrs: bool = True) -> vector_module.GeoVectorDataset:
        """Convert a :class:`MetDataset` to a :class:`GeoVectorDataset` by raveling data.

        If :attr:`data` is lazy, it will be loaded.

        Parameters
        ----------
        transfer_attrs : bool, optional
            Transfer attributes from :attr:`data` to output :class:`GeoVectorDataset`.
            By default, True, meaning that attributes are transferred.

        Returns
        -------
        GeoVectorDataset
            Converted :class:`GeoVectorDataset`. The variables on the returned instance
            include all of those on the input instance, plus the four core spatial temporal
            variables.

        Examples
        --------
        >>> from pycontrails.datalib.ecmwf import ERA5
        >>> times = "2022-03-01",  "2022-03-01T01"
        >>> variables = ["air_temperature", "specific_humidity"]
        >>> levels = [250, 200]
        >>> era5 = ERA5(time=times, variables=variables, pressure_levels=levels)
        >>> met = era5.open_metdataset()
        >>> met.to_vector(transfer_attrs=False)
        GeoVectorDataset [6 keys x 4152960 length, 0 attributes]
            Keys: longitude, latitude, level, time, air_temperature, ..., specific_humidity
            Attributes:
            time                [2022-03-01 00:00:00, 2022-03-01 01:00:00]
            longitude           [-180.0, 179.75]
            latitude            [-90.0, 90.0]
            altitude            [10362.8, 11783.9]

        """
        coords_keys = self.data.dims
        indexes = self.indexes
        coords_vals = [indexes[key].values for key in coords_keys]
        coords_meshes = np.meshgrid(*coords_vals, indexing="ij")
        raveled_coords = (mesh.ravel() for mesh in coords_meshes)
        data = dict(zip(coords_keys, raveled_coords, strict=True))

        out = vector_module.GeoVectorDataset(data, copy=False)
        for key, da in self.data.items():
            # The call to .values here will load the data if it is lazy
            out[key] = da.values.ravel()  # type: ignore[index]

        if transfer_attrs:
            out.attrs.update(self.attrs)  # type: ignore[arg-type]

        return out

    def _get_pycontrails_attr_template(
        self,
        name: str,
        supported: tuple[str, ...],
        examples: dict[str, str],
    ) -> str:
        """Look up an attribute with a custom error message."""
        try:
            out = self.attrs[name]
        except KeyError as e:
            msg = f"Specify '{name}' attribute on underlying dataset."

            for i, (k, v) in enumerate(examples.items()):
                if i == 0:
                    msg = f"{msg} For example, set attrs['{name}'] = '{k}' for {v}."
                else:
                    msg = f"{msg} Set attrs['{name}'] = '{k}' for {v}."
            raise KeyError(msg) from e

        if out not in supported:
            warnings.warn(
                f"Unknown {name} '{out}'. Data may not be processed correctly. "
                f"Known {name}s are {supported}. Contact the pycontrails "
                "developers if you believe this is an error."
            )

        return out

    @property
    def provider_attr(self) -> str:
        """Look up the 'provider' attribute with a custom error message.

        Returns
        -------
        str
            Provider of the data. If not one of 'ECMWF' or 'NCEP',
            a warning is issued.
        """
        supported = ("ECMWF", "NCEP")
        examples = {"ECMWF": "data provided by ECMWF", "NCEP": "GFS data"}
        return self._get_pycontrails_attr_template("provider", supported, examples)

    @property
    def dataset_attr(self) -> str:
        """Look up the 'dataset' attribute with a custom error message.

        Returns
        -------
        str
            Dataset of the data. If not one of 'ERA5', 'HRES', 'IFS',
            or 'GFS', a warning is issued.
        """
        supported = ("ERA5", "HRES", "IFS", "GFS")
        examples = {
            "ERA5": "ECMWF ERA5 reanalysis data",
            "HRES": "ECMWF HRES forecast data",
            "GFS": "NCEP GFS forecast data",
        }
        return self._get_pycontrails_attr_template("dataset", supported, examples)

    @property
    def product_attr(self) -> str:
        """Look up the 'product' attribute with a custom error message.

        Returns
        -------
        str
            Product of the data. If not one of 'forecast', 'ensemble', or 'reanalysis',
            a warning is issued.

        """
        supported = ("reanalysis", "forecast", "ensemble")
        examples = {
            "reanalysis": "ECMWF ERA5 reanalysis data",
            "ensemble": "ECMWF ERA5 ensemble member data",
        }
        return self._get_pycontrails_attr_template("product", supported, examples)

    @overload
    def standardize_variables(
        self, variables: Iterable[MetVariable], inplace: Literal[False] = ...
    ) -> Self: ...

    @overload
    def standardize_variables(
        self, variables: Iterable[MetVariable], inplace: Literal[True]
    ) -> None: ...

    def standardize_variables(
        self, variables: Iterable[MetVariable], inplace: bool = False
    ) -> Self | None:
        """Standardize variable names.

        .. versionchanged:: 0.54.7

            By default, this method returns a new :class:`MetDataset` instead
            of renaming in place. To retain the old behavior, set ``inplace=True``.

        Parameters
        ----------
        variables : Iterable[MetVariable]
            Data source variables
        inplace : bool, optional
            If True, rename variables in place. Otherwise, return a new
            :class:`MetDataset` with renamed variables.

        See Also
        --------
        :func:`standardize_variables`
        """
        data_renamed = standardize_variables(self.data, variables)

        if inplace:
            self.data = data_renamed
            return None

        return type(self)._from_fastpath(data_renamed, cachestore=self.cachestore)

    @classmethod
    def from_coords(
        cls,
        longitude: npt.ArrayLike | float,
        latitude: npt.ArrayLike | float,
        level: npt.ArrayLike | float,
        time: npt.ArrayLike | np.datetime64,
    ) -> Self:
        r"""Create a :class:`MetDataset` containing a coordinate skeleton from coordinate arrays.

        Parameters
        ----------
        longitude, latitude : npt.ArrayLike | float
            Horizontal coordinates, in [:math:`\deg`]
        level : npt.ArrayLike | float
            Vertical coordinate, in [:math:`hPa`]
        time: npt.ArrayLike | np.datetime64,
            Temporal coordinates, in [:math:`UTC`]. Will be sorted.

        Returns
        -------
        Self
            MetDataset with no variables.

        Examples
        --------
        >>> # Create skeleton MetDataset
        >>> longitude = np.arange(0, 10, 0.5)
        >>> latitude = np.arange(0, 10, 0.5)
        >>> level = [250, 300]
        >>> time = np.datetime64("2019-01-01")
        >>> met = MetDataset.from_coords(longitude, latitude, level, time)
        >>> met
        MetDataset with data:
        <xarray.Dataset> Size: 360B
        Dimensions:       (longitude: 20, latitude: 20, level: 2, time: 1)
        Coordinates:
          * longitude     (longitude) float64 160B 0.0 0.5 1.0 1.5 ... 8.0 8.5 9.0 9.5
          * latitude      (latitude) float64 160B 0.0 0.5 1.0 1.5 ... 8.0 8.5 9.0 9.5
          * level         (level) float64 16B 250.0 300.0
          * time          (time) datetime64[ns] 8B 2019-01-01
            air_pressure  (level) float32 8B 2.5e+04 3e+04
            altitude      (level) float32 8B 1.036e+04 9.164e+03
        Data variables:
            *empty*

        >>> met.shape
        (20, 20, 2, 1)

        >>> met.size
        800

        >>> # Fill it up with some constant data
        >>> met["temperature"] = xr.DataArray(np.full(met.shape, 234.5), coords=met.coords)
        >>> met["humidity"] = xr.DataArray(np.full(met.shape, 0.5), coords=met.coords)
        >>> met
        MetDataset with data:
        <xarray.Dataset> Size: 13kB
        Dimensions:       (longitude: 20, latitude: 20, level: 2, time: 1)
        Coordinates:
          * longitude     (longitude) float64 160B 0.0 0.5 1.0 1.5 ... 8.0 8.5 9.0 9.5
          * latitude      (latitude) float64 160B 0.0 0.5 1.0 1.5 ... 8.0 8.5 9.0 9.5
          * level         (level) float64 16B 250.0 300.0
          * time          (time) datetime64[ns] 8B 2019-01-01
            air_pressure  (level) float32 8B 2.5e+04 3e+04
            altitude      (level) float32 8B 1.036e+04 9.164e+03
        Data variables:
            temperature   (longitude, latitude, level, time) float64 6kB 234.5 ... 234.5
            humidity      (longitude, latitude, level, time) float64 6kB 0.5 0.5 ... 0.5

        >>> # Convert to a GeoVectorDataset
        >>> vector = met.to_vector()
        >>> vector.dataframe.head()
        longitude  latitude  level       time  temperature  humidity
        0        0.0       0.0  250.0 2019-01-01        234.5       0.5
        1        0.0       0.0  300.0 2019-01-01        234.5       0.5
        2        0.0       0.5  250.0 2019-01-01        234.5       0.5
        3        0.0       0.5  300.0 2019-01-01        234.5       0.5
        4        0.0       1.0  250.0 2019-01-01        234.5       0.5
        """
        input_data = {
            "longitude": longitude,
            "latitude": latitude,
            "level": level,
            "time": time,
        }

        # clean up input into coords
        coords: dict[str, np.ndarray] = {}
        for key, val in input_data.items():
            dtype = "datetime64[ns]" if key == "time" else COORD_DTYPE
            arr: np.ndarray = np.asarray(val, dtype=dtype)  # type: ignore[call-overload]

            if arr.ndim == 0:
                arr = arr.reshape(1)
            elif arr.ndim > 1:
                raise ValueError(f"{key} has too many dimensions")

            arr = np.sort(arr)
            if arr.size == 0:
                raise ValueError(f"Coordinate {key} must be nonempty.")

            coords[key] = arr

        return cls(xr.Dataset({}, coords=coords))

    @classmethod
    def from_zarr(cls, store: Any, **kwargs: Any) -> Self:
        """Create a :class:`MetDataset` from a path to a Zarr store.

        Parameters
        ----------
        store : Any
            Path to Zarr store. Passed into :func:`xarray.open_zarr`.
        **kwargs : Any
            Other keyword only arguments passed into :func:`xarray.open_zarr`.

        Returns
        -------
        Self
            MetDataset with data from Zarr store.
        """
        kwargs.setdefault("storage_options", {"read_only": True})
        ds = xr.open_zarr(store, **kwargs)
        return cls(ds)


class MetDataArray(MetBase):
    """Meteorological DataArray of single variable.

    Wrapper around :class:`xarray.DataArray` to enforce certain
    variables and dimensions for internal usage.

    .. versionchanged:: 0.54.4

        Remove ``validate`` parameter. Validation is now always performed.

    Parameters
    ----------
    data : ArrayLike
        xr.DataArray or other array-like data source.
        When array-like input is provided, input ``**kwargs`` passed directly to
        xr.DataArray constructor.
    cachestore : :class:`CacheStore`, optional
        Cache datastore for staging intermediates with :meth:`save`.
        Defaults to DiskCacheStore.
    wrap_longitude : bool, optional
        Wrap data along the longitude dimension. If True, duplicate and shift longitude
        values (ie, -180 -> 180) to ensure that the longitude dimension covers the entire
        interval ``[-180, 180]``. Defaults to False.
    copy : bool, optional
        Copy `data` parameter on construction, by default `True`. If `data` is lazy-loaded
        via `dask`, this parameter has no effect. If `data` is already loaded into memory,
        a copy of the data (rather than a view) may be created if `True`.
    name : Hashable, optional
        Name of the data variable. If not specified, the name will be set to "met".

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> rng = np.random.default_rng(seed=456)

    >>> # Cook up random xarray object
    >>> coords = {
    ...     "longitude": np.arange(-20, 20),
    ...     "latitude": np.arange(-30, 30),
    ...     "level": [220, 240, 260, 280],
    ...     "time": [np.datetime64("2021-08-01T12", "ns"), np.datetime64("2021-08-01T16", "ns")]
    ...     }
    >>> da = xr.DataArray(rng.random((40, 60, 4, 2)), dims=coords.keys(), coords=coords)

    >>> # Cast to `MetDataArray` in order to interpolate
    >>> from pycontrails import MetDataArray
    >>> mda = MetDataArray(da)
    >>> mda.interpolate(-11.4, 5.7, 234, np.datetime64("2021-08-01T13"))
    array([0.52358215])

    >>> mda.interpolate(-11.4, 5.7, 234, np.datetime64("2021-08-01T13"), method='nearest')
    array([0.4188465])

    >>> da.sel(longitude=-11, latitude=6, level=240, time=np.datetime64("2021-08-01T12")).item()
    0.41884649899766946
    """

    __slots__ = ()

    data: xr.DataArray

    def __init__(
        self,
        data: xr.DataArray,
        cachestore: CacheStore | None = None,
        wrap_longitude: bool = False,
        copy: bool = True,
        name: Hashable | None = None,
    ) -> None:
        self.cachestore = cachestore

        if copy:
            self.data = data.copy()
            self._preprocess_dims(wrap_longitude)
        elif wrap_longitude:
            raise ValueError("Set 'copy=True' when using 'wrap_longitude=True'.")
        else:
            self.data = data
            self._validate_dims()

        # Priority: name > data.name > "met"
        self.data.name = name or self.data.name or "met"

    @property
    def values(self) -> np.ndarray:
        """Return underlying numpy array.

        This methods loads :attr:`data` if it is not already in memory.

        Returns
        -------
        np.ndarray
            Underlying numpy array

        See Also
        --------
        :meth:`xarray.Dataset.load`
        :meth:`xarray.DataArray.load`

        """
        if not self.in_memory:
            self._check_memory("Extracting numpy array from")
            self.data.load()

        return self.data.values

    @property
    def name(self) -> Hashable:
        """Return the DataArray name.

        Returns
        -------
        Hashable
            DataArray name
        """
        return self.data.name

    @property
    def binary(self) -> bool:
        """Determine if all data is a binary value (0, 1).

        Returns
        -------
        bool
            True if all data values are binary value (0, 1)
        """
        return np.array_equal(self.data, self.data.astype(bool))

    @property
    @override
    def size(self) -> int:
        return self.data.size

    @property
    @override
    def shape(self) -> tuple[int, int, int, int]:
        # https://github.com/python/mypy/issues/1178
        return typing.cast(tuple[int, int, int, int], self.data.shape)

    @property
    def in_memory(self) -> bool:
        """Check if underlying :attr:`data` is loaded into memory.

        This method uses protected attributes of underlying `xarray` objects, and may be subject
        to deprecation.

        .. versionchanged:: 0.26.0

            Rename from ``is_loaded`` to ``in_memory``.

        Returns
        -------
        bool
            If underlying data exists as an `np.ndarray` in memory.
        """
        return self.data._in_memory

    @overload
    def interpolate(
        self,
        longitude: float | npt.NDArray[np.floating],
        latitude: float | npt.NDArray[np.floating],
        level: float | npt.NDArray[np.floating],
        time: np.datetime64 | npt.NDArray[np.datetime64],
        *,
        method: str = ...,
        bounds_error: bool = ...,
        fill_value: float | np.float64 | None = ...,
        localize: bool = ...,
        lowmem: bool = ...,
        indices: interpolation.RGIArtifacts | None = ...,
        return_indices: Literal[False] = ...,
    ) -> npt.NDArray[np.floating]: ...

    @overload
    def interpolate(
        self,
        longitude: float | npt.NDArray[np.floating],
        latitude: float | npt.NDArray[np.floating],
        level: float | npt.NDArray[np.floating],
        time: np.datetime64 | npt.NDArray[np.datetime64],
        *,
        method: str = ...,
        bounds_error: bool = ...,
        fill_value: float | np.float64 | None = ...,
        localize: bool = ...,
        lowmem: bool = ...,
        indices: interpolation.RGIArtifacts | None = ...,
        return_indices: Literal[True],
    ) -> tuple[npt.NDArray[np.floating], interpolation.RGIArtifacts]: ...

    def interpolate(
        self,
        longitude: float | npt.NDArray[np.floating],
        latitude: float | npt.NDArray[np.floating],
        level: float | npt.NDArray[np.floating],
        time: np.datetime64 | npt.NDArray[np.datetime64],
        *,
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: float | np.float64 | None = np.nan,
        localize: bool = False,
        lowmem: bool = False,
        indices: interpolation.RGIArtifacts | None = None,
        return_indices: bool = False,
    ) -> npt.NDArray[np.floating] | tuple[npt.NDArray[np.floating], interpolation.RGIArtifacts]:
        """Interpolate values over underlying DataArray.

        Zero dimensional coordinates are reshaped to 1D arrays.

        If ``lowmem == False``, method automatically loads underlying :attr:`data` into
        memory. Otherwise, method iterates through smaller subsets of :attr:`data` and releases
        subsets from memory once interpolation against each subset is finished.

        If ``method == "nearest"``, the out array will have the same ``dtype`` as
        the underlying :attr:`data`.

        If ``method == "linear"``, the out array will be promoted to the most
        precise ``dtype`` of:

        - underlying :attr:`data`
        - :attr:`data.longitude`
        - :attr:`data.latitude`
        - :attr:`data.level`
        - ``longitude``
        - ``latitude``

        .. versionadded:: 0.24

            This method can now handle singleton dimensions with ``method == "linear"``.
            Previously these degenerate dimensions caused nan values to be returned.

        Parameters
        ----------
        longitude : float | npt.NDArray[np.floating]
            Longitude values to interpolate. Assumed to be 0 or 1 dimensional.
        latitude : float | npt.NDArray[np.floating]
            Latitude values to interpolate. Assumed to be 0 or 1 dimensional.
        level : float | npt.NDArray[np.floating]
            Level values to interpolate. Assumed to be 0 or 1 dimensional.
        time : np.datetime64 | npt.NDArray[np.datetime64]
            Time values to interpolate. Assumed to be 0 or 1 dimensional.
        method: str, optional
            Additional keyword arguments to pass to
            :class:`scipy.interpolate.RegularGridInterpolator`.
            Defaults to "linear".
        bounds_error: bool, optional
            Additional keyword arguments to pass to
            :class:`scipy.interpolate.RegularGridInterpolator`.
            Defaults to ``False``.
        fill_value: float | np.float64, optional
            Additional keyword arguments to pass to
            :class:`scipy.interpolate.RegularGridInterpolator`.
            Set to None to extrapolate outside the boundary when ``method`` is ``nearest``.
            Defaults to ``np.nan``.
        localize: bool, optional
            Experimental. If True, downselect gridded data to smallest bounding box containing
            all points.  By default False.
        lowmem: bool, optional
            Experimental. If True, iterate through points binned by the time coordinate of the
            grided data, and downselect gridded data to the smallest bounding box containing
            each binned set of point *before loading into memory*. This can significantly reduce
            memory consumption with large numbers of points at the cost of increased runtime.
            By default False.
        indices: tuple | None, optional
            Experimental. See :func:`interpolation.interp`. None by default.
        return_indices: bool, optional
            Experimental. See :func:`interpolation.interp`. False by default.
            Note that values returned differ when ``lowmem=True`` and ``lowmem=False``,
            so output should only be re-used in calls with the same ``lowmem`` value.

        Returns
        -------
        np.ndarray
            Interpolated values

        See Also
        --------
        :meth:`GeoVectorDataset.intersect_met`

        Examples
        --------
        >>> from datetime import datetime
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pycontrails.datalib.ecmwf import ERA5

        >>> times = (datetime(2022, 3, 1, 12),  datetime(2022, 3, 1, 15))
        >>> variables = "air_temperature"
        >>> levels = [200, 250, 300]
        >>> era5 = ERA5(times, variables, levels)
        >>> met = era5.open_metdataset()
        >>> mda = met["air_temperature"]

        >>> # Interpolation at a grid point agrees with value
        >>> mda.interpolate(1, 2, 300, np.datetime64('2022-03-01T14:00'))
        array([241.91972984])

        >>> da = mda.data
        >>> da.sel(longitude=1, latitude=2, level=300, time=np.datetime64('2022-03-01T14')).item()
        241.9197298421629

        >>> # Interpolation off grid
        >>> mda.interpolate(1.1, 2.1, 290, np.datetime64('2022-03-01 13:10'))
        array([239.83793798])

        >>> # Interpolate along path
        >>> longitude = np.linspace(1, 2, 10)
        >>> latitude = np.linspace(2, 3, 10)
        >>> level = np.linspace(200, 300, 10)
        >>> time = pd.date_range("2022-03-01T14", periods=10, freq="5min")
        >>> mda.interpolate(longitude, latitude, level, time)
        array([220.44347694, 223.08900738, 225.74338924, 228.41642088,
               231.10858599, 233.54857391, 235.71504913, 237.86478872,
               239.99274623, 242.10792167])

        >>> # Can easily switch to alternative low-memory implementation
        >>> mda.interpolate(longitude, latitude, level, time, lowmem=True)
        array([220.44347694, 223.08900738, 225.74338924, 228.41642088,
               231.10858599, 233.54857391, 235.71504913, 237.86478872,
               239.99274623, 242.10792167])
        """
        if lowmem:
            return self._interp_lowmem(
                longitude,
                latitude,
                level,
                time,
                method=method,
                bounds_error=bounds_error,
                fill_value=fill_value,
                indices=indices,
                return_indices=return_indices,
            )

        # Load if necessary
        if not self.in_memory:
            self._check_memory("Interpolation over")
            self.data.load()

        # Convert all inputs to 1d arrays
        # Not validating against ndim >= 2
        longitude, latitude, level, time = np.atleast_1d(longitude, latitude, level, time)

        # Pass off to the interp function, which does all the heavy lifting
        return interpolation.interp(
            longitude=longitude,
            latitude=latitude,
            level=level,
            time=time,
            da=self.data,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
            localize=localize,
            indices=indices,
            return_indices=return_indices,
        )

    def _interp_lowmem(
        self,
        longitude: float | npt.NDArray[np.floating],
        latitude: float | npt.NDArray[np.floating],
        level: float | npt.NDArray[np.floating],
        time: np.datetime64 | npt.NDArray[np.datetime64],
        *,
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: float | np.float64 | None = np.nan,
        indices: interpolation.RGIArtifacts | None = None,
        return_indices: bool = False,
    ) -> npt.NDArray[np.floating] | tuple[npt.NDArray[np.floating], interpolation.RGIArtifacts]:
        """Interpolate values against underlying DataArray.

        This method is used by :meth:`interpolate` when ``lowmem=True``.
        Parameters and return types are identical to :meth:`interpolate`, except
        that the ``localize`` keyword argument is omitted.
        """
        # Convert all inputs to 1d arrays
        # Not validating against ndim >= 2
        longitude, latitude, level, time = np.atleast_1d(longitude, latitude, level, time)

        if bounds_error:
            _lowmem_boundscheck(time, self.data)

        # Create buffers for holding interpolation output
        # Use np.full rather than np.empty so points not covered
        # by masks are filled with correct out-of-bounds values.
        out = np.full(longitude.shape, fill_value, dtype=self.data.dtype)
        if return_indices:
            rgi_artifacts = interpolation.RGIArtifacts(
                xi_indices=np.full((4, longitude.size), -1, dtype=np.int64),
                norm_distances=np.full((4, longitude.size), np.nan, dtype=np.float64),
                out_of_bounds=np.full((longitude.size,), True, dtype=np.bool_),
            )

        # Iterate over portions of points between adjacent time steps in gridded data
        for mask in _lowmem_masks(time, self.data["time"].values):
            if mask is None or not np.any(mask):
                continue

            lon_sl = longitude[mask]
            lat_sl = latitude[mask]
            lev_sl = level[mask]
            t_sl = time[mask]
            if indices is not None:
                indices_sl = interpolation.RGIArtifacts(
                    xi_indices=indices.xi_indices[:, mask],
                    norm_distances=indices.norm_distances[:, mask],
                    out_of_bounds=indices.out_of_bounds[mask],
                )
            else:
                indices_sl = None

            coords = {"longitude": lon_sl, "latitude": lat_sl, "level": lev_sl, "time": t_sl}
            if any(np.all(np.isnan(coord)) for coord in coords.values()):
                continue
            da = interpolation._localize(self.data, coords)
            if not da._in_memory:
                logger.debug(
                    "Loading %s MB subset of %s into memory.",
                    round(da.nbytes / 1_000_000, 2),
                    da.name,
                )
                da.load()

            if return_indices:
                out[mask], rgi_sl = interpolation.interp(
                    longitude=lon_sl,
                    latitude=lat_sl,
                    level=lev_sl,
                    time=t_sl,
                    da=da,
                    method=method,
                    bounds_error=bounds_error,
                    fill_value=fill_value,
                    localize=False,  # would be no-op; da is localized already
                    indices=indices_sl,
                    return_indices=return_indices,
                )
                rgi_artifacts.xi_indices[:, mask] = rgi_sl.xi_indices
                rgi_artifacts.norm_distances[:, mask] = rgi_sl.norm_distances
                rgi_artifacts.out_of_bounds[mask] = rgi_sl.out_of_bounds
            else:
                out[mask] = interpolation.interp(
                    longitude=lon_sl,
                    latitude=lat_sl,
                    level=lev_sl,
                    time=t_sl,
                    da=da,
                    method=method,
                    bounds_error=bounds_error,
                    fill_value=fill_value,
                    localize=False,  # would be no-op; da is localized already
                    indices=indices_sl,
                    return_indices=return_indices,
                )

        if return_indices:
            return out, rgi_artifacts
        return out

    def _check_memory(self, msg_start: str) -> None:
        """Check the memory usage of the underlying data.

        If the data is larger than 4 GB, a warning is issued. If the data is
        larger than 32 GB, a RuntimeError is raised.
        """
        n_bytes = self.data.nbytes
        mb = round(n_bytes / int(1e6), 2)
        logger.debug("Loading %s into memory consumes %s MB.", self.name, mb)

        n_gb = n_bytes // int(1e9)
        if n_gb <= 4:
            return

        # Prevent something stupid
        msg = (
            f"{msg_start} MetDataArray {self.name} requires loading "
            f"at least {n_gb} GB of data into memory. Downselect data if possible. "
            "If working with a GeoVectorDataset instance, this can be achieved "
            "with the method 'downselect_met'."
        )

        if n_gb > 32:
            raise RuntimeError(msg)
        warnings.warn(msg)

    def save(self, **kwargs: Any) -> list[str]:
        """Save intermediate to :attr:`cachestore` as netcdf.

        Load and restore using :meth:`load`.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed directly to :func:`xarray.save_mfdataset`

        Returns
        -------
        list[str]
            Returns filenames of saved files
        """
        dataset = self.data.to_dataset()
        return self._save(dataset, **kwargs)

    @classmethod
    def load(
        cls,
        hash: str,
        cachestore: CacheStore | None = None,
        chunks: dict[str, int] | None = None,
    ) -> Self:
        """Load saved intermediate from :attr:`cachestore`.

        Parameters
        ----------
        hash : str
            Saved hash to load.
        cachestore : CacheStore, optional
            Cache datastore to use for sourcing files.
            Defaults to DiskCacheStore.
        chunks : dict[str, int], optional
            Chunks kwarg passed to :func:`xarray.open_mfdataset()` when opening files.

        Returns
        -------
        MetDataArray
            New MetDataArray with loaded data.
        """
        cachestore = cachestore or DiskCacheStore()
        chunks = chunks or {}
        data = _load(hash, cachestore, chunks)
        return cls(data[next(iter(data.data_vars))])

    @property
    def proportion(self) -> float:
        """Compute proportion of points with value 1.

        Returns
        -------
        float
            Proportion of points with value 1

        Raises
        ------
        NotImplementedError
            If instance does not contain binary data.
        """
        if not self.binary:
            raise NotImplementedError("proportion method is only implemented for binary fields")

        return self.data.sum().values.item() / self.data.count().values.item()  # type: ignore[operator]

    def find_edges(self) -> Self:
        """Find edges of regions.

        Returns
        -------
        Self
            MetDataArray with a binary field, 1 on the edge of the regions,
            0 outside and inside the regions.

        Raises
        ------
        NotImplementedError
            If the instance is not binary.
        """
        if not self.binary:
            raise NotImplementedError("find_edges method is only implemented for binary fields")

        # edge detection algorithm using differentiation to reduce the areas to lines
        def _edges(da: xr.DataArray) -> xr.DataArray:
            lat_diff = da.differentiate("latitude")
            lon_diff = da.differentiate("longitude")
            diff = da.where((lat_diff != 0) | (lon_diff != 0), 0)

            # TODO: what is desired behavior here?
            # set boundaries to close contour regions
            diff[dict(longitude=0)] = da[dict(longitude=0)]
            diff[dict(longitude=-1)] = da[dict(longitude=-1)]
            diff[dict(latitude=0)] = da[dict(latitude=0)]
            diff[dict(latitude=-1)] = da[dict(latitude=-1)]

            return diff

        # load data into memory (required for value assignment in _edges()
        self.data.load()

        data = self.data.groupby("level", squeeze=False).map(_edges)
        return type(self)(data, cachestore=self.cachestore)

    def to_polygon_feature(
        self,
        level: float | int | None = None,
        time: np.datetime64 | datetime | None = None,
        fill_value: float = np.nan,
        iso_value: float | None = None,
        min_area: float = 0.0,
        epsilon: float = 0.0,
        lower_bound: bool = True,
        precision: int | None = None,
        interiors: bool = True,
        convex_hull: bool = False,
        include_altitude: bool = False,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create GeoJSON Feature artifact from spatial array on a single level and time slice.

        Computed polygons always contain an exterior linear ring as defined by the
        `GeoJSON Polygon specification <https://www.rfc-editor.org/rfc/rfc7946.html#section-3.1.6>`.
        Polygons may also contain interior linear rings (holes). This method does not support
        nesting beyond the GeoJSON specification. See the :mod:`pycontrails.core.polygon`
        for additional polygon support.

        .. versionchanged:: 0.25.12

            Previous implementation include several additional parameters which have
            been removed:

            - The ``approximate`` parameter
            - An ``path`` parameter to save output as JSON
            - Passing arbitrary kwargs to :func:`skimage.measure.find_contours`.

            New implementation includes new parameters previously lacking:

            - ``fill_value``
            - ``min_area``
            - ``include_altitude``

        .. versionchanged:: 0.38.0

            Change default value of ``epsilon`` from 0.15 to 0.

        .. versionchanged:: 0.41.0

            Convert continuous fields to binary fields before computing polygons.
            The parameters ``max_area`` and ``epsilon`` are now expressed in terms of
            longitude/latitude units instead of pixels.

        Parameters
        ----------
        level : float, optional
            Level slice to create polygons.
            If the "level" coordinate is length 1, then the single level slice will be selected
            automatically.
        time : datetime, optional
            Time slice to create polygons.
            If the "time" coordinate is length 1, then the single time slice will be selected
            automatically.
        fill_value : float, optional
            Value used for filling missing data and for padding the underlying data array.
            Set to ``np.nan`` by default, which ensures that regions with missing data are
            never included in polygons.
        iso_value : float, optional
            Value in field to create iso-surface.
            Defaults to the average of the min and max value of the array. (This is the
            same convention as used by ``skimage``.)
        min_area : float, optional
            Minimum area of each polygon. Polygons with area less than ``min_area`` are
            not included in the output. The unit of this parameter is in longitude/latitude
            degrees squared. Set to 0 to omit any polygon filtering based on a minimal area
            conditional. By default, 0.0.
        epsilon : float, optional
            Control the extent to which the polygon is simplified. A value of 0 does not alter
            the geometry of the polygon. The unit of this parameter is in longitude/latitude
            degrees. By default, 0.0.
        lower_bound : bool, optional
            Whether to use ``iso_value`` as a lower or upper bound on values in polygon interiors.
            By default, True.
        precision : int, optional
            Number of decimal places to round coordinates to. If None, no rounding is performed.
        interiors : bool, optional
            If True, include interior linear rings (holes) in the output. True by default.
        convex_hull : bool, optional
            EXPERIMENTAL. If True, compute the convex hull of each polygon. Only implemented
            for depth=1. False by default. A warning is issued if the underlying algorithm
            fails to make valid polygons after computing the convex hull.
        include_altitude : bool, optional
            If True, include the array altitude [:math:`m`] as a z-coordinate in the
            `GeoJSON output <https://www.rfc-editor.org/rfc/rfc7946#section-3.1.1>`.
            False by default.
        properties : dict, optional
            Additional properties to include in the GeoJSON output. By default, None.

        Returns
        -------
        dict[str, Any]
            Python representation of GeoJSON Feature with MultiPolygon geometry.

        Notes
        -----
        :class:`Cocip` and :class:`CocipGrid` set some quantities to 0 and other quantities
        to ``np.nan`` in regions where no contrails form. When computing polygons from
        :class:`Cocip` or :class:`CocipGrid` output, take care that the choice of
        ``fill_value`` correctly includes or excludes contrail-free regions. See the
        :class:`Cocip` documentation for details about ``np.nan`` in model output.

        See Also
        --------
        :meth:`to_polyhedra`
        :func:`polygons.find_multipolygons`

        Examples
        --------
        >>> from pprint import pprint
        >>> from pycontrails.datalib.ecmwf import ERA5
        >>> era5 = ERA5("2022-03-01", variables="air_temperature", pressure_levels=250)
        >>> mda = era5.open_metdataset()["air_temperature"]
        >>> mda.shape
        (1440, 721, 1, 1)

        >>> pprint(mda.to_polygon_feature(iso_value=239.5, precision=2, epsilon=0.1))
        {'geometry': {'coordinates': [[[[167.88, -22.5],
                                        [167.75, -22.38],
                                        [167.62, -22.5],
                                        [167.75, -22.62],
                                        [167.88, -22.5]]],
                                      [[[43.38, -33.5],
                                        [43.5, -34.12],
                                        [43.62, -33.5],
                                        [43.5, -33.38],
                                        [43.38, -33.5]]]],
                      'type': 'MultiPolygon'},
         'properties': {},
         'type': 'Feature'}

        """
        if convex_hull and interiors:
            raise ValueError("Set 'interiors=False' to use the 'convex_hull' parameter.")

        arr, altitude = _extract_2d_arr_and_altitude(self, level, time)
        if not include_altitude:
            altitude = None  # this logic used below
        elif altitude is None:
            raise ValueError(
                "The parameter 'include_altitude' is True, but altitude is not "
                "found on MetDataArray instance. Either set altitude, or pass "
                "include_altitude=False."
            )

        if not np.isnan(fill_value):
            np.nan_to_num(arr, copy=False, nan=fill_value)

        # default iso_value
        if iso_value is None:
            iso_value = (np.nanmax(arr) + np.nanmin(arr)) / 2
            warnings.warn(f"The 'iso_value' parameter was not specified. Using value: {iso_value}")

        # We'll get a nice error message if dependencies are not installed
        from pycontrails.core import polygon

        # Convert to nested lists of coordinates for GeoJSON representation
        indexes = self.indexes
        longitude = indexes["longitude"].to_numpy()
        latitude = indexes["latitude"].to_numpy()

        mp = polygon.find_multipolygon(
            arr,
            threshold=iso_value,
            min_area=min_area,
            epsilon=epsilon,
            lower_bound=lower_bound,
            interiors=interiors,
            convex_hull=convex_hull,
            longitude=longitude,
            latitude=latitude,
            precision=precision,
        )

        return polygon.multipolygon_to_geojson(mp, altitude, properties)

    def to_polygon_feature_collection(
        self,
        time: np.datetime64 | datetime | None = None,
        fill_value: float = np.nan,
        iso_value: float | None = None,
        min_area: float = 0.0,
        epsilon: float = 0.0,
        lower_bound: bool = True,
        precision: int | None = None,
        interiors: bool = True,
        convex_hull: bool = False,
        include_altitude: bool = False,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create GeoJSON FeatureCollection artifact from spatial array at time slice.

        See the :meth:`to_polygon_feature` method for a description of the parameters.

        Returns
        -------
        dict[str, Any]
            Python representation of GeoJSON FeatureCollection. This dictionary is
            comprised of individual GeoJON Features, one per :attr:`self.data["level"]`.
        """
        base_properties = properties or {}
        features = []
        for level in self.data["level"]:
            properties = base_properties.copy()
            properties.update(level=level.item())
            properties.update({f"level_{k}": v for k, v in self.data["level"].attrs.items()})

            feature = self.to_polygon_feature(
                level=level,
                time=time,
                fill_value=fill_value,
                iso_value=iso_value,
                min_area=min_area,
                epsilon=epsilon,
                lower_bound=lower_bound,
                precision=precision,
                interiors=interiors,
                convex_hull=convex_hull,
                include_altitude=include_altitude,
                properties=properties,
            )
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    @overload
    def to_polyhedra(
        self,
        *,
        time: datetime | None = ...,
        iso_value: float = ...,
        simplify_fraction: float = ...,
        lower_bound: bool = ...,
        return_type: Literal["geojson"],
        path: str | None = ...,
        altitude_scale: float = ...,
        output_vertex_normals: bool = ...,
        closed: bool = ...,
    ) -> dict: ...

    @overload
    def to_polyhedra(
        self,
        *,
        time: datetime | None = ...,
        iso_value: float = ...,
        simplify_fraction: float = ...,
        lower_bound: bool = ...,
        return_type: Literal["mesh"],
        path: str | None = ...,
        altitude_scale: float = ...,
        output_vertex_normals: bool = ...,
        closed: bool = ...,
    ) -> o3d.geometry.TriangleMesh: ...

    def to_polyhedra(
        self,
        *,
        time: datetime | None = None,
        iso_value: float = 0.0,
        simplify_fraction: float = 1.0,
        lower_bound: bool = True,
        return_type: str = "geojson",
        path: str | None = None,
        altitude_scale: float = 1.0,
        output_vertex_normals: bool = False,
        closed: bool = True,
    ) -> dict | o3d.geometry.TriangleMesh:
        """Create a collection of polyhedra from spatial array corresponding to a single time slice.

        Parameters
        ----------
        time : datetime, optional
            Time slice to create mesh.
            If the "time" coordinate is length 1, then the single time slice will be selected
            automatically.
        iso_value : float, optional
            Value in field to create iso-surface. Defaults to 0.
        simplify_fraction : float, optional
            Apply `open3d` `simplify_quadric_decimation` method to simplify the polyhedra geometry.
            This parameter must be in the half-open interval (0.0, 1.0].
            Defaults to 1.0, corresponding to no reduction.
        lower_bound : bool, optional
            Whether to use ``iso_value`` as a lower or upper bound on values in polyhedra interiors.
            By default, True.
        return_type : str, optional
            Must be one of "geojson" or "mesh". Defaults to "geojson".
            If "geojson", this method returns a dictionary representation of a geojson MultiPolygon
            object whose polygons are polyhedra faces.
            If "mesh", this method returns an `open3d` `TriangleMesh` instance.
        path : str, optional
            Output geojson or mesh to file.
            If `return_type` is "mesh", see `Open3D File I/O for Mesh
            <http://www.open3d.org/docs/release/tutorial/geometry/file_io.html#Mesh>`_ for
            file type options.
        altitude_scale : float, optional
            Rescale the altitude dimension of the mesh, [:math:`m`]
        output_vertex_normals : bool, optional
            If ``path`` is defined, write out vertex normals.
            Defaults to False.
        closed : bool, optional
            If True, pad spatial array along all axes to ensure polyhedra are "closed".
            This flag often gives rise to cleaner visualizations. Defaults to True.

        Returns
        -------
        dict | open3d.geometry.TriangleMesh
            Python representation of geojson object or `Open3D Triangle Mesh
            <http://www.open3d.org/docs/release/tutorial/geometry/mesh.html>`_ depending on the
            `return_type` parameter.

        Raises
        ------
        ModuleNotFoundError
            Method requires the `vis` optional dependencies
        ValueError
            If input parameters are invalid.

        See Also
        --------
        :meth:`to_polygon_feature`
        :func:`skimage.measure.marching_cubes`
        :class:`open3d.geometry.TriangleMesh`

        Notes
        -----
        Uses the `scikit-image Marching Cubes  <https://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html>`_
        algorithm to reconstruct a surface from the point-cloud like arrays.
        """
        try:
            from skimage import measure
        except ModuleNotFoundError as e:
            dependencies.raise_module_not_found_error(
                name="MetDataArray.to_polyhedra method",
                package_name="scikit-image",
                pycontrails_optional_package="vis",
                module_not_found_error=e,
            )

        try:
            import open3d as o3d
        except ModuleNotFoundError as e:
            dependencies.raise_module_not_found_error(
                name="MetDataArray.to_polyhedra method",
                package_name="open3d",
                pycontrails_optional_package="open3d",
                module_not_found_error=e,
            )

        if len(self.data["level"]) == 1:
            raise ValueError(
                "Found single `level` coordinate in DataArray. This method requires at least two."
            )

        # select time
        if time is None and len(self.data["time"]) == 1:
            time = self.data["time"].values[0]

        if time is None:
            raise ValueError(
                "time input must be defined when the length of the time coordinates are > 1"
            )

        if simplify_fraction > 1 or simplify_fraction <= 0:
            raise ValueError("Parameter `simplify_fraction` must be in the interval (0, 1].")

        return_types = ["geojson", "mesh"]
        if return_type not in return_types:
            raise ValueError(f"Parameter `return_type` must be one of {', '.join(return_types)}")

        # 3d array of longitude, latitude, altitude values
        volume = self.data.sel(time=time).values

        # invert if iso_value is an upper bound on interior values
        if not lower_bound:
            volume = -volume
            iso_value = -iso_value

        # convert from array index back to coordinates
        longitude = self.indexes["longitude"].values
        latitude = self.indexes["latitude"].values
        altitude = units.pl_to_m(self.indexes["level"].values)

        # Pad volume on all axes to close the volumes
        if closed:
            # pad values to domain
            longitude0 = longitude[0] - (longitude[1] - longitude[0])
            longitude1 = longitude[-1] + longitude[-1] - longitude[-2]
            longitude = np.pad(longitude, pad_width=1, constant_values=(longitude0, longitude1))

            latitude0 = latitude[0] - (latitude[1] - latitude[0])
            latitude1 = latitude[-1] + latitude[-1] - latitude[-2]
            latitude = np.pad(latitude, pad_width=1, constant_values=(latitude0, latitude1))

            altitude0 = altitude[0] - (altitude[1] - altitude[0])
            altitude1 = altitude[-1] + altitude[-1] - altitude[-2]
            altitude = np.pad(altitude, pad_width=1, constant_values=(altitude0, altitude1))

            # Pad along axes to ensure polygons are closed
            volume = np.pad(volume, pad_width=1, constant_values=iso_value)

        # Use marching cubes to obtain the surface mesh
        # Coordinates of verts are indexes of volume array
        verts, faces, normals, _ = measure.marching_cubes(
            volume, iso_value, allow_degenerate=False, gradient_direction="ascent"
        )

        # Convert from indexes to longitude, latitude, altitude values
        verts[:, 0] = longitude[verts[:, 0].astype(int)]
        verts[:, 1] = latitude[verts[:, 1].astype(int)]
        verts[:, 2] = altitude[verts[:, 2].astype(int)]

        # rescale altitude
        verts[:, 2] = verts[:, 2] * altitude_scale

        # create mesh in open3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

        # simplify mesh according to sim
        if simplify_fraction < 1:
            target_n_triangles = int(faces.shape[0] * simplify_fraction)
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_n_triangles)
            mesh.compute_vertex_normals()

        if path is not None:
            path = str(pathlib.Path(path).absolute())

        if return_type == "geojson":
            verts = np.round(
                np.asarray(mesh.vertices), decimals=4
            )  # rounding to reduce the size of resultant json arrays
            faces = np.asarray(mesh.triangles)

            # TODO: technically this is not valid GeoJSON because each polygon (triangle)
            # does not have the last element equal to the first (not a linear ring)
            # but it still works for now in Deck.GL
            coords = [[verts[face].tolist()] for face in faces]

            geojson = {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "MultiPolygon", "coordinates": coords},
            }

            if path is not None:
                if not path.endswith(".json"):
                    path += ".json"
                with open(path, "w") as file:
                    json.dump(geojson, file)
            return geojson

        if path is not None:
            o3d.io.write_triangle_mesh(
                path,
                mesh,
                write_ascii=False,
                compressed=True,
                write_vertex_normals=output_vertex_normals,
                write_vertex_colors=False,
                write_triangle_uvs=True,
                print_progress=False,
            )
        return mesh

    @override
    def broadcast_coords(self, name: str) -> xr.DataArray:
        da = xr.ones_like(self.data) * self.data[name]
        da.name = name

        return da


def _is_wrapped(longitude: np.ndarray) -> bool:
    """Check if ``longitude`` covers ``[-180, 180]``."""
    return longitude[0] <= -180.0 and longitude[-1] >= 180.0


def _is_zarr(ds: xr.Dataset | xr.DataArray) -> bool:
    """Check if ``ds`` appears to be Zarr-based.

    Neither ``xarray`` nor ``dask`` readily expose such information, so
    implementation is very narrow and brittle.
    """
    if isinstance(ds, xr.Dataset):
        # Attempt 1: xarray binds the zarr close function to the instance
        # This gives us an indicator of the data is zarr based
        try:
            if ds._close.__func__.__qualname__ == "ZarrStore.close":  # type: ignore
                return True
        except AttributeError:
            pass

        # Grab the first data variable, get underlying DataArray
        ds = ds[next(iter(ds.data_vars))]

    # Attempt 2: Examine the dask graph
    darr = ds.variable._data  # dask array in some cases
    try:
        # Get the first dask instruction
        dask0 = darr.dask[next(iter(darr.dask))]  # type: ignore[union-attr]
    except AttributeError:
        return False
    return dask0.array.array.array.__class__.__name__ == "ZarrArrayWrapper"


def shift_longitude(data: XArrayType, bound: float = -180.0) -> XArrayType:
    """Shift longitude values from any input domain to [bound, 360 + bound) domain.

    Sorts data by ascending longitude values.


    Parameters
    ----------
    data : XArrayType
        :class:`xr.Dataset` or :class:`xr.DataArray` with longitude dimension
    bound : float, optional
        Lower bound of the domain.
        Output domain will be [bound, 360 + bound).
        Defaults to -180, which results in longitude domain [-180, 180).


    Returns
    -------
    XArrayType
        :class:`xr.Dataset` or :class:`xr.DataArray` with longitude values on [a, 360 + a).
    """
    return data.assign_coords(
        longitude=((data["longitude"].values - bound) % 360.0) + bound
    ).sortby("longitude", ascending=True)


def _wrap_longitude(data: XArrayType) -> XArrayType:
    """Wrap longitude grid coordinates.

    This function assumes the longitude dimension on ``data``:

    - is sorted
    - is contained in [-180, 180]

    These assumptions are checked by :class:`MetDataset` and :class`MetDataArray`
    constructors.

    .. versionchanged:: 0.26.0

        This function now ensures every value in the interval ``[-180, 180]``
        is covered by the longitude dimension of the returned object. See
        :meth:`MetDataset.is_wrapped` for more details.


    Parameters
    ----------
    data : XArrayType
        :class:`xr.Dataset` or :class:`xr.DataArray` with longitude dimension

    Returns
    -------
    XArrayType
        Copy of :class:`xr.Dataset` or :class:`xr.DataArray` with wrapped longitude values.

    Raises
    ------
    ValueError
        If longitude values are already wrapped.
    """
    lon = data._indexes["longitude"].index.to_numpy()  # type: ignore[attr-defined]
    if _is_wrapped(lon):
        raise ValueError("Longitude values are already wrapped")

    lon0 = lon[0]
    lon1 = lon[-1]

    # Try to prevent something stupid
    if lon1 - lon0 < 330.0:
        warnings.warn("Wrapping longitude will create a large spatial gap of more than 30 degrees.")

    objs = [data]
    if lon0 > -180.0:  # if the lowest longitude is not low enough, duplicate highest
        dup1 = data.sel(longitude=[lon1]).assign_coords(longitude=[lon1 - 360.0])
        objs.insert(0, dup1)
    if lon1 < 180.0:  # if the highest longitude is not highest enough, duplicate lowest
        dup0 = data.sel(longitude=[lon0]).assign_coords(longitude=[lon0 + 360.0])
        objs.append(dup0)

    # Because we explicitly raise a ValueError if longitude already wrapped,
    # we know that len(objs) > 1, so the concatenation here is nontrivial
    wrapped = xr.concat(objs, dim="longitude")
    wrapped["longitude"] = wrapped["longitude"].astype(lon.dtype, copy=False)

    # If there is only one longitude chunk in parameter data, increment
    # data.chunks can be None, using getattr for extra safety
    # NOTE: This probably doesn't seem to play well with Zarr data ...
    # we don't want to be rechunking them.
    # Ideally we'd raise if data was Zarr-based
    chunks = getattr(data, "chunks", None) or {}
    chunks = dict(chunks)  # chunks can be frozen
    lon_chunks = chunks.get("longitude", ())
    if len(lon_chunks) == 1:
        chunks["longitude"] = (lon_chunks[0] + len(objs) - 1,)
        wrapped = wrapped.chunk(chunks)

    return wrapped


def _extract_2d_arr_and_altitude(
    mda: MetDataArray,
    level: float | int | None,
    time: np.datetime64 | datetime | None,
) -> tuple[np.ndarray, float | None]:
    """Extract underlying 2D array indexed by longitude and latitude.

    Parameters
    ----------
    mda : MetDataArray
        MetDataArray to extract from
    level : float | int | None
        Pressure level to slice at
    time : np.datetime64 | datetime
        Time to slice at

    Returns
    -------
    arr : np.ndarray
        Copy of 2D array at given level and time
    altitude : float | None
        Altitude of slice [:math:`m`]. None if "altitude" not found on ``mda``
        (ie, for surface level :class:`MetDataArray`).
    """
    # Determine level if not specified
    if level is None:
        level_coord = mda.indexes["level"].values
        if len(level_coord) == 1:
            level = level_coord[0]
        else:
            raise ValueError(
                "Parameter 'level' must be defined when the length of the 'level' "
                "coordinates is not 1."
            )

    # Determine time if not specified
    if time is None:
        time_coord = mda.indexes["time"].values
        if len(time_coord) == 1:
            time = time_coord[0]
        else:
            raise ValueError(
                "Parameter 'time' must be defined when the length of the 'time' "
                "coordinates is not 1."
            )

    da = mda.data.sel(level=level, time=time)
    arr = da.values.copy()
    if arr.ndim != 2:
        raise RuntimeError("Malformed data array")

    try:
        altitude = da["altitude"].values.item()  # item not implemented on dask arrays
    except KeyError:
        altitude = None
    else:
        altitude = round(altitude)  # type: ignore[call-overload]

    return arr, altitude  # type: ignore[return-value]


def downselect(data: XArrayType, bbox: tuple[float, ...]) -> XArrayType:
    """Downselect :class:`xr.Dataset` or :class:`xr.DataArray` with spatial bounding box.

    Parameters
    ----------
    data : XArrayType
        xr.Dataset or xr.DataArray to downselect
    bbox : tuple[float, ...]
        Tuple of coordinates defining a spatial bounding box in WGS84 coordinates.

        - For 2D queries, ``bbox`` takes the form ``(west, south, east, north)``
        - For 3D queries, ``bbox`` takes the form
          ``(west, south, min-level, east, north, max-level)``

        with level defined in [:math:`hPa`].

    Returns
    -------
    XArrayType
        Downselected xr.Dataset or xr.DataArray

    Raises
    ------
    ValueError
        If parameter ``bbox`` has wrong length.
    """
    if len(bbox) == 4:
        west, south, east, north = bbox
        level_min = -np.inf
        level_max = np.inf

    elif len(bbox) == 6:
        west, south, level_min, east, north, level_max = bbox

    else:
        raise ValueError(
            f"bbox {bbox} is not length 4 [west, south, east, north] "
            "or length 6 [west, south, min-level, east, north, max-level]"
        )

    if west <= east:
        # Return a view of the data
        # If data is lazy, this will not load the data
        return data.sel(
            longitude=slice(west, east),
            latitude=slice(south, north),
            level=slice(level_min, level_max),
        )

    # In this case, the bbox spans the antimeridian
    # If data is lazy, this will load the data (data.where is not lazy AFAIK)
    cond = (
        (data["latitude"] >= south)
        & (data["latitude"] <= north)
        & (data["level"] >= level_min)
        & (data["level"] <= level_max)
        & ((data["longitude"] >= west) | (data["longitude"] <= east))
    )
    return data.where(cond, drop=True)


def standardize_variables(ds: xr.Dataset, variables: Iterable[MetVariable]) -> xr.Dataset:
    """Rename all variables in dataset from short name to standard name.

    This function does not change any variables in ``ds`` that are not found in ``variables``.

    When there are multiple variables with the same short name, the last one is used.

    Parameters
    ----------
    ds : DatasetType
        An :class:`xr.Dataset`.
    variables : Iterable[MetVariable]
        Data source variables

    Returns
    -------
    DatasetType
        Dataset with variables renamed to standard names
    """
    variables_dict: dict[Hashable, str] = {v.short_name: v.standard_name for v in variables}
    name_dict = {var: variables_dict[var] for var in ds.data_vars if var in variables_dict}
    return ds.rename(name_dict)


def originates_from_ecmwf(met: MetDataset | MetDataArray) -> bool:
    """Check if data appears to have originated from an ECMWF source.

    .. versionadded:: 0.27.0

        Experimental. Implementation is brittle.

    Parameters
    ----------
    met : MetDataset | MetDataArray
        Dataset or array to inspect.

    Returns
    -------
    bool
        True if data appears to be derived from an ECMWF source.

    See Also
    --------
    - :class:`ERA5`
    - :class:`HRES`

    """
    if isinstance(met, MetDataset):
        try:
            return met.provider_attr == "ECMWF"
        except KeyError:
            pass
    return "ecmwf" in met.attrs.get("history", "")


def _load(hash: str, cachestore: CacheStore, chunks: dict[str, int]) -> xr.Dataset:
    """Load xarray data from hash.

    Parameters
    ----------
    hash : str
        Description
    cachestore : CacheStore
        Description
    chunks : dict[str, int]
        Description

    Returns
    -------
    xr.Dataset
        Description
    """
    disk_path = cachestore.get(f"{hash}*.nc")
    return xr.open_mfdataset(disk_path, chunks=chunks)


def _add_vertical_coords(data: XArrayType) -> XArrayType:
    """Add "air_pressure" and "altitude" coordinates to data.

    .. versionchanged:: 0.52.1
        Ensure that the ``dtype`` of the additional vertical coordinates agree
        with the ``dtype`` of the underlying gridded data.
    """

    data["level"].attrs.update(units="hPa", long_name="Pressure", positive="down")

    # XXX: use the dtype of the data to determine the precision of these coordinates
    # There are two competing conventions here:
    # - coordinate data should be float64
    # - gridded data is typically float32
    # - air_pressure and altitude often play both roles
    # It is more important for air_pressure and altitude to be grid-aligned than to be
    # coordinate-aligned, so we use the dtype of the data to determine the precision of
    # these coordinates
    dtype = (
        np.result_type(*data.data_vars.values(), np.float32)
        if isinstance(data, xr.Dataset)
        else data.dtype
    )

    level = data["level"].values

    if "air_pressure" not in data.coords:
        data = data.assign_coords(air_pressure=("level", level * 100.0))
        data.coords["air_pressure"].attrs.update(
            standard_name=AirPressure.standard_name,
            long_name=AirPressure.long_name,
            units=AirPressure.units,
        )
    if data.coords["air_pressure"].dtype != dtype:
        data.coords["air_pressure"] = data.coords["air_pressure"].astype(dtype, copy=False)

    if "altitude" not in data.coords:
        data = data.assign_coords(altitude=("level", units.pl_to_m(level)))
        data.coords["altitude"].attrs.update(
            standard_name=Altitude.standard_name,
            long_name=Altitude.long_name,
            units=Altitude.units,
        )
    if data.coords["altitude"].dtype != dtype:
        data.coords["altitude"] = data.coords["altitude"].astype(dtype, copy=False)

    return data


def _lowmem_boundscheck(time: npt.NDArray[np.datetime64], da: xr.DataArray) -> None:
    """Extra bounds check required with low-memory interpolation strategy.

    Because the main loop in `_interp_lowmem` processes points between time steps
    in gridded data, it will never encounter points that are out-of-bounds in time
    and may fail to produce requested out-of-bounds errors.
    """
    da_time = da["time"].to_numpy()
    if not np.all((time >= da_time.min()) & (time <= da_time.max())):
        axis = da.get_axis_num("time")
        msg = f"One of the requested xi is out of bounds in dimension {axis}"
        raise ValueError(msg)


def _lowmem_masks(
    time: npt.NDArray[np.datetime64], t_met: npt.NDArray[np.datetime64]
) -> Generator[npt.NDArray[np.bool_], None, None]:
    """Generate sequence of masks for low-memory interpolation."""
    t_met_max = t_met.max()
    t_met_min = t_met.min()
    inbounds = (time >= t_met_min) & (time <= t_met_max)
    if not np.any(inbounds):
        return

    earliest = np.nanmin(time)
    istart = 0 if earliest < t_met_min else np.flatnonzero(t_met <= earliest).max()
    latest = np.nanmax(time)
    iend = t_met.size - 1 if latest > t_met_max else np.flatnonzero(t_met >= latest).min()
    if istart == iend:
        yield inbounds
        return

    # Sequence of masks covers elements in time in the interval [t_met[istart], t_met[iend]].
    # The first iteration masks elements in the interval [t_met[istart], t_met[istart+1]]
    # (inclusive of both endpoints).
    # Subsequent iterations mask elements in the interval (t_met[i], t_met[i+1]]
    # (inclusive of right endpoint only).
    for i in range(istart, iend):
        mask = ((time >= t_met[i]) if i == istart else (time > t_met[i])) & (time <= t_met[i + 1])
        if np.any(mask):
            yield mask


def maybe_downselect_mds(
    big_mds: MetDataset,
    little_mds: MetDataset | None,
    t0: np.datetime64,
    t1: np.datetime64,
) -> MetDataset:
    """Possibly downselect ``big_mds`` in the time domain to cover ``[t0, t1]``.

    If possible, ``little_mds`` is recycled to avoid re-loading data.

    This implementation assumes ``t0 <= t1``, but this is not enforced.

    If ``little_mds`` already covers the time range, it is returned as-is.

    If ``big_mds`` doesn't cover the time range, no error is raised.

    Parameters
    ----------
    big_mds : MetDataset
        Larger MetDataset
    little_mds : MetDataset | None
        Smaller MetDataset. This is assumed to be a subset of ``big_mds``,
        though the implementation may work if this is not the case.
    t0, t1 : np.datetime64
        Time range to cover

    Returns
    -------
    MetDataset
        MetDataset covering the time range ``[t0, t1]`` comprised of data from
        ``little_mds`` when possible, otherwise from ``big_mds``.
    """
    if little_mds is None:
        big_time = big_mds.indexes["time"].values
        i0 = np.searchsorted(big_time, t0, side="right").item()
        i0 = max(0, i0 - 1)
        i1 = np.searchsorted(big_time, t1, side="left").item()
        i1 = min(i1 + 1, big_time.size)
        return MetDataset._from_fastpath(big_mds.data.isel(time=slice(i0, i1)))

    little_time = little_mds.indexes["time"].values
    if t0 >= little_time[0] and t1 <= little_time[-1]:
        return little_mds

    big_time = big_mds.indexes["time"].values
    i0 = np.searchsorted(big_time, t0, side="right").item()
    i0 = max(0, i0 - 1)
    i1 = np.searchsorted(big_time, t1, side="left").item()
    i1 = min(i1 + 1, big_time.size)
    big_ds = big_mds.data.isel(time=slice(i0, i1))
    big_time = big_ds._indexes["time"].index.values  # type: ignore[attr-defined]

    # Select exactly the times in big_ds that are not in little_ds
    _, little_indices, big_indices = np.intersect1d(
        little_time, big_time, assume_unique=True, return_indices=True
    )
    little_ds = little_mds.data.isel(time=little_indices)
    filt = np.ones_like(big_time, dtype=bool)
    filt[big_indices] = False
    big_ds = big_ds.isel(time=filt)

    # Manually load relevant parts of big_ds into memory before xr.concat
    # It appears that without this, xr.concat will forget the in-memory
    # arrays in little_ds
    for var, da in little_ds.items():
        if da._in_memory:
            da2 = big_ds[var]
            if not da2._in_memory:
                da2.load()

    ds = xr.concat([little_ds, big_ds], dim="time")
    if not ds._indexes["time"].index.is_monotonic_increasing:  # type: ignore[attr-defined]
        # Rarely would we enter this: t0 would have to be before the first
        # time in little_mds, and the various advection-based models generally
        # proceed forward in time.
        ds = ds.sortby("time")
    return MetDataset._from_fastpath(ds)
