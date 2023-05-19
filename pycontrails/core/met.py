"""Meteorology data models."""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import typing
import warnings
from abc import ABC, abstractmethod
from contextlib import ExitStack
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt
import xarray as xr
from overrides import overrides

from pycontrails.core import interpolation
from pycontrails.core import vector as vector_module
from pycontrails.core.cache import CacheStore, DiskCacheStore
from pycontrails.core.met_var import AirPressure, Altitude, MetVariable
from pycontrails.physics import units
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

    #: DataArray or Dataset
    data: XArrayType

    #: Cache datastore to use for :meth:`save` or :meth:`load`
    cachestore: CacheStore | None

    #: Default dimension order for DataArray or Dataset (x, y, z, t)
    dim_order: list[Hashable] = [
        "longitude",
        "latitude",
        "level",
        "time",
    ]

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
        for dim in self.dim_order:
            if dim not in self.data.dims:
                if dim == "level":
                    raise ValueError(
                        f"Meteorology data must contain dimension '{dim}'. "
                        "For single level data, set 'level' coordinate to constant -1 "
                        "using `ds = ds.expand_dims({'level': [-1]})`"
                    )
                else:
                    raise ValueError(f"Meteorology data must contain dimension '{dim}'.")

    def _validate_longitude(self) -> None:
        """Check longitude bounds.

        Assumes ``longitude`` dimension is already sorted.

        Raises
        ------
        ValueError
            If longitude values are not contained in the interval [-180, 180].
        """
        longitude = self.variables["longitude"].values
        if longitude.dtype != COORD_DTYPE:
            raise ValueError(
                "Longitude values must be of type float64. "
                "Initiate with 'copy=True' to convert to float64. "
                "Initiate with 'validate=False' to skip validation."
            )

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
        latitude = self.variables["latitude"].values
        if latitude.dtype != COORD_DTYPE:
            raise ValueError(
                "Latitude values must be of type float64. "
                "Initiate with 'copy=True' to convert to float64. "
                "Initiate with 'validate=False' to skip validation."
            )

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
        if not np.all(np.diff(self.variables["time"]) > np.timedelta64(0, "ns")):
            raise ValueError("Coordinate `time` not sorted. Initiate with `copy=True`.")
        for coord in self.dim_order[:3]:  # exclude time, the 4th dimension
            if not np.all(np.diff(self.variables[coord]) > 0.0):
                raise ValueError(f"Coordinate '{coord}' not sorted. Initiate with 'copy=True'.")

    def _validate_transpose(self) -> None:
        """Check that data is transposed according to :attr:`dim_order`."""

        dims_tuple = tuple(self.dim_order)

        def _check_da(da: xr.DataArray, key: str | None = None) -> None:
            if da.dims != dims_tuple:
                if key is not None:
                    msg = (
                        "Data dimension not transposed on variable '{key}'. Initiate with"
                        " `copy=True`."
                    )
                else:
                    msg = "Data dimension not transposed. Initiate with `copy=True`."
                raise ValueError(msg)

        data = self.data
        if isinstance(data, xr.DataArray):
            _check_da(data)
            return

        for key, da in self.data.data_vars.items():
            _check_da(da, key)

    def _validate_dims(self) -> None:
        """Apply all validators."""
        self._validate_dim_contains_coords()

        # Apply this one first: validate_longitude and validate_latitude assume sorted
        self._validate_sorting()
        self._validate_longitude()
        self._validate_latitude()
        self._validate_transpose()

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
        for coord in ("longitude", "latitude", "level"):
            arr = self.variables[coord].values
            if arr.dtype != COORD_DTYPE:
                self.data[coord] = arr.astype(COORD_DTYPE)

        # Ensure time is np.datetime64[ns]
        self.data["time"] = self.data["time"].astype("datetime64[ns]", copy=False)

        # sortby to ensure each coordinate has ascending order
        self.data = self.data.sortby(self.dim_order, ascending=True)

        if not self.is_wrapped:
            # Ensure longitude is contained in interval [-180, 180)
            # If longitude has value at 180, we might not want to shift it?
            lon = self.variables["longitude"].values

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
        dim_order = self.dim_order + [d for d in self.data.dims if d not in self.dim_order]
        self.data = self.data.transpose(*dim_order)

        # single level data
        if self.is_single_level:
            # add level attributes to reflect surface level
            self.data["level"].attrs.update(units="", long_name="Single Level")
            return

        # pressure level data
        level = self.variables["level"].values

        # add pressure level attributes
        self.data["level"].attrs.update(units="hPa", long_name="Pressure", positive="down")

        # add altitude and air_pressure

        # XXX: use the dtype of the data to determine the precision of these coordinates
        # There are two competing conventions here:
        # - coordinate data should be float64
        # - gridded data is typically float32
        # - air_pressure and altitude often play both roles
        # It is more important for air_pressure and altitude to be grid-aligned than to be
        # coordinate-aligned, so we use the dtype of the data to determine the precision of
        # these coordinates
        if isinstance(self.data, xr.Dataset):
            dtype = np.result_type(*self.data.data_vars.values(), np.float32)
        else:
            dtype = self.data.dtype

        level = level.astype(dtype)
        air_pressure = level * 100.0
        altitude = units.pl_to_m(level)
        self.data = self.data.assign_coords({"air_pressure": ("level", air_pressure)})
        self.data = self.data.assign_coords({"altitude": ("level", altitude)})

        # add air_pressure units and long name attributes
        self.data.coords["air_pressure"].attrs.update(
            standard_name=AirPressure.standard_name,
            long_name=AirPressure.long_name,
            units=AirPressure.units,
        )
        # add altitude units and long name attributes
        self.data.coords["altitude"].attrs.update(
            standard_name=Altitude.standard_name,
            long_name=Altitude.long_name,
            units=Altitude.units,
        )

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
        return {
            "longitude": self.variables["longitude"].values,
            "latitude": self.variables["latitude"].values,
            "level": self.variables["level"].values,
            "time": self.variables["time"].values,
        }

    @property
    def variables(self) -> dict[Hashable, xr.Variable]:
        """Low level access to underlying :attr:`data` variables.

        This method is typically is faster for accessing coordinate variables.

        .. versionadded:: 0.25.2

        Returns
        -------
        dict[Hashable, xr.Variable]
            Dictionary of variables. The type is actually..

                xarray.core.utils.Frozen[Any, xr.Variable]

            In practice, this behaves like a dictionary.

        Examples
        --------
        >>> from pycontrails.datalib.ecmwf import ERA5
        >>> times = (datetime(2022, 3, 1, 12),  datetime(2022, 3, 1, 13))
        >>> variables = "air_temperature", "specific_humidity"
        >>> levels = [200, 300]
        >>> era5 = ERA5(times, variables, levels)
        >>> mds = era5.open_metdataset()
        >>> mds.variables["level"].values  # faster access than mds.data["level"]
        array([200., 300.])

        >>> mda = mds["air_temperature"]
        >>> mda.variables["level"].values  # faster access than mda.data["level"]
        array([200., 300.])
        """
        if isinstance(self.data, xr.Dataset):
            return self.data.variables  # type: ignore[return-value]
        return self.data.coords.variables

    @property
    def is_wrapped(self) -> bool:
        """Check if the longitude dimension covers the closed interval ``[-180, 180]``.

        Assumes the longitude dimension is sorted (this is established by the
        :class:`MetDataset` or :class:`MetDataArray` constructor).

        .. versionchanged 0.26.0::

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
        longitude = self.variables["longitude"].values
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
        level = self.variables["level"].values
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

        ..versionchanged::0.34.1

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
        times, datasets = zip(*dataset.groupby("time", squeeze=False))

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
        """Pass through to `self.data.attrs`."""
        return self.data.attrs

    @abstractmethod
    def downselect(self, bbox: list[float]) -> MetBase:
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
        MetBase
            Return downselected data
        """

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
    >>> da[5:10, 5:10, 1, 1].values
    array([[224.0896 , 224.41374, 224.75946, 225.16237, 225.60507],
           [224.09457, 224.42038, 224.76526, 225.16817, 225.61089],
           [224.10037, 224.42618, 224.77106, 225.17314, 225.61586],
           [224.10617, 224.43282, 224.7777 , 225.17812, 225.62166],
           [224.11115, 224.44028, 224.7835 , 225.18393, 225.62663]],
          dtype=float32)

    >>> # Mean temperature over entire array
    >>> da.mean().load().item()
    223.5083
    """

    data: xr.Dataset

    def __init__(
        self,
        data: xr.Dataset,
        cachestore: CacheStore | None = None,
        wrap_longitude: bool = False,
        copy: bool = True,
    ):
        # init cache
        self.cachestore = cachestore

        # if input is already a Dataset, copy into data
        if not isinstance(data, xr.Dataset):
            raise ValueError("Input `data` must be an xarray Dataset")

        # copy Dataset into data
        if copy:
            self.data = data.copy()
            self._preprocess_dims(wrap_longitude)

        else:
            if wrap_longitude:
                raise ValueError("Set 'copy=True' when using 'wrap_longitude=True'.")
            self.data = data
            self._validate_dims()

    def __getitem__(self, key: Hashable) -> MetDataArray:
        """Return DataArray of variable `key` cast to a `MetDataArray` object.

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
            If `key` not found in :attr:`data`
        """
        try:
            da = self.data[key]
        except KeyError as e:
            raise KeyError(
                f"Variable {key} not found. Available variables: {', '.join(self.data.data_vars)}. "
                "To get items (e.g. `time` or `level`) from underlying `xr.Dataset` object, "
                "use the `data` attribute."
            ) from e
        return MetDataArray(da, copy=False, validate=False)

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
        value: MetDataArray | xr.DataArray | np.ndarray | xr.Variable,
    ) -> None:
        """Shortcut to set data variable on :attr:`data`.

        Warns if ``key`` is already present in dataset.

        Parameters
        ----------
        key : Hashable | list[Hashable] | Mapping
            Variable name
        value : MetDataArray | ArrayLike | xr.Variable
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
        """Check if key `key` is in :attr:`data`.

        Parameters
        ----------
        key : Hashable
            Key to check

        Returns
        -------
        bool
            True if `key` is in :attr:`data`, False otherwise
        """
        return key in self.data

    @property
    @overrides
    def shape(self) -> tuple[int, int, int, int]:
        sizes = self.data.sizes
        return sizes["longitude"], sizes["latitude"], sizes["level"], sizes["time"]

    @property
    @overrides
    def size(self) -> int:
        return np.prod(self.shape).item()

    def copy(self) -> MetDataset:
        """Create a copy of the current class.

        Returns
        -------
        MetDataset
            MetDataset copy
        """
        return MetDataset(
            self.data,
            cachestore=self.cachestore,
            copy=True,  # True by default, but being extra explicit
        )

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
        if isinstance(vars, MetVariable):
            vars = [vars]

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
    ) -> MetDataset:
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
        MetDataset
            New MetDataArray with loaded data.
        """
        cachestore = cachestore or DiskCacheStore()
        chunks = chunks or {}
        data = _load(hash, cachestore, chunks)
        return cls(data)

    def wrap_longitude(self) -> MetDataset:
        """Wrap longitude coordinates.

        Returns
        -------
        MetDataset
            Copy of MetDataset with wrapped longitude values.
            Returns copy of current MetDataset when longitude values are already wrapped
        """
        return MetDataset(
            _wrap_longitude(self.data),
            cachestore=self.cachestore,
        )

    @overrides
    def broadcast_coords(self, name: str) -> xr.DataArray:
        da = xr.ones_like(self.data[list(self.data.keys())[0]]) * self.data[name]
        da.name = name

        return da

    @overrides
    def downselect(self, bbox: list[float]) -> MetDataset:
        data = downselect(self.data, bbox)
        return MetDataset(data, cachestore=self.cachestore, copy=False)

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
        GeoVectorDataset [6 keys x 4152960 length, 1 attributes]
            Keys: longitude, latitude, level, time, air_temperature, ..., specific_humidity
            Attributes:
            time                [2022-03-01 00:00:00, 2022-03-01 01:00:00]
            longitude           [-180.0, 179.75]
            latitude            [-90.0, 90.0]
            altitude            [10362.848672411146, 11783.938524404566]
            crs                 EPSG:4326

        """
        coords_keys = [str(key) for key in self.data.dims]  # str not in Hashable
        coords_vals = [self.variables[key].values for key in coords_keys]
        coords_meshes = np.meshgrid(*coords_vals, indexing="ij")
        raveled_coords = [mesh.ravel() for mesh in coords_meshes]
        data = dict(zip(coords_keys, raveled_coords))

        vector = vector_module.GeoVectorDataset(data, copy=False)
        for key in self:
            # The call to .values here will load the data if it is lazy
            vector[key] = self[key].data.values.ravel()

        if transfer_attrs:
            # vector.attrs expects keys to be strings .... we'll get an error
            # if we cannot cast here
            vector.attrs.update({str(k): v for k, v in self.attrs.items()})
        return vector

    @classmethod
    def from_coords(
        cls,
        longitude: npt.ArrayLike | float,
        latitude: npt.ArrayLike | float,
        level: npt.ArrayLike | float,
        time: npt.ArrayLike | np.datetime64,
    ) -> MetDataset:
        """Create a :class:`MetDataset` containing a coordinate skeleton from coordinate arrays.

        Parameters
        ----------
        longitude, latitude : npt.ArrayLike | float
            Horizontal coordinates, in [:math:`degrees`]
        level : npt.ArrayLike | float
            Vertical coordinate, in [:math:`hPa`]
        time: npt.ArrayLike | np.datetime64,
            Temporal coordinates, in [:math:`UTC`]. Will be sorted.

        Returns
        -------
        MetDataset
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
        <xarray.Dataset>
        Dimensions:       (longitude: 20, latitude: 20, level: 2, time: 1)
        Coordinates:
          * longitude     (longitude) float64 0.0 0.5 1.0 1.5 2.0 ... 8.0 8.5 9.0 9.5
          * latitude      (latitude) float64 0.0 0.5 1.0 1.5 2.0 ... 7.5 8.0 8.5 9.0 9.5
          * level         (level) float64 250.0 300.0
          * time          (time) datetime64[ns] 2019-01-01
            air_pressure  (level) float32 2.5e+04 3e+04
            altitude      (level) float32 1.036e+04 9.164e+03
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
        <xarray.Dataset>
        Dimensions:       (longitude: 20, latitude: 20, level: 2, time: 1)
        Coordinates:
          * longitude     (longitude) float64 0.0 0.5 1.0 1.5 2.0 ... 8.0 8.5 9.0 9.5
          * latitude      (latitude) float64 0.0 0.5 1.0 1.5 2.0 ... 7.5 8.0 8.5 9.0 9.5
          * level         (level) float64 250.0 300.0
          * time          (time) datetime64[ns] 2019-01-01
            air_pressure  (level) float32 2.5e+04 3e+04
            altitude      (level) float32 1.036e+04 9.164e+03
        Data variables:
            temperature   (longitude, latitude, level, time) float64 234.5 ... 234.5
            humidity      (longitude, latitude, level, time) float64 0.5 0.5 ... 0.5 0.5

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
            val = np.asarray(val)
            if val.ndim == 0:
                val = val.reshape(1)
            elif val.ndim > 1:
                raise ValueError(f"{key} has too many dimensions")

            val = np.sort(val)
            if val.size == 0:
                raise ValueError(f"Coordinate {key} must be nonempty.")

            # Check dtypes
            if key == "time":
                val = val.astype("datetime64[ns]", copy=False)
            else:
                val = val.astype(COORD_DTYPE, copy=False)

            coords[key] = val

        return cls(xr.Dataset({}, coords=coords))

    @classmethod
    def from_zarr(cls, store: Any, **kwargs: Any) -> MetDataset:
        """Create a :class:`MetDataset` from a path to a Zarr store.

        Parameters
        ----------
        store : Any
            Path to Zarr store. Passed into :func:`xarray.open_zarr`.
        **kwargs : Any
            Other keyword only arguments passed into :func:`xarray.open_zarr`.

        Returns
        -------
        MetDataset
            MetDataset with data from Zarr store.
        """
        kwargs.setdefault("storage_options", {"read_only": True})
        ds = xr.open_zarr(store, **kwargs)
        return cls(ds)


class MetDataArray(MetBase):
    """Meteorological DataArray of single variable.

    Wrapper around xr.DataArray to enforce certain
    variables and dimensions for internal usage.

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
    validate : bool, optional
        Confirm that the parameter `data` has correct specification. This automatically handled
        in the case that `copy=True`. Validation only introduces a very small overhead.
        This parameter should only be set to `False` if working with data derived from an
        existing MetDataset or :class`MetDataArray`. By default `True`.
    **kwargs
        If `data` input is not a xr.DataArray, `data` will be passed
        passed directly to xr.DataArray constructor with these keyword arguments.

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

    data: xr.DataArray

    def __init__(
        self,
        data: xr.DataArray,
        cachestore: CacheStore | None = None,
        wrap_longitude: bool = False,
        copy: bool = True,
        validate: bool = True,
        **kwargs: Any,
    ) -> None:
        # init cache
        self.cachestore = cachestore

        # try to create DataArray out of input data and **kwargs
        if not isinstance(data, xr.DataArray):
            data = xr.DataArray(data, **kwargs)

        if copy:
            self.data = data.copy()
            self._preprocess_dims(wrap_longitude)
        else:
            if wrap_longitude:
                raise ValueError("Set 'copy=True' when using 'wrap_longitude=True'.")
            self.data = data
            if validate:
                self._validate_dims()

        # if name is specified in kwargs, overwrite any other name in DataArray
        if "name" in kwargs:
            self.data.name = kwargs["name"]

        # if at this point now "name" exists on DataArray, set default
        if self.data.name is None:
            self.data.name = "met"

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
        - :meth:`xr.Dataset.load`
        - :meth:`xr.DataArray.load`
        """
        if not self.in_memory:
            self._check_memory("Extracting numpy array from")
            self.data = self.data.load()

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
    @overrides
    def size(self) -> int:
        return self.data.size

    @property
    @overrides
    def shape(self) -> tuple[int, int, int, int]:
        # https://github.com/python/mypy/issues/1178
        return typing.cast(typing.Tuple[int, int, int, int], self.data.shape)

    def copy(self) -> MetDataArray:
        """Create a copy of the current class.

        Returns
        -------
        MetDataArray
            MetDataArray copy
        """
        return MetDataArray(self.data, cachestore=self.cachestore, copy=True)

    def wrap_longitude(self) -> MetDataArray:
        """Wrap longitude coordinates.

        Returns
        -------
        MetDataArray
            Copy of MetDataArray with wrapped longitude values.
            Returns copy of current MetDataArray when longitude values are already wrapped
        """
        return MetDataArray(_wrap_longitude(self.data), cachestore=self.cachestore)

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
        longitude: float | npt.NDArray[np.float_],
        latitude: float | npt.NDArray[np.float_],
        level: float | npt.NDArray[np.float_],
        time: np.datetime64 | npt.NDArray[np.datetime64],
        *,
        method: str = ...,
        bounds_error: bool = ...,
        fill_value: float | np.float64 | None = ...,
        localize: bool = ...,
        indices: interpolation.RGIArtifacts | None = None,
        return_indices: Literal[False] = ...,
    ) -> npt.NDArray[np.float_]:
        ...

    @overload
    def interpolate(
        self,
        longitude: float | npt.NDArray[np.float_],
        latitude: float | npt.NDArray[np.float_],
        level: float | npt.NDArray[np.float_],
        time: np.datetime64 | npt.NDArray[np.datetime64],
        *,
        method: str = ...,
        bounds_error: bool = ...,
        fill_value: float | np.float64 | None = ...,
        localize: bool = ...,
        indices: interpolation.RGIArtifacts | None = None,
        return_indices: Literal[True],
    ) -> tuple[npt.NDArray[np.float_], interpolation.RGIArtifacts]:
        ...

    def interpolate(
        self,
        longitude: float | npt.NDArray[np.float_],
        latitude: float | npt.NDArray[np.float_],
        level: float | npt.NDArray[np.float_],
        time: np.datetime64 | npt.NDArray[np.datetime64],
        *,
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: float | np.float64 | None = np.nan,
        localize: bool = False,
        indices: interpolation.RGIArtifacts | None = None,
        return_indices: bool = False,
    ) -> npt.NDArray[np.float_] | tuple[npt.NDArray[np.float_], interpolation.RGIArtifacts]:
        """Interpolate values over underlying DataArray.

        Zero dimensional coordinates are reshaped to 1D arrays.

        Method automatically loads underlying :attr:`data` into memory.

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
        longitude : float | npt.NDArray[np.float_]
            Longitude values to interpolate. Assumed to be 0 or 1 dimensional.
        latitude : float | npt.NDArray[np.float_]
            Latitude values to interpolate. Assumed to be 0 or 1 dimensional.
        level : float | npt.NDArray[np.float_]
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
        indices: tuple | None, optional
            Experimental. See :func:`interpolation.interp`. None by default.
        return_indices: bool, optional
            Experimental. See :func:`interpolation.interp`. False by default.

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
        array([241.91974], dtype=float32)

        >>> da = mda.data
        >>> da.sel(longitude=1, latitude=2, level=300, time=np.datetime64('2022-03-01T14')).item()
        241.91974

        >>> # Interpolation off grid
        >>> mda.interpolate(1.1, 2.1, 290, np.datetime64('2022-03-01 13:10'))
        array([239.83794], dtype=float32)

        >>> # Interpolate along path
        >>> longitude = np.linspace(1, 2, 10)
        >>> latitude = np.linspace(2, 3, 10)
        >>> level = np.linspace(200, 300, 10)
        >>> time = pd.date_range("2022-03-01T14", periods=10, freq="5T")
        >>> mda.interpolate(longitude, latitude, level, time)
        array([220.44348, 223.089  , 225.7434 , 228.41643, 231.10858, 233.54858,
               235.71506, 237.86479, 239.99275, 242.10793], dtype=float32)
        """
        # Load if necessary
        if not self.in_memory:
            self._check_memory("Interpolation over")
            self.data = self.data.load()

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

    def _check_memory(self, msg_start: str) -> None:
        n_bytes = self.data.nbytes
        n_gb = n_bytes // int(1e9)
        msg = (
            f"{msg_start} MetDataArray {self.name} requires loading "
            f"at least {n_gb} GB of data into memory. Downselect data if possible. "
            "If working with a GeoVectorDataset instance, this can be achieved "
            "with the method 'downselect_met'."
        )

        if n_gb > 32:  # Prevent something stupid
            raise RuntimeError(msg)
        if n_gb > 4:
            warnings.warn(msg)

        mb = round(n_bytes / int(1e6), 2)
        logger.debug("Loading %s into memory consumes %s MB.", self.name, mb)

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
    ) -> MetDataArray:
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
        return cls(data[list(data.data_vars)[0]])

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

        return self.data.sum().values.item() / self.data.count().values.item()

    def find_edges(self) -> MetDataArray:
        """Find edges of regions.

        Returns
        -------
        MetDataArray
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

        return MetDataArray(self.data.groupby("level").map(_edges), cachestore=self.cachestore)

    def to_polygon_feature(
        self,
        level: float | int | None = None,
        time: np.datetime64 | datetime | None = None,
        fill_value: float = 0.0,
        iso_value: float | None = None,
        min_area: float = 0.0,
        epsilon: float = 0.0,
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
            Expected to be less than the ``iso_value`` parameter. By default, 0.0.
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

        See Also
        --------
        :meth:`to_polyhedra`
        :func:`pycontrails.core.polygons.find_multipolygons`

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

        np.nan_to_num(arr, copy=False, nan=fill_value)

        # default iso_value
        if iso_value is None:
            iso_value = (np.max(arr) + np.min(arr)) / 2
            warnings.warn(f"The 'iso_value' parameter was not specified. Using value: {iso_value}")

        # We'll get a nice error message if dependencies are not installed
        from pycontrails.core import polygon

        # Convert to nested lists of coordinates for GeoJSON representation
        longitude: npt.NDArray[np.float_] = self.variables["longitude"].values
        latitude: npt.NDArray[np.float_] = self.variables["latitude"].values

        mp = polygon.find_multipolygon(
            arr,
            threshold=iso_value,
            min_area=min_area,
            epsilon=epsilon,
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
        fill_value: float = 0.0,
        iso_value: float | None = None,
        min_area: float = 0.0,
        epsilon: float = 0.0,
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
        return_type: Literal["geojson"],
        path: str | None = ...,
        altitude_scale: float = ...,
        output_vertex_normals: bool = ...,
        closed: bool = ...,
    ) -> dict:
        ...

    @overload
    def to_polyhedra(
        self,
        *,
        time: datetime | None = ...,
        iso_value: float = ...,
        simplify_fraction: float = ...,
        return_type: Literal["mesh"],
        path: str | None = ...,
        altitude_scale: float = ...,
        output_vertex_normals: bool = ...,
        closed: bool = ...,
    ) -> "o3d.geometry.TriangleMesh":
        ...

    def to_polyhedra(
        self,
        *,
        time: datetime | None = None,
        iso_value: float = 0.0,
        simplify_fraction: float = 1.0,
        return_type: str = "geojson",
        path: str | None = None,
        altitude_scale: float = 1.0,
        output_vertex_normals: bool = False,
        closed: bool = True,
    ) -> dict | "o3d.geometry.TriangleMesh":
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
        dict | :class:`o3d.geometry.TriangleMesh`
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
        :meth:`to_polygons`
        `skimage.measure.marching_cubes <https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes>`_

        Notes
        -----
        Uses the `scikit-image Marching Cubes  <https://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html>`_
        algorithm to reconstruct a surface from the point-cloud like arrays.
        """  # noqa: E501
        try:
            from skimage import measure
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "This method requires the `skimage` module from scikit-learn, which can be "
                "installed using `pip install pycontrails[vis]`"
            ) from e

        try:
            import open3d as o3d
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "This method requires the `open3d` module, which can be installed "
                "with `pip install pycontrails[open3d]` or `pip install open3d`."
            ) from e

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

        # convert from array index back to coordinates
        longitude = self.variables["longitude"].values
        latitude = self.variables["latitude"].values
        altitude = units.pl_to_m(self.variables["level"].values)

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

    @overrides
    def broadcast_coords(self, name: str) -> xr.DataArray:
        da = xr.ones_like(self.data) * self.data[name]
        da.name = name

        return da

    @overrides
    def downselect(self, bbox: list[float]) -> MetDataArray:
        data = downselect(self.data, bbox)
        return MetDataArray(data, cachestore=self.cachestore)


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
        dask0 = darr.dask[next(iter(darr.dask))]  # first dask instruction
        return dask0.array.array.array.__class__.__name__ == "ZarrArrayWrapper"
    except AttributeError:
        return False


def shift_longitude(data: XArrayType) -> XArrayType:
    """Shift longitude values from [0, 360) to [-180, 180) domain.

    Sorts data by ascending longitude values.

    Parameters
    ----------
    data : XArrayType
        :class:`xr.Dataset` or :class:`xr.DataArray` with longitude dimension


    Returns
    -------
    XArrayType
        :class:`xr.Dataset` or :class:`xr.DataArray` with longitude values on [-180, 180).
    """
    return data.assign_coords(
        longitude=((data["longitude"].values + 180.0) % 360.0) - 180.0
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
    if isinstance(data, xr.Dataset):
        lon = data.variables["longitude"].values
    else:
        lon = data.coords.variables["longitude"].values

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
    lon_chunks = chunks.get("longitude", tuple())
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
        level_coord = mda.variables["level"].values
        if len(level_coord) == 1:
            level = level_coord[0]
        else:
            raise ValueError(
                "Parameter 'level' must be defined when the length of the 'level' "
                "coordinates is not 1."
            )

    # Determine time if not specified
    if time is None:
        time_coord = mda.variables["time"].values
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
        altitude = round(altitude)

    return arr, altitude


def downselect(data: XArrayType, bbox: list[float]) -> XArrayType:
    """Downselect :class:`xr.Dataset` or :class:`xr.DataArray` with spatial bounding box.

    Parameters
    ----------
    data : XArrayType
        xr.Dataset or xr.DataArray to downselect
    bbox : list[float]
        List of coordinates defining a spatial bounding box in WGS84 coordinates.
        For 2D queries, list is [west, south, east, north].
        For 3D queries, list is [west, south, min-level, east, north, max-level]
        with level defined in [:math:`hPa`].

    Returns
    -------
    XArrayType
        Downselected xr.Dataset or xr.DataArray

    Raises
    ------
    ValueError
        If parameter `bbox` has wrong length.
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

    cond = (
        (data["latitude"] >= south)
        & (data["latitude"] <= north)
        & (data["level"] >= level_min)
        & (data["level"] <= level_max)
    )

    # wrapping longitude
    if west <= east:
        cond = cond & (data["longitude"] >= west) & (data["longitude"] <= east)
    else:
        cond = cond & ((data["longitude"] >= west) | (data["longitude"] <= east))

    return data.where(cond, drop=True)


def standardize_variables(ds: DatasetType, variables: Iterable[MetVariable]) -> DatasetType:
    """Rename all variables in dataset from short name to standard name.

    This function does not change any variables in ``ds`` that are not found in ``variables``.

    When there are multiple variables with the same short name, the last one is used.

    Parameters
    ----------
    ds : DatasetType
        An :class:`xr.Dataset` or :class:`MetDataset`. When a :class:`MetDataset` is
        passed, the underlying :class:`xr.Dataset` is modified in place.
    variables : Iterable[MetVariable]
        Data source variables

    Returns
    -------
    DatasetType
        Dataset with variables renamed to standard names
    """
    if isinstance(ds, xr.Dataset):
        return _standardize_variables(ds, variables)

    ds.data = _standardize_variables(ds.data, variables)
    return ds


def _standardize_variables(ds: xr.Dataset, variables: Iterable[MetVariable]) -> xr.Dataset:
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
    return met.attrs.get("met_source") in ("ERA5", "HRES") or "ecmwf" in met.attrs.get(
        "history", ""
    )


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
