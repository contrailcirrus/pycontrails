"""Lightweight data structures for vector paths."""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from typing import Any, Dict, Generator, Iterable, Iterator, Sequence, Type, TypeVar, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
from overrides import overrides

from pycontrails.core import coordinates, interpolation
from pycontrails.core import met as met_module
from pycontrails.physics import units
from pycontrails.utils import json as json_module

logger = logging.getLogger(__name__)

#: Vector types
VectorDatasetType = TypeVar("VectorDatasetType", bound="VectorDataset")
GeoVectorDatasetType = TypeVar("GeoVectorDatasetType", bound="GeoVectorDataset")


class AttrDict(Dict[str, Any]):
    """Thin wrapper around dict to warn when setting a key that already exists."""

    def __setitem__(self, k: str, v: Any) -> None:
        """Warn when setting values that already contain values.

        Parameters
        ----------
        k : str
            Key
        v : Any
            Value
        """
        if k in self and self[k] is not None and self[k] is not v:
            warnings.warn(
                f"Overwriting attr key `{k}`. Use `.update({k}=...)` to suppress warning."
            )

        super().__setitem__(k, v)

    def setdefault(self, k: str, default: Any = None) -> Any:
        """Thin wrapper around ``dict.setdefault``.

        Overwrites value if value is None.

        Parameters
        ----------
        k : str
            Key
        default : Any, optional
            Default value for key ``k``

        Returns
        -------
        Any
            Value at ``k``
        """
        ret = self.get(k, None)
        if ret is not None:
            return ret

        self[k] = default
        return default


class VectorDataDict(Dict[str, np.ndarray]):
    """Thin wrapper around ``dict[str, np.ndarray]`` to ensure consistency.

    Parameters
    ----------
    data : dict[str, np.ndarray], optional
        Dictionary input
    **kwargs : np.ndarray
        Keyword arguments, override values in ``data``
    """

    __slots__ = ("_size",)

    #: Length of the data
    _size: int

    def __init__(self, data: dict[str, np.ndarray] | None = None, **kwargs: np.ndarray) -> None:
        if data is None:
            data = {}

        super().__init__(data, **kwargs)

        # validate any arrays, first one defines size()
        for arr in self.values():
            self._validate_array(arr)

    def __setitem__(self, k: str, v: npt.ArrayLike) -> None:
        """Set new key-value pair to instance and warn when overwriting existing key.

        This method casts ``v`` to a ``np.ndarray`` and ensures that the array size is
        consistent with the instance.

        Parameters
        ----------
        k : str
            Key
        v : np.ndarray
            Values

        See Also
        --------
        :meth:`update`
        """
        v = np.asarray(v)  # asarray does NOT copy
        self._validate_array(v)

        if k in self and len(self[k]) and self[k] is not v:
            warnings.warn(
                f"Overwriting data in key `{k}`. Use `.update({k}=...)` to suppress warning."
            )

        super().__setitem__(k, v)

    def __delitem__(self, k: str) -> None:
        super().__delitem__(k)

        # if not data keys left, set size to 0
        if not len(self):
            del self._size

    def setdefault(self, k: str, default: npt.ArrayLike | None = None) -> np.ndarray:
        """Thin wrapper around ``dict.setdefault``.

        The main purpose of overriding is to run :meth:`_validate_array()` on set.

        Parameters
        ----------
        k : str
            Key
        default : npt.ArrayLike, optional
            Default value for key ``k``

        Returns
        -------
        Any
            Value at ``k``
        """
        ret = self.get(k, None)
        if ret is not None:
            return ret

        if default is None:
            default = np.array([])

        self[k] = default
        return self[k]

    def update(  # type: ignore[override]
        self, other: dict[str, npt.ArrayLike] | None = None, **kwargs: npt.ArrayLike
    ) -> None:
        """Update values without warning if overwriting.

        This method casts values in  ``other`` to np.ndarray arrays and ensures that the
        array sizes are consistent with the instance.

        Parameters
        ----------
        other : dict[str, npt.ArrayLike] | None, optional
            Fields to update as dict
        **kwargs : npt.ArrayLike
            Fields to update as kwargs
        """
        other = other or {}
        other_arrs = {k: np.asarray(v) for k, v in other.items()}
        for arr in other_arrs.values():
            self._validate_array(arr)

        super().update(other_arrs)

        # validate any kwarg arrays
        kwargs_arr = {k: np.asarray(v) for k, v in kwargs.items()}
        for arr in kwargs_arr.values():
            self._validate_array(arr)

        super().update(kwargs_arr)

    def _validate_array(self, arr: np.ndarray) -> None:
        """Ensure that `arr` is compatible with instance.

        Set attribute `_size` if it has not yet been defined.

        Parameters
        ----------
        arr : np.ndarray
            Array to validate

        Raises
        ------
        ValueError
            If `arr` is not compatible with instance.
        """
        if arr.ndim != 1:
            raise ValueError("All np.arrays must have dimension 1.")
        if getattr(self, "_size", 0) != 0:
            if arr.size != self._size:
                raise ValueError(f"Incompatible array sizes: {arr.size} and {self._size}.")
        else:
            self._size = arr.size


def _empty_vector_dict(keys: Iterable[str]) -> VectorDataDict:
    """Create instance of VectorDataDict with variables defined by `keys` and size 0.

    Parameters
    ----------
    keys : Iterable[str]
        Keys to include in empty VectorDataset instance.

    Returns
    -------
    VectorDataDict
        Empty :class:`VectorDataDict` instance.
    """
    keys = keys or set()
    data = VectorDataDict({key: np.array([]) for key in keys})

    # The default dtype is float64
    # Time is special and should have a non-default dtype of datetime64[ns]
    if "time" in data:
        data.update(time=np.array([], dtype="datetime64[ns]"))

    return data


class VectorDataset:
    """Base class to hold 1D arrays of consistent size.

    Parameters
    ----------
    data : dict[str, npt.ArrayLike] | pd.DataFrame | VectorDataDict | VectorDataset | None, optional
        Initial data, by default None
    attrs : dict[str, Any] | AttrDict, optional
        Dictionary of attributes, by default None
    copy : bool, optional
        Copy data on class creation, by default True
    **attrs_kwargs : Any
        Additional attributes passed as keyword arguments

    Raises
    ------
    ValueError
        If "time" variable cannot be converted to numpy array.
    """

    __slots__ = ("data", "attrs")

    #: Vector data with labels as keys and np.ndarrays as values
    data: VectorDataDict

    #: Generic dataset attributes
    attrs: AttrDict

    def __init__(
        self,
        data: dict[str, npt.ArrayLike]
        | pd.DataFrame
        | VectorDataDict
        | VectorDataset
        | None = None,
        attrs: dict[str, Any] | AttrDict | None = None,
        copy: bool = True,
        **attrs_kwargs: Any,
    ) -> None:
        # Casting from one VectorDataset type to another
        # e.g., flight = Flight(...); vector = VectorDataset(flight)
        if isinstance(data, VectorDataset):
            attrs = {**data.attrs, **(attrs or {})}
            data = data.data

        if data is None:
            self.data = VectorDataDict()

        elif isinstance(data, pd.DataFrame):
            # Take extra caution with a time column

            if "time" in data:
                time = data["time"]

                if not hasattr(time, "dt"):
                    # If the time column is a string, we try to convert it to a datetime
                    # If it fails (for example, a unix integer time), we raise an error
                    # and let the user figure it out.
                    try:
                        time = pd.to_datetime(time)
                    except ValueError:
                        raise ValueError(
                            "The 'time' field must hold datetime-like values. "
                            'Try data["time"] = pd.to_datetime(data["time"], unit=...) '
                            "with the appropriate unit."
                        )

                # If the time column contains a timezone, the call to `to_numpy`
                # will convert it to an array of object. We do not want this, so
                # we raise an error in this case. Timezone issues are complicated,
                # and so it is better for the user to handle them rather than try
                # to address them here.
                if time.dt.tz is not None:
                    raise ValueError(
                        "The 'time' field must be timezone naive. "
                        "This can be achieved with: "
                        'data["time"] = data["time"].dt.tz_localize(None)'
                    )

                data = {col: ser.to_numpy(copy=copy) for col, ser in data.items() if col != "time"}
                data["time"] = time.to_numpy(copy=copy)
            else:
                data = {col: ser.to_numpy(copy=copy) for col, ser in data.items()}

            self.data = VectorDataDict(data)

        elif isinstance(data, VectorDataDict) and not copy:
            self.data = data

        # For anything else, we assume it is a dictionary of array-like and attach it
        # This doesn't quite work for casting one VectorDataset to another subclass
        # (we don't support .items() here), but we could easily accommodate that if
        # needed
        else:
            self.data = VectorDataDict({k: np.array(v, copy=copy) for k, v in data.items()})

        # set attributes
        if attrs is None:
            self.attrs = AttrDict()

        elif isinstance(attrs, AttrDict) and not copy:
            self.attrs = attrs

        #  shallow copy if dict
        else:
            self.attrs = AttrDict(attrs.copy())

        # update with kwargs
        self.attrs.update(attrs_kwargs)

    # ------------
    # dict-like methods
    # ------------
    def __getitem__(self, key: str) -> np.ndarray:
        """Get values from :attr:`data`.

        Parameters
        ----------
        key : str
            Key to get from :attr:`data`

        Returns
        -------
        np.ndarray
            Values at :attr:`data[key]`
        """
        return self.data[key]

    def get(self, key: str, default_value: Any = None) -> Any:
        """Get values from :attr:`data` with default_value if `key` not in `data`.

        Parameters
        ----------
        key : str
            Key to get from :attr:`data`
        default_value : Any, optional
            Return `default_value` if `key` not in :attr:`data`, by default `None`

        Returns
        -------
        Any
            Values at :attr:`data[key]` or `default_value`
        """
        return self.data.get(key, default_value)

    def __setitem__(self, key: str, values: npt.ArrayLike) -> None:
        """Set values at key `key` on :attr:`data`.

        Parameters
        ----------
        key : str
            Key name in :attr:`data`
        values : npt.ArrayLike
            Values to set to :attr:`data`. Array size must be compatible with existing data.
        """
        self.data[key] = values

    def __delitem__(self, key: str) -> None:
        """Delete values at key `key` on :attr:`data`.

        Parameters
        ----------
        key : str
            Key name in :attr:`data`
        """
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys in :attr:`data`.

        Returns
        -------
        Iterator[str]
            Keys in :attr:`data`
        """
        return iter(self.data)

    def __contains__(self, key: str) -> bool:
        """Check if key `key` is in :attr:`data`.

        Parameters
        ----------
        key : str
            Key to check

        Returns
        -------
        bool
            True if `key` is in :attr:`data`, False otherwise
        """
        return key in self.data

    def update(
        self,
        other: dict[str, npt.ArrayLike] | None = None,
        **kwargs: npt.ArrayLike,
    ) -> None:
        """Update values in  :attr:`data` dict without warning if overwriting.

        Parameters
        ----------
        other : dict[str, npt.ArrayLike] | None, optional
            Fields to update as dict
        **kwargs : npt.ArrayLike
            Fields to update as kwargs
        """
        self.data.update(other, **kwargs)

    def setdefault(self, key: str, default: npt.ArrayLike | None = None) -> np.ndarray:
        """Shortcut to :attr:`data.setdefault`.

        Parameters
        ----------
        key : str
            Key in :attr:`data` dict.
        default : npt.ArrayLike, optional
            Values to use as default, if key is not defined

        Returns
        -------
        np.ndarray
            Values at ``key``
        """
        return self.data.setdefault(key, default)

    def get_data_or_attr(self, key: str) -> Any:
        """Get value from :attr:`data` or :attr:`attrs`.

        This method first checks if ``key`` is in :attr:`data` and returns the value if so.
        If ``key`` is not in :attr:`data`, then this method checks if ``key`` is in :attr:`attrs`
        and returns the value if so. If ``key`` is not in :attr:`data` or :attr:`attrs`,
        then a ``KeyError`` is raised.

        Parameters
        ----------
        key : str
            Key to get from :attr:`data` or :attr:`attrs`

        Returns
        -------
        Any
            Value at :attr:`data[key]` or :attr:`attrs[key]`

        Examples
        --------
        >>> vector = VectorDataset({"a": [1, 2, 3]}, attrs={"b": 4})
        >>> vector.get_data_or_attr("a")
        array([1, 2, 3])

        >>> vector.get_data_or_attr("b")
        4

        >>> vector.get_data_or_attr("c")
        Traceback (most recent call last):
        ...
        KeyError: 'c'

        """
        try:
            return self.data[key]
        except KeyError:
            return self.attrs[key]

    # ------------

    def __len__(self) -> int:
        """Length of each array in :attr:`data`.

        Returns
        -------
        int
            Length of each array in :attr:`data`
        """
        return self.size

    def _display_attrs(self) -> dict[str, str]:
        """Return properties used in `repr` constructions`.

        Returns
        -------
        dict[str, str]
            Properties used in :meth:`__repr__` and :meth:`_repr_html_`.
        """

        # Clip any attribute value that is too long
        def str_clip(v: Any) -> str:
            s = str(v)
            if len(s) < 80:
                return s
            return f"{s[:77]}..."

        return {k: str_clip(v) for k, v in self.attrs.items()}

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        n_attrs = len(self.attrs)
        n_keys = len(self.data)
        _repr = f"{class_name} [{n_keys} keys x {self.size} length, {n_attrs} attributes]"

        keys = list(self.data.keys())
        keys = keys[0:5] + ["..."] + keys[-1:] if len(keys) > 5 else keys
        _repr += f"\n\tKeys: {', '.join(keys)}"

        attrs = self._display_attrs()
        _repr += "\n\tAttributes:\n"
        _repr += "\n".join([f"\t{k:20}{v}" for k, v in attrs.items()])

        return _repr

    def _repr_html_(self) -> str:
        name = type(self).__name__
        n_attrs = len(self.attrs)
        n_keys = len(self.data)
        attrs = self._display_attrs()
        size = self.size

        title = f"<b>{name}</b> [{n_keys} keys x {size} length, {n_attrs} attributes]<br/ ><br/>"

        # matching pd.DataFrame styling
        header = '<tr style="border-bottom:1px solid silver"><th colspan="2">Attributes</th></tr>'
        rows = [f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in attrs.items()]
        table = f"<table>{header + ''.join(rows)}</table>"
        return title + table + self.dataframe._repr_html_()

    def __bool__(self) -> bool:
        """Check if :attr:`data` is nonempty..

        Returns
        -------
        bool
            True if non-empty values are set in :attr:`data`
        """
        return self.size > 0

    def __add__(self: VectorDatasetType, other: VectorDatasetType | None) -> VectorDatasetType:
        """Concatenate two compatible instances of VectorDataset.

        In this context, compatibility means that both have identical :attr:`data` keys.

        This operator behaves similarly to the ``__add__`` method on python lists.

        If self is an empty VectorDataset, return other. This is useful when
        calling :keyword:`sum` with an empty initial value.

        Parameters
        ----------
        other : VectorDatasetType
            Other values to concatenate

        Returns
        -------
        VectorDatasetType
            Concatenated values.

        Raises
        ------
        KeyError
            If `other` has different :attr:`data` keys than self.
        """
        # Short circuit: If other is empty or None, return self. The order here can matter.
        # We let self (so the left addend) take priority.
        if not other:
            return self
        if not self:
            return other

        return type(self).sum((self, other))

    @classmethod
    def sum(
        cls: Type[VectorDatasetType],
        vectors: Sequence[VectorDataset],
        infer_attrs: bool = True,
    ) -> VectorDatasetType:
        """Sum a list of :class:`VectorDataset` instances.

        Parameters
        ----------
        vectors : Sequence[VectorDataset]
            List of :class:`VectorDataset` instances to concatenate.
        infer_attrs : bool, optional
            If True, infer attributes from the first VectorDataset in the list.

        Returns
        -------
        VectorDataset
            Sum of all VectorDataset instances in ``vectors``.

        Raises
        ------
        KeyError
            If incompatible :attr:`data` keys are found among ``vectors``.

        Examples
        --------
        >>> from pycontrails import VectorDataset
        >>> v1 = VectorDataset({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> v2 = VectorDataset({"a": [7, 8, 9], "b": [10, 11, 12]})
        >>> v3 = VectorDataset({"a": [13, 14, 15], "b": [16, 17, 18]})
        >>> v = VectorDataset.sum([v1, v2, v3])
        >>> v.dataframe
            a   b
        0   1   4
        1   2   5
        2   3   6
        3   7  10
        4   8  11
        5   9  12
        6  13  16
        7  14  17
        8  15  18

        """
        if not vectors:
            return cls()
        keys = vectors[0].data.keys()
        for v in vectors[1:]:
            if v.data.keys() != keys:
                raise KeyError("Summands have incompatible keys.")

        def concat(key: str) -> np.ndarray:
            values = [v[key] for v in vectors]
            return np.concatenate(values)

        data = {key: concat(key) for key in keys}

        if infer_attrs:
            return cls(data, attrs=vectors[0].attrs, copy=False)
        return cls(data, copy=False)

    def __eq__(self: VectorDatasetType, other: object) -> bool:
        """Determine if two instances are equal.

        NaN values are considered equal in this comparison.

        Parameters
        ----------
        other : object
            VectorDatasetType to compare with

        Returns
        -------
        bool
            True if both instances have identical :attr:`data` and :attr:`attrs`.
        """
        if isinstance(other, VectorDataset):
            # assert attrs equal
            for key in self.attrs:
                if isinstance(self.attrs[key], np.ndarray):
                    # equal_nan not supported for non-numeric data
                    equal_nan = not np.issubdtype(self.attrs[key].dtype, "O")
                    try:
                        eq = np.array_equal(self.attrs[key], other.attrs[key], equal_nan=equal_nan)
                    except KeyError:
                        return False
                else:
                    eq = self.attrs[key] == other.attrs[key]

                if not eq:
                    return False

            # assert data equal
            for key in self:
                # equal_nan not supported for non-numeric data (e.g. strings)
                equal_nan = not np.issubdtype(self[key].dtype, "O")
                try:
                    eq = np.array_equal(self[key], other[key], equal_nan=equal_nan)
                except KeyError:
                    return False

                if not eq:
                    return False

            return True
        return False

    @property
    def size(self) -> int:
        """Length of each array in :attr:`data`.

        Returns
        -------
        int
            Length of each array in :attr:`data`.
        """
        return getattr(self.data, "_size", 0)

    @property
    def shape(self) -> tuple[int]:
        """Shape of each array in :attr:`data`.

        Returns
        -------
        tuple[int]
            Shape of each array in :attr:`data`.
        """
        return (self.size,)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Shorthand property to access :meth:`to_dataframe` with `copy=False`.

        Returns
        -------
        pd.DataFrame
            Equivalent to the output from :meth:`to_dataframe()`
        """
        return self.to_dataframe(copy=False)

    @property
    def hash(self) -> str:
        """Generate a unique hash for this class instance.

        Returns
        -------
        str
            Unique hash for flight instance (sha1)
        """
        _hash = json.dumps(self.data, cls=json_module.NumpyEncoder)
        return hashlib.sha1(bytes(_hash, "utf-8")).hexdigest()

    # ------------
    # Utilities
    # ------------

    def copy(self: VectorDatasetType) -> VectorDatasetType:
        """Return a copy of this VectorDatasetType class.

        Returns
        -------
        VectorDatasetType
            Copy of class
        """
        return type(self)(data=self.data, attrs=self.attrs, copy=True)

    def select(self: VectorDataset, keys: Iterable[str], copy: bool = True) -> VectorDataset:
        """Return new class instance only containing specified keys.

        Parameters
        ----------
        keys : Iterable[str]
            An iterable of keys to filter by.
        copy : bool, optional
            Copy data on selection.
            Defaults to True.

        Returns
        -------
        VectorDataset
            VectorDataset containing only data associated to ``keys``.
            Note that this method always returns a :class:`VectorDataset`, even if
            the calling class is a proper subclass of :class:`VectorDataset`.
        """
        data = {key: self[key] for key in keys}
        return VectorDataset(data=data, attrs=self.attrs, copy=copy)

    def filter(
        self: VectorDatasetType, mask: npt.NDArray[np.bool_], copy: bool = True
    ) -> VectorDatasetType:
        """Filter :attr:`data` according to a boolean array ``mask``.

        Entries corresponding to ``mask == True`` are kept.

        Parameters
        ----------
        mask : npt.NDArray[np.bool_]
            Boolean array with compatible shape.
        copy : bool, optional
            Copy data on filter. Defaults to True. See
            `numpy best practices <https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding>`_
            for insight into whether copy is appropriate.

        Returns
        -------
        VectorDatasetType
            Containing filtered data

        Raises
        ------
        TypeError
            If ``mask`` is not a boolean array.
        """  # noqa: E501
        self.data._validate_array(mask)
        if mask.dtype != bool:
            raise TypeError("Parameter `mask` must be a boolean array.")

        data = {key: value[mask] for key, value in self.data.items()}
        return type(self)(data=data, attrs=self.attrs, copy=copy)

    def sort(self: VectorDatasetType, by: str | list[str]) -> VectorDatasetType:
        """Sort data by key(s).

        This method always creates a copy of the data.

        Parameters
        ----------
        by : str | list[str]
            Key or list of keys to sort by.

        Returns
        -------
        VectorDatasetType
            Instance with sorted data.
        """
        return type(self)(data=self.dataframe.sort_values(by=by), attrs=self.attrs)

    def ensure_vars(self, vars: str | Sequence[str], raise_error: bool = True) -> bool:
        """Ensure variables exist in column of :attr:`data` or :attr:`attrs`.

        Parameters
        ----------
        vars : str | Sequence[str]
            A single string variable name or a sequence of string variable names.
        raise_error : bool, optional
            Raise KeyError if data does not contain variables.
            Defaults to True.

        Returns
        -------
        bool
            True if all variables exist.
            False otherwise.

        Raises
        ------
        KeyError
            Raises when dataset does not contain variable in ``vars``
        """
        if isinstance(vars, str):
            vars = (vars,)

        for v in vars:
            if v in self or v in self.attrs:
                continue
            if raise_error:
                raise KeyError(
                    f"{type(self).__name__} instance does not contain data or attr '{v}'"
                )
            return False

        return True

    def broadcast_attrs(self, keys: str | Iterable[str], overwrite: bool = False) -> None:
        """Attach values from `keys` in :attr:`attrs` onto :attr:`data`.

        If possible, use ``dtype = np.float32`` when broadcasting. If not possible,
        use whatever ``dtype`` is inferred from the data by :func:`numpy.full`.

        Parameters
        ----------
        keys : str | Iterable[str]
            Keys to broadcast
        overwrite : bool, optional
            If True, overwrite existing values in :attr:`data`. By default False.

        Raises
        ------
        KeyError
            Not all `keys` found in :attr:`attrs`.
        """
        if isinstance(keys, str):
            keys = [keys]
        else:
            keys = list(keys)  # necessary since we iterate over it twice

        # Validate everything up front to avoid partial broadcasting
        for key in keys:
            if key not in self.attrs:
                raise KeyError(f"{type(self)} does not contain attr `{key}`")

        # Do the broadcasting
        for attr in keys:
            if attr in self.data and not overwrite:
                warnings.warn(
                    f"Found duplicate key {attr} in attrs and data. "
                    "Set `overwrite=True` parameter to force overwrite."
                )
            else:
                scalar = self.attrs[attr]
                min_dtype = np.min_scalar_type(scalar)
                if np.can_cast(min_dtype, np.float32):
                    self.data.update({attr: np.full(self.size, scalar, dtype=np.float32)})
                else:
                    self.data.update({attr: np.full(self.size, scalar)})

    def broadcast_numeric_attrs(
        self, ignore_keys: str | Iterable[str] | None = None, overwrite: bool = False
    ) -> None:
        """Attach numeric values in :attr:`attrs` onto :attr:`data`.

        Iterate through values in :attr:`attrs` and attach `float` and `int` values
        to `data`.

        This method modifies object in place.

        Parameters
        ----------
        ignore_keys: str | Iterable[str], optional
            Do not broadcast selected keys.
            Defaults to None.
        overwrite : bool, optional
            If True, overwrite existing values in :attr:`data`. By default False.
        """
        if ignore_keys is None:
            ignore_keys = ()
        elif isinstance(ignore_keys, str):
            ignore_keys = (ignore_keys,)

        # Somewhat brittle: Only checking for int or float type
        numeric_attrs = [
            attr
            for attr, val in self.attrs.items()
            if (isinstance(val, (int, float)) and attr not in ignore_keys)
        ]
        self.broadcast_attrs(numeric_attrs, overwrite)

    # ------------
    # I / O
    # ------------

    def to_dataframe(self, copy: bool = True) -> pd.DataFrame:
        """Create DataFrame in which each key-value pair in :attr:`data` is a column.

        DataFrame does **not** copy data by default.
        Use the ``copy`` parameter to copy data values on creation.

        Parameters
        ----------
        copy : bool, optional
            Copy data on DataFrame creation.

        Returns
        -------
        pd.DataFrame
            DataFrame holding key-values as columns.
        """
        df = pd.DataFrame(self.data, copy=copy)
        df.attrs = self.attrs
        return df

    @classmethod
    def create_empty(
        cls: Type[VectorDatasetType],
        keys: Iterable[str],
        attrs: dict[str, Any] | None = None,
        **attrs_kwargs: Any,
    ) -> VectorDatasetType:
        """Create instance with variables defined by `keys` and size 0.

        If instance requires additional variables to be defined, these keys will automatically
        be attached to returned instance.

        Parameters
        ----------
        keys : Iterable[str]
            Keys to include in empty VectorDataset instance.
        attrs : dict[str, Any] | None, optional
            Attributes to attach instance.
        **attrs_kwargs : Any
            Define attributes as keyword arguments.

        Returns
        -------
        VectorDatasetType
            Empty VectorDataset instance.
        """
        return cls(data=_empty_vector_dict(keys or set()), attrs=attrs, copy=False, **attrs_kwargs)

    def generate_splits(
        self: VectorDatasetType, n_splits: int, copy: bool = True
    ) -> Generator[VectorDatasetType, None, None]:
        """Split instance into ``n_split`` sub-vectors.

        Parameters
        ----------
        n_splits : int
            Number of splits.
        copy : bool, optional
            Passed into :meth:`filter`. Defaults to True. Recommend to keep as True
            based on `numpy best practices <https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding>`_.

        Returns
        -------
        Generator[VectorDatasetType, None, None]
            Generator of split vectors.

        See Also
        --------
        :func:`numpy.array_split`
        """  # noqa: E501
        full_index = np.arange(self.size)
        index_splits = np.array_split(full_index, n_splits)
        for index in index_splits:
            filt = np.zeros(self.size, dtype=bool)
            filt[index] = True
            yield self.filter(filt, copy=copy)


class GeoVectorDataset(VectorDataset):
    """Base class to hold 1D geospatial arrays of consistent size.

    GeoVectorDataset is required to have geospatial coordinate keys defined
    in :attr:`required_keys`.

    Expect latitude-longitude CRS in WGS 84.
    Expect altitude in [:math:`m`].
    Expect level in [:math:`hPa`].

    Each spatial variable is expected to have "float32" or "float64" ``dtype``.
    The time variable is expected to have "datetime64[ns]" ``dtype``.

    Use the attribute :attr:`attr["crs"]` to specify coordinate reference system
    using `PROJ <https://proj.org/>`_ or `EPSG <https://epsg.org/home.html>`_ syntax.

    Parameters
    ----------
    data : dict[str, npt.ArrayLike] | pd.DataFrame | VectorDataDict | VectorDataset | None, optional
        Data dictionary or :class:`pandas.DataFrame` .
        Must include keys/columns ``time``, ``latitude``, ``longitude``, ``altitude`` or ``level``.
        Keyword arguments for ``time``, ``latitude``, ``longitude``, ``altitude`` or ``level``
        override ``data`` inputs. Expects ``altitude`` in meters and ``time``
        as a DatetimeLike (or array that can processed with :meth:`pd.to_datetime`).
        Additional waypoint-specific data can be included as additional keys/columns.
    longitude : npt.ArrayLike, optional
        Longitude data.
        Defaults to None.
    latitude : npt.ArrayLike, optional
        Latitude data.
        Defaults to None.
    altitude : npt.ArrayLike, optional
        Altitude data, [:math:`m`].
        Defaults to None.
    level : npt.ArrayLike, optional
        Level data, [:math:`hPa`].
        Defaults to None.
    time : npt.ArrayLike, optional
        Time data.
        Expects an array of DatetimeLike values,
        or array that can proccessed with :meth:`pd.to_datetime`.
        Defaults to None.
    attrs : dict[Hashable, Any] | AttrDict, optional
        Additional properties as a dictionary.
        Defaults to {}.
    copy : bool, optional
        Copy data on class creation.
        Defaults to True.
    **attrs_kwargs : Any
        Additional properties passed as keyword arguments.

    Raises
    ------
    KeyError
        Raises if ``data`` input does not contain at least ``time``, ``latitude``, ``longitude``,
        (``altitude`` or ``level``).
    """

    #: Required keys for creating GeoVectorDataset
    required_keys: set[str] = {
        "latitude",
        "longitude",
        "time",
    }

    def __init__(
        self,
        data: dict[str, npt.ArrayLike]
        | pd.DataFrame
        | VectorDataDict
        | VectorDataset
        | None = None,
        longitude: npt.ArrayLike | None = None,
        latitude: npt.ArrayLike | None = None,
        altitude: npt.ArrayLike | None = None,
        level: npt.ArrayLike | None = None,
        time: npt.ArrayLike | None = None,
        attrs: dict[str, Any] | AttrDict | None = None,
        copy: bool = True,
        **attrs_kwargs: Any,
    ) -> None:
        # shortcut to `GeoVectorDataset.create_empty` by just using `GeoVectorDataset()`
        if (
            data is None
            and longitude is None
            and latitude is None
            and altitude is None
            and level is None
            and time is None
        ):
            keys = self.required_keys.union(["altitude"])
            data = _empty_vector_dict(keys)

        super().__init__(data=data, attrs=attrs, copy=copy, **attrs_kwargs)

        # using the self[key] syntax specifically to run qc on assigment
        if longitude is not None:
            self["longitude"] = np.array(longitude, copy=copy)

        if latitude is not None:
            self["latitude"] = np.array(latitude, copy=copy)

        if time is not None:
            self["time"] = np.array(time, copy=copy)

        if altitude is not None:
            self["altitude"] = np.array(altitude, copy=copy)

        if level is not None:
            self["level"] = np.array(level, copy=copy)

        # warn if level and altitude are both defined
        if "level" not in self and "altitude" not in self and "altitude_ft" not in self:
            raise KeyError(
                f"{self.__class__.__name__} requires either "
                "`level` or `altitude` or `altitude_ft` data key"
            )

        # TODO: do we want to throw a warning if both level and altitude are specified?
        # This is the default behavior in Cocip
        # elif "level" in self and "altitude" in self:
        #     warnings.warn(
        #         f"{self.__class__.__name__} contains both `level` and `altitude` keys. "
        #         "This can result in ambiguous behavior."
        #     )

        # confirm that input dataframe has required columns
        if not self.required_keys.issubset(self):
            missing_keys = self.required_keys.difference(self)
            raise KeyError(
                f"Missing required data keys: {missing_keys}. "
                f"Use {self.__class__.__name__}.create_empty() to create an empty dataset."
            )

        # Parse time: If time is not np.datetime64, we try to coerce it to be
        # by pumping it through pd.to_datetime.
        time = self["time"]
        if not np.issubdtype(time.dtype, np.datetime64):
            warnings.warn("Time data is not np.datetime64. Attempting to coerce.")
            try:
                pd_time = pd.to_datetime(self["time"])
            except ValueError as e:
                raise ValueError("Could not coerce time data to datetime64.") from e
            np_time = pd_time.to_numpy(dtype="datetime64[ns]")
            self.update(time=np_time)
        elif time.dtype != "datetime64[ns]":
            self.update(time=time.astype("datetime64[ns]"))

        # Ensure spatial coordinates are float32 or float64
        float_dtype = (np.float32, np.float64)
        for coord in ("longitude", "latitude", "altitude", "level", "altitude_ft"):
            try:
                arr = self[coord]
            except KeyError:
                continue
            if arr.dtype not in float_dtype:
                self.update({coord: arr.astype(np.float64)})

        # set CRS to "EPSG:4326" by default
        crs = self.attrs.setdefault("crs", "EPSG:4326")

        if crs == "EPSG:4326":
            longitude = self["longitude"]
            if np.any(longitude > 180.0) or np.any(longitude < -180.0):
                raise ValueError("EPSG:4326 longitude coordinates should lie between [-180, 180).")
            latitude = self["latitude"]
            if np.any(latitude > 90.0) or np.any(latitude < -90.0):
                raise ValueError("EPSG:4326 latitude coordinates should lie between [-90, 90].")

    @overrides
    def _display_attrs(self) -> dict[str, str]:
        try:
            time0, time1 = np.nanmin(self["time"]), np.nanmax(self["time"])
            lon0, lon1 = np.nanmin(self["longitude"]), np.nanmax(self["longitude"])
            lat0, lat1 = np.nanmin(self["latitude"]), np.nanmax(self["latitude"])
            alt0, alt1 = np.nanmin(self.altitude), np.nanmax(self.altitude)
            attrs = {
                "time": f"[{pd.Timestamp(time0)}, {pd.Timestamp(time1)}]",
                "longitude": f"[{lon0}, {lon1}]",
                "latitude": f"[{lat0}, {lat1}]",
                "altitude": f"[{alt0}, {alt1}]",
            }
        except Exception:
            attrs = {}

        attrs.update(super()._display_attrs())
        return attrs

    @property
    def level(self) -> npt.NDArray[np.float_]:
        """Get pressure ``level`` values for points.

        Automatically calculates pressure level using :func:`units.m_to_pl` using ``altitude`` key.

        Note that if ``level`` key exists in :attr:`data`, the data at the ``level``
        key will be returned. This allows an override of the default calculation
        of pressure level from altitude.

        Returns
        -------
        npt.NDArray[np.float_]
            Point pressure level values, [:math:`hPa`]
        """
        try:
            return self["level"]
        except KeyError:
            return units.m_to_pl(self.altitude)

    @property
    def altitude(self) -> npt.NDArray[np.float_]:
        """Get altitude.

        Automatically calculates altitude using :func:`units.pl_to_m` using ``level`` key.

        Note that if ``altitude`` key exists in :attr:`data`, the data at the ``altitude``
        key will be returned. This allows an override of the default calculation of altitude
        from pressure level.

        Returns
        -------
        npt.NDArray[np.float_]
            Altitude, [:math:`m`]
        """
        try:
            return self["altitude"]
        except KeyError:
            # Implementation note: explicitly look for "level" or "altitude_ft" key
            # here to avoid getting stuck in an infinite loop when .level or .altitude_ft
            # are called.
            if (level := self.get("level")) is not None:
                return units.pl_to_m(level)
            return units.ft_to_m(self["altitude_ft"])

    @property
    def air_pressure(self) -> npt.NDArray[np.float_]:
        """Get ``air_pressure`` values for points.

        Returns
        -------
        npt.NDArray[np.float_]
            Point air pressure values, [:math:`Pa`]
        """
        try:
            return self["air_pressure"]
        except KeyError:
            return 100.0 * self.level

    @property
    def altitude_ft(self) -> npt.NDArray[np.float_]:
        """Get altitude in feet.

        Returns
        -------
        npt.NDArray[np.float_]
            Altitude, [:math:`ft`]
        """
        try:
            return self["altitude_ft"]
        except KeyError:
            return units.m_to_ft(self.altitude)

    @property
    def constants(self) -> dict[str, Any]:
        """Return a dictionary of constant attributes and data values.

        Includes :attr:`attrs` and values from columns in :attr:`data` with a unique
        value.

        Returns
        -------
        dict[str, Any]
            Properties and their constant values
        """
        constants = {}

        # get constant data values that are not nan
        for key in set(self.data) - self.required_keys:
            unique = np.unique(self[key])
            if len(unique) == 1 and (isinstance(unique[0], str) or ~np.isnan(unique[0])):
                constants[key] = unique[0]

        # add attributes
        constants.update(self.attrs)

        # clean strings values by removing whitespace
        # convert any numpy items to python objects
        def _cleanup(v: Any) -> Any:
            if isinstance(v, str):
                return v.strip()
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.floating):
                return float(v)
            if isinstance(v, np.bool_):
                return bool(v)
            return v

        return {k: _cleanup(v) for k, v in constants.items()}

    @property
    def coords(self) -> dict[str, np.ndarray]:
        """Get geospatial coordinates for compatibility with MetDataArray.

        Returns
        -------
        pd.DataFrame
            :class:`pd.DataFrame` with columns `longitude`, `latitude`, `level`, and `time`.
        """
        return {
            "longitude": self["longitude"],
            "latitude": self["latitude"],
            "level": self.level,
            "time": self["time"],
        }

    # ------------
    # Utilities
    # ------------

    def transform_crs(
        self: GeoVectorDatasetType, crs: str, copy: bool = True
    ) -> GeoVectorDatasetType:
        """Transform trajectory data from one coordinate reference system (CRS) to another.

        Parameters
        ----------
        crs : str
            Target CRS. Passed into to :class:`pyproj.Transformer`. The source CRS
            is inferred from the :attr:`attrs["crs"]` attribute.
        copy : bool, optional
            Copy data on transformation. Defaults to True.

        Returns
        -------
        GeoVectorDatasetType
            Converted dataset with new coordinate reference system.
            :attr:`attrs["crs"]` reflects new crs.
        """
        transformer = pyproj.Transformer.from_crs(self.attrs["crs"], crs, always_xy=True)
        lon, lat = transformer.transform(self["longitude"], self["latitude"])

        if copy:
            ret = self.copy()
        else:
            ret = self

        ret.update(longitude=lon, latitude=lat)
        ret.attrs.update(crs=crs)
        return ret

    # ------------
    # Met
    # ------------

    def coords_intersect_met(
        self, met: met_module.MetDataset | met_module.MetDataArray
    ) -> npt.NDArray[np.bool_]:
        """Return boolean mask of data inside the bounding box defined by ``met``.

        Parameters
        ----------
        met : MetDataset | MetDataArray
            MetDataset or MetDataArray to compare.

        Returns
        -------
        npt.NDArray[np.bool_]
            True if point is inside the bounding box defined by ``met``.
        """

        lat_intersect = coordinates.intersect_domain(
            met.variables["latitude"].values,
            self["latitude"],
        )
        lon_intersect = coordinates.intersect_domain(
            met.variables["longitude"].values,
            self["longitude"],
        )
        level_intersect = coordinates.intersect_domain(
            met.variables["level"].values,
            self.level,
        )
        time_intersect = coordinates.intersect_domain(
            met.variables["time"].values,
            self["time"],
        )

        return lat_intersect & lon_intersect & level_intersect & time_intersect

    def intersect_met(
        self,
        mda: met_module.MetDataArray,
        *,
        longitude: npt.NDArray[np.float_] | None = None,
        latitude: npt.NDArray[np.float_] | None = None,
        level: npt.NDArray[np.float_] | None = None,
        time: npt.NDArray[np.datetime64] | None = None,
        use_indices: bool = False,
        **interp_kwargs: Any,
    ) -> npt.NDArray[np.float_]:
        """Intersect waypoints with MetDataArray.

        Parameters
        ----------
        mda : MetDataArray
            MetDataArray containing a meteorological variable at spatio-temporal coordinates.
        longitude : npt.NDArray[np.float_], optional
            Override existing coordinates for met interpolation
        latitude : npt.NDArray[np.float_], optional
            Override existing coordinates for met interpolation
        level : npt.NDArray[np.float_], optional
            Override existing coordinates for met interpolation
        time : npt.NDArray[np.datetime64], optional
            Override existing coordinates for met interpolation
        use_indices : bool, optional
            Experimental.
        **interp_kwargs : Any
            Additional keyword arguments to pass to :meth:`MetDataArray.intersect_met`.
            Examples include ``method``, ``bounds_error``, and ``fill_value``. If an error such as
            `ValueError: One of the requested xi is out of bounds in dimension 2` occurs, try
            calling this function with ``bounds_error=False``. In addition, setting
            ``fill_value=0.0`` will replace NaN values with 0.0.

        Returns
        -------
        npt.NDArray[np.float_]
            Interpolated values

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

        >>> # Example flight
        >>> df = pd.DataFrame()
        >>> df['longitude'] = np.linspace(0, 50, 10)
        >>> df['latitude'] = np.linspace(0, 10, 10)
        >>> df['altitude'] = 11000
        >>> df['time'] = pd.date_range("2022-03-01T00", "2022-03-01T02", periods=10)
        >>> fl = Flight(df)

        >>> # Intersect
        >>> fl.intersect_met(met['air_temperature'], method='nearest')
        array([231.6297 , 230.72604, 232.2432 , 231.88339, 231.0643 , 231.59073,
               231.65126, 231.93065, 232.03345, 231.65955], dtype=float32)

        >>> fl.intersect_met(met['air_temperature'], method='linear')
        array([225.77794, 225.13908, 226.23122, 226.31831, 225.56102, 225.81192,
               226.03194, 226.22057, 226.0377 , 225.63226], dtype=float32)

        >>> # Interpolate and attach to `Flight` instance
        >>> for key in met:
        ...     fl[key] = fl.intersect_met(met[key])

        >>> # Show the final three columns of the dataframe
        >>> fl.dataframe.iloc[:, -3:].head()
                         time  air_temperature  specific_humidity
        0 2022-03-01 00:00:00       225.777939           0.000132
        1 2022-03-01 00:13:20       225.139084           0.000132
        2 2022-03-01 00:26:40       226.231216           0.000107
        3 2022-03-01 00:40:00       226.318314           0.000171
        4 2022-03-01 00:53:20       225.561020           0.000109

        """
        # Override use_indices in certain situations
        if use_indices:
            # Often the single_level data we use has time shifted
            # Don't allow it for now. We could do something smarter here!
            if mda.is_single_level:
                use_indices = False

            # Cannot both override some coordinate AND pass indices.
            elif any(c is not None for c in (longitude, latitude, level, time)):
                # Should we warn?! Or is this "convenience"?
                use_indices = False

        longitude = longitude if longitude is not None else self["longitude"]
        latitude = latitude if latitude is not None else self["latitude"]
        level = level if level is not None else self.level
        time = time if time is not None else self["time"]

        if not use_indices:
            return mda.interpolate(longitude, latitude, level, time, **interp_kwargs)

        indices = self._get_indices()
        already_has_indices = indices is not None
        out, indices = mda.interpolate(
            longitude,
            latitude,
            level,
            time,
            indices=indices,
            return_indices=True,
            **interp_kwargs,
        )
        if not already_has_indices:
            self._put_indices(indices)
        return out

    def _put_indices(self, indices: interpolation.RGIArtifacts) -> None:
        """Set entries of ``indices`` onto underlying :attr:`data.

        Each entry of ``indices`` are unpacked assuming certain conventions
        for its structure. A ValueError is raise if these conventions are not
        satisfied.

        .. versionadded:: 0.26.0

            Experimental


        Parameters
        ----------
        indices : interpolation.RGIArtifacts
            The indices to store.
        """
        indices_x, indices_y, indices_z, indices_t = indices.xi_indices
        distances_x, distances_y, distances_z, distances_t = indices.norm_distances
        out_of_bounds = indices.out_of_bounds

        self["_indices_x"] = indices_x
        self["_indices_y"] = indices_y
        self["_indices_z"] = indices_z
        self["_indices_t"] = indices_t
        self["_distances_x"] = distances_x
        self["_distances_y"] = distances_y
        self["_distances_z"] = distances_z
        self["_distances_t"] = distances_t
        self["_out_of_bounds"] = out_of_bounds

    def _get_indices(self) -> interpolation.RGIArtifacts | None:
        """Get entries from call to :meth:`_put_indices`.

        .. versionadded:: 0.26.0

            Experimental

        Returns
        -------
        tuple | None
            Previously cached output of
            :meth:`scipy.interpolate.RegularGridInterpolator._find_indices`,
            or None if cached output is not present on instance.
        """
        try:
            indices_x = self["_indices_x"]
            indices_y = self["_indices_y"]
            indices_z = self["_indices_z"]
            indices_t = self["_indices_t"]
            distances_x = self["_distances_x"]
            distances_y = self["_distances_y"]
            distances_z = self["_distances_z"]
            distances_t = self["_distances_t"]
            out_of_bounds = self["_out_of_bounds"]
        except KeyError:
            return None

        indices = np.asarray([indices_x, indices_y, indices_z, indices_t])
        distances = np.asarray([distances_x, distances_y, distances_z, distances_t])

        return interpolation.RGIArtifacts(indices, distances, out_of_bounds)

    def _invalidate_indices(self) -> None:
        """Remove any cached indices from :attr:`data."""
        for key in (
            "_indices_x",
            "_indices_y",
            "_indices_z",
            "_indices_t",
            "_distances_x",
            "_distances_y",
            "_distances_z",
            "_distances_t",
            "_out_of_bounds",
        ):
            self.data.pop(key, None)

    @overload
    def downselect_met(
        self,
        met: met_module.MetDataset,
        *,
        longitude_buffer: tuple[float, float] = ...,
        latitude_buffer: tuple[float, float] = ...,
        level_buffer: tuple[float, float] = ...,
        time_buffer: tuple[np.timedelta64, np.timedelta64] = ...,
        copy: bool = ...,
    ) -> met_module.MetDataset:
        ...

    @overload
    def downselect_met(
        self,
        met: met_module.MetDataArray,
        *,
        longitude_buffer: tuple[float, float] = ...,
        latitude_buffer: tuple[float, float] = ...,
        level_buffer: tuple[float, float] = ...,
        time_buffer: tuple[np.timedelta64, np.timedelta64] = ...,
        copy: bool = ...,
    ) -> met_module.MetDataArray:
        ...

    def downselect_met(
        self,
        met: met_module.MetDataType,
        *,
        longitude_buffer: tuple[float, float] = (0.0, 0.0),
        latitude_buffer: tuple[float, float] = (0.0, 0.0),
        level_buffer: tuple[float, float] = (0.0, 0.0),
        time_buffer: tuple[np.timedelta64, np.timedelta64] = (
            np.timedelta64(0, "h"),
            np.timedelta64(0, "h"),
        ),
        copy: bool = True,
    ) -> met_module.MetDataType:
        """Downselect ``met`` to encompass a spatiotemporal region of the data.

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
        copy : bool
            If returned object is a copy or view of the original. True by default.

        Returns
        -------
        MetDataset | MetDataArray
            Copy of downselected MetDataset or MetDataArray.
        """
        lon_slice = coordinates.slice_domain(
            met.variables["longitude"].values,
            self["longitude"],
            buffer=longitude_buffer,
        )
        lat_slice = coordinates.slice_domain(
            met.variables["latitude"].values,
            self["latitude"],
            buffer=latitude_buffer,
        )
        time_slice = coordinates.slice_domain(
            met.variables["time"].values,
            self["time"],
            buffer=time_buffer,
        )

        # single level data have "level" == [-1]
        if met.is_single_level:
            level_slice = slice(None)
        else:
            level_slice = coordinates.slice_domain(
                met.variables["level"].values,
                self.level,
                buffer=level_buffer,
            )
        logger.debug("Downselect met at %s %s %s %s", lon_slice, lat_slice, level_slice, time_slice)

        data = met.data.isel(
            longitude=lon_slice,
            latitude=lat_slice,
            level=level_slice,
            time=time_slice,
        )
        return type(met)(data, copy=copy)

    # ------------
    # I / O
    # ------------

    @classmethod
    @overrides
    def create_empty(
        cls: Type[GeoVectorDatasetType],
        keys: Iterable[str] | None = None,
        attrs: dict[str, Any] | None = None,
        **attrs_kwargs: Any,
    ) -> GeoVectorDatasetType:
        keys = cls.required_keys.union(keys or set())
        keys.add("altitude")
        return super().create_empty(keys, attrs, **attrs_kwargs)

    def to_geojson_points(self) -> dict[str, Any]:
        """Return dataset as GeoJSON FeatureCollection of Points.

        Each Feature has a properties attribute that includes `time` and
        other data besides `latitude`, `longitude`, and `altitude` in :attr:`data`.

        Returns
        -------
        dict[str, Any]
            Python representation of GeoJSON FeatureCollection
        """
        return json_module.dataframe_to_geojson_points(self.dataframe)

    def to_pseudo_mercator(self: GeoVectorDatasetType, copy: bool = True) -> GeoVectorDatasetType:
        """Convert data from :attr:`attrs["crs"]` to Pseudo Mercator (EPSG:3857).

        Parameters
        ----------
        copy : bool, optional
            Copy data on tranformation.
            Defaults to True.

        Returns
        -------
        GeoVectorDatasetType
        """
        return self.transform_crs("EPSG:3857", copy=copy)
