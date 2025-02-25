"""Lightweight data structures for vector paths."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import warnings
from collections.abc import Generator, Iterable, Iterator, Sequence
from typing import Any, overload

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

from pycontrails.core import coordinates, interpolation
from pycontrails.core import met as met_module
from pycontrails.physics import units
from pycontrails.utils import dependencies
from pycontrails.utils import json as json_utils

logger = logging.getLogger(__name__)


class AttrDict(dict[str, Any]):
    """Thin wrapper around dict to warn when setting a key that already exists."""

    __slots__ = ()

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


class VectorDataDict(dict[str, np.ndarray]):
    """Thin wrapper around ``dict[str, np.ndarray]`` to ensure consistency.

    Parameters
    ----------
    data : dict[str, np.ndarray], optional
        Dictionary input. A shallow copy is always made.
    """

    __slots__ = ("_size",)

    #: Length of the data
    _size: int

    def __init__(self, data: dict[str, np.ndarray] | None = None) -> None:
        super().__init__(data or {})

        # validate any arrays, first one defines _size attribute
        for arr in self.values():
            self._validate_array(arr)

    def __setitem__(self, k: str, v: npt.ArrayLike) -> None:
        """Set new key-value pair to instance and warn when overwriting existing key.

        This method casts ``v`` to an :class:`numpy.ndarray` and ensures that the array size is
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

        # if no keys remain, delete _size attribute
        if not self:
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

        This method casts values in  ``other`` to :class:`numpy.ndarray` and
        ensures that the array sizes are consistent with the instance.

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
        """Ensure that ``arr`` is compatible (1 dimensional of equal size) with instance.

        Set attribute ``_size`` if it has not yet been defined.

        Parameters
        ----------
        arr : np.ndarray
            Array to validate

        Raises
        ------
        ValueError
            If ``arr`` is not compatible with instance.
        """
        if arr.ndim != 1:
            raise ValueError("All np.arrays must have dimension 1.")

        size = getattr(self, "_size", 0)
        if not size:
            self._size = arr.size
            return

        if arr.size != size:
            raise ValueError(f"Incompatible array sizes: {arr.size} and {size}.")


def _empty_vector_dict(keys: Iterable[str]) -> dict[str, np.ndarray]:
    """Create a dictionary with keys defined by ``keys`` and empty arrays.

    Parameters
    ----------
    keys : Iterable[str]
        Keys to include in dictionary.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with empty arrays.
    """
    data = {key: np.array([]) for key in keys}

    # The default dtype is float64
    # Time is special and should have a non-default dtype of datetime64[ns]
    if "time" in data:
        data.update(time=np.array([], dtype="datetime64[ns]"))

    return data


class VectorDataset:
    """Base class to hold 1D arrays of consistent size.

    Parameters
    ----------
    data : dict[str, npt.ArrayLike] | pd.DataFrame | VectorDataset | None, optional
        Initial data, by default None. A shallow copy is always made. Use the ``copy``
        parameter to copy the underlying array data.
    attrs : dict[str, Any] | None, optional
        Dictionary of attributes, by default None. A shallow copy is always made.
    copy : bool, optional
        Copy individual arrays on instantiation, by default True.
    **attrs_kwargs : Any
        Additional attributes passed as keyword arguments.

    Raises
    ------
    ValueError
        If "time" variable cannot be converted to numpy array.
    """

    __slots__ = ("attrs", "data")

    #: Generic dataset attributes
    attrs: AttrDict

    #: Vector data with labels as keys and :class:`numpy.ndarray` as values
    data: VectorDataDict

    def __init__(
        self,
        data: dict[str, npt.ArrayLike] | pd.DataFrame | VectorDataset | None = None,
        *,
        attrs: dict[str, Any] | None = None,
        copy: bool = True,
        **attrs_kwargs: Any,
    ) -> None:
        # Set data: always shallow copy
        # -----------------------------

        # Casting from one VectorDataset type to another
        # e.g., flight = Flight(...); vector = VectorDataset(flight)
        if isinstance(data, VectorDataset):
            attrs = {**data.attrs, **(attrs or {})}
            if copy:
                self.data = VectorDataDict({k: v.copy() for k, v in data.data.items()})
            else:
                self.data = VectorDataDict(data.data)

        elif data is None:
            self.data = VectorDataDict()

        elif isinstance(data, pd.DataFrame):
            attrs = {**data.attrs, **(attrs or {})}

            # Take extra caution with a time column
            try:
                time = data["time"]
            except KeyError:
                self.data = VectorDataDict({k: v.to_numpy(copy=copy) for k, v in data.items()})
            else:
                time = _handle_time_column(time)
                data = {k: v.to_numpy(copy=copy) for k, v in data.items() if k != "time"}
                data["time"] = time.to_numpy(copy=copy)
                self.data = VectorDataDict(data)

        # For anything else, we assume it is a dictionary of array-like and attach it
        else:
            self.data = VectorDataDict({k: np.array(v, copy=copy) for k, v in data.items()})

        # Set attributes: always shallow copy
        # -----------------------------------

        self.attrs = AttrDict(attrs or {})  # type: ignore[arg-type]
        self.attrs.update(attrs_kwargs)

    @classmethod
    def _from_fastpath(
        cls,
        data: dict[str, np.ndarray],
        attrs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create new instance from consistent data.

        This is a low-level method that bypasses the standard constructor in certain
        special cases. It is intended for internal use only.

        In essence, this method skips any validation from __init__ and directly sets
        ``data`` and ``attrs``. This is useful when creating a new instance from an existing
        instance the data has already been validated.
        """
        obj = cls.__new__(cls)

        obj.data = VectorDataDict(data)
        obj.attrs = AttrDict(attrs or {})

        for key, value in kwargs.items():
            try:
                setattr(obj, key, value)
            # If key not present in __slots__ of class (or parents), it's intended for attrs
            except AttributeError:
                obj.attrs[key] = value

        return obj

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
        """Get values from :attr:`data` with ``default_value`` if ``key`` not in :attr:`data`.

        Parameters
        ----------
        key : str
            Key to get from :attr:`data`
        default_value : Any, optional
            Return ``default_value`` if `key` not in :attr:`data`, by default ``None``

        Returns
        -------
        Any
            Values at :attr:`data[key]` or ``default_value``
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
        """Shortcut to :meth:`VectorDataDict.setdefault`.

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

    __marker = object()

    def get_data_or_attr(self, key: str, default: Any = __marker) -> Any:
        """Get value from :attr:`data` or :attr:`attrs`.

        This method first checks if ``key`` is in :attr:`data` and returns the value if so.
        If ``key`` is not in :attr:`data`, then this method checks if ``key`` is in :attr:`attrs`
        and returns the value if so. If ``key`` is not in :attr:`data` or :attr:`attrs`,
        then the ``default`` value is returned if provided. Otherwise a :class:`KeyError` is raised.

        Parameters
        ----------
        key : str
            Key to get from :attr:`data` or :attr:`attrs`
        default : Any, optional
            Default value to return if ``key`` is not in :attr:`data` or :attr:`attrs`.

        Returns
        -------
        Any
            Value at :attr:`data[key]` or :attr:`attrs[key]`

        Raises
        ------
        KeyError
            If ``key`` is not in :attr:`data` or :attr:`attrs` and ``default`` is not provided.

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
        KeyError: "Key 'c' not found in data or attrs."

        >>> vector.get_data_or_attr("c", default=5)
        5

        See Also
        --------
        get_constant
        """
        marker = self.__marker

        out = self.get(key, marker)
        if out is not marker:
            return out

        out = self.attrs.get(key, marker)
        if out is not marker:
            return out

        if default is not marker:
            return default

        msg = f"Key '{key}' not found in data or attrs."
        raise KeyError(msg)

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
        """Return properties used in `repr` constructions.

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

        keys = list(self)
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

    def __add__(self, other: Self | None) -> Self:
        """Concatenate two compatible instances of VectorDataset.

        In this context, compatibility means that both have identical :attr:`data` keys.

        This operator behaves similarly to the ``__add__`` method on python lists.

        If self is an empty VectorDataset, return other. This is useful when
        calling :keyword:`sum` with an empty initial value.

        Parameters
        ----------
        other : Self | None
            Other values to concatenate

        Returns
        -------
        Self
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
        cls,
        vectors: Sequence[VectorDataset],
        infer_attrs: bool = True,
        fill_value: float | None = None,
    ) -> Self:
        """Sum a list of :class:`VectorDataset` instances.

        Parameters
        ----------
        vectors : Sequence[VectorDataset]
            List of :class:`VectorDataset` instances to concatenate.
        infer_attrs : bool, optional
            If True, infer attributes from the first element in the sequence.
        fill_value : float, optional
            Fill value to use when concatenating arrays. By default None, which raises
            an error if incompatible keys are found.

        Returns
        -------
        VectorDataset
            Sum of all instances in ``vectors``.

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
        if cls not in (VectorDataset, GeoVectorDataset):
            msg = (
                "Method 'sum' is only available on 'VectorDataset' and 'GeoVectorDataset'. "
                "To sum 'Flight' instances, use 'Fleet.from_seq'."
            )
            raise TypeError(msg)

        vectors = [v for v in vectors if v is not None]  # remove None values

        if not vectors:
            return cls()

        keys: Iterable[str]
        if fill_value is None:
            keys = vectors[0].data.keys()
            for v in vectors[1:]:
                if v.data.keys() != keys:
                    diff = set(v).symmetric_difference(keys)
                    msg = f"Summands have incompatible keys. Difference: {diff}"
                    raise KeyError(msg)

        else:
            keys = set().union(*[v.data.keys() for v in vectors])

        def _get(k: str, v: VectorDataset) -> np.ndarray:
            # Could also use VectorDataset.get() here, but we want to avoid creating
            # an unused array if the key is present in the VectorDataset.
            try:
                return v[k]
            except KeyError:
                return np.full(v.size, fill_value)

        def concat(key: str) -> np.ndarray:
            values = [_get(key, v) for v in vectors]
            return np.concatenate(values)

        data = {key: concat(key) for key in keys}
        attrs = vectors[0].attrs if infer_attrs else None

        return cls._from_fastpath(data, attrs)

    def __eq__(self, other: object) -> bool:
        """Determine if two instances are equal.

        NaN values are considered equal in this comparison.

        Parameters
        ----------
        other : object
            VectorDataset to compare with

        Returns
        -------
        bool
            True if both instances have identical :attr:`data` and :attr:`attrs`.
        """
        if not isinstance(other, VectorDataset):
            return False

        # Check attrs
        if self.attrs.keys() != other.attrs.keys():
            return False

        for key, val in self.attrs.items():
            if isinstance(val, np.ndarray):
                # equal_nan not supported for non-numeric data
                equal_nan = not np.issubdtype(val.dtype, "O")
                if not np.array_equal(val, other.attrs[key], equal_nan=equal_nan):
                    return False
            elif val != other.attrs[key]:
                return False

        # Check data
        if self.data.keys() != other.data.keys():
            return False

        for key, val in self.data.items():
            # equal_nan not supported for non-numeric data (e.g. strings)
            equal_nan = not np.issubdtype(val.dtype, "O")
            if not np.array_equal(val, other[key], equal_nan=equal_nan):
                return False

        return True

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
        """Shorthand property to access :meth:`to_dataframe` with ``copy=False``.

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
        _hash = json.dumps(self.data, cls=json_utils.NumpyEncoder)
        return hashlib.sha1(bytes(_hash, "utf-8")).hexdigest()

    # ------------
    # Utilities
    # ------------

    def copy(self, **kwargs: Any) -> Self:
        """Return a copy of this instance.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed into the constructor of the returned class.

        Returns
        -------
        Self
            Copy of class
        """
        data = {key: value.copy() for key, value in self.data.items()}
        return type(self)._from_fastpath(data, self.attrs, **kwargs)

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
        data = {key: np.array(self[key], copy=copy) for key in keys}
        return VectorDataset._from_fastpath(data, self.attrs)

    def filter(self, mask: npt.NDArray[np.bool_], copy: bool = True, **kwargs: Any) -> Self:
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
        **kwargs : Any
            Additional keyword arguments passed into the constructor of the returned class.

        Returns
        -------
        Self
            Containing filtered data

        Raises
        ------
        TypeError
            If ``mask`` is not a boolean array.
        """
        self.data._validate_array(mask)
        if mask.dtype != bool:
            raise TypeError("Parameter `mask` must be a boolean array.")

        data = {key: np.array(value[mask], copy=copy) for key, value in self.data.items()}
        return type(self)._from_fastpath(data, self.attrs, **kwargs)

    def sort(self, by: str | list[str]) -> Self:
        """Sort data by key(s).

        This method always creates a copy of the data by calling
        :meth:`pandas.DataFrame.sort_values`.

        Parameters
        ----------
        by : str | list[str]
            Key or list of keys to sort by.

        Returns
        -------
        Self
            Instance with sorted data.
        """
        return type(self)(data=self.dataframe.sort_values(by=by), attrs=self.attrs, copy=False)

    def ensure_vars(self, vars: str | Iterable[str], raise_error: bool = True) -> bool:
        """Ensure variables exist in column of :attr:`data` or :attr:`attrs`.

        Parameters
        ----------
        vars : str | Iterable[str]
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
                msg = f"{type(self).__name__} instance does not contain data or attr '{v}'"
                raise KeyError(msg)
            return False

        return True

    def broadcast_attrs(
        self,
        keys: str | Iterable[str],
        overwrite: bool = False,
        raise_error: bool = True,
    ) -> None:
        """Attach values from ``keys`` in :attr:`attrs` onto :attr:`data`.

        If possible, use ``dtype = np.float32`` when broadcasting. If not possible,
        use whatever ``dtype`` is inferred from the data by :func:`numpy.full`.

        Parameters
        ----------
        keys : str | Iterable[str]
            Keys to broadcast
        overwrite : bool, optional
            If True, overwrite existing values in :attr:`data`. By default False.
        raise_error : bool, optional
            Raise KeyError if :attr:`self.attrs` does not contain some of ``keys``.

        Raises
        ------
        KeyError
            Not all ``keys`` found in :attr:`attrs`.
        """
        if isinstance(keys, str):
            keys = (keys,)

        # Validate everything up front to avoid partial broadcasting
        for key in keys:
            try:
                scalar = self.attrs[key]
            except KeyError as exc:
                if raise_error:
                    raise KeyError(f"{type(self)} does not contain attr `{key}`") from exc
                continue

            if key in self.data and not overwrite:
                warnings.warn(
                    f"Found duplicate key {key} in attrs and data. "
                    "Set `overwrite=True` parameter to force overwrite."
                )
                continue

            min_dtype = np.min_scalar_type(scalar)
            dtype = np.float32 if np.can_cast(min_dtype, np.float32) else None
            self.data.update({key: np.full(self.size, scalar, dtype=dtype)})

    def broadcast_numeric_attrs(
        self, ignore_keys: str | Iterable[str] | None = None, overwrite: bool = False
    ) -> None:
        """Attach numeric values in :attr:`attrs` onto :attr:`data`.

        Iterate through values in :attr:`attrs` and attach :class:`float` and
        :class:`int` values to ``data``.

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
        numeric_attrs = (
            attr
            for attr, val in self.attrs.items()
            if (isinstance(val, int | float | np.number) and attr not in ignore_keys)
        )
        self.broadcast_attrs(numeric_attrs, overwrite)

    def get_constant(self, key: str, default: Any = __marker) -> Any:
        """Get a constant value from :attr:`attrs` or :attr:`data`.

        - If ``key`` is found in :attr:`attrs`, the value is returned.
        - If ``key`` is found in :attr:`data`, the common value is returned if all
          values are equal.
        - If ``key`` is not found in :attr:`attrs` or :attr:`data` and a ``default`` is provided,
          the ``default`` is returned.
        - Otherwise, a KeyError is raised.

        Parameters
        ----------
        key : str
            Key to look for.
        default : Any, optional
            Default value to return if ``key`` is not found in :attr:`attrs` or :attr:`data`.

        Returns
        -------
        Any
            The constant value for ``key``.

        Raises
        ------
        KeyError
            If ``key`` is not found in :attr:`attrs` or the values in :attr:`data` are not equal
            and ``default`` is not provided.

        Examples
        --------
        >>> vector = VectorDataset({"a": [1, 1, 1], "b": [2, 2, 3]})
        >>> vector.get_constant("a")
        np.int64(1)
        >>> vector.get_constant("b")
        Traceback (most recent call last):
        ...
        KeyError: "A constant key 'b' not found in attrs or data"
        >>> vector.get_constant("b", 3)
        3

        See Also
        --------
        get_data_or_attr
        GeoVectorDataset.constants
        """
        marker = self.__marker

        out = self.attrs.get(key, marker)
        if out is not marker:
            return out

        arr: np.ndarray = self.data.get(key, marker)  # type: ignore[arg-type]
        if arr is not marker:
            try:
                vals = np.unique(arr)
            except TypeError:
                # A TypeError can occur if the arr has object dtype and contains None
                # Handle this case by returning None
                if arr.dtype == object and np.all(arr == None):  # noqa: E711
                    return None
                raise

            if len(vals) == 1:
                return vals[0]

        if default is not marker:
            return default

        msg = f"A constant key '{key}' not found in attrs or data"
        raise KeyError(msg)

    # ------------
    # I / O
    # ------------

    def to_dataframe(self, copy: bool = True) -> pd.DataFrame:
        """Create :class:`pd.DataFrame` in which each key-value pair in :attr:`data` is a column.

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

    def to_dict(self) -> dict[str, Any]:
        """Create dictionary with :attr:`data` and :attr:`attrs`.

        If geo-spatial coordinates (e.g. ``"latitude"``, ``"longitude"``, ``"altitude"``)
        are present, round to a reasonable precision. If a ``"time"`` variable is present,
        round to unix seconds. When the instance is a :class:`GeoVectorDataset`,
        disregard any ``"altitude"`` or ``"level"`` coordinate and only include
        ``"altitude_ft"`` in the output.

        Returns
        -------
        dict[str, Any]
            Dictionary with :attr:`data` and :attr:`attrs`.

        See Also
        --------
        :meth:`from_dict`

        Examples
        --------
        >>> import pprint
        >>> from pycontrails import Flight
        >>> fl = Flight(
        ...     longitude=[-100, -110],
        ...     latitude=[40, 50],
        ...     level=[200, 200],
        ...     time=[np.datetime64("2020-01-01T09"), np.datetime64("2020-01-01T09:30")],
        ...     aircraft_type="B737",
        ... )
        >>> fl = fl.resample_and_fill("5min")
        >>> pprint.pprint(fl.to_dict())
        {'aircraft_type': 'B737',
         'altitude_ft': [38661.0, 38661.0, 38661.0, 38661.0, 38661.0, 38661.0, 38661.0],
         'latitude': [40.0, 41.724, 43.428, 45.111, 46.769, 48.399, 50.0],
         'longitude': [-100.0,
                       -101.441,
                       -102.959,
                       -104.563,
                       -106.267,
                       -108.076,
                       -110.0],
         'time': [1577869200,
                  1577869500,
                  1577869800,
                  1577870100,
                  1577870400,
                  1577870700,
                  1577871000]}
        """
        np_encoder = json_utils.NumpyEncoder()

        # round latitude, longitude, and altitude
        precision = {"longitude": 3, "latitude": 3, "altitude_ft": 0}

        def encode(key: str, obj: Any) -> Any:
            # Try to handle some pandas objects
            if hasattr(obj, "to_numpy"):
                obj = obj.to_numpy()

            # Convert numpy objects to python objects
            if isinstance(obj, np.ndarray | np.generic):
                # round time to unix seconds
                if key == "time":
                    return np_encoder.default(obj.astype("datetime64[s]").astype(int))

                # round specific keys in precision
                try:
                    d = precision[key]
                except KeyError:
                    return np_encoder.default(obj)

                return np_encoder.default(obj.astype(float).round(d))

            # Pass through everything else
            return obj

        data = {k: encode(k, v) for k, v in self.data.items()}
        attrs = {k: encode(k, v) for k, v in self.attrs.items()}

        # Only include one of the vertical coordinate keys
        if isinstance(self, GeoVectorDataset):
            data.pop("altitude", None)
            data.pop("level", None)
            if "altitude_ft" not in data:
                data["altitude_ft"] = self.altitude_ft.round(precision["altitude_ft"]).tolist()

        # Issue warning if any keys are duplicated
        common_keys = data.keys() & attrs.keys()
        if common_keys:
            warnings.warn(
                f"Found duplicate keys in data and attrs: {common_keys}. "
                "Data keys will overwrite attrs keys in returned dictionary."
            )

        return {**attrs, **data}

    @classmethod
    def create_empty(
        cls,
        keys: Iterable[str],
        attrs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create instance with variables defined by ``keys`` and size 0.

        If instance requires additional variables to be defined, these keys will automatically
        be attached to returned instance.

        Parameters
        ----------
        keys : Iterable[str]
            Keys to include in empty VectorDataset instance.
        attrs : dict[str, Any] | None, optional
            Attributes to attach instance.
        **kwargs : Any
            Additional keyword arguments passed into the constructor of the returned class.

        Returns
        -------
        Self
            Empty VectorDataset instance.
        """
        data = _empty_vector_dict(keys)
        return cls._from_fastpath(data, attrs, **kwargs)

    @classmethod
    def from_dict(cls, obj: dict[str, Any], copy: bool = True, **obj_kwargs: Any) -> Self:
        """Create instance from dict representation containing data and attrs.

        Parameters
        ----------
        obj : dict[str, Any]
            Dict representation of VectorDataset (e.g. :meth:`to_dict`)
        copy : bool, optional
            Passed to :class:`VectorDataset` constructor.
            Defaults to True.
        **obj_kwargs : Any
            Additional properties passed as keyword arguments.

        Returns
        -------
        Self
            VectorDataset instance.

        See Also
        --------
        :meth:`to_dict`
        """
        data = {}
        attrs = {}

        for k, v in {**obj, **obj_kwargs}.items():
            if isinstance(v, list | np.ndarray):
                data[k] = v
            else:
                attrs[k] = v

        return cls(data=data, attrs=attrs, copy=copy)

    def generate_splits(self, n_splits: int, copy: bool = True) -> Generator[Self, None, None]:
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
        Generator[Self, None, None]
            Generator of split vectors.

        See Also
        --------
        :func:`numpy.array_split`
        """
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

    Parameters
    ----------
    data : dict[str, npt.ArrayLike] | pd.DataFrame | VectorDataset | None, optional
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
    altitude_ft : npt.ArrayLike, optional
        Altitude data, [:math:`ft`].
        Defaults to None.
    level : npt.ArrayLike, optional
        Level data, [:math:`hPa`].
        Defaults to None.
    time : npt.ArrayLike, optional
        Time data.
        Expects an array of DatetimeLike values,
        or array that can processed with :meth:`pd.to_datetime`.
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

    __slots__ = ()

    #: Required keys for creating GeoVectorDataset
    required_keys = "longitude", "latitude", "time"

    #: At least one of these vertical-coordinate keys must also be included
    vertical_keys = "altitude", "level", "altitude_ft"

    def __init__(
        self,
        data: dict[str, npt.ArrayLike] | pd.DataFrame | VectorDataset | None = None,
        *,
        longitude: npt.ArrayLike | None = None,
        latitude: npt.ArrayLike | None = None,
        altitude: npt.ArrayLike | None = None,
        altitude_ft: npt.ArrayLike | None = None,
        level: npt.ArrayLike | None = None,
        time: npt.ArrayLike | None = None,
        attrs: dict[str, Any] | None = None,
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
            keys = *self.required_keys, "altitude"
            self.data = VectorDataDict(_empty_vector_dict(keys))
            self.attrs = AttrDict(attrs or {})  # type: ignore[arg-type]
            self.attrs.update(attrs_kwargs)
            return

        super().__init__(data=data, attrs=attrs, copy=copy, **attrs_kwargs)

        # using the self[key] syntax specifically to run qc on assignment
        if longitude is not None:
            self["longitude"] = np.array(longitude, copy=copy)

        if latitude is not None:
            self["latitude"] = np.array(latitude, copy=copy)

        if time is not None:
            self["time"] = np.array(time, copy=copy)

        if altitude is not None:
            self["altitude"] = np.array(altitude, copy=copy)
            if altitude_ft is not None or level is not None:
                warnings.warn(
                    "Altitude data provided. Ignoring altitude_ft and level inputs.",
                )
        elif altitude_ft is not None:
            self["altitude_ft"] = np.array(altitude_ft, copy=copy)
            if level is not None:
                warnings.warn(
                    "Altitude_ft data provided. Ignoring level input.",
                )
        elif level is not None:
            self["level"] = np.array(level, copy=copy)

        # Confirm that input has required keys
        if not all(key in self for key in self.required_keys):
            raise KeyError(
                f"{self.__class__.__name__} requires all of the following keys: "
                f"{', '.join(self.required_keys)}"
            )

        # Confirm that input has at least one vertical key
        if not any(key in self for key in self.vertical_keys):
            raise KeyError(
                f"{self.__class__.__name__} requires at least one of the following keys: "
                f"{', '.join(self.vertical_keys)}"
            )

        # Parse time: If time is not np.datetime64, we try to coerce it to be
        # by pumping it through pd.to_datetime.
        time = self["time"]
        if not np.issubdtype(time.dtype, np.datetime64):
            warnings.warn("Time data is not np.datetime64. Attempting to coerce.")
            try:
                pd_time = _handle_time_column(pd.Series(self["time"]))
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

        longitude = self["longitude"]
        if np.any(longitude > 180.0) or np.any(longitude < -180.0):
            raise ValueError("EPSG:4326 longitude coordinates should lie between [-180, 180).")
        latitude = self["latitude"]
        if np.any(latitude > 90.0) or np.any(latitude < -90.0):
            raise ValueError("EPSG:4326 latitude coordinates should lie between [-90, 90].")

    @override
    def _display_attrs(self) -> dict[str, str]:
        try:
            time0 = pd.Timestamp(np.nanmin(self["time"]))
            time1 = pd.Timestamp(np.nanmax(self["time"]))
            lon0 = round(np.nanmin(self["longitude"]), 3)
            lon1 = round(np.nanmax(self["longitude"]), 3)
            lat0 = round(np.nanmin(self["latitude"]), 3)
            lat1 = round(np.nanmax(self["latitude"]), 3)
            alt0 = round(np.nanmin(self.altitude), 1)
            alt1 = round(np.nanmax(self.altitude), 1)

            attrs = {
                "time": f"[{time0}, {time1}]",
                "longitude": f"[{lon0}, {lon1}]",
                "latitude": f"[{lat0}, {lat1}]",
                "altitude": f"[{alt0}, {alt1}]",
            }
        except Exception:
            attrs = {}

        attrs.update(super()._display_attrs())
        return attrs

    @property
    def level(self) -> npt.NDArray[np.floating]:
        """Get pressure ``level`` values for points.

        Automatically calculates pressure level using :func:`units.m_to_pl` using ``altitude`` key.

        Note that if ``level`` key exists in :attr:`data`, the data at the ``level``
        key will be returned. This allows an override of the default calculation
        of pressure level from altitude.

        Returns
        -------
        npt.NDArray[np.floating]
            Point pressure level values, [:math:`hPa`]
        """
        try:
            return self["level"]
        except KeyError:
            return units.m_to_pl(self.altitude)

    @property
    def altitude(self) -> npt.NDArray[np.floating]:
        """Get altitude.

        Automatically calculates altitude using :func:`units.pl_to_m` using ``level`` key.

        Note that if ``altitude`` key exists in :attr:`data`, the data at the ``altitude``
        key will be returned. This allows an override of the default calculation of altitude
        from pressure level.

        Returns
        -------
        npt.NDArray[np.floating]
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
    def air_pressure(self) -> npt.NDArray[np.floating]:
        """Get ``air_pressure`` values for points.

        Returns
        -------
        npt.NDArray[np.floating]
            Point air pressure values, [:math:`Pa`]
        """
        try:
            return self["air_pressure"]
        except KeyError:
            return 100.0 * self.level

    @property
    def altitude_ft(self) -> npt.NDArray[np.floating]:
        """Get altitude in feet.

        Returns
        -------
        npt.NDArray[np.floating]
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
        for key in set(self).difference(self.required_keys):
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

    def transform_crs(self, crs: str) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Transform trajectory data from one coordinate reference system (CRS) to another.

        Parameters
        ----------
        crs : str
            Target CRS. Passed into to :class:`pyproj.Transformer`. The source CRS
            is assumed to be EPSG:4326.
        copy : bool, optional
            Copy data on transformation. Defaults to True.

        Returns
        -------
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
            New x and y coordinates in the target CRS.
        """
        try:
            import pyproj
        except ModuleNotFoundError as exc:
            dependencies.raise_module_not_found_error(
                name="GeoVectorDataset.transform_crs method",
                package_name="pyproj",
                module_not_found_error=exc,
                pycontrails_optional_package="pyproj",
            )

        crs_from = "EPSG:4326"
        transformer = pyproj.Transformer.from_crs(crs_from, crs, always_xy=True)
        return transformer.transform(self["longitude"], self["latitude"])

    def T_isa(self) -> npt.NDArray[np.floating]:
        """Calculate the ICAO standard atmosphere temperature at each point.

        Returns
        -------
        npt.NDArray[np.floating]
            ISA temperature, [:math:`K`]

        See Also
        --------
        :func:`pycontrails.physics.units.m_to_T_isa`
        """
        return units.m_to_T_isa(self.altitude)

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
        indexes = met.indexes

        lat_intersect = coordinates.intersect_domain(
            indexes["latitude"].to_numpy(),
            self["latitude"],
        )
        lon_intersect = coordinates.intersect_domain(
            indexes["longitude"].to_numpy(),
            self["longitude"],
        )
        level_intersect = coordinates.intersect_domain(
            indexes["level"].to_numpy(),
            self.level,
        )
        time_intersect = coordinates.intersect_domain(
            indexes["time"].to_numpy(),
            self["time"],
        )

        return lat_intersect & lon_intersect & level_intersect & time_intersect

    def intersect_met(
        self,
        mda: met_module.MetDataArray,
        *,
        longitude: npt.NDArray[np.floating] | None = None,
        latitude: npt.NDArray[np.floating] | None = None,
        level: npt.NDArray[np.floating] | None = None,
        time: npt.NDArray[np.datetime64] | None = None,
        use_indices: bool = False,
        **interp_kwargs: Any,
    ) -> npt.NDArray[np.floating]:
        """Intersect waypoints with MetDataArray.

        Parameters
        ----------
        mda : MetDataArray
            MetDataArray containing a meteorological variable at spatio-temporal coordinates.
        longitude : npt.NDArray[np.floating], optional
            Override existing coordinates for met interpolation
        latitude : npt.NDArray[np.floating], optional
            Override existing coordinates for met interpolation
        level : npt.NDArray[np.floating], optional
            Override existing coordinates for met interpolation
        time : npt.NDArray[np.datetime64], optional
            Override existing coordinates for met interpolation
        use_indices : bool, optional
            Experimental.
        **interp_kwargs : Any
            Additional keyword arguments to pass to :meth:`MetDataArray.intersect_met`.
            Examples include ``method``, ``bounds_error``, and ``fill_value``. If an error such as

            .. code-block:: python

                ValueError: One of the requested xi is out of bounds in dimension 2

            occurs, try calling this function with ``bounds_error=False``. In addition,
            setting ``fill_value=0.0`` will replace NaN values with 0.0.

        Returns
        -------
        npt.NDArray[np.floating]
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
        array([231.62969892, 230.72604651, 232.24318771, 231.88338483,
               231.06429438, 231.59073409, 231.65125393, 231.93064004,
               232.03344087, 231.65954432])

        >>> fl.intersect_met(met['air_temperature'], method='linear')
        array([225.77794552, 225.13908414, 226.231218  , 226.31831528,
               225.56102321, 225.81192149, 226.03192642, 226.22056121,
               226.03770174, 225.63226188])

        >>> # Interpolate and attach to `Flight` instance
        >>> for key in met:
        ...     fl[key] = fl.intersect_met(met[key])

        >>> # Show the final three columns of the dataframe
        >>> fl.dataframe.iloc[:, -3:].head()
                         time  air_temperature  specific_humidity
        0 2022-03-01 00:00:00       225.777946           0.000132
        1 2022-03-01 00:13:20       225.139084           0.000132
        2 2022-03-01 00:26:40       226.231218           0.000107
        3 2022-03-01 00:40:00       226.318315           0.000171
        4 2022-03-01 00:53:20       225.561022           0.000109

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
    ) -> met_module.MetDataset: ...

    @overload
    def downselect_met(
        self,
        met: met_module.MetDataArray,
        *,
        longitude_buffer: tuple[float, float] = ...,
        latitude_buffer: tuple[float, float] = ...,
        level_buffer: tuple[float, float] = ...,
        time_buffer: tuple[np.timedelta64, np.timedelta64] = ...,
    ) -> met_module.MetDataArray: ...

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
    ) -> met_module.MetDataType:
        """Downselect ``met`` to encompass a spatiotemporal region of the data.

        .. versionchanged:: 0.54.5

            Returned object is no longer copied.

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
        indexes = met.indexes
        lon_slice = coordinates.slice_domain(
            indexes["longitude"].to_numpy(),
            self["longitude"],
            buffer=longitude_buffer,
        )
        lat_slice = coordinates.slice_domain(
            indexes["latitude"].to_numpy(),
            self["latitude"],
            buffer=latitude_buffer,
        )
        time_slice = coordinates.slice_domain(
            indexes["time"].to_numpy(),
            self["time"],
            buffer=time_buffer,
        )

        # single level data have "level" == [-1]
        if met.is_single_level:
            level_slice = slice(None)
        else:
            level_slice = coordinates.slice_domain(
                indexes["level"].to_numpy(),
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
        return type(met)._from_fastpath(data)

    # ------------
    # I / O
    # ------------

    @classmethod
    @override
    def create_empty(
        cls,
        keys: Iterable[str] | None = None,
        attrs: dict[str, Any] | None = None,
        **attrs_kwargs: Any,
    ) -> Self:
        keys = *cls.required_keys, "altitude", *(keys or ())
        return super().create_empty(keys, attrs, **attrs_kwargs)

    def to_geojson_points(self) -> dict[str, Any]:
        """Return dataset as GeoJSON FeatureCollection of Points.

        Each Feature has a properties attribute that includes ``time`` and
        other data besides ``latitude``, ``longitude``, and ``altitude`` in :attr:`data`.

        Returns
        -------
        dict[str, Any]
            Python representation of GeoJSON FeatureCollection
        """
        return json_utils.dataframe_to_geojson_points(self.dataframe)

    # ------------
    # Vector to grid
    # ------------
    def to_lon_lat_grid(
        self,
        agg: dict[str, str],
        *,
        spatial_bbox: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
        spatial_grid_res: float = 0.5,
    ) -> xr.Dataset:
        """
        Convert vectors to a longitude-latitude grid.

        See Also
        --------
        vector_to_lon_lat_grid
        """
        return vector_to_lon_lat_grid(
            self, agg=agg, spatial_bbox=spatial_bbox, spatial_grid_res=spatial_grid_res
        )


def vector_to_lon_lat_grid(
    vector: GeoVectorDataset,
    agg: dict[str, str],
    *,
    spatial_bbox: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    spatial_grid_res: float = 0.5,
) -> xr.Dataset:
    r"""
    Convert vectors to a longitude-latitude grid.

    Parameters
    ----------
    vector: GeoVectorDataset
        Contains the longitude, latitude and variables for aggregation.
    agg: dict[str, str]
        Variable name and the function selected for aggregation,
        i.e. ``{"segment_length": "sum"}``.
    spatial_bbox: tuple[float, float, float, float]
        Spatial bounding box, ``(lon_min, lat_min, lon_max, lat_max)``, [:math:`\deg`].
        By default, the entire globe is used.
    spatial_grid_res: float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    xr.Dataset
        Aggregated variables in a longitude-latitude grid.

    Examples
    --------
    >>> rng = np.random.default_rng(234)
    >>> vector = GeoVectorDataset(
    ...     longitude=rng.uniform(-10, 10, 10000),
    ...     latitude=rng.uniform(-10, 10, 10000),
    ...     altitude=np.zeros(10000),
    ...     time=np.zeros(10000).astype("datetime64[ns]"),
    ... )
    >>> vector["foo"] = rng.uniform(0, 1, 10000)
    >>> ds = vector.to_lon_lat_grid({"foo": "sum"}, spatial_bbox=(-10, -10, 9.5, 9.5))
    >>> da = ds["foo"]
    >>> da.coords
    Coordinates:
      * longitude  (longitude) float64 320B -10.0 -9.5 -9.0 -8.5 ... 8.0 8.5 9.0 9.5
      * latitude   (latitude) float64 320B -10.0 -9.5 -9.0 -8.5 ... 8.0 8.5 9.0 9.5

    >>> da.values.round(2)
    array([[2.23, 0.67, 1.29, ..., 4.66, 3.91, 1.93],
           [4.1 , 3.84, 1.34, ..., 3.24, 1.71, 4.55],
           [0.78, 3.25, 2.33, ..., 3.78, 2.93, 2.33],
           ...,
           [1.97, 3.02, 1.84, ..., 2.37, 3.87, 2.09],
           [3.74, 1.6 , 4.01, ..., 4.6 , 4.27, 3.4 ],
           [2.97, 0.12, 1.33, ..., 3.54, 0.74, 2.59]], shape=(40, 40))

    >>> da.sum().item() == vector["foo"].sum()
    np.True_

    """
    df = vector.select(("longitude", "latitude", *agg), copy=False).dataframe

    # Create longitude and latitude coordinates
    assert spatial_grid_res > 0.01, "spatial_grid_res must be greater than 0.01"
    west, south, east, north = spatial_bbox
    lon_coords = np.arange(west, east + 0.01, spatial_grid_res)
    lat_coords = np.arange(south, north + 0.01, spatial_grid_res)
    shape = lon_coords.size, lat_coords.size

    # Convert vector to lon-lat grid
    idx_lon = np.searchsorted(lon_coords, df["longitude"]) - 1
    idx_lat = np.searchsorted(lat_coords, df["latitude"]) - 1

    df_agg = df.groupby([idx_lon, idx_lat]).agg(agg)
    index = df_agg.index.get_level_values(0), df_agg.index.get_level_values(1)

    out = xr.Dataset(coords={"longitude": lon_coords, "latitude": lat_coords})
    for name, col in df_agg.items():
        arr = np.zeros(shape, dtype=col.dtype)
        arr[index] = col
        out[name] = (("longitude", "latitude"), arr)

    return out


def _handle_time_column(time: pd.Series) -> pd.Series:
    """Ensure that pd.Series has compatible Timestamps.

    Parameters
    ----------
    time : pd.Series
        Pandas dataframe column labeled "time".

    Returns
    -------
    pd.Series
        Parsed pandas time series.

    Raises
    ------
    ValueError
        When time series can't be parsed, or is not timezone naive.
    """
    if not hasattr(time, "dt"):
        time = _parse_pandas_time(time)

    # Translate all times to UTC and then remove timezone.
    # If the time column contains a timezone, the call to `to_numpy`
    # will convert it to an array of object.
    # Note `.tz_convert(None)` automatically converts to UTC first.
    if time.dt.tz is not None:
        time = time.dt.tz_convert(None)

    return time


def _parse_pandas_time(time: pd.Series) -> pd.Series:
    """Parse pandas dataframe column labelled "time".

    Parameters
    ----------
    time : pd.Series
        Time series

    Returns
    -------
    pd.Series
        Parsed time series

    Raises
    ------
    ValueError
        When series values can't be inferred.
    """
    try:
        # If the time series is a string, try to convert it to a datetime
        if time.dtype == "O":
            return pd.to_datetime(time)

        # If the time is an int, try to parse it as unix time
        if np.issubdtype(time.dtype, np.integer):
            return _parse_unix_time(time)

    except ValueError as exc:
        msg = (
            "The 'time' field must hold datetime-like values. "
            'Try data["time"] = pd.to_datetime(data["time"], unit=...) '
            "with the appropriate unit."
        )
        raise ValueError(msg) from exc

    raise ValueError("Unsupported time format")


def _parse_unix_time(time: list[int] | npt.NDArray[np.int_] | pd.Series) -> pd.Series:
    """Parse array of int times as unix epoch timestamps.

    Attempts to parse the time in "s", "ms", "us", "ns"

    Parameters
    ----------
    time : list[int] | npt.NDArray[np.int_] | pd.Series
        Sequence of unix timestamps

    Returns
    -------
    pd.Series
        Series of timezone naive pandas Timestamps

    Raises
    ------
    ValueError
        When unable to parse time as unix epoch timestamp
    """
    units = "s", "ms", "us", "ns"
    for unit in units:
        try:
            out = pd.to_datetime(time, unit=unit, utc=True)
        except ValueError:
            continue

        # make timezone naive
        out = out.dt.tz_convert(None)

        # make sure time is reasonable
        if (pd.Timestamp("1980-01-01") <= out).all() and (out <= pd.Timestamp("2030-01-01")).all():
            return out

    raise ValueError(
        f"Unable to parse time parameter '{time}' as unix epoch timestamp between "
        "1980-01-01 and 2030-01-01"
    )
