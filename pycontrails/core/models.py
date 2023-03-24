"""Physical model data structures."""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, List, NoReturn, Sequence, Type, TypeVar, Union, overload

import numpy as np
import xarray as xr

from pycontrails.core.fleet import Fleet
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataArray, MetDataset, MetVariable, originates_from_ecmwf
from pycontrails.core.met_var import SpecificHumidity
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.utils.json import NumpyEncoder
from pycontrails.utils.types import type_guard

logger = logging.getLogger(__name__)

#: Model input source types
ModelInput = Union[MetDataset, GeoVectorDataset, Flight, Sequence[Flight], None]

#: Model output source types
ModelOutput = Union[MetDataArray, MetDataset, GeoVectorDataset, Flight, List[Flight], NoReturn]

#: Model attribute source types
SourceType = Union[MetDataset, GeoVectorDataset, Flight, Fleet]

_Source = TypeVar("_Source")

# ------------
# Model Params
# ------------


@dataclass
class ModelParams:
    """Class for constructing model parameters.

    Implementing classes must still use the @dataclass operator.
    """

    #: Copy input ``source`` data on eval
    copy_source: bool = True

    # -----------
    # Interpolate
    # -----------

    #: Interpolation method. Supported methods include "linear", "nearest", "slinear",
    #: "cubic", and "quintic". See :class:`scipy.interpolate.RegularGridInterpolator`
    #: for the description of each method. Not all methods are supported by all
    #: met grids. For example, the "cubic" method requires at least 4 points per
    #: dimension.
    interpolation_method: str = "linear"

    #: If True, points lying outside interpolation will raise an error
    interpolation_bounds_error: bool = False

    #: Used for outside interpolation value if :attr:`interpolation_bounds_error` is False
    interpolation_fill_value: float = np.nan

    #: Experimental. See :mod:`pycontrails.core.interpolation`.
    interpolation_localize: bool = False

    #: Experimental. See :mod:`pycontrails.core.interpolation`.
    interpolation_use_indices: bool = False

    # -----------
    # Meteorology
    # -----------

    #: Call :meth:`_verify_met` on model instantiation
    verify_met: bool = True

    #: Downselect input MetDataset to region around ``source``, if input.
    downselect_met: bool = True

    #: Met longitude buffer for input to :meth:`Flight.downselect_met(longitude_buffer=...)`,
    #: in WGS84 coordinates
    #: Only applies when :attr:`downselect_met` is True.
    met_longitude_buffer: tuple[float, float] = (0.0, 0.0)

    #: Met latitude buffer for input to :meth:`Flight.downselect_met(latitude_buffer=...)`,
    #: in WGS84 coordinates
    #: Only applies when :attr:`downselect_met` is True.
    met_latitude_buffer: tuple[float, float] = (0.0, 0.0)

    #: Met level buffer for input to :meth:`Flight.downselect_met(level_buffer=...)`,
    #: in [:math:`hPa`]
    #: Only applies when :attr:`downselect_met` is True.
    met_level_buffer: tuple[float, float] = (0.0, 0.0)

    #: Met time buffer for input to :meth:`Flight.downselect_met(time_buffer=...)`
    #: Only applies when :attr:`downselect_met` is True.
    met_time_buffer: tuple[np.timedelta64, np.timedelta64] = (
        np.timedelta64(0, "h"),
        np.timedelta64(0, "h"),
    )

    def as_dict(self) -> dict[str, Any]:
        """Convert object to dictionary.

        We use this method instead of  `dataclasses.asdict`
        to use a shallow/unrecursive copy.
        This will return values as Any instead of dict.

        Returns
        -------
        dict[str, Any]
            Dictionary version of self.
        """
        return dict((field.name, getattr(self, field.name)) for field in fields(self))


# ------
# Models
# ------


class Model(ABC):
    """Base class for physical models.

    Implementing classes must implement the :meth:`eval` method
    """

    __slots__ = ("params", "met", "source")

    #: Default model parameter dataclass
    default_params: Type[ModelParams] = ModelParams

    #: Instantiated model parameters, in dictionary form
    params: dict[str, Any]

    #: Data evaluated in model
    source: SourceType

    #: Meteorology data
    met: MetDataset | None

    #: Require meteorology is not None on __init__()
    met_required: bool = False

    #: Required meteorology variables
    #: Each element in the list is a MetVariable or a list[MetVariable].
    #: If a list[MetVariable], only one variable in the list is required by input :attr:`met`.
    met_variables: tuple[MetVariable | tuple[MetVariable, ...], ...]

    #: Minimal set of required parameters if processing already complete on met input.
    processed_met_variables: tuple[MetVariable, ...]

    #: Optional meteorology variables
    optional_met_variables: tuple[MetVariable | tuple[MetVariable, ...], ...]

    def __init__(
        self,
        met: MetDataset | None = None,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ) -> None:
        # Load base params, override default and user params
        self._load_params(params, **params_kwargs)

        # Do *not* copy met on input
        self.met = met

        # require met inputs
        if self.met_required:
            self.require_met()

        # verify met variables
        if self.params["verify_met"]:
            self._verify_met()

        # gut check - warn if humidity_scaling param is present for ECMWF met data
        if self.params.get("humidity_scaling", None) is None:
            if (
                self.met is not None
                and SpecificHumidity in getattr(self, "met_variables", [])
                and originates_from_ecmwf(self.met)
            ):
                warnings.warn(
                    "Met data appears to have originated from ECMWF and no humidity "
                    "scaling is enabled. For ECMWF data, consider using one of "
                    "'ConstantHumidityScaling', 'ExponentialBoostHumidityScaling', or "
                    "'ExponentialBoostLatitudeCorrectionHumidityScaling'. For example: \n"
                    ">>> from pycontrails.models import ConstantHumidityScaling\n"
                    f">>> {type(self).__name__}(met=met, ..., humidity_scaling=ConstantHumidityScaling(rhi_adj=0.99))"  # noqa: E501
                )

    def __repr__(self) -> str:
        params = getattr(self, "params", {})
        return f"{type(self).__name__} model\n\t{self.long_name}\n\tParams: {params}\n"

    @property
    @abstractmethod
    def name(self) -> str:
        """Get model name for use as a data key in xr.DataArray or Flight."""

    @property
    @abstractmethod
    def long_name(self) -> str:
        """Get long name descriptor, annotated on xr.DataArray outputs."""

    @property
    def hash(self) -> str:
        """Generate a unique hash for model instance.

        Returns
        -------
        str
            Unique hash for model instance (sha1)
        """
        params = json.dumps(self.params, sort_keys=True, cls=NumpyEncoder)
        _hash = self.name + params
        if self.met is not None:
            _hash += self.met.hash
        if hasattr(self, "source"):
            _hash += self.source.hash

        return hashlib.sha1(bytes(_hash, "utf-8")).hexdigest()

    def _verify_met(self) -> None:
        """Verify integrity of :attr:`met`.

        This method confirms that :attr:`met` contains each variable in
        :attr:`met_variables`. If this check fails, and :attr:`processed_met_variables`
        is defined, confirm :attr:`met` contains each variable there.

        Does not raise errors if :attr:`met` is None.

        Raises
        ------
        KeyError
            Raises KeyError if data does not contain variables :attr:`met_variables`
        """
        if self.met is not None:
            if hasattr(self, "met_variables"):
                # Try to verify met_variables
                try:
                    self.met.ensure_vars(self.met_variables)
                except KeyError as e1:
                    # If that fails, try to verify processed_met_variables
                    if hasattr(self, "processed_met_variables"):
                        try:
                            self.met.ensure_vars(self.processed_met_variables)
                        except KeyError as e2:
                            raise e2 from e1
                    else:
                        raise e1

    def _load_params(self, params: dict[str, Any] | None = None, **params_kwargs: Any) -> None:
        """Load parameters to model :attr:`params`.

        Load order:

        1. :attr:`default_params` instantiated as dict
        2. ``params`` input dict
        3. ``params_kwargs`` override keys in params

        Parameters
        ----------
        params : dict[str, Any], optional
            Model parameter dictionary.
            Defaults to {}
        **params_kwargs : Any
            Override keys in ``params`` with keyword arguments.

        Raises
        ------
        KeyError
            Unknown parameter passed into model
        """
        self.params = self.default_params().as_dict()
        self.update_params(params, **params_kwargs)

    @abstractmethod
    def eval(self, source: Any = None, **params: Any) -> ModelOutput:
        """Abstract method to handle evaluation.

        Implementing classes should override call signature to overload ``source`` inputs
        and model outputs.

        Parameters
        ----------
        source : ModelInput, optional
            Dataset defining coordinates to evaluate model.
            Defined by implementing class, but must be a subset of ModelInput.
            If None, :attr:`met` is assumed to be evaluation points.
        **params : Any
            Overwrite model parameters before eval

        Returns
        -------
        ModelOutput
            Return type depends on implementing model
        """

    # ---------
    # Utilities
    # ---------

    @property
    def interp_kwargs(self) -> dict[str, Any]:
        """Shortcut to create interpolation arguments from :attr:`params`.

        Creates the required dict format for ``RegularGridInterpolator``
        attached to ``MetDataArray``.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys "method", "bounds_error", and "fill_value" as determined by
            :attr:`params`.
        """
        return {
            "method": self.params["interpolation_method"],
            "bounds_error": self.params["interpolation_bounds_error"],
            "fill_value": self.params["interpolation_fill_value"],
            "localize": self.params["interpolation_localize"],
            "use_indices": self.params["interpolation_use_indices"],
        }

    def require_met(self) -> MetDataset:
        """Ensure that :attr:`met` is a MetDataset.

        Returns
        -------
        MetDataset
            Returns reference to :attr:`met`.
            This is helpful for type narrowing :attr:`met` when meteorology is required.

        Raises
        ------
        ValueError
            Raises when :attr:`met` is None.
        """
        return type_guard(
            self.met,
            MetDataset,
            f"Meteorology is required for this model. Specify with {type(self).__name__}(met=...) ",
        )

    def require_source_type(self, type_: Type[_Source] | tuple[Type[_Source], ...]) -> _Source:
        """Ensure that :attr:`source` is ``type_``.

        Returns
        -------
        _Source
            Returns reference to :attr:`source`.
            This is helpful for type narrowing :attr:`source` to specific type(s).

        Raises
        ------
        ValueError
            Raises when :attr:`source` is not ``_type_``.
        """
        return type_guard(getattr(self, "source", None), type_, f"Source must be of type {type_}")

    @overload
    def _get_source(self, source: MetDataset | None) -> MetDataset:
        ...

    @overload
    def _get_source(self, source: GeoVectorDataset) -> GeoVectorDataset:
        ...

    @overload
    def _get_source(self, source: Sequence[Flight]) -> Fleet:
        ...

    def _get_source(self, source: ModelInput) -> SourceType:
        # Turn Fleet into Flight
        if isinstance(source, Sequence):
            # TODO: fix type guard here
            return Fleet.from_seq(source, copy=self.params["copy_source"])

        if isinstance(source, GeoVectorDataset):
            if self.params["copy_source"]:
                source = source.copy()

            if isinstance(source, Flight):
                if "flight_id" not in source:
                    # if flight_id in attrs, broadcast
                    if "flight_id" in source.attrs:
                        source.broadcast_attrs("flight_id")

                    # if no flight_id exists, add one as 0
                    else:
                        warnings.warn(
                            "Source flight does not contain `flight_id` data or attr. "
                            "Adding `flight_id` of 0"
                        )
                        source["flight_id"] = np.zeros(len(source), dtype=int)

            return source

        if isinstance(source, (MetDataset, GeoVectorDataset)):
            if self.params["copy_source"]:
                source = source.copy()

            return source

        # fall back on met data
        if source is None:
            self.met = self.require_met()

            # create a dataset with the same coords as self.met, but empty data_vars
            return MetDataset(xr.Dataset(coords=self.met.data.coords))

        raise TypeError(f"Unable to handle input eval data {source}")

    def set_source(self, source: ModelInput = None) -> None:
        """Attach original or copy of input ``source`` to :attr:`source`.

        Parameters
        ----------
        source : MetDataset | GeoVectorDataset | Flight | Iterable[Flight] | None
            Parameter ``source`` passed in :meth:`eval`.
            If None, an empty MetDataset with coordinates like :attr:`met` is set to :attr:`source`.

        See Also
        --------
        - :meth:`eval`
        """
        self.source = self._get_source(source)

    def update_params(self, params: dict[str, Any] | None = None, **params_kwargs: Any) -> None:
        """Update model parameters on :attr:`params`.

        Parameters
        ----------
        params : dict[str, Any], optional
            Model parameters to update, as dictionary.
            Defaults to {}
        **params_kwargs : Any
            Override keys in ``params`` with keyword arguments.
        """
        update_param_dict(self.params, params or {})
        update_param_dict(self.params, params_kwargs)

    def downselect_met(self) -> None:
        """Downselect :attr:`met` domain to the max/min bounds of :attr:`source`.

        Override this method if special handling is needed in met down-selection.

        - :attr:`source` must be defined before calling :meth:`downselect_met`.
        - This method copies and re-assigns :attr:`met` using :meth:`met.copy()`
        to avoid side-effects.

        Raises
        ------
        ValueError
            Raised if :attr:`source` is not defined.
            Raised if :attr:`source` is not  GeoVectorDataset.
        TypeError
            Raised if :attr:`met` is not a MetDataset.
        """
        if not hasattr(self, "source"):
            raise ValueError("Attribute `source` must be defined before calling `downselect_met`")

        # TODO: This could be generalized for a MetDataset source
        if not isinstance(self.source, GeoVectorDataset):
            raise ValueError("Attribute `source` must be a GeoVectorDataset")

        if self.met is None:
            return

        # return if downselect_met is False
        if not self.params["downselect_met"]:
            logger.debug("Avoiding downselecting met because params['downselect_met'] is False")
            return

        logger.debug("Downselecting met in model %s", self.name)

        # get buffers from params
        buffers = {
            "longitude_buffer": self.params.get("met_longitude_buffer"),
            "latitude_buffer": self.params.get("met_latitude_buffer"),
            "level_buffer": self.params.get("met_level_buffer"),
            "time_buffer": self.params.get("met_time_buffer"),
        }
        kwargs = {k: v for k, v in buffers.items() if v is not None}

        # `downselect_met(met=...)` copies `met`
        self.met = self.source.downselect_met(self.met, **kwargs)

    def set_source_met(
        self,
        optional: bool = False,
        variable: MetVariable | Sequence[MetVariable] | None = None,
    ) -> None:
        """Ensure or interpolate each required :attr:`met_variables` on :attr:`source` .

        For each variable in :attr:`met_variables`, check :attr:`source` for data variable
        with the same name.

        For :class:`GeoVectorDataset` sources, try to interpolate :attr:`met`
        if variable does not exist.

        For :class:`MetDataset` sources, try to get data from :attr:`met`
        if variable does not exist.

        Parameters
        ----------
        optional : bool, optional
            Include :attr:`optional_met_variables`
        variable : MetVariable | Sequence[MetVariable] | None, optional
            MetVariable to set, from :attr:`met_variables`.
            If None, set all variables in :attr:`met_variables`
            and :attr:`optional_met_variables` if ``optional`` is True.

        Raises
        ------
        ValueError
            Variable does not exist and :attr:`source` is a MetDataset.
        KeyError
            Variable not found in :attr:`source` or :attr:`met`.
        """
        variables: Sequence[MetVariable | tuple[MetVariable, ...]]
        if variable is None:
            if optional:
                variables = [*self.met_variables, *self.optional_met_variables]
            else:
                variables = list(self.met_variables)
        elif isinstance(variable, MetVariable):
            variables = [variable]
        else:
            variables = variable

        for var in variables:
            # If var is a tuple of options, check if at least one of them exists in source
            if isinstance(var, tuple):
                for v in var:
                    if v.standard_name in self.source:
                        continue

            # Check if var exists in source
            elif var.standard_name in self.source:
                continue

            # Otherwise, interpolate / set from met
            if not isinstance(self.met, MetDataset):
                _raise_missing_met_var(var)

            # take the first var name output from ensure_vars
            met_key = self.met.ensure_vars(var)[0]

            # interpolate GeoVectorDataset
            if isinstance(self.source, GeoVectorDataset):
                interpolate_met(self.met, self.source, met_key, **self.interp_kwargs)
                continue

            # set MetDataset
            if not isinstance(self.source, MetDataset):
                raise TypeError(f"Unknown source type: {type(self.source)}")

            da = self.met.data[met_key].reset_coords(drop=True)
            try:
                # This case is when self.source is a subgrid of self.met
                # The call to .sel will raise a KeyError if this is not the case
                self.source[met_key] = da.sel(self.source.coords)

            except KeyError:
                # This call to DataArray.interp was added in pycontrails 0.28.1
                # For arbitrary grids, use xr.DataArray.interp
                # Extract certain parameters to pass into interp
                kwargs = {
                    "method": self.params["interpolation_method"],
                    "kwargs": {
                        "bounds_error": self.params["interpolation_bounds_error"],
                        "fill_value": self.params["interpolation_fill_value"],
                    },
                    "assume_sorted": True,
                }
                # Correct dtype if promoted
                # Somewhat of a pain: dask believes the dtype is float32, but
                # when it is actually computed, it comes out as float64
                # Call load() here to smooth over this issue
                # https://github.com/pydata/xarray/issues/4770
                # There is also an issue in which xarray assumes non-singleton
                # dimensions. This causes issues when the ``da`` variable has
                # a scalar dimension, or the ``self.source`` variable coincides
                # with an edge of the ``da`` variable. For now, we try an additional
                # sel over just the time dimension, which is the most common case.
                # This stuff isn't so well unit tested in pycontrails, and the xarray
                # and scipy interpolate conventions are always changing, so more
                # issues may arise here in the future.
                coords = self.source.coords
                try:
                    da = da.sel(time=coords["time"])
                except KeyError:
                    pass
                else:
                    del coords["time"]

                interped = da.interp(coords, **kwargs).load()
                interped = interped.astype(da.dtype, copy=False)
                self.source[met_key] = interped

    # Following python implementation
    # https://github.com/python/cpython/blob/618b7a8260bb40290d6551f24885931077309590/Lib/collections/__init__.py#L231
    __marker = object()

    def get_source_param(self, key: str, default: Any = __marker) -> Any:
        """Get source data with default set by parameter key.

        Retrieves data with the following hierarchy:

        1. :attr:`source.data[key]`. Returns ``np.ndarray | xr.DataArray``.
        2. :attr:`source.attrs[key]`
        3. :attr:`params[key]`
        4. ``default``

        In case 3., the value of :attr:`params[key]` is attached to :attr:`source.attrs[key]`.

        Parameters
        ----------
        key : str
            Key to retrieve
        default : Any
            Default value if key is not found.

        Returns
        -------
        Any
            Value(s) found for key in source data, source attrs, or model params

        Raises
        ------
        KeyError
            Raises KeyError if key is not found in any location and ``default`` is not provided.
        """
        if key in self.source.data:
            return self.source.data[key]

        if key in self.source.attrs:
            return self.source.attrs[key]

        if key in self.params:
            out = self.params[key]

            # Set parameter to source attr for better post model evaluation tracking
            self.source.attrs[key] = out

            return out

        if default is not self.__marker:
            return default

        raise KeyError(f"key {key} not found in source data, attrs, or model params")

    def _cleanup_indices(self) -> None:
        """Cleanup indices artifacts if ``params["interpolation_use_indices"]`` is True."""
        if self.params["interpolation_use_indices"] and isinstance(self.source, GeoVectorDataset):
            self.source._invalidate_indices()


def _raise_missing_met_var(var: MetVariable | Sequence[MetVariable]) -> NoReturn:
    """Raise KeyError on missing met variable.

    Parameters
    ----------
    var : MetVariable | list[MetVariable]
        Met variable

    Raises
    ------
    KeyError
    """
    if isinstance(var, MetVariable):
        raise KeyError(
            f"Variable `{var.standard_name}` not found. Either pass parameter `met`"
            f"in model constructor, or define `{var.standard_name}` data on input data."
        )
    missing_keys = [v.standard_name for v in var]
    raise KeyError(
        f"One of `{missing_keys}` is required. Either pass parameter `met`"
        f"in model constructor, or define one of `{missing_keys}` data on input data."
    )


def interpolate_met(
    met: MetDataset | None,
    vector: GeoVectorDataset,
    met_key: str,
    vector_key: str | None = None,
    **interp_kwargs: Any,
) -> np.ndarray:
    """Interpolate ``vector`` against ``met`` gridded data.

    If ``vector_key`` (=``met_key`` by default) already exists,
    then break and return values at ``vector_key``.

    Mutates parameter ``vector`` in place by attaching new key
    and returns values.

    Parameters
    ----------
    met : MetDataset | None
        Met data to interpolate against
    vector : GeoVectorDataset
        Flight or GeoVectorDataset instance
    met_key : str
        Key of met variable in ``met``.
    vector_key : str, optional
        Key of variable to attach to ``vector``.
        By default, use ``met_key``.
    **interp_kwargs : Any,
        Additional keyword only arguments passed to `intersect_met`. For example,
        `level=np.array([...])`.

    Returns
    -------
    np.ndarray
        Interpolated values.

    Raises
    ------
    KeyError
        Parameter ``met_key`` not found in ``met``.
    """
    if vector_key is None:
        vector_key = met_key

    if (out := vector.get(vector_key, None)) is not None:
        return out

    if met is None:
        raise ValueError(f"No variable key {vector_key} in ``vector`` and ``met`` is None")

    try:
        mda = met[met_key]
    except KeyError as exc:
        raise KeyError(f"No variable key {met_key} in ``met``") from exc

    out = vector.intersect_met(mda, **interp_kwargs)
    vector[vector_key] = out
    return out


def update_param_dict(param_dict: dict[str, Any], new_params: dict[str, Any]) -> None:
    """Update parameter dictionary in place.

    Parameters
    ----------
    param_dict : dict[str, Any]
        Active model parameter dictionary
    new_params : dict[str, Any]
        Model parameters to update, as a dictionary

    Raises
    ------
    KeyError
        Raises when ``new_params`` key is not found in ``param_dict``

    """
    for param, value in new_params.items():
        if param not in param_dict:
            raise KeyError(
                f"Unknown parameter '{param}' passed into model. Possible "
                f"parameters include {', '.join(param_dict)}."
            )
        param_dict[param] = value
