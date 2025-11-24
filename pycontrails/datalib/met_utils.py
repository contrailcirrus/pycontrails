"""Tooling and support for meteorology data."""

import warnings

import dask.array
import numpy as np
import numpy.typing as npt
import xarray as xr

from pycontrails.utils import array


def _interp_artifacts(
    xp: npt.NDArray[np.floating], x: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """Compute the indices and distances for linear interpolation."""
    idx = array.searchsorted2d(xp, x)
    out_of_bounds = (idx == 0) | (idx == xp.shape[1])
    idx.clip(1, xp.shape[1] - 1, out=idx)

    x0 = np.take_along_axis(xp, idx - 1, axis=1)
    x1 = np.take_along_axis(xp, idx, axis=1)
    dist = (x.reshape(1, -1) - x0) / (x1 - x0)

    return idx, dist, out_of_bounds


def _interp_on_chunk(ds_chunk: xr.Dataset, target_pl: npt.NDArray[np.floating]) -> xr.Dataset:
    """Interpolate the data on a chunk to the target pressure levels.

    Parameters
    ----------
    ds_chunk : xr.Dataset
        Chunk of the dataset. The last dimension must be "model_level".
        The dataset from which ``ds_chunk`` is taken must not split the
        "model_level" dimension across chunks.
    target_pl : npt.NDArray[np.floating]
        Target pressure levels, [:math:`hPa`].

    Returns
    -------
    xr.Dataset
        Interpolated data on the target pressure levels. This has the same
        dimensions as ``ds_chunk`` except that the "model_level" dimension
        is replaced with "level". The shape of the "level" dimension is
        the length of ``target_pl``.
    """
    if any(da_chunk.dims[-1] != "model_level" for da_chunk in ds_chunk.values()):
        msg = "The last dimension of the dataset must be 'model_level'"
        raise ValueError(msg)

    pl_chunk = ds_chunk["pressure_level"]

    # Put the model_level column in the second dimension
    # And stack the horizontal dimensions into the first dimension
    xp = pl_chunk.values.reshape(-1, len(pl_chunk["model_level"]))

    # AFAICT, metview performs linear interpolation in xp and target_pl by default
    # However, the conversion_from_ml_to_pl.py script in https://confluence.ecmwf.int/x/JJh0CQ
    # suggests interpolating in the log space. If using consecutive model levels,
    # the difference between the two methods is negligible. We use the log space
    # method here for consistency with the ECMWF script. This only changes
    # the `dist` calculation below.
    idx, dist, out_of_bounds = _interp_artifacts(np.log(xp), np.log(target_pl))

    shape4d = pl_chunk.shape[:-1] + target_pl.shape
    idx = idx.reshape(shape4d)
    dist = dist.reshape(shape4d)
    out_of_bounds = out_of_bounds.reshape(shape4d)

    interped_dict = {}

    for name, da in ds_chunk.items():
        if name == "pressure_level":
            continue

        fp = da.values
        f0 = np.take_along_axis(fp, idx - 1, axis=-1)
        f1 = np.take_along_axis(fp, idx, axis=-1)
        interped = f0 + dist * (f1 - f0)
        interped[out_of_bounds] = np.nan  # we could extrapolate here like RGI(..., fill_value=None)

        coords = {k: da.coords[k] for k in da.dims[:-1]}
        coords["level"] = target_pl

        interped_dict[name] = xr.DataArray(
            interped,
            dims=tuple(coords),
            coords=coords,
            attrs=da.attrs,
        )

    return xr.Dataset(interped_dict)


def _build_template(ds: xr.Dataset, target_pl: npt.NDArray[np.floating]) -> xr.Dataset:
    """Build the template dataset for the interpolated data."""
    coords = {k: ds.coords[k] for k in ds.dims if k != "model_level"} | {"level": target_pl}

    dims = tuple(coords)
    shape = tuple(len(v) for v in coords.values())

    vars = {
        k: (dims, dask.array.empty(shape=shape, dtype=da.dtype))
        for k, da in ds.items()
        if k != "pressure_level"
    }

    chunks = {k: v for k, v in ds.chunks.items() if k != "model_level"}
    chunks["level"] = (len(target_pl),)

    return xr.Dataset(data_vars=vars, coords=coords, attrs=ds.attrs).chunk(chunks)


def ml_to_pl(
    ds: xr.Dataset,
    target_pl: npt.ArrayLike,
) -> xr.Dataset:
    r"""Interpolate model-level meteorology data to pressure levels.

    The implementation is here is consistent with ECMWF's
    `suggested implementation <https://confluence.ecmwf.int/x/JJh0CQ#heading-Step2Interpolatevariablesonmodellevelstocustompressurelevels>`_.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with model-level meteorology data. Must include a "model_level" dimension
        which is not split across chunks and a "pressure_level_ variable with pressure in hPa.
        Can include any number of other variables.
        Any `non-dimension coordinates <https://docs.xarray.dev/en/latest/user-guide/terminology.html#term-Non-dimension-coordinate>`_
        will be dropped.
    target_pl : npt.ArrayLike
        Target pressure levels, [:math:`hPa`]. Will be promoted to a 1D array if a scalar
        is provided and flattened to 1D if multidimensional array is provided.

    Returns
    -------
    xr.Dataset
        Interpolated data on the target pressure levels. This has the same
        dimensions as ``ds`` except that the "model_level" dimension
        is replaced with "level". The shape of the "level" dimension is
        the length of ``target_pl``. If ``ds`` is dask-backed, the output
        will be as well. Call ``.compute()`` to compute the result eagerly.

    """
    if np.isnan(target_pl).any():
        msg = "Target pressure levels must not contain NaN values."
        raise ValueError(msg)

    if "pressure_level" not in ds:
        msg = "The dataset must contain a 'pressure_level' variable"
        raise ValueError(msg)

    ds = ds.reset_coords(drop=True)

    # If there are any variables which do not have the "model_level" dimension,
    # issue a warning and drop them
    for name, da in ds.items():
        if "model_level" not in da.dims:
            msg = f"Variable '{name}' does not have a 'model_level' dimension"
            warnings.warn(msg)
            ds = ds.drop_vars([name])

    if not ds.data_vars:
        msg = "Dataset has no variables with a 'model_level' dimension"
        raise ValueError(msg)

    # IMPORTANT: model_level must be the last dimension for _interp_on_chunk
    ds = ds.transpose(..., "model_level")

    # Raise if chunks over model level
    if ds.chunks and len(ds.chunks["model_level"]) > 1:
        msg = "The 'model_level' dimension must not be split across chunks"
        raise ValueError(msg)

    target_pl = np.asarray(target_pl, dtype=ds["pressure_level"].dtype)
    target_pl = np.atleast_1d(target_pl).ravel()
    template = _build_template(ds, target_pl)

    return xr.map_blocks(_interp_on_chunk, ds, (target_pl,), template=template)
