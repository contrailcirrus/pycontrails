"""Public API."""

from __future__ import annotations

import logging
from importlib import metadata

import dask

# Work around for https://github.com/pydata/xarray/issues/7259
# Only occurs for xarray 2022.11 and above
try:
    import netCDF4
except ModuleNotFoundError:
    pass

from pycontrails.core.cache import DiskCacheStore, GCPCacheStore
from pycontrails.core.datalib import MetDataSource
from pycontrails.core.fleet import Fleet
from pycontrails.core.flight import Aircraft, Flight
from pycontrails.core.fuel import Fuel, HydrogenFuel, JetA, SAFBlend
from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.core.met_var import MetVariable
from pycontrails.core.models import Model, ModelParams
from pycontrails.core.vector import GeoVectorDataset, VectorDataset

__version__ = metadata.version("pycontrails")

log = logging.getLogger(__name__)
log.debug("Init pycontrails")

# Hardcode the dask warning silence config
dask.config.set({"array.slicing.split_large_chunks": False})


__all__ = [
    "Aircraft",
    "DiskCacheStore",
    "Fleet",
    "Flight",
    "Fuel",
    "GCPCacheStore",
    "GeoVectorDataset",
    "HydrogenFuel",
    "JetA",
    "MetDataArray",
    "MetDataset",
    "MetDataSource",
    "MetVariable",
    "Model",
    "ModelParams",
    "SAFBlend",
    "VectorDataset",
]
