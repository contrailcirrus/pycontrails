from __future__ import annotations

import os
from datetime import datetime
from typing import TypeVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import DiskCacheStore, MetDataset, MetVariable
from pycontrails.core.met_var import AirTemperature, SurfacePressure
from pycontrails.datalib.ecmwf import (
    ECMWF_VARIABLES,
    ERA5,
    HRES,
    MODEL_LEVELS_PATH,
    ERA5ModelLevel,
    HRESModelLevel,
    ml_to_pl,
    model_level_pressure,
    model_level_reference_pressure,
)
from pycontrails.datalib.ecmwf.hres import get_forecast_filename

AnyERA5DatalibClass = TypeVar("AnyERA5DatalibClass", type[ERA5], type[ERA5ModelLevel])
AnyHRESDatalibClass = TypeVar("AnyHRESDatalibClass", type[HRES], type[HRESModelLevel])
AnyModelLevelDatalibClass = TypeVar(
    "AnyModelLevelDatalibClass", type[ERA5ModelLevel], type[HRESModelLevel]
)
AnyECMWFDatalibClass = TypeVar("AnyECMWFDatalibClass", AnyERA5DatalibClass, AnyHRESDatalibClass)


def test_environ_keys() -> None:
    """Test CDS Keys."""
    assert os.environ["CDSAPI_URL"] == "FAKE"
    assert os.environ["CDSAPI_KEY"] == "FAKE"


def test_ECMWF_variables() -> None:
    """Test ECMWF_VARIABLES."""
    assert isinstance(ECMWF_VARIABLES[0], MetVariable)
    assert isinstance(AirTemperature, MetVariable)

    # confirm properties
    assert AirTemperature.ecmwf_id == 130
    assert AirTemperature.standard_name == "air_temperature"
    assert AirTemperature.short_name == "t"
    assert AirTemperature.units == "K"
    assert AirTemperature.ecmwf_link == "https://apps.ecmwf.int/codes/grib/param-db?id=130"


##############################
# All ECMWF datalibs
##############################


@pytest.mark.parametrize("datalib", [ERA5, HRES, ERA5ModelLevel, HRESModelLevel])
def test_single_time_input(datalib: AnyECMWFDatalibClass) -> None:
    """Test TimeInput parsing."""
    # accept single time
    dl = datalib(time=datetime(2019, 5, 31, 0), variables=["vo"], pressure_levels=[200])
    assert dl.timesteps == [datetime(2019, 5, 31, 0)]
    dl = datalib(time=[datetime(2019, 5, 31, 0)], variables=["vo"], pressure_levels=[200])
    assert dl.timesteps == [datetime(2019, 5, 31, 0)]

    # accept single time with minutes defined
    dl = datalib(time=datetime(2019, 5, 31, 0, 29), variables=["vo"], pressure_levels=[200])
    assert dl.timesteps == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]
    dl = datalib(time=[datetime(2019, 5, 31, 0, 29)], variables=["vo"], pressure_levels=[200])
    assert dl.timesteps == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]


@pytest.mark.parametrize("datalib", [ERA5, HRES, ERA5ModelLevel, HRESModelLevel])
def test_time_input_wrong_length(datalib: AnyECMWFDatalibClass) -> None:
    """Test TimeInput parsing."""

    # throw ValueError for length == 0
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        datalib(time=[], variables=["vo"], pressure_levels=[200])

    # throw ValueError for length > 2
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        datalib(
            time=[datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 0)],
            variables=["vo"],
            pressure_levels=[200],
        )


@pytest.mark.parametrize("datalib", [ERA5, HRES, ERA5ModelLevel, HRESModelLevel])
def test_time_input_two_times_nominal(datalib: AnyECMWFDatalibClass) -> None:
    """Test TimeInput parsing for datalibs with 1 hour default timestep."""

    # accept pair (start, end)
    dl = datalib(
        time=(datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 3)),
        variables=["vo"],
        pressure_levels=[200],
    )
    assert dl.timesteps == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 1),
        datetime(2019, 5, 31, 2),
        datetime(2019, 5, 31, 3),
    ]

    dl = ERA5(
        time=(datetime(2019, 5, 31, 0, 29), datetime(2019, 5, 31, 2, 40)),
        variables=["vo"],
        pressure_levels=[200],
    )
    assert dl.timesteps == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 1),
        datetime(2019, 5, 31, 2),
        datetime(2019, 5, 31, 3),
    ]


@pytest.mark.parametrize("datalib", [ERA5, HRES, ERA5ModelLevel, HRESModelLevel])
def test_time_input_numpy_pandas(datalib: AnyECMWFDatalibClass) -> None:
    """Test TimeInput parsing for alternate time input formats."""

    # support alternate types for input
    dl = datalib(
        time=pd.to_datetime(datetime(2019, 5, 31, 0, 29)),
        variables=["vo"],
        pressure_levels=[200],
    )
    assert dl.timesteps == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]
    dl = datalib(time=np.datetime64("2019-05-31T00:29:00"), variables=["vo"], pressure_levels=[200])
    assert dl.timesteps == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]
    dl = datalib(
        time=(np.datetime64("2019-05-31T00:29:00"), np.datetime64("2019-05-31T02:40:00")),
        variables=["vo"],
        pressure_levels=[200],
    )
    assert dl.timesteps == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 1),
        datetime(2019, 5, 31, 2),
        datetime(2019, 5, 31, 3),
    ]


@pytest.mark.parametrize("datalib", [ERA5, HRES, ERA5ModelLevel, HRESModelLevel])
def test_inputs(datalib: AnyECMWFDatalibClass) -> None:
    """Test datalib __init__ with different but equivalent inputs."""
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        datalib(time=[], variables=[], pressure_levels=[17, 18, 19])

    d1 = datalib(time=datetime(2000, 1, 1), variables="vo", pressure_levels=200)
    d2 = datalib(time=[datetime(2000, 1, 1)], variables=["vo"], pressure_levels=[200])
    d3 = datalib(
        time=np.datetime64("2000-01-01 00:00:00"),
        variables=["vo"],
        pressure_levels=[200],
    )
    d4 = datalib(
        time=np.array([np.datetime64("2000-01-01 00:00:00")]),
        variables=["vo"],
        pressure_levels=[200],
    )
    assert d1.variables == d2.variables == d3.variables == d4.variables
    assert d1.pressure_levels == d2.pressure_levels == d3.pressure_levels == d4.pressure_levels
    assert d1.timesteps == d2.timesteps == d3.timesteps == d4.timesteps


##########################
# All model level datalibs
##########################


@pytest.mark.parametrize("datalib", [ERA5ModelLevel, HRESModelLevel])
def test_model_level_retrieved_levels(datalib: AnyModelLevelDatalibClass) -> None:
    """Test model levels included in request."""
    with pytest.raises(ValueError, match="Retrieval model_levels must be between"):
        dl = datalib(time=datetime(2000, 1, 1), variables="vo", model_levels=[0, 1, 2])
    with pytest.raises(ValueError, match="Retrieval model_levels must be between"):
        dl = datalib(time=datetime(2000, 1, 1), variables="vo", model_levels=[136, 137, 138])

    dl = datalib(time=datetime(2000, 1, 1), variables="vo")
    assert dl.model_levels == list(range(1, 138))

    dl = datalib(time=datetime(2000, 1, 1), variables="vo", model_levels=[3, 4, 5])
    assert dl.model_levels == [3, 4, 5]


@pytest.mark.parametrize("datalib", [ERA5ModelLevel, HRESModelLevel])
def test_model_level_pressure_levels(datalib: AnyModelLevelDatalibClass) -> None:
    """Test pressure level inputs for model-level datalibs."""
    dl = datalib(time=datetime(2000, 1, 1), variables="vo")
    assert dl.pressure_levels == model_level_reference_pressure(20_000, 50_000)


@pytest.mark.parametrize("datalib", [ERA5ModelLevel, HRESModelLevel])
def test_model_level_single_level_variables(datalib: AnyModelLevelDatalibClass) -> None:
    """Test supported single-level variables."""
    dl = datalib(time=datetime(2000, 1, 1), variables="vo")
    assert dl.single_level_variables == []


@pytest.mark.parametrize("datalib", [ERA5ModelLevel, HRESModelLevel])
def test_model_level_open_metdataset_errors(
    datalib: AnyModelLevelDatalibClass, met_ecmwf_pl_path: str
) -> None:
    """Test open_metdataset error handing."""
    dl = datalib(time=(datetime(2000, 1, 1), datetime(2000, 1, 2)), variables=["t", "q"])
    ds = xr.open_dataset(met_ecmwf_pl_path)
    with pytest.raises(ValueError, match="Parameter 'dataset' is not supported"):
        dl.open_metdataset(dataset=ds)

    dl = datalib(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
    )
    dl.cachestore = None
    with pytest.raises(ValueError, match="Cachestore is required"):
        dl.open_metdataset()


##################################################
# ERA5 datalibs (pressure levels and model levels)
##################################################


@pytest.mark.parametrize("datalib", [ERA5, ERA5ModelLevel])
def test_time_input_two_times_era5_ensemble(datalib: AnyERA5DatalibClass) -> None:
    """Test TimeInput parsing for ERA5 ensemble datalibs (3 hour default timestep)."""

    # accept pair (start, end)
    dl = datalib(
        time=(datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 6)),
        variables=["vo"],
        pressure_levels=[200],
        product_type="ensemble_members",
    )
    assert dl.timesteps == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 3),
        datetime(2019, 5, 31, 6),
    ]

    dl = datalib(
        time=(datetime(2019, 5, 31, 0, 29), datetime(2019, 5, 31, 5, 40)),
        variables=["vo"],
        pressure_levels=[200],
        product_type="ensemble_members",
    )
    assert dl.timesteps == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 3),
        datetime(2019, 5, 31, 6),
    ]


@pytest.mark.parametrize("datalib", [ERA5, ERA5ModelLevel])
@pytest.mark.parametrize(
    ("product", "grid", "expected", "warn"),
    [
        ("reanalysis", None, 0.25, False),
        ("reanalysis", 0.1, 0.1, True),
        ("reanalysis", 1.0, 1.0, False),
        ("ensemble_members", None, 0.5, False),
        ("ensemble_members", 0.1, 0.1, True),
        ("ensemble_members", 1.0, 1.0, False),
    ],
)
def test_era5_grid(
    datalib: AnyERA5DatalibClass, product: str, grid: float, expected: float, warn: bool
) -> None:
    """Test horizontal resolution."""
    if warn:
        with pytest.warns(UserWarning, match="The highest resolution available"):
            dl = datalib(
                time=datetime(2000, 1, 1),
                variables="vo",
                pressure_levels=[200],
                product_type=product,
                grid=grid,
            )
    else:
        dl = datalib(
            time=datetime(2000, 1, 1),
            variables="vo",
            pressure_levels=[200],
            product_type=product,
            grid=grid,
        )
    assert dl.grid == expected


#############################
# ERA5 pressure level datalib
#############################


def test_ERA5_repr() -> None:
    """Test ERA5 __repr__."""
    era5 = ERA5(time=datetime(2000, 1, 1), variables="vo", pressure_levels=200)
    out = repr(era5)
    assert "ERA5" in out
    assert "Dataset:" in out


@pytest.mark.usefixtures("_dask_single_threaded")
def test_ERA5_cachestore(met_ecmwf_pl_path: str, override_cache: DiskCacheStore) -> None:
    """Test ERA5 cachestore input."""

    # clear cache to start
    override_cache.clear()

    era5 = ERA5(
        time=datetime(2000, 1, 1),
        variables="vo",
        pressure_levels=200,
        cachestore=override_cache,
    )
    cachepath = era5.create_cachepath(datetime(2000, 1, 1))
    assert "20000101-00-era5pl0.25reanalysis.nc" in cachepath

    # load actual data
    assert override_cache.size == 0
    times = [datetime(2019, 5, 31, 5, 0, 0)]
    pressure_levels = [300, 250]
    variables = ["air_temperature", "relative_humidity", "specific_humidity"]
    era5 = ERA5(
        time=times,
        variables=variables,
        pressure_levels=pressure_levels,
        paths=met_ecmwf_pl_path,
        cachestore=override_cache,
    )
    mds = era5.open_metdataset()
    assert isinstance(mds, MetDataset)
    assert override_cache.size > 0

    # allow cache to be None
    pre_init_size = DiskCacheStore().size
    era5 = ERA5(
        time=times,
        variables=variables,
        pressure_levels=pressure_levels,
        paths=met_ecmwf_pl_path,
        cachestore=None,
    )
    mds = era5.open_metdataset()
    assert isinstance(mds, MetDataset)
    post_init_size = DiskCacheStore().size
    assert pre_init_size == post_init_size


@pytest.mark.usefixtures("_dask_single_threaded")
def test_ERA5_pressure_levels(met_ecmwf_pl_path: str, override_cache: DiskCacheStore) -> None:
    """Test ERA5 pressure_level parsing."""
    times = [datetime(2019, 5, 31, 5, 0, 0)]
    pressure_levels = [300, 250]
    variables = ["air_temperature", "relative_humidity", "specific_humidity"]
    era5 = ERA5(time=times, variables=variables, pressure_levels=pressure_levels)

    # properties
    assert 300 in era5.pressure_levels
    assert AirTemperature in era5.pressure_level_variables
    assert AirTemperature in era5.supported_variables
    assert isinstance(era5._cachepaths[0], str)
    assert "2019" in era5._cachepaths[0]

    # load
    era5 = ERA5(
        time=times,
        variables=variables,
        pressure_levels=pressure_levels,
        paths=met_ecmwf_pl_path,
        cachestore=override_cache,
    )
    met = era5.open_metdataset()
    assert isinstance(met, MetDataset)

    # preprocess
    assert met.data.attrs["pycontrails_version"] == "0.34.0"
    assert 250 in met.data["level"]
    assert 200 not in met.data["level"]
    assert met.data.longitude.min().values == -160

    # Consistent level and time with input
    np.testing.assert_array_equal(met.data["level"], sorted(pressure_levels))
    np.testing.assert_array_equal(met.data["time"], np.array(era5.timesteps, dtype="datetime64"))


def test_ERA5_single_levels(met_ecmwf_sl_path: str, override_cache: DiskCacheStore) -> None:
    """Test ERA5 surface level parsing."""
    times = [datetime(2019, 5, 31, 5, 0, 0)]
    variables = ["surface_air_pressure"]
    era5 = ERA5(times, variables)

    # init
    assert era5.dataset == "reanalysis-era5-single-levels"
    assert era5.pressure_levels == [-1]

    # properties
    assert SurfacePressure in era5.single_level_variables
    assert SurfacePressure in era5.supported_variables
    assert isinstance(era5._cachepaths[0], str)
    assert "2019" in era5._cachepaths[0]

    # load
    era5 = ERA5(time=times, variables=variables, paths=met_ecmwf_sl_path, cachestore=override_cache)
    met = era5.open_metdataset()
    assert isinstance(met, MetDataset)

    # preprocess
    assert met.data.attrs["pycontrails_version"] == "0.34.0"
    assert "surface_air_pressure" in met.data
    assert -1 in met.data["level"]
    assert 200 not in met.data["level"]
    assert met.data.longitude.min().values == -160

    # consistency
    assert np.all(met.data["level"].values == era5.pressure_levels)
    # dealing with discrepancy between datetime and np.datetime64
    assert np.all(met.data["time"].values == np.array([np.datetime64(t) for t in era5.timesteps]))


def test_ERA5_radiation_processing(rad_cocip1: MetDataset) -> None:
    """Test ERA5 radiation processing, as displaying in cocip rad data."""

    assert rad_cocip1.data["top_net_thermal_radiation"].attrs.get("_pycontrails_modified", False)
    assert rad_cocip1.data["top_net_solar_radiation"].attrs.get("_pycontrails_modified", False)

    assert rad_cocip1.data["top_net_thermal_radiation"].attrs["units"] == "W m**-2"
    assert rad_cocip1.data["top_net_solar_radiation"].attrs["units"] == "W m**-2"


def test_ERA5_hash() -> None:
    """Test ERA5 hash."""

    era5 = ERA5(time=datetime(2022, 3, 5), variables=["t", "r"], pressure_levels=200)
    era52 = ERA5(time=datetime(2022, 3, 5), variables=["t", "r"], pressure_levels=[200, 300])
    assert era5.hash == "6939450efa6c348471dea359a0055f49d72922e6"
    assert era52.hash == "58af5be8d2826bbacee14aa471da6d51060b4f8c"
    assert era5.hash != era52.hash


@pytest.mark.usefixtures("_dask_single_threaded")
def test_ERA5_paths_with_time(
    met_ecmwf_pl_path: str,
    met_ecmwf_sl_path: str,
    override_cache: DiskCacheStore,
) -> None:
    """Test ERA5 paths input with time."""

    time = datetime(2019, 5, 31, 5, 0, 0)

    # these should both work without warning
    era5pl = ERA5(
        time=time,
        variables=["air_temperature"],
        pressure_levels=[300, 250],
        paths=met_ecmwf_pl_path,
        cachestore=override_cache,
    )
    metpl = era5pl.open_metdataset()
    assert metpl.data.attrs["pycontrails_version"] == "0.34.0"

    era5sl = ERA5(
        time=time,
        variables=["surface_air_pressure"],
        paths=met_ecmwf_sl_path,
        cachestore=override_cache,
    )
    metsl = era5sl.open_metdataset()
    assert metsl.data.attrs["pycontrails_version"] == "0.34.0"

    # these files should be getting cached by default
    assert override_cache.size > 0


@pytest.mark.usefixtures("_dask_single_threaded")
def test_ERA5_paths_without_time(
    met_ecmwf_pl_path: str,
    override_cache: DiskCacheStore,
) -> None:
    """Test ERA5 paths input with time=None."""

    era5pl = ERA5(
        time=None,
        variables=["air_temperature"],
        pressure_levels=[300, 250],
        paths=met_ecmwf_pl_path,
        cachestore=override_cache,
    )
    metpl = era5pl.open_metdataset()
    ds = xr.open_dataset(met_ecmwf_pl_path)  # open manually for comparison
    assert era5pl.timesteps
    assert len(era5pl.timesteps) == len(ds["time"])
    assert (metpl.data["time"].values == ds["time"].values).all()

    # allow cache to be None
    pre_init_size = DiskCacheStore().size
    era5pl = ERA5(
        time=None,
        variables=["air_temperature"],
        pressure_levels=[300, 250],
        paths=met_ecmwf_pl_path,
        cachestore=None,
    )
    metpl = era5pl.open_metdataset()
    post_init_size = DiskCacheStore().size
    assert pre_init_size == post_init_size


@pytest.mark.usefixtures("_dask_single_threaded")
def test_ERA5_paths_with_error(
    met_ecmwf_pl_path: str,
    override_cache: DiskCacheStore,
) -> None:
    """Test ERA5 issues errors with paths input."""

    era5pl = ERA5(
        time="2019-06-02 00:00:00, ",
        variables=["air_temperature"],
        pressure_levels=[300, 250],
        paths=met_ecmwf_pl_path,
        cachestore=override_cache,
    )
    with pytest.raises(KeyError, match="2019-06-02T00:00:00"):
        era5pl.open_metdataset()

    time = datetime(2019, 5, 31, 5, 0, 0)

    # check variables
    era5pl = ERA5(
        time=time,
        variables=["z"],
        pressure_levels=[300, 250],
        paths=met_ecmwf_pl_path,
        cachestore=override_cache,
    )
    with pytest.raises(KeyError, match="z"):
        era5pl.open_metdataset()

    # check pressure levels
    era5pl = ERA5(
        time=time,
        variables=["air_temperature"],
        pressure_levels=[400, 300, 250],
        paths=met_ecmwf_pl_path,
        cachestore=override_cache,
    )
    with pytest.raises(KeyError, match="400"):
        era5pl.open_metdataset()


def test_ERA5_dataset(met_ecmwf_pl_path: str, met_ecmwf_sl_path: str) -> None:
    """Test ERA5 paths input."""

    dspl = xr.open_dataset(met_ecmwf_pl_path)
    times = datetime(2019, 5, 31, 5, 0, 0)

    # these should both work without warning
    era5pl = ERA5(
        time=times,
        variables=["air_temperature"],
        pressure_levels=[300, 250],
    )
    metpl = era5pl.open_metdataset(dataset=dspl)

    assert metpl.data.attrs["pycontrails_version"] == "0.34.0"
    assert 250 in metpl.data["level"].values
    assert 300 in metpl.data["level"].values

    dssl = xr.open_dataset(met_ecmwf_sl_path)
    era5sl = ERA5(time=times, variables=["surface_air_pressure"])
    metsl = era5sl.open_metdataset(dataset=dssl)
    assert metsl.data.attrs["pycontrails_version"] == "0.34.0"
    assert -1 in metsl.data["level"].values
    assert "surface_air_pressure" in metsl.data.data_vars

    # make sure there is sanity checking

    # check time
    era5pl = ERA5(
        time="2019-06-02 00:00:00, ",
        variables=["air_temperature"],
        pressure_levels=[300, 250],
    )
    with pytest.raises(KeyError, match="2019-06-02T00:00:00"):
        metpl = era5pl.open_metdataset(dataset=dspl)

    # check variables
    era5pl = ERA5(
        time=times,
        variables=["z"],
        pressure_levels=[300, 250],
    )
    with pytest.raises(KeyError, match="z"):
        metpl = era5pl.open_metdataset(dataset=dspl)

    # check pressure levels
    era5pl = ERA5(
        time=times,
        variables=["air_temperature"],
        pressure_levels=[400, 300, 250],
    )
    with pytest.raises(KeyError, match="400"):
        metpl = era5pl.open_metdataset(dataset=dspl)


@pytest.mark.parametrize("variables", ["t", "tsr"])
@pytest.mark.parametrize("product_type", ["reanalysis", "ensemble_mean", "ensemble_members"])
def test_ERA5_set_met_source_metadata(product_type: str, variables: str) -> None:
    """Test ERA5.set_met_source_metadata method."""

    era5 = ERA5(
        time=datetime(2000, 1, 1),
        variables=variables,
        pressure_levels=200 if variables == "t" else -1,
        product_type=product_type,
    )

    ds = xr.Dataset()
    era5.set_metadata(ds)

    assert ds.attrs["provider"] == "ECMWF"
    assert ds.attrs["dataset"] == "ERA5"
    assert ds.attrs["product"] == product_type.split("_")[0]


@pytest.mark.usefixtures("_dask_single_threaded")
def test_ERA5_met_source_open_metdataset(met_ecmwf_pl_path: str) -> None:
    """Test the met_source attribute on the MetDataset arising from ERA5."""
    era5 = ERA5(
        paths=met_ecmwf_pl_path,
        time=datetime(2019, 5, 31, 5),
        variables="t",
        pressure_levels=(300, 250, 225),
        cachestore=None,
    )
    mds = era5.open_metdataset()

    assert mds.provider_attr == "ECMWF"
    assert mds.dataset_attr == "ERA5"
    assert mds.product_attr == "reanalysis"


##########################
# ERA5 model-level datalib
##########################


def test_model_level_era5_repr() -> None:
    """Test model level ERA5 repr."""
    era5 = ERA5ModelLevel(
        time=datetime(2000, 1, 1),
        variables="vo",
    )
    out = repr(era5)
    assert "ERA5" in out
    assert "Dataset" in out
    assert "Product type" in out


def test_model_level_era5_product_type() -> None:
    """Test model level ERA5 product type validation."""
    with pytest.raises(ValueError, match="Unknown product_type"):
        ERA5ModelLevel(time=datetime(2000, 1, 1), variables="vo", product_type="foo")

    with pytest.raises(ValueError, match="No ensemble members available"):
        ERA5ModelLevel(
            time=datetime(2000, 1, 1),
            variables="vo",
            product_type="reanalysis",
            ensemble_members=[0, 1, 2],
        )


def test_model_level_era5_ensemble_member_selection() -> None:
    """Test model level ERA5 ensemble member selection."""
    era5 = ERA5ModelLevel(
        time=datetime(2000, 1, 1),
        variables="vo",
        product_type="ensemble_members",
    )
    assert era5.ensemble_members == list(range(10))

    era5 = ERA5ModelLevel(
        time=datetime(2000, 1, 1),
        variables="vo",
        product_type="ensemble_members",
        ensemble_members=[1, 3, 4],
    )
    assert era5.ensemble_members == [1, 3, 4]


@pytest.mark.parametrize(
    ("product", "timestep_freq", "raises"),
    [
        ("reanalysis", "1h", False),
        ("reanalysis", "4h", False),
        ("reanalysis", "30min", True),
        ("reanalysis", "90min", True),
        ("ensemble_members", "3h", False),
        ("ensemble_members", "12h", False),
        ("ensemble_members", "1h", True),
        ("ensemble_members", "4h", True),
    ],
)
def test_model_level_era_timestep_freq(product: str, timestep_freq: str, raises: bool) -> None:
    """Test timestep frequency selection and validation."""
    if raises:
        with pytest.raises(ValueError, match=f"Product {product} has timestep frequency"):
            hres = ERA5ModelLevel(
                time=datetime(2000, 1, 1),
                variables=["t", "q"],
                timestep_freq=timestep_freq,
                product_type=product,
            )
    else:
        hres = ERA5ModelLevel(
            time=datetime(2000, 1, 1),
            variables=["t", "q"],
            timestep_freq=timestep_freq,
            product_type=product,
        )
        assert hres.timesteps == [datetime(2000, 1, 1)]


def test_model_level_era5_dataset() -> None:
    """Test CDS dataset property."""
    hres = ERA5ModelLevel(
        time=datetime(2000, 1, 1),
        variables=["t", "q"],
    )
    assert hres.dataset == "reanalysis-era5-complete"


def test_model_level_era5_cachepath() -> None:
    """Test cachepath creation."""
    era5 = ERA5ModelLevel(time=(datetime(2000, 1, 1), datetime(2000, 1, 2)), variables=["t", "q"])
    p = era5.create_cachepath(datetime(2000, 1, 1))
    assert "era5ml-6e5805300d4358fb27ced4fe5efe610b.nc" in p

    p1 = era5.create_cachepath(datetime(2000, 1, 1, 1))
    assert p1 != p

    era5 = ERA5ModelLevel(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
        pressure_levels=[150, 200, 250],
    )
    p1 = era5.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    era5 = ERA5ModelLevel(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["ciwc"],
    )
    p1 = era5.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    era5 = ERA5ModelLevel(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
        grid=1.0,
    )
    p1 = era5.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    era5.cachestore = None
    with pytest.raises(ValueError, match="Cachestore is required"):
        era5.create_cachepath(datetime(2000, 1, 1))


def test_model_level_era5_nominal_mars_request() -> None:
    """Test MARS request generation for nominal reanalysis."""
    era5 = ERA5ModelLevel(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2, 6)),
        variables=["t", "q"],
        model_levels=[1, 2, 3],
        timestep_freq="6h",
    )
    request = era5.mars_request(era5.timesteps)
    assert request == {
        "class": "ea",
        "date": "2000-01-01/2000-01-02",
        "expver": "1",
        "levelist": "1/2/3",
        "levtype": "ml",
        "param": "130/133",
        "time": "00:00:00/06:00:00/12:00:00/18:00:00",
        "type": "an",
        "grid": "0.25/0.25",
        "format": "netcdf",
        "stream": "oper",
    }


def test_model_level_era5_ensemble_mars_request() -> None:
    """Test MARS request generation for ensemble members."""
    era5 = ERA5ModelLevel(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2, 6)),
        variables=["t", "q"],
        model_levels=[1, 2, 3],
        timestep_freq="6h",
        product_type="ensemble_members",
        ensemble_members=[1, 4, 5],
    )
    request = era5.mars_request(era5.timesteps)
    assert request == {
        "class": "ea",
        "date": "2000-01-01/2000-01-02",
        "expver": "1",
        "levelist": "1/2/3",
        "levtype": "ml",
        "param": "130/133",
        "time": "00:00:00/06:00:00/12:00:00/18:00:00",
        "type": "an",
        "grid": "0.5/0.5",
        "format": "netcdf",
        "stream": "enda",
        "number": "1/4/5",
    }


def test_model_level_era5_set_metadata() -> None:
    """Test metadata setting."""
    era5 = ERA5ModelLevel(time=(datetime(2000, 1, 1), datetime(2000, 1, 2)), variables=["t", "q"])
    ds = xr.Dataset()
    era5.set_metadata(ds)
    assert ds.attrs["provider"] == "ECMWF"
    assert ds.attrs["dataset"] == "ERA5"
    assert ds.attrs["product"] == "reanalysis"

    era5 = ERA5ModelLevel(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
        product_type="ensemble_members",
    )
    ds = xr.Dataset()
    era5.set_metadata(ds)
    assert ds.attrs["provider"] == "ECMWF"
    assert ds.attrs["dataset"] == "ERA5"
    assert ds.attrs["product"] == "ensemble"

    era5.product_type = "foo"
    ds = xr.Dataset()
    with pytest.raises(ValueError, match="Unknown product type"):
        era5.set_metadata(ds)


##################################################
# HRES datalibs (pressure levels and model levels)
##################################################


@pytest.mark.parametrize("datalib", [HRES, HRESModelLevel])
def test_HRES_inputs(datalib: AnyHRESDatalibClass) -> None:
    """Test HRES time parsing."""
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        datalib(time=[], variables=[], pressure_levels=[17, 18, 19])

    times = (datetime(2019, 5, 31, 2, 29), datetime(2019, 5, 31, 4, 29))
    pressure_levels = [300, 250]
    variables = ["air_temperature", "specific_humidity"]
    dl = datalib(times, variables, pressure_levels=pressure_levels)

    assert dl.forecast_time == datetime(2019, 5, 31, 0, 0)
    assert dl.step_offset == 2
    assert dl.steps == [2, 3, 4, 5]

    times_str = ("2019-05-31 02:29:00", "2019-05-31 04:29:00")
    pressure_levels = [300, 250]
    variables = ["air_temperature", "specific_humidity"]
    forecast_time = "2019-05-30 12:00:00"
    dl = datalib(
        time=times_str,
        variables=variables,
        pressure_levels=pressure_levels,
        forecast_time=forecast_time,
    )

    assert dl.forecast_time == datetime(2019, 5, 30, 12, 0)
    assert dl.step_offset == 14
    assert dl.steps == [14, 15, 16, 17]


#############################
# HRES pressure level datalib
#############################


def test_HRES_repr() -> None:
    """Test HRES __repr__."""
    hres = HRES(time=datetime(2000, 1, 1), variables="vo", pressure_levels=200)
    out = repr(hres)
    assert "HRES" in out
    assert "Forecast time:" in out


def test_HRES_hash() -> None:
    """Test HRES hash."""

    hres = HRES(time=datetime(2022, 3, 5), variables=["t", "r"], pressure_levels=200)
    hres2 = HRES(time=datetime(2022, 3, 5), variables=["t", "r"], pressure_levels=[200, 300])
    assert hres.hash == "9f6c8ef71eed695d9e79093ee4cf750dec00895a"
    assert hres2.hash == "178f8cb671e8ac9e5ba086db7cc888d467ec11c2"
    assert hres.hash != hres2.hash


def test_HRES_dissemination_filename() -> None:
    """Test HRES dissemination filename creation."""

    forecast_time = datetime(2022, 12, 1, 0)

    timestep = datetime(2022, 12, 2, 8)
    forecast_time_str = get_forecast_filename(forecast_time, timestep)
    assert forecast_time_str == "A1D12010000120208001"

    # 0th step
    timestep = datetime(2022, 12, 1, 0)
    forecast_time_str = get_forecast_filename(forecast_time, timestep)
    assert forecast_time_str == "A1D12010000120100011"

    # 6th hour forecast
    forecast_time = datetime(2022, 12, 1, 6)
    timestep = datetime(2022, 12, 2, 0)
    forecast_time_str = get_forecast_filename(forecast_time, timestep)
    assert forecast_time_str == "A1S12010600120200001"

    # 6th hour forecast, 0th step
    forecast_time = datetime(2022, 12, 1, 6)
    timestep = datetime(2022, 12, 1, 6)
    forecast_time_str = get_forecast_filename(forecast_time, timestep)
    assert forecast_time_str == "A1S12010600120106011"


def test_HRES_dissemination_filename_errors() -> None:
    """Test HRES dissemination filename creation errors."""

    # forecast time error
    forecast_time = datetime(2022, 12, 1, 2)
    timestep = datetime(2022, 12, 1, 6)
    with pytest.raises(ValueError, match="hour 0, 6, 12"):
        get_forecast_filename(forecast_time, timestep)

    # timestep before forecast
    forecast_time = datetime(2022, 12, 2, 0)
    timestep = datetime(2022, 12, 1, 6)
    with pytest.raises(ValueError, match="timestep must be on or after forecast time"):
        get_forecast_filename(forecast_time, timestep)


@pytest.mark.parametrize("variables", ["t", "tsr"])
@pytest.mark.parametrize("field_type", ["fc", "an", "pf"])
def test_hres_set_met_source_metadata(field_type: str, variables: str) -> None:
    """Test HRES.get_met_source_metadata method."""

    hres = HRES(
        time=datetime(2000, 1, 1),
        variables=variables,
        pressure_levels=200 if variables == "t" else -1,
        field_type=field_type,
    )

    ds = xr.Dataset()
    hres.set_metadata(ds)

    assert ds.attrs["provider"] == "ECMWF"
    assert ds.attrs["dataset"] == "HRES"
    if field_type == "pc":
        assert ds.attrs["product"] == "ensemble"
    else:
        assert ds.attrs["product"] == "forecast"


##########################
# HRES model-level datalib
##########################


def test_model_level_hres_repr() -> None:
    hres = HRESModelLevel(
        time=datetime(2000, 1, 1),
        variables=["t", "q"],
    )
    out = repr(hres)
    assert "HRESModelLevel" in out
    assert "Forecast time" in out
    assert "Steps" in out


def test_model_level_hres_grid() -> None:
    """Test horizontal resolution."""
    hres = HRESModelLevel(
        time=datetime(2000, 1, 1),
        variables=["t", "q"],
    )
    assert hres.grid == 0.1

    hres = HRESModelLevel(time=datetime(2000, 1, 1), variables=["t", "q"], grid=1.0)
    assert hres.grid == 1.0

    with pytest.warns(UserWarning, match="The highest resolution available"):
        hres = HRESModelLevel(time=datetime(2000, 1, 1), variables=["t", "q"], grid=0.01)
    assert hres.grid == 0.01


def test_model_level_hres_forecast_time_validation() -> None:
    """Test forecast time validation."""
    hres = HRESModelLevel(
        time=datetime(2000, 1, 1), variables=["t", "q"], forecast_time="2000-01-01 00:00:00"
    )
    assert hres.forecast_time == datetime(2000, 1, 1, 0)

    hres = HRESModelLevel(
        time=datetime(2000, 1, 1), variables=["t", "q"], forecast_time="1999-12-31 12:00:00"
    )
    assert hres.forecast_time == datetime(1999, 12, 31, 12)

    with pytest.raises(ValueError, match="Requested times requires forecast steps out to"):
        hres = HRESModelLevel(
            time=datetime(2000, 1, 1), variables=["t", "q"], forecast_time="1999-12-21 12:00:00"
        )

    with pytest.raises(ValueError, match="Forecast hour must be one of"):
        hres = HRESModelLevel(
            time=datetime(2000, 1, 1), variables=["t", "q"], forecast_time="1999-12-31 18:00:00"
        )

    with pytest.raises(ValueError, match="Selected forecast time"):
        hres = HRESModelLevel(
            time=datetime(2000, 1, 1), variables=["t", "q"], forecast_time="2000-1-1 12:00:00"
        )


@pytest.mark.parametrize(
    ("time", "timestep_freq", "raises"),
    [
        ([datetime(2000, 1, 4, 23), datetime(2000, 1, 5, 0)], "1h", False),
        ([datetime(2000, 1, 5, 0), datetime(2000, 1, 5, 3)], "1h", True),
        ([datetime(2000, 1, 6, 21), datetime(2000, 1, 7, 0)], "3h", False),
        ([datetime(2000, 1, 7, 0), datetime(2000, 1, 7, 3)], "3h", True),
        ([datetime(2000, 1, 7, 0), datetime(2000, 1, 7, 6)], "6h", False),
    ],
)
def test_model_level_hres_timestep_freq(
    time: list[datetime], timestep_freq: str, raises: bool
) -> None:
    """Test timestep frequency selection and validation."""
    if raises:
        with pytest.raises(ValueError, match="Forecast out to step"):
            hres = HRESModelLevel(
                time=time,
                variables=["t", "q"],
                forecast_time="2000-01-01 00:00:00",
                timestep_freq=timestep_freq,
            )
    else:
        hres = HRESModelLevel(
            time=time,
            variables=["t", "q"],
            forecast_time="2000-01-01 00:00:00",
            timestep_freq=timestep_freq,
        )
        assert hres.timesteps == time


def test_model_level_hres_get_forecast_step() -> None:
    """Test forecast step calculation."""
    hres = HRESModelLevel(
        time=datetime(2000, 1, 1), variables=["t", "q"], forecast_time="2000-01-01 00:00:00"
    )
    assert hres.get_forecast_steps([datetime(2000, 1, 1, 0)]) == [0]
    assert hres.get_forecast_steps([datetime(2000, 1, 1, 3)]) == [3]
    assert hres.get_forecast_steps([datetime(1999, 12, 31, 22)]) == [-2]
    with pytest.raises(ValueError, match="Time-to-step conversion returned fractional"):
        hres.get_forecast_steps([datetime(2000, 1, 1, 0, 30)])


def test_model_level_hres_cachepath() -> None:
    """Test cachepath creation."""
    hres = HRESModelLevel(time=(datetime(2000, 1, 1), datetime(2000, 1, 2)), variables=["t", "q"])
    p = hres.create_cachepath(datetime(2000, 1, 1))
    assert "hresml-c7ef25c716b87726c24ca848a3f32e79.nc" in p

    p1 = hres.create_cachepath(datetime(2000, 1, 1, 1))
    assert p1 != p

    hres = HRESModelLevel(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
        forecast_time="1999-12-31 12:00:00",
    )
    p1 = hres.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    hres = HRESModelLevel(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
        pressure_levels=[150, 200, 250],
    )
    p1 = hres.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    hres = HRESModelLevel(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["ciwc"],
    )
    p1 = hres.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    hres = HRESModelLevel(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
        grid=1.0,
    )
    p1 = hres.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    hres.cachestore = None
    with pytest.raises(ValueError, match="Cachestore is required"):
        hres.create_cachepath(datetime(2000, 1, 1))


def test_model_level_hres_mars_request() -> None:
    """Test MARS request formatting."""
    hres = HRESModelLevel(
        time=(datetime(2000, 1, 1, 3), datetime(2000, 1, 1, 6)),
        forecast_time="2000-01-01 00:00:00",
        variables=["t", "q"],
        model_levels=[1, 2, 3],
    )
    request = hres.mars_request(hres.timesteps)
    assert request == ",\n".join(
        [
            "retrieve",
            "class=od",
            "date=2000-01-01",
            "expver=1",
            "levelist=1/2/3",
            "levtype=ml",
            "param=130/133/152",
            "step=3/4/5/6",
            "stream=oper",
            "time=00:00:00",
            "type=fc",
            "grid=0.1/0.1",
            "format=netcdf",
        ]
    )


def test_model_level_hres_set_metadata() -> None:
    """Test metadata setting."""
    hres = HRESModelLevel(time=datetime(2000, 1, 1), variables=["t", "q"])
    ds = xr.Dataset()
    hres.set_metadata(ds)
    assert ds.attrs["provider"] == "ECMWF"
    assert ds.attrs["dataset"] == "HRES"
    assert ds.attrs["product"] == "forecast"
    assert ds.attrs["radiation_accumulated"]


def test_model_level_pressure_agreement() -> None:
    pl1 = model_level_reference_pressure()
    assert isinstance(pl1, list)
    assert len(pl1) == 137

    sp = xr.DataArray(1013.25 * 100.0)
    model_levels = range(1, 138)
    da = model_level_pressure(sp, model_levels)
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("model_level",)

    pl2 = da.values.round().astype(int).tolist()
    assert pl1 == pl2


def test_model_level_pressure_agrees_with_ecmwf() -> None:
    """Test pressure level at model levels agrees with published ECMWF values."""
    sp = xr.DataArray(1013.25 * 100.0)
    model_levels = range(1, 138)
    s1 = model_level_pressure(sp, model_levels).to_series().round(4)

    s2 = (
        pd.read_csv(MODEL_LEVELS_PATH, index_col=0)["pf [hPa]"]
        .loc[1:137]
        .rename_axis("model_level")
        .rename(None)
    )

    pd.testing.assert_series_equal(s1, s2, atol=5e-4, rtol=0.0)


def test_ml_to_pl_conversion_output(era5_ml: xr.Dataset, lnsp: xr.DataArray) -> None:
    """Test ml_to_pl conversion output."""
    target_pl = [200, 210, 220, 230, 240, 250]
    ds = ml_to_pl(era5_ml, target_pl, lnsp=lnsp)
    assert isinstance(ds, xr.Dataset)
    np.testing.assert_array_equal(ds["level"], target_pl)

    # No null values for these pressure levels
    for v in ds.data_vars:
        assert not ds[v].isnull().any()


def test_ml_to_pl_conversion_output_with_null(era5_ml: xr.Dataset, lnsp: xr.DataArray) -> None:
    """Test ml_to_pl conversion with null values in the output."""
    target_pl = [190, 195, 200]
    ds = ml_to_pl(era5_ml, target_pl, lnsp=lnsp)
    assert isinstance(ds, xr.Dataset)
    np.testing.assert_array_equal(ds["level"], target_pl)

    # Most the values on PL 190 are null
    # Some on PL 195 are null
    # And none on PL 200 are null
    for v in ds.data_vars:
        assert ds[v].sel(level=190).isnull().mean() == 0.925
        assert ds[v].sel(level=195).isnull().mean() == 0.841666666666666666
        assert not ds[v].sel(level=200).isnull().any()


def test_ml_to_pl_close_to_era5_pl(
    era5_ml: xr.Dataset,
    lnsp: xr.DataArray,
    met_ecmwf_pl_path: str,
) -> None:
    """Comfirm that the ml_to_pl conversion is close to what the CDS API provides."""
    era5_ml = era5_ml.rename(valid_time="time")
    lnsp = lnsp.rename(valid_time="time")
    ds_pl = xr.open_dataset(met_ecmwf_pl_path).sel(time=era5_ml["time"])

    target_pl = ds_pl["level"].values
    np.testing.assert_array_equal(target_pl, [300, 250, 225])

    ds_ml = ml_to_pl(era5_ml, target_pl, lnsp=lnsp)

    # 19 nulls got introduced in the conversion, all on level 300
    for v in ds_ml.data_vars:
        assert ds_ml[v].sel(level=[300]).isnull().sum() == 19
        assert ds_ml[v].sel(level=[225, 250]).notnull().all()

        # Fill the nulls in the converted dataset with the values ds_pl
        # and compare the two
        # We don't expect equality to be close to exact here -- the ERA5 PL data
        # and the ERA5 ML data were generated separately and are not expected to
        # be derived from a common source
        # Still, the agreement isn't bad
        da = ds_ml[v].fillna(ds_pl[v])
        xr.testing.assert_allclose(da, ds_pl[v], rtol=0.001, atol=1e-5, check_dim_order=False)
