from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import DiskCacheStore, MetDataset, MetVariable
from pycontrails.core.met_var import AirTemperature, SurfacePressure
from pycontrails.datalib.ecmwf import ECMWF_VARIABLES, ERA5, HRES
from pycontrails.datalib.ecmwf.hres import get_forecast_filename


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


def test_ERA5_single_time_input() -> None:
    """Test TimeInput parsing."""
    # accept single time
    era5 = ERA5(time=datetime(2019, 5, 31, 0), variables=["vo"], pressure_levels=[200])
    assert era5.timesteps == [datetime(2019, 5, 31, 0)]
    era5 = ERA5(time=[datetime(2019, 5, 31, 0)], variables=["vo"], pressure_levels=[200])
    assert era5.timesteps == [datetime(2019, 5, 31, 0)]

    # accept single time with minutes defined
    era5 = ERA5(time=datetime(2019, 5, 31, 0, 29), variables=["vo"], pressure_levels=[200])
    assert era5.timesteps == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]
    era5 = ERA5(time=[datetime(2019, 5, 31, 0, 29)], variables=["vo"], pressure_levels=[200])
    assert era5.timesteps == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]


def test_ERA5_time_input_wrong_length() -> None:
    """Test TimeInput parsing."""

    # throw ValueError for length == 0
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        ERA5(time=[], variables=["vo"], pressure_levels=[200])

    # throw ValueError for length > 2
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        ERA5(
            time=[datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 0)],
            variables=["vo"],
            pressure_levels=[200],
        )


def test_ERA5_time_input_two_times() -> None:
    """Test TimeInput parsing."""

    # accept pair (start, end)
    era5 = ERA5(
        time=(datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 3)),
        variables=["vo"],
        pressure_levels=[200],
    )
    assert era5.timesteps == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 1),
        datetime(2019, 5, 31, 2),
        datetime(2019, 5, 31, 3),
    ]

    era5 = ERA5(
        time=(datetime(2019, 5, 31, 0, 29), datetime(2019, 5, 31, 2, 40)),
        variables=["vo"],
        pressure_levels=[200],
    )
    assert era5.timesteps == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 1),
        datetime(2019, 5, 31, 2),
        datetime(2019, 5, 31, 3),
    ]


def test_ERA5_time_input_numpy_pandas() -> None:
    """Test TimeInput parsing."""

    # support alternate types for input
    era5 = ERA5(
        time=pd.to_datetime(datetime(2019, 5, 31, 0, 29)),
        variables=["vo"],
        pressure_levels=[200],
    )
    assert era5.timesteps == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]
    era5 = ERA5(time=np.datetime64("2019-05-31T00:29:00"), variables=["vo"], pressure_levels=[200])
    assert era5.timesteps == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]
    era5 = ERA5(
        time=(np.datetime64("2019-05-31T00:29:00"), np.datetime64("2019-05-31T02:40:00")),
        variables=["vo"],
        pressure_levels=[200],
    )
    assert era5.timesteps == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 1),
        datetime(2019, 5, 31, 2),
        datetime(2019, 5, 31, 3),
    ]


def test_ERA5_inputs() -> None:
    """Test ERA5 __init__."""
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        ERA5(time=[], variables=[], pressure_levels=[17, 18, 19])

    e1 = ERA5(time=datetime(2000, 1, 1), variables="vo", pressure_levels=200)
    e2 = ERA5(time=[datetime(2000, 1, 1)], variables=["vo"], pressure_levels=[200])
    e3 = ERA5(
        time=np.datetime64("2000-01-01 00:00:00"),
        variables=["vo"],
        pressure_levels=[200],
    )
    e4 = ERA5(
        time=np.array([np.datetime64("2000-01-01 00:00:00")]),
        variables=["vo"],
        pressure_levels=[200],
    )
    assert e1.variables == e2.variables
    assert e1.pressure_levels == e2.pressure_levels
    assert e1.timesteps == e2.timesteps == e3.timesteps == e4.timesteps


def test_ERA5_repr() -> None:
    """Test ERA5 __repr__."""
    era5 = ERA5(time=datetime(2000, 1, 1), variables="vo", pressure_levels=200)
    out = repr(era5)
    assert "ERA5" in out
    assert "Dataset:" in out


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


def test_HRES_repr() -> None:
    """Test HRES __repr__."""
    hres = HRES(time=datetime(2000, 1, 1), variables="vo", pressure_levels=200)
    out = repr(hres)
    assert "HRES" in out
    assert "Forecast time:" in out


def test_HRES_inputs() -> None:
    """Test HRES time parsing."""
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        HRES(time=[], variables=[], pressure_levels=[17, 18, 19])

    times = (datetime(2019, 5, 31, 2, 29), datetime(2019, 5, 31, 4, 29))
    pressure_levels = [300, 250]
    variables = ["air_temperature", "relative_humidity", "specific_humidity"]
    hres = HRES(times, variables, pressure_levels=pressure_levels)

    assert hres.forecast_time == datetime(2019, 5, 31, 0, 0)
    assert hres.step_offset == 2
    assert hres.steps == [2, 3, 4, 5]

    times_str = ("2019-05-31 02:29:00", "2019-05-31 04:29:00")
    pressure_levels = [300, 250]
    variables = ["air_temperature", "relative_humidity", "specific_humidity"]
    forecast_time = "2019-05-30 12:00:00"
    hres = HRES(
        time=times_str,
        variables=variables,
        pressure_levels=pressure_levels,
        forecast_time=forecast_time,
    )

    assert hres.forecast_time == datetime(2019, 5, 30, 12, 0)
    assert hres.step_offset == 14
    assert hres.steps == [14, 15, 16, 17]


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
@pytest.mark.parametrize("product_type", ["reanalysis", "ensemble_mean", "ensemble_members"])
def test_era5_set_met_source_metadata(product_type: str, variables: str) -> None:
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


def test_era5_met_source_open_metdataset(met_ecmwf_pl_path: str) -> None:
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


@pytest.mark.parametrize("variables", ["t", "tsr"])
@pytest.mark.parametrize("field_type", ["fc", "an", "pf"])
def test_hres_set_met_source_metadata(field_type: str, variables: str) -> None:
    """Test HRES.get_met_source_metadata method."""

    era5 = HRES(
        time=datetime(2000, 1, 1),
        variables=variables,
        pressure_levels=200 if variables == "t" else -1,
        field_type=field_type,
    )

    ds = xr.Dataset()
    era5.set_metadata(ds)

    assert ds.attrs["provider"] == "ECMWF"
    assert ds.attrs["dataset"] == "HRES"
    if field_type == "pc":
        assert ds.attrs["product"] == "ensemble"
    else:
        assert ds.attrs["product"] == "forecast"
