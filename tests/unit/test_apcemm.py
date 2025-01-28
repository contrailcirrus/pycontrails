from __future__ import annotations

import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails.core import Flight, MetDataset
from pycontrails.models.apcemm import APCEMM
from pycontrails.models.humidity_scaling import (
    ConstantHumidityScaling,
    ExponentialBoostHumidityScaling,
)
from pycontrails.models.ps_model import PSFlight
from pycontrails.physics import constants
from tests.unit import get_static_path

APCEMM_GIT_COMMIT = "9d8e1eeaa61cbdee1b1d03c65b5b033ded9159e4"


@pytest.fixture()
def apcemm_paths() -> tuple[str, str]:
    """Return APCEMM executable and root directory, if found.

    This test looks for an ``APCEMM`` executable on the ``PATH`` using
    :py:func:`shutil.which` and attempts to set the APCEMM root directory
    based on the location of the executable.

    For APCEMM tests to run, APCEMM must be in the ``build`` subdirectory
    of an otherwise-unmodified APCEMM git repository, and the repository
    must be at the appropriate git hash.
    """

    # GitPython required to inspect repositories
    # https://pypi.org/project/GitPython/
    git = pytest.importorskip("git")

    # Attempt to find APCEMM executable
    apcemm_path = shutil.which("APCEMM")
    if apcemm_path is None:
        pytest.skip("APCEMM executable not found")

    # If found, check that it is in a directory called build...
    dirname = os.path.dirname(apcemm_path)
    if os.path.basename(dirname) != "build":
        msg = "APCEMM executable is not in a directory called 'build'"
        raise ValueError(msg)

    # ... and check that the parent of the build directory is a git repository
    apcemm_root = os.path.dirname(dirname)
    try:
        repo = git.Repo(apcemm_root)
    except git.InvalidGitRepositoryError as exc:
        msg = f"{apcemm_root} is not a valid git repository"
        raise ValueError(msg) from exc

    # Check commit hash
    if repo.head.object.hexsha != APCEMM_GIT_COMMIT:
        msg = "APCEMM repository has wrong commit hash"
        raise ValueError(msg)

    # Check repository state:
    # - no untracked files outside of build directory
    if any(f.split(os.path.sep)[0] != "build" for f in repo.untracked_files):
        msg = "APCEMM repository has untracked files outside build directory"
        raise ValueError(msg)
    # - no unstaged changes to working directory
    if len(repo.index.diff(None)) != 0:
        msg = "APCEMM working directory contains unstaged changes"
        raise ValueError(msg)
    # - no uncommitted changes in staging area
    if len(repo.index.diff(repo.head.object.hexsha)) != 0:
        msg = "APCEMM working directory contains staged changes"
        raise ValueError(msg)

    return apcemm_path, apcemm_root


@pytest.fixture()
def met_apcemm() -> MetDataset:
    """MetDataset for APCEMM simulations."""
    path = get_static_path("met-era5-cocip1.nc")
    ds = xr.open_dataset(path).astype("float32")
    ds = ds[
        [
            "air_temperature",
            "specific_humidity",
            "geopotential",
            "eastward_wind",
            "northward_wind",
            "lagrangian_tendency_of_air_pressure",
        ]
    ]
    ds["geopotential_height"] = ds["geopotential"] / constants.g
    ds["air_pressure"] = ds["air_pressure"].astype("float32")
    ds["altitude"] = ds["altitude"].astype("float32")
    ds.attrs.update(provider="ECMWF", dataset="ERA5", product="reanalysis")
    return MetDataset(ds)


@pytest.fixture()
def flight_apcemm() -> Flight:
    """Synthetic flight for generating APCEMM output."""
    attrs = {
        "aircraft_type": "B738",
        "flight_id": "test",
    }
    df = pd.DataFrame()
    df["longitude"] = np.linspace(-29, -32, 11)
    df["latitude"] = np.linspace(56, 57, 11)
    df["altitude"] = np.linspace(10900, 10900, 11)
    df["time"] = pd.date_range("2019-01-01T00:15:00", "2019-01-01T02:30:00", periods=11)
    return Flight(df, attrs=attrs)


@pytest.fixture()
def apcemm_persistent(
    flight_apcemm: Flight, met_apcemm: MetDataset, apcemm_paths: tuple[str, str]
) -> tuple[APCEMM, Flight]:
    """Return `APCEMM` instance and result from evaluation on `flight_apcemm`."""
    apcemm_path, apcemm_root = apcemm_paths
    params = {
        "max_age": np.timedelta64(10, "m"),  # to limit runtime
        "dt_lagrangian": np.timedelta64(1, "m"),
        "dt_input_met": np.timedelta64(5, "m"),
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "aircraft_performance": PSFlight(),
        "waypoints": [1, 7],
        "overwrite": True,
    }
    model = APCEMM(met_apcemm, apcemm_path=apcemm_path, apcemm_root=apcemm_root, params=params)
    result = model.eval(flight_apcemm)
    return model, result


@pytest.mark.parametrize(
    "drop_var",
    [
        "air_temperature",
        "specific_humidity",
        ("geopotential", "geopotential_height"),
        "eastward_wind",
        "northward_wind",
        "lagrangian_tendency_of_air_pressure",
        None,
    ],
)
def test_apcemm_validate_met(
    drop_var: str | tuple[str] | None, met_apcemm: MetDataset, apcemm_paths: tuple[str, str]
) -> None:
    """Test APCEMM met validation."""
    apcemm_path, apcemm_root = apcemm_paths

    if drop_var is not None:
        met = MetDataset(met_apcemm.data.drop_vars(drop_var))
        with pytest.raises(KeyError, match="Dataset does not contain"):
            _ = APCEMM(
                met=met,
                apcemm_path=apcemm_path,
                apcemm_root=apcemm_root,
                humidity_scaling=ConstantHumidityScaling(),
            )
    else:
        _ = APCEMM(
            met=met_apcemm,
            apcemm_path=apcemm_path,
            apcemm_root=apcemm_root,
            humidity_scaling=ConstantHumidityScaling(),
        )


@pytest.mark.parametrize(
    "drop_var",
    [
        (),
        ("geopotential",),
        ("geopotential_height",),
        ("geopotential", "geopotential_height"),
    ],
)
def test_apcemm_ensure_geopotential_height(
    drop_var: tuple[str], met_apcemm: MetDataset, apcemm_paths: tuple[str, str]
) -> None:
    """Test APCEMM met validation."""
    apcemm_path, apcemm_root = apcemm_paths
    met = MetDataset(met_apcemm.data.drop_vars(drop_var))

    if len(drop_var) < 2:
        model = APCEMM(
            met=met,
            apcemm_path=apcemm_path,
            apcemm_root=apcemm_root,
            humidity_scaling=ConstantHumidityScaling(),
        )
        assert "geopotential_height" in model.met

    else:
        with pytest.raises(KeyError, match="Dataset does not contain"):
            _ = APCEMM(
                met=met,
                apcemm_path=apcemm_path,
                apcemm_root=apcemm_root,
                humidity_scaling=ConstantHumidityScaling(),
            )
        with pytest.raises(ValueError, match="APCEMM MetDataset must contain"):
            _ = APCEMM(
                met=met,
                apcemm_path=apcemm_path,
                apcemm_root=apcemm_root,
                humidity_scaling=ConstantHumidityScaling(),
                verify_met=False,
            )


@pytest.mark.parametrize(
    ("params", "valid"),
    [
        (
            {},
            True,
        ),
        (
            {"horiz_diff": 30.0},
            True,
        ),
        (
            {"rhw": 0.50},
            False,
        ),
        (
            {"output_directory": "foo", "max_age": np.timedelta64(20, "h")},
            False,
        ),
    ],
)
def test_apcemm_input_param_validation(
    params: dict[str, Any], valid: bool, met_apcemm: MetDataset, apcemm_paths: tuple[str, str]
) -> None:
    """Test validation of APCEMM input parameter overrides."""
    apcemm_path, _ = apcemm_paths
    if not valid:
        with pytest.raises(ValueError, match="Cannot override APCEMM input"):
            model = APCEMM(
                met=met_apcemm,
                apcemm_path=apcemm_path,
                apcemm_input_params=params,
                humidity_scaling=ConstantHumidityScaling(),
            )
        return

    model = APCEMM(
        met=met_apcemm,
        apcemm_path=apcemm_path,
        apcemm_input_params=params,
        humidity_scaling=ConstantHumidityScaling(),
    )
    assert model.apcemm_input_params == params


def test_apcemm_default_root_directory(
    met_apcemm: MetDataset, apcemm_paths: tuple[str, str]
) -> None:
    """Test APCEMM default root directory."""
    apcemm_path, _ = apcemm_paths
    model = APCEMM(
        met=met_apcemm, apcemm_path=apcemm_path, humidity_scaling=ConstantHumidityScaling()
    )
    assert "input_background_conditions" not in model.apcemm_input_params
    assert "input_engine_emissions" not in model.apcemm_input_params


def test_apcemm_root_directory_param(met_apcemm: MetDataset, apcemm_paths: tuple[str, str]) -> None:
    """Test apcemm root directory set by constructor parameter."""
    apcemm_path, apcemm_root = apcemm_paths
    model = APCEMM(
        met=met_apcemm,
        apcemm_path=apcemm_path,
        apcemm_root=apcemm_root,
        humidity_scaling=ConstantHumidityScaling(),
    )
    assert model.apcemm_input_params["input_background_conditions"] == os.path.join(
        apcemm_root, "input_data", "init.txt"
    )
    assert model.apcemm_input_params["input_engine_emissions"] == os.path.join(
        apcemm_root, "input_data", "ENG_EI.txt"
    )


def test_apcemm_root_directory_yaml(met_apcemm: MetDataset, apcemm_paths: tuple[str, str]) -> None:
    """Test APCEMM root directory based on input parameters."""
    apcemm_path, apcemm_root = apcemm_paths
    model = APCEMM(
        met=met_apcemm,
        apcemm_path=apcemm_path,
        apcemm_root="foo",
        apcemm_input_params=dict(input_background_conditions="bar", input_engine_emissions="baz"),
        humidity_scaling=ConstantHumidityScaling(),
    )
    assert model.apcemm_input_params["input_background_conditions"] == "bar"
    assert model.apcemm_input_params["input_engine_emissions"] == "baz"


def test_apcemm_overwrite_protection(
    apcemm_persistent: tuple[APCEMM, Flight],
    flight_apcemm: Flight,
    met_apcemm: MetDataset,
    apcemm_paths: tuple[str, str],
) -> None:
    """Test APCEMM overwrite protection."""

    # check that fixture has already generated output
    model, _ = apcemm_persistent
    assert os.path.exists(model.apcemm_file(1))

    # re-running simulation on same waypoint should produce an error
    apcemm_path, apcemm_root = apcemm_paths
    params = {
        "max_age": np.timedelta64(10, "m"),  # to limit runtime
        "dt_lagrangian": np.timedelta64(1, "m"),
        "dt_input_met": np.timedelta64(5, "m"),
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "aircraft_performance": PSFlight(),
    }
    model = APCEMM(met_apcemm, apcemm_path=apcemm_path, apcemm_root=apcemm_root, params=params)
    with pytest.raises(ValueError, match="APCEMM run directory already exists"):
        _ = model.eval(flight_apcemm, waypoints=[1])


def test_apcemm_output_status(apcemm_persistent: tuple[APCEMM, Flight]) -> None:
    """Test APCEMM simulation status output."""
    _, result = apcemm_persistent
    assert result.dataframe.iloc[0]["status"] == "NoSimulation"
    assert result.dataframe.iloc[1]["status"] == "NoPersistence"
    assert result.dataframe.iloc[7]["status"] == "Incomplete"
    assert result.dataframe.iloc[-1]["status"] == "NoSimulation"


def test_apcemm_output_time_series(apcemm_persistent: tuple[APCEMM, Flight]) -> None:
    """Test APCEMM time series output.

    Current tests presence and shape but not values.
    """
    model, _ = apcemm_persistent
    df = model.vortex
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (600, 20)
    assert set(df["waypoint"]) == {1, 7}


def test_apcemm_output_contrail(apcemm_persistent: tuple[APCEMM, Flight]) -> None:
    """Test APCEMM contrail output.

    Current checks that all files are present and can be opened by xarray.
    """
    model, _ = apcemm_persistent
    df = model.contrail
    assert isinstance(df, pd.DataFrame)
    assert set(df["waypoint"]) == {7}
    assert df.shape == (11, 3)
    for _, row in df.iterrows():
        path = row["path"]
        ds = xr.open_dataset(path, decode_times=False)
        assert len(ds.coords) == 4
        assert all(c in ds.coords for c in ["x", "y", "r", "t"])
        assert len(ds.variables) == 24
