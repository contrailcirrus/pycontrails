import os
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails.core import Flight, MetDataset
from pycontrails.models.apcemm import APCEMM
from pycontrails.models.apcemm.interface import APCEMMYaml
from pycontrails.models.humidity_scaling import (
    ConstantHumidityScaling,
    ExponentialBoostHumidityScaling,
)
from pycontrails.models.ps_model import PSFlight

from .conftest import get_static_path

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
    apcemm = shutil.which("APCEMM")
    if apcemm is None:
        pytest.skip("APCEMM executable not found")

    # If found, check that it is in a directory called build...
    dirname = os.path.dirname(apcemm)
    if os.path.basename(dirname) != "build":
        pytest.skip("APCEMM executable is not in a directory called 'build'")

    # ... and check that the parent of the build directory is a git repository
    apcemm_root = os.path.dirname(dirname)
    try:
        repo = git.Repo(apcemm_root)
    except git.InvalidGitRepositoryError:
        pytest.skip(f"{apcemm_root} is not a valid git repository")

    # Check commit hash
    if repo.head.object.hexsha != APCEMM_GIT_COMMIT:
        pytest.skip("APCEMM repository has wrong commit hash")

    # Check repository state:
    # - no untracked files outside of build directory
    if any(f.split(os.path.sep)[0] != "build" for f in repo.untracked_files):
        pytest.skip("APCEMM repository has untracked files outside build directory")
    # - no unstaged changes to working directory
    if len(repo.index.diff(None)) != 0:
        pytest.skip("APCEMM working directory contains unstaged changes")
    # - no uncommitted changes in staging area
    if len(repo.index.diff(repo.head.object.hexsha)) != 0:
        pytest.skip("APCEMM working directory contains staged changes")

    return apcemm, apcemm_root


@pytest.fixture()
def met_apcemm() -> MetDataset:
    """MetDataset for APCEMM simulations."""
    path = get_static_path("met-era5-cocip1.nc")
    ds = xr.open_dataset(path).astype("float32")
    ds = ds[
        [
            "air_temperature",
            "specific_humidity",
            "eastward_wind",
            "northward_wind",
            "lagrangian_tendency_of_air_pressure",
        ]
    ]
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
    df["longitude"] = np.linspace(-29, -32, 20)
    df["latitude"] = np.linspace(56, 57, 20)
    df["altitude"] = np.linspace(10900, 10900, 20)
    df["time"] = pd.date_range("2019-01-01T00:15:00", "2019-01-01T02:30:00", periods=20)
    return Flight(df, attrs=attrs)


@pytest.fixture()
def apcemm_persistent(
    flight_apcemm: Flight, met_apcemm: MetDataset, apcemm_paths: tuple[str, str]
) -> tuple[APCEMM, Flight]:
    """Return `APCEMM` instance and result from evaluation on `flight_apcemm`."""
    apcemm, apcemm_root = apcemm_paths
    params = {
        "max_age": np.timedelta64(10, "m"),  # to limit runtime
        "dt_lagrangian": np.timedelta64(1, "m"),
        "dt_input_met": np.timedelta64(5, "m"),
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "aircraft_performance": PSFlight(),
        "segments": [1, 13],
        "overwrite": True,
    }
    model = APCEMM(met_apcemm, apcemm=apcemm, apcemm_root=apcemm_root, params=params)
    result = model.eval(flight_apcemm)
    return model, result


@pytest.mark.parametrize(
    "drop_var",
    [
        "air_temperature",
        "specific_humidity",
        "eastward_wind",
        "northward_wind",
        "lagrangian_tendency_of_air_pressure",
        None,
    ],
)
def test_apcemm_validate_met(
    drop_var: str | None, met_apcemm: MetDataset, apcemm_paths: tuple[str, str]
) -> None:
    """Test APCEMM met validation."""
    apcemm, apcemm_root = apcemm_paths

    if drop_var is not None:
        met = MetDataset(met_apcemm.data.drop_vars(drop_var))
        with pytest.raises(KeyError, match="Dataset does not contain"):
            _ = APCEMM(
                met=met,
                apcemm=apcemm,
                apcemm_root=apcemm_root,
                humidity_scaling=ConstantHumidityScaling(),
            )
    else:
        _ = APCEMM(
            met=met_apcemm,
            apcemm=apcemm,
            apcemm_root=apcemm_root,
            humidity_scaling=ConstantHumidityScaling(),
        )


@pytest.mark.parametrize(
    ("dt_lagrangian", "dt_input_met", "downsampling"),
    [
        (np.timedelta64(30, "m"), np.timedelta64(1, "h"), 2),
        (np.timedelta64(1, "h"), np.timedelta64(1, "h"), 1),
        (np.timedelta64(2, "h"), np.timedelta64(1, "h"), None),
        (np.timedelta64(45, "h"), np.timedelta64(1, "h"), None),
    ],
)
def test_apcemm_validate_downsampling(
    dt_lagrangian: np.timedelta64,
    dt_input_met: np.timedelta64,
    downsampling: int | None,
    met_apcemm: MetDataset,
    apcemm_paths: tuple[str, str],
) -> None:
    """Test timestep validation."""
    apcemm, apcemm_root = apcemm_paths

    if downsampling is None:
        with pytest.raises(ValueError, match="Timestep for Lagrangian trajectories"):
            _ = APCEMM(
                met=met_apcemm,
                apcemm=apcemm,
                apcemm_root=apcemm_root,
                humidity_scaling=ConstantHumidityScaling(),
                dt_lagrangian=dt_lagrangian,
                dt_input_met=dt_input_met,
            )

    else:
        model = APCEMM(
            met=met_apcemm,
            apcemm=apcemm,
            apcemm_root=apcemm_root,
            humidity_scaling=ConstantHumidityScaling(),
            dt_lagrangian=dt_lagrangian,
            dt_input_met=dt_input_met,
        )
        assert model._trajectory_downsampling == downsampling


def test_apcemm_default_root_directory(
    met_apcemm: MetDataset, apcemm_paths: tuple[str, str]
) -> None:
    """Test APCEMM default root directory."""
    apcemm, _ = apcemm_paths
    model = APCEMM(met=met_apcemm, apcemm=apcemm, humidity_scaling=ConstantHumidityScaling())
    assert model.yaml.apcemm_root == os.path.expanduser("~/APCEMM")


def test_apcemm_root_directory_param(met_apcemm: MetDataset, apcemm_paths: tuple[str, str]) -> None:
    """Test apcemm root directory set by constructor parameter."""
    apcemm, apcemm_root = apcemm_paths
    model = APCEMM(
        met=met_apcemm,
        apcemm=apcemm,
        apcemm_root=apcemm_root,
        humidity_scaling=ConstantHumidityScaling(),
    )
    assert model.yaml.apcemm_root == apcemm_root


def test_apcemm_root_directory_yaml(met_apcemm: MetDataset, apcemm_paths: tuple[str, str]) -> None:
    """Test APCEMM root directory based on YAML parameter."""
    apcemm, apcemm_root = apcemm_paths
    model = APCEMM(
        met=met_apcemm,
        apcemm=apcemm,
        apcemm_root="foo",
        yaml=APCEMMYaml(apcemm_root=apcemm_root),
        humidity_scaling=ConstantHumidityScaling(),
    )
    assert model.yaml.apcemm_root == apcemm_root


def test_apcemm_overwrite_protection(
    flight_apcemm: Flight, met_apcemm: MetDataset, apcemm_paths: tuple[str, str]
) -> None:
    """Test APCEMM overwrite protection."""
    apcemm, apcemm_root = apcemm_paths
    params = {
        "max_age": np.timedelta64(10, "m"),  # to limit runtime
        "dt_lagrangian": np.timedelta64(1, "m"),
        "dt_input_met": np.timedelta64(5, "m"),
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "aircraft_performance": PSFlight(),
    }
    model = APCEMM(met_apcemm, apcemm=apcemm, apcemm_root=apcemm_root, params=params)

    # already generated by fixture, so should produce error
    with pytest.raises(ValueError, match="APCEMM run directory already exists"):
        _ = model.eval(flight_apcemm, segments=[1])


def test_apcemm_output_status(apcemm_persistent: tuple[APCEMM, Flight]) -> None:
    """Test APCEMM simulation status output."""
    _, result = apcemm_persistent
    assert result.dataframe.iloc[0]["status"] == "NoSimulation"
    assert result.dataframe.iloc[1]["status"] == "NoPersistence"
    assert result.dataframe.iloc[13]["status"] == "Incomplete"
    assert result.dataframe.iloc[-1]["status"] == "N/A"


def test_apcemm_output_time_series(apcemm_persistent: tuple[APCEMM, Flight]) -> None:
    """Test APCEMM time series output.

    Current tests presence and shape but not values.
    """
    model, _ = apcemm_persistent
    df = model.vortex
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (600, 20)
    assert set(df["waypoint"]) == set([1, 13])


def test_apcemm_output_contrail(apcemm_persistent: tuple[APCEMM, Flight]) -> None:
    """Test APCEMM contrail output.

    Current checks that all files are present and can be opened by xarray.
    """
    model, _ = apcemm_persistent
    df = model.contrail
    assert isinstance(df, pd.DataFrame)
    assert set(df["waypoint"]) == set([13])
    assert df.shape == (11, 3)
    for _, row in df.iterrows():
        path = row["path"]
        ds = xr.open_dataset(path, decode_times=False)
        assert len(ds.coords) == 4
        assert all(c in ds.coords for c in ["x", "y", "r", "t"])
        assert len(ds.variables) == 24
