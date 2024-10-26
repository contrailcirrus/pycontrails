"""LibRadtran utilities."""

import itertools
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from copy import copy

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pycontrails.core.cache import CacheStore
from pycontrails.physics import constants


def get_lrt_folder() -> str:
    """Get libRadtran root directory."""
    user_location = os.path.expanduser("~/.pylrtrc")
    try:
        with open(user_location) as f:
            return f.read().strip()
    except FileNotFoundError as exc:
        msg = "No default location for LibRadTran found. Place the path in ~/.pylrtrc."
        raise FileNotFoundError(msg) from exc


def cldprp(ds: xr.Dataset, param: str = "y") -> xr.Dataset:
    """Fill dataset with cloud optical properties."""
    iwc = 1.0
    re = ds["re"].to_numpy()
    mu = ds["mu"].to_numpy()
    if param == "y":
        habit = ds["habit"].to_numpy()
        input_str = "\n".join(
            [f"{m*1e3:.8f} {iwc:.8f} {r:.8f} {h:d}" for h, m, r in itertools.product(habit, mu, re)]
        )
    else:
        input_str = "\n".join(
            [f"{m*1e3:.8f} {iwc:.8f} {r:.8f}" for m, r in itertools.product(mu, re)]
        )

    with tempfile.NamedTemporaryFile(mode="w", delete=True, delete_on_close=False) as input:
        input.write(input_str)
        input.close()
        rundir = os.path.join(get_lrt_folder(), "bin")
        result = subprocess.run(
            ["./cldprp", f"-{param}", input.name], cwd=rundir, capture_output=True, check=False
        )

    if result.returncode != 0:
        msg = (
            f"cldprp exited with return code {result.returncode}. "
            f"Contents of stderr: {result.stderr.decode()}"
        )
        raise ChildProcessError(msg)

    shape = (-1, 9) if param in ["k", "y"] else (-1, 7) if param in ["f"] else (-1, 6)
    out = np.fromstring(result.stdout.decode(), sep=" ").reshape(shape)
    iwc_out = out[:, 1]
    re_out = out[:, 2]
    beta = out[:, 3]
    g = out[:, 4]
    omega = out[:, 5]
    q = 1e-6 * 4.0 * constants.rho_ice * re_out * beta / (3.0 * iwc_out)

    if param == "y":
        shape_y_out = (ds.sizes["habit"], ds.sizes["mu"], ds.sizes["re"])
        ds_out = xr.Dataset(
            data_vars={
                "q_ext": (("habit", "mu", "re"), q.reshape(shape_y_out)),
                "omega": (("habit", "mu", "re"), omega.reshape(shape_y_out)),
                "g": (("habit", "mu", "re"), g.reshape(shape_y_out)),
            },
            coords={"habit": habit, "mu": mu, "re": re},
        )
        ds_out["habit"].attrs = ds["habit"].attrs
    else:
        shape_out = (ds.sizes["mu"], ds.sizes["re"])
        ds_out = xr.Dataset(
            data_vars={
                "q_ext": (("mu", "re"), q.reshape(shape_out)),
                "omega": (("mu", "re"), omega.reshape(shape_out)),
                "g": (("mu", "re"), g.reshape(shape_out)),
            },
            coords={"mu": mu, "re": re},
        )

    ds_out["q_ext"].attrs = {
        "long_name": "extinction efficiency",
        "units": "nondim",
    }
    ds_out["omega"].attrs = {
        "long_name": "single scattering albedo",
        "units": "nondim",
    }
    ds_out["g"].attrs = {
        "long_name": "asymmetry factor",
        "units": "nondim",
    }
    ds_out["mu"].attrs = ds["mu"].attrs
    ds_out["re"].attrs = ds["re"].attrs

    return ds_out


def prepare_input(
    locations: pd.DataFrame,
    atmospheres: pd.DataFrame,
    surfaces: pd.DataFrame,
    clouds: pd.DataFrame,
    options: dict[str, str],
    cachestore: CacheStore,
) -> list[str]:
    """Set up libRadtran input and return list of run directories."""

    scene_loc = locations.groupby(level=0, sort=True)
    scene_atm = atmospheres.groupby(level=0, sort=True)
    scene_sfc = surfaces.groupby(level=0, sort=True)
    scene_cld = clouds.groupby(level=0, sort=True)

    if not (
        scene_loc.groups.keys()
        == scene_atm.groups.keys()
        == scene_sfc.groups.keys()
        == scene_cld.groups.keys()
    ):
        msg = "Inputs provide data for different scenes"
        raise ValueError(msg)

    rundirs = []
    for (scene, location), (_, atmosphere), (_, surface), (_, cloud) in zip(
        scene_loc, scene_atm, scene_sfc, scene_cld, strict=True
    ):
        rundir = cachestore.path(str(scene))
        write_input(location, atmosphere, surface, cloud, options, rundir)
        rundirs.append(rundir)

    return rundirs


def _check_set_option(options: dict[str, str], key: str, value: str) -> None:
    """Set options with error on attempted overwrite."""
    if key in options:
        msg = f"Attempting to overwrite libRadtran option {key} with value {value}"
        raise ValueError(msg)
    options[key] = value


def write_input(
    location: pd.DataFrame,
    atmosphere: pd.DataFrame,
    surface: pd.DataFrame,
    cloud: pd.DataFrame,
    options: dict[str, str],
    rundir: str,
) -> None:
    """Write libRadtran input to run directory."""
    os.makedirs(rundir, exist_ok=True)

    def _path(name: str) -> str:
        return os.path.join(rundir, name)

    # avoid modifying input
    options = copy(options)

    # Location
    if n := len(location) != 1:
        msg = f"Expecting exactly one set of coordinates but found {n}"
        raise ValueError(msg)
    data = location.iloc[0]
    for key, value in data.items():
        _check_set_option(options, key, value)

    # Atmosphere
    if n := len(atmosphere) != 1:
        msg = f"Expecting exactly one set of atmosphere profiles but found {n}"
        raise ValueError(msg)
    data = atmosphere.iloc[0]
    path = _path("atmosphere")
    with open(path, "wb") as f:
        atmstr = "\n".join(
            [
                " {:.8f} {:.8f} {:.8f} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}".format(
                    data["z"][alt],
                    data["p"][alt],
                    data["t"][alt],
                    data["n"][alt],
                    data["n_o3"][alt],
                    data["n_o2"][alt],
                    data["n_v"][alt],
                    data["n_co2"][alt],
                )
                for alt in range(len(data["z"]))
            ]
        )
        f.write(atmstr.encode("ascii"))
    _check_set_option(options, "atmosphere_file", path)

    # Surface
    if n := len(surface) != 1:
        msg = f"Expecting exactly one set of surface properties but found {n}"
        raise ValueError(msg)
    data = surface.iloc[0]
    for key, value in data.items():
        _check_set_option(options, key, value)

    # Cloud profiles
    for (_, name), data in cloud.iterrows():
        path = _path(name)
        with open(path, "wb") as f:
            cloudstr = "\n".join(
                [
                    " {:.8f} {:.8f} {:.8f}".format(
                        data["z"][alt], data["cwc"][alt], data["re"][alt]
                    )
                    for alt in range(len(data["cwc"]))
                ]
            )
            f.write(cloudstr.encode("ascii"))
        _check_set_option(options, f"profile_file {name}", " ".join(["1D", path]))
        for opt in data["options"]:
            words = opt.split()
            key = " ".join([words[0], name])
            value = " ".join(words[1:])
            _check_set_option(options, key, value)

    # Input file
    inputstr = "\n".join([f"{key} {value}" for key, value in options.items()])
    stdin_log = _path("stdin")
    with open(stdin_log, "wb") as f:
        f.write(inputstr.encode("ascii"))


def run(rundir: str) -> float:
    """Run libRadtran."""

    start = time.perf_counter()

    def _path(name: str) -> str:
        return os.path.join(rundir, name)

    lrt_rundir = os.path.join(get_lrt_folder(), "bin")
    stdin_log = _path("stdin")
    stdout_log = _path("stdout")
    stderr_log = _path("stderr")
    with (
        open(stdin_log) as stdin,
        open(stdout_log, "w") as stdout,
        open(stderr_log, "w") as stderr,
    ):
        result = subprocess.run(
            ["./uvspec"], stdin=stdin, stdout=stdout, stderr=stderr, cwd=lrt_rundir, check=False
        )
        if result.returncode != 0:
            msg = (
                f"libRadtran calculation exited with return code {result.returncode}. "
                f"Check input at {stdin_log} and logs at {stdout_log} and {stderr_log}."
            )
            raise ChildProcessError(msg)

    return time.perf_counter() - start


def parse_stdout(path: str) -> npt.NDArray[np.float64]:
    """Parse libRadtran output file."""
    fname = os.path.join(path, "stdout")
    return np.loadtxt(fname).astype(np.float64)


def apply_expand_index(
    df: pd.DataFrame, fun: Callable[[pd.Series], pd.DataFrame], by: str | list[str] | None = None
) -> pd.DataFrame:
    """Apply function that returns DataFrame to rows of a DataFrame."""
    msg = "Fix this to use iterrows instead of groupby!"
    raise ValueError(msg)
    groups = df.groupby(by or df.index)
    if groups.size().max() > 1:
        msg = "DataFrame must have unique index or unique grouping key must be provided."
        raise ValueError(msg)
    return groups.apply(lambda df: fun(df.iloc[0]))
