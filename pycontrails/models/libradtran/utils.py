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
    scene_options: pd.DataFrame,
    profiles: pd.DataFrame,
    clouds: pd.DataFrame | None,
    static_options: dict[str, str],
    cachestore: CacheStore,
) -> list[str]:
    """Set up libRadtran input and return list of run directories."""

    if not scene_options.index.equals(profiles.index):
        msg = "Inputs `scene_options` and `profiles` must share a common index."
        raise ValueError(msg)

    scene_clouds = clouds.groupby(level=0) if clouds is not None else None

    rundirs = []
    for scene, options in scene_options.iterrows():
        rundir = cachestore.path(str(scene))
        cloud_profiles = (
            scene_clouds.get_group(scene)
            if scene_clouds is not None and scene in scene_clouds.groups
            else None
        )
        write_input(options, profiles.loc[scene], cloud_profiles, static_options, rundir)
        rundirs.append(rundir)

    return rundirs


def _check_set_option(options: dict[str, str], key: str, value: str) -> None:
    """Set options with error on attempted overwrite."""
    if key in options:
        msg = f"Attempting to overwrite libRadtran option {key} with value {value}"
        raise ValueError(msg)
    options[key] = value


def write_input(
    scene_options: pd.Series,
    profiles: pd.Series,
    clouds: pd.DataFrame | None,
    static_options: dict[str, str],
    rundir: str,
) -> None:
    """Write libRadtran input to run directory."""
    os.makedirs(rundir, exist_ok=True)

    def _path(name: str) -> str:
        return os.path.join(rundir, name)

    # avoid modifying input
    options = copy(static_options)

    # Scene options
    for key, value in scene_options.items():
        _check_set_option(options, key, value)

    # Atmosphere
    path = _path("atmosphere")
    with open(path, "wb") as f:
        atmstr = "\n".join(
            [
                " {:.8f} {:.8f} {:.8f} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}".format(
                    profiles["z"][alt],
                    profiles["p"][alt],
                    profiles["t"][alt],
                    profiles["n"][alt],
                    profiles["n_o3"][alt],
                    profiles["n_o2"][alt],
                    profiles["n_v"][alt],
                    profiles["n_co2"][alt],
                )
                for alt in range(len(profiles["z"]))
            ]
        )
        f.write(atmstr.encode("ascii"))
    _check_set_option(options, "atmosphere_file", path)

    # Cloud profiles
    cloud_iter = clouds.iterrows() if clouds is not None else []
    for (_, name), profile in cloud_iter:
        path = _path(name)
        with open(path, "wb") as f:
            cloudstr = "\n".join(
                [
                    " {:.8f} {:.8f} {:.8f}".format(
                        profile["z"][alt], profile["cwc"][alt], profile["re"][alt]
                    )
                    for alt in range(len(profile["cwc"]))
                ]
            )
            f.write(cloudstr.encode("ascii"))
        _check_set_option(options, f"profile_file {name}", " ".join(["1D", path]))
        for opt in profile["options"]:
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
    df: pd.DataFrame, fun: Callable[[pd.Series], pd.DataFrame | None]
) -> pd.DataFrame | None:
    """Apply function that returns a DataFrame to rows of a DataFrame."""
    out = []
    for idx, row in df.iterrows():
        row_out = fun(row)
        if row_out is None or len(row_out) == 0:
            continue
        row_out = pd.concat({idx: row_out})
        out.append(row_out)
    if len(out) > 0:
        return pd.concat(out)
    return None
