"""LibRadtran utilities."""

import itertools
import os
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

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


def write_atmosphere_file(
    path: str,
    z: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    n: npt.NDArray[np.float64],
    n_o3: npt.NDArray[np.float64],
    n_o2: npt.NDArray[np.float64],
    n_v: npt.NDArray[np.float64],
    n_co2: npt.NDArray[np.float64],
) -> None:
    """Write libRadtran atmosphere file."""
    with open(path, "wb") as f:
        atmstr = "\n".join(
            [
                (
                    f" {z[alt]/1e3:.8f} {p[alt]/1e2:.8f} {t[alt]:.8f}"
                    f" {n[alt]/1e6:.6e} {n_o3[alt]/1e6:.6e} {n_o2[alt]/1e6:.6e}"
                    f" {n_v[alt]/1e6:.6e} {n_co2[alt]/1e6:.6e}"
                )
                for alt in range(z.size)
            ]
        )
        f.write(atmstr.encode("ascii"))


def write_cloud_file(
    path: str,
    z: npt.NDArray[np.float64],
    cwc: npt.NDArray[np.float64],
    re: npt.NDArray[np.float64],
) -> None:
    """Write libRadtran cloud file."""
    with open(path, "wb") as f:
        cloudstr = "\n".join(
            [f" {z[alt]/1e3:.8f} {cwc[alt]*1e3:.8f} {re[alt]*1e6:.8f}" for alt in range(z.size)]
        )
        f.write(cloudstr.encode("ascii"))


def write_input(path: str, options: dict[str, str]) -> None:
    """Write libRadtran input file."""
    with open(path, "wb") as f:
        inputstr = "\n".join([f"{key} {value}" for key, value in options.items()])
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


def check_set(current: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Set entry with error on attempted overwrite."""
    if key in current:
        msg = f"Attempting to overwrite {key}: {current[key]} with value {value}"
        raise ValueError(msg)
    current[key] = value
    return current


def check_merge(current: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Merge dicts with error on attempted overwrite."""
    for key, value in new.items():
        check_set(current, key, value)
    return current
