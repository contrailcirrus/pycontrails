"""LibRadtran utilities."""

import itertools
import os
import subprocess
import tempfile
from copy import copy
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


def run(
    location: dict[str, Any],
    met_profile: dict[str, Any],
    surface_options: dict[str, Any],
    cloud_profiles: list[dict[str, Any]],
    output_dir: str,
    options: dict[str, str],
) -> Any:
    """Run libRadtran."""

    os.makedirs(output_dir, exist_ok=True)

    def _path(name: str) -> str:
        return os.path.join(output_dir, name)

    folder = get_lrt_folder()
    options = copy(options)  # avoid modifying input

    # Location
    for key, value in location.items():
        if key in options:
            msg = f"Attempting to override {key} with value from location"
            raise ValueError(msg)
        options[key] = value

    # Atmosphere
    path = _path("atmosphere")
    with open(path, "wb") as f:
        atmstr = "\n".join(
            [
                " {:.8f} {:.8f} {:.8f} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}".format(
                    met_profile["z"][alt],
                    met_profile["p"][alt],
                    met_profile["t"][alt],
                    met_profile["n"][alt],
                    met_profile["n_o3"][alt],
                    met_profile["n_o2"][alt],
                    met_profile["n_v"][alt],
                    met_profile["n_co2"][alt],
                )
                for alt in range(len(met_profile["z"]))
            ]
        )
        f.write(atmstr.encode("ascii"))
    options["atmosphere_file"] = path

    # Surface
    for key, value in surface_options.items():
        if key in options:
            msg = f"Attempting to override {key} with value from surface_options"
            raise ValueError(msg)
        options[key] = value

    # Cloud profiles
    for i, profile in enumerate(cloud_profiles):
        name = f"cloud_profile_{i}"
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
        options[f"profile_file {name}"] = " ".join(["1D", path])
        for opt in profile["options"]:
            words = opt.split()
            key = " ".join([words[0], name])
            value = " ".join(words[1:])
            options[key] = value

    inputstr = "\n".join([f"{key} {value}" for key, value in options.items()])
    stdin_log = _path("stdin")
    with open(stdin_log, "wb") as f:
        f.write(inputstr.encode("ascii"))

    rundir = os.path.join(folder, "bin")
    stdout_log = _path("stdout")
    stderr_log = _path("stderr")
    with (
        open(stdin_log) as stdin,
        open(stdout_log, "w") as stdout,
        open(stderr_log, "w") as stderr,
    ):
        result = subprocess.run(
            ["./uvspec"], stdin=stdin, stdout=stdout, stderr=stderr, cwd=rundir, check=False
        )
        if result.returncode != 0:
            msg = (
                f"libRadtran calculation exited with return code {result.returncode}. "
                f"Check input at {stdin_log} and logs at {stdout_log} and {stderr_log}."
            )
            raise ChildProcessError(msg)

    return output_dir


def parse_stdout(path: str) -> npt.NDArray[np.float64]:
    """Parse libRadtran output file."""
    fname = os.path.join(path, "stdout")
    return np.loadtxt(fname).astype(np.float64)
