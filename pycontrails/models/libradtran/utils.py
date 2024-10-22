"""LibRadtran utilities."""

import os
import subprocess
from typing import Any

import numpy as np
import numpy.typing as npt

DEFAULT_OPTIONS = {
    "rte_solver": "disort",
    "source": "thermal",
    "mol_abs_param": "reptran fine",
    "number_of_streams": "16",
    "zout": "TOA",
    "umu": "1",
    "phi": "0",
    "output_user": "lambda uu",
    "output_quantity": "brightness",
}


def get_lrt_folder() -> str:
    """Get libRadtran root directory."""
    user_location = os.path.expanduser("~/.pylrtrc")
    try:
        with open(user_location) as f:
            return f.read().strip()
    except FileNotFoundError as exc:
        msg = "No default location for LibRadTran found. Place the path in ~/.pylrtrc."
        raise FileNotFoundError(msg) from exc


def run(
    location: dict[str, Any],
    met_profile: dict[str, Any],
    surface_options: dict[str, Any],
    cloud_profiles: list[dict[str, Any]],
    output_dir: str,
    static_options: dict[str, Any],
) -> Any:
    """Run libRadtran."""

    os.makedirs(output_dir, exist_ok=True)

    def _path(name: str) -> str:
        return os.path.join(output_dir, name)

    folder = get_lrt_folder()
    options = DEFAULT_OPTIONS | static_options

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
