"""Unit tests for the package."""

import pathlib


def get_static_path(filename: str | pathlib.Path) -> pathlib.Path:
    """Return a path to file in ``/tests/static/`` directory.

    Parameters
    ----------
    filename : str | pathlib.Path
        Filename to prefix

    Returns
    -------
    pathlib.Path
    """
    parent = pathlib.Path(__file__).parent
    return parent / "static" / filename
