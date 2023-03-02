"""Temp utilities."""

import logging
import os
import tempfile
from contextlib import contextmanager
from typing import Generator

LOG = logging.getLogger(__name__)


def temp_filename() -> str:
    """Get a filename in the host computers temp directory.

    More robust than using tempfile.NamedTemporaryFile()

    Returns
    -------
    str
        Temp filename
    """
    return os.path.join(tempfile.gettempdir(), os.urandom(24).hex())


def remove_tempfile(temp_filename: str) -> None:
    """Remove temp file.

    Parameters
    ----------
    temp_filename : str
        Temp filename
    """
    try:
        os.unlink(temp_filename)
    except OSError as e:
        LOG.debug(f"Failed to delete temp files with error: {e}")


@contextmanager
def temp_file() -> Generator[str, None, None]:
    """Get context manager for temp file creation and cleanup."""
    try:
        filename = temp_filename()
        yield filename
    finally:
        remove_tempfile(filename)
