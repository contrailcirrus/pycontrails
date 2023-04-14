"""Temp utilities."""

import logging
import os
import tempfile
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


def temp_filename() -> str:
    """Get a filename in the host computers temp directory.

    More robust than using tempfile.NamedTemporaryFile()

    Returns
    -------
    str
        Temp filename

    See Also
    --------
    temp_file : Context manager for temp file creation and cleanup
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
        logger.warning("Failed to delete temp file %s with error %s", temp_filename, e)


@contextmanager
def temp_file() -> Generator[str, None, None]:
    """Get context manager for temp file creation and cleanup."""
    filename = temp_filename()
    try:
        yield filename
    finally:
        remove_tempfile(filename)
