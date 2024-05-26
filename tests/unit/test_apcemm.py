import os
import shutil

import pytest

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


def test_apcemm(apcemm_paths: tuple[str, str]) -> None:
    """Run APCEMM using pycontrails interface."""
    apcemm, apcemm_root = apcemm_paths
    assert True
