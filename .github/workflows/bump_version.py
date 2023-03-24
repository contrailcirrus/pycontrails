"""Bump version in pyproject.toml."""

import argparse
import pathlib
import re

PYPROJECT_TOML = pathlib.Path(__file__).parents[2] / "pyproject.toml"


def check_version_format(version: str) -> str:
    """Check that version has the form v1.2.3.

    Returns the version without the leading 'v'.
    """
    if not version.startswith("v"):
        raise ValueError("Version must start with 'v'")
    version = version[1:]
    if not re.match(r"\d+\.\d+\.\d+", version):
        raise ValueError("Version must be in the form x.y.z")
    return version


def check_and_bump_version(version: str) -> None:
    """Check if the new version is consecutive with the current one and bump it."""

    # get target version
    version = check_version_format(version)
    target = version.split(".")
    tmajor, tminor, tpatch = (int(v) for v in target)

    # get version in pyproject.toml
    txt = PYPROJECT_TOML.read_text()
    match = re.search(r"version\s*=\s*\"(\d+)\.(\d+)\.(\d+)\"", txt)
    assert match is not None
    major, minor, patch = (int(v) for v in match.groups())

    # check if the new version is compatible with the current one
    # this doesn't handle tags like "1.0.0rc1" or "1.0.0.dev1"
    is_bump_patch = tmajor == major and tminor == minor and tpatch == patch + 1
    is_bump_minor = tmajor == major and tminor == minor + 1 and tpatch == 0
    is_bump_major = tmajor == major + 1 and tminor == 0 and tpatch == 0

    if not (is_bump_patch or is_bump_minor or is_bump_major):
        raise ValueError(
            "Versions must be consecutive!"
            f" The current version is '{major}.{minor}.{patch}',"
            f" the target version is '{version}'."
        )

    # bump to target version
    print(f"Updating version from {major}.{minor}.{patch} to {version}")
    txt = txt.replace(match.group(), f'version = "{version}"')
    PYPROJECT_TOML.write_text(txt)


parser = argparse.ArgumentParser()
parser.add_argument("version", help="New version; should start with 'v'")
args = parser.parse_args()

check_and_bump_version(args.version)
