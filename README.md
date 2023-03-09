# **pycontrails**

> Python library for contrail modeling and analysis.

[![Unit test](https://github.com/contrailcirrus/pycontrails/actions/workflows/test.yml/badge.svg)](https://github.com/contrailcirrus/pycontrails/actions/workflows/test.yml)
[![Docs](https://github.com/contrailcirrus/pycontrails/actions/workflows/docs.yml/badge.svg)](https://github.com/contrailcirrus/pycontrails/actions/workflows/docs.yml)
[![Chat](https://img.shields.io/matrix/pycontrails-community:matrix.org.svg?label=Chat&logo=matrix)](https://matrix.to/#/#pycontrails-community:matrix.org)

## Documentation

**pycontrails** documentation available at [https://py.contrails.earth](https://py.contrails.earth/).

See [docs/README.md](docs/README.md) for documentation development instructions.

## Requires

- [git](https://git-scm.com/)
- [Python 3.x](https://www.python.org/downloads/) (3.9 or later)

## Environment

Create a dedicated virtual environment:

```bash
# create environment in <DIR>
$ python3 -m venv <DIR>

# activate environment (Unix-like)
$ source <DIR>/bin/activate
```

If using [Anaconda](https://www.anaconda.com/) / [Miniconda](https://docs.conda.io/en/latest/miniconda.html) Python, create a dedicated Anaconda environment:

```bash
# create conda environment
$ conda create -n contrails python=3.10

# activate environment
$ conda activate contrails
```

## Install

After activating the virtual environment, clone the [pycontrails repository](https://github.com/contrailcirrus/pycontrails) onto your machine:

```bash
$ cd <install-path>
$ git clone git@github.com:contrailcirrus/pycontrails.git
$ cd pycontrails
```

Install dependencies individually using `pip`:

```bash
# Core dependencies
$ pip install -e .             # Base installation

# Optional dependencies
$ pip install -e ".[ecmwf]"    # ECMWF datalib interfaces
$ pip install -e ".[gfs]"      # GFS datalib interfaces
$ pip install -e ".[gcp]"      # Google Cloud Platform caching interface
$ pip install -e ".[vis]"      # For polygon construction methods and plotting support 
$ pip install -e ".[zarr]"     # Load data from remote Zarr stores
$ pip install -e ".[jupyter]"  # Install Jupyter lab
$ pip install -e ".[dev]"      # Development support

# These packages may not support the latest python version
$ pip install -e ".[accf]"     # ACCF model support
$ pip install -e ".[open3d]"   # For polyhedra construction methods
```

Or install all dependencies with shortcut:

```bash
$ make pip-install  # install each dependency group
```

## Extensions

### BADA

To install the [pycontrails-BADA](https://github.com/contrailcirrus/pycontrails-bada) extension, run:

```
$ pip install -e ".[bada]"
```


## Develop

> [GNU Make](https://www.gnu.org/software/make/) is used for scripting tasks.
> See [Makefile](Makefile) for the source of each recipe.

After activating the virtual environment, clone the [pycontrails repository](https://github.com/contrailcirrus/pycontrails) onto your machine:

```bash
$ cd <install-path>
$ git clone https://github.com/contrailcirrus/pycontrails.git
$ cd pycontrails
```

### Setup

#### pip

From the root of the git repository, run setup using:

```bash
$ make dev-install  # install pip dependencies and pre-commit hooks
```

### Test

Lint the repository with `flake8`:

```bash
$ make flake8
```

Run type checking with `mypy`:

```bash
$ make mypy
```

Run unit tests with `pytest`:

```bash
$ make pytest
```

Run all code quality checks in local environment:

```bash
$ make test
```

## Common Issues

> To Add

## License

> Currently unlicensed
