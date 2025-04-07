# pycontrails

> Python library for modeling aviation climate impacts

|               |                                                                   |
|---------------|-------------------------------------------------------------------|
| **Version**   | [![PyPI version](https://img.shields.io/pypi/v/pycontrails.svg)](https://pypi.python.org/pypi/pycontrails)  [![conda-forge version](https://anaconda.org/conda-forge/pycontrails/badges/version.svg)](https://anaconda.org/conda-forge/pycontrails) [![Supported python versions](https://img.shields.io/pypi/pyversions/pycontrails.svg)](https://pypi.python.org/pypi/pycontrails) |
| **Citation**  | [![DOI](https://zenodo.org/badge/617248930.svg)](https://zenodo.org/badge/latestdoi/617248930) |
| **Tests**     | [![Unit test](https://github.com/contrailcirrus/pycontrails/actions/workflows/test.yaml/badge.svg)](https://github.com/contrailcirrus/pycontrails/actions/workflows/test.yaml) [![Docs](https://github.com/contrailcirrus/pycontrails/actions/workflows/docs.yaml/badge.svg?event=push)](https://github.com/contrailcirrus/pycontrails/actions/workflows/docs.yaml) [![Release](https://github.com/contrailcirrus/pycontrails/actions/workflows/release.yaml/badge.svg)](https://github.com/contrailcirrus/pycontrails/actions/workflows/release.yaml) [![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/contrailcirrus/pycontrails/badge)](https://securityscorecards.dev/viewer?uri=github.com/contrailcirrus/pycontrails)|
| **License**   | [![Apache License 2.0](https://img.shields.io/pypi/l/pycontrails.svg)](https://github.com/contrailcirrus/pycontrails/blob/main/LICENSE) |
| **Community** | [![Github Discussions](https://img.shields.io/github/discussions/contrailcirrus/pycontrails)](https://github.com/contrailcirrus/pycontrails/discussions) [![Github Issues](https://img.shields.io/github/issues/contrailcirrus/pycontrails)](https://github.com/contrailcirrus/pycontrails/issues) [![Github PRs](https://img.shields.io/github/issues-pr/contrailcirrus/pycontrails)](https://github.com/contrailcirrus/pycontrails/pulls) |

**pycontrails** is an open source project and Python package for modeling aircraft contrails and other
aviation related climate impacts.

`pycontrails` defines common [data structures](https://py.contrails.org/api.html#data) and [interfaces](https://py.contrails.org/api.html#datalib) to efficiently build and run [models](https://py.contrails.org/api.html#models) of aircraft performance, emissions, and radiative forcing.

## Documentation

Documentation and examples available at [py.contrails.org](https://py.contrails.org/).

<!-- Try out an [interactive Colab Notebook](). -->

## Install

### Install with pip

You can install pycontrails from PyPI with `pip` (Python 3.10 or later required):

```bash
$ pip install pycontrails

# install with all optional dependencies
$ pip install "pycontrails[complete]"
```

Install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/contrailcirrus/pycontrails.git
```

### Install with conda

You can install pycontrails from the [conda-forge](https://conda-forge.org/) channel with `conda` (or other `conda`-like package managers such as `mamba`):

```bash
conda install -c conda-forge pycontrails
```

The conda-forge package includes all optional runtime dependencies.

See more installation options in the [install documentation](https://py.contrails.org/install).

## Get Involved

- Ask questions, discuss models, and present ideas in [GitHub Discussions](https://github.com/contrailcirrus/pycontrails/discussions).
- Report bugs or suggest changes in [GitHub Issues](https://github.com/contrailcirrus/pycontrails/issues).
- Review the [contributing guidelines](https://py.contrails.org/contributing.html) and contribute improvements as [Pull Requests](https://github.com/contrailcirrus/pycontrails/pulls).

## License

[Apache License 2.0](https://github.com/contrailcirrus/pycontrails/blob/main/LICENSE)

Additional attributions in [NOTICE](https://github.com/contrailcirrus/pycontrails/blob/main/NOTICE).
