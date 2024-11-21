
Install
=======

conda install
-------------

Install the latest release from `conda-forge <https://conda-forge.org>`__ using ``conda``:

.. code-block:: bash

    $ conda install -c conda-forge pycontrails


The conda-forge package includes all optional runtime dependencies.

pip install
-----------

With Python 3.10 or later, install the latest release from PyPI using ``pip``:

.. code-block:: bash

    # core installation
    $ pip install pycontrails

    # install with all optional dependencies
    $ pip install "pycontrails[complete]"

Wheels are currently built and tested for python 3.10 - 3.13 on Linux, macOS, and Windows.

Install the latest development version directly from GitHub:

.. code-block:: bash

    $ pip install git+https://github.com/contrailcirrus/pycontrails.git


Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

The ``pycontrails`` package uses optional dependencies for specific features:

.. code-block:: bash

    # install all optional runtime dependencies
    $ pip install "pycontrails[complete]"

    # install specific optional dependencies
    $ pip install "pycontrails[dev]"      # Development dependencies
    $ pip install "pycontrails[docs]"     # Documentation / Sphinx dependencies
    $ pip install "pycontrails[ecmwf]"    # ECMWF datalib interfaces
    $ pip install "pycontrails[gcp]"      # Google Cloud Platform caching interface
    $ pip install "pycontrails[gfs]"      # GFS datalib interfaces
    $ pip install "pycontrails[jupyter]"  # Jupyter notebook and lab interface
    $ pip install "pycontrails[vis]"      # Polygon construction methods and plotting support
    $ pip install "pycontrails[zarr]"     # Load data from remote Zarr stores

    # These packages may not support the latest python version
    # and are excluded from "complete"
    $ pip install "pycontrails[open3d]"   # Polyhedra contruction methods

See ``[project.optional-dependencies]`` in `pyproject.toml <https://github.com/contrailcirrus/pycontrails/blob/main/pyproject.toml>`__
for the latest optional dependencies.


Pre-built wheels
~~~~~~~~~~~~~~~~

Wheels for common platforms are available on `PyPI <https://pypi.org/project/pycontrails/>`__. Currently, wheels are available for:

- Linux (x86_64)
- macOS (x86_64 and arm64)
- Windows (x86_64)

It's possible to build the wheels from source using `Cython <https://cython.org/>`__ and a C compiler with the usual ``pip install`` process. The source distribution on PyPI includes the C files, so it's not necessary to have Cython installed to build from source.


Extensions
----------

Some features of ``pycontrails`` are written as extensions that can be added manually:

.. _bada-install:

BADA
~~~~

    This extension is private due to license restrictions

`pycontrails-bada <https://github.com/contrailcirrus/pycontrails-bada>`__ is an extension to
interface with `BADA <https://www.eurocontrol.int/model/bada>`__ aircraft performance data.

Reach out to `info@contrails.org <mailto:info@contrails.org>`__ to request access.

Once provided access, install using:

1. Follow the instructions for your platform to install the `gcloud CLI <https://cloud.google.com/sdk/docs/install>`__
2. Login to `gcloud`. For alternate auth methods, see `Google Artifact Registry keyring docs <https://cloud.google.com/artifact-registry/docs/python/authentication#keyring>`__

.. code-block:: bash

    gcloud auth login

3. Install `keyring <https://pypi.org/project/keyring/>`__ for Google Artifact Registry

.. code-block:: bash

    pip install keyring keyrings.google-artifactregistry-auth

4. Install ``pycontrails-bada`` package:

.. code-block:: bash

    pip install --index-url https://us-central1-python.pkg.dev/contrails-301217/pycontrails/simple \
        pycontrails-bada

.. code-block:: bash

    # or at a tag
    pip install --index-url https://us-central1-python.pkg.dev/contrails-301217/pycontrails/simple \
        "pycontrails-bada==0.6.0"


Cirium
~~~~~~

    This extension is private due to license restrictions

`pycontrails-cirium <https://github.com/contrailcirrus/pycontrails-cirium>`__ is an extension
to the `Cirium <https://www.cirium.com/>`__ database of jet engines.

.. code-block:: bash

    pip install "pycontrails-cirium @ git+ssh://git@github.com/contrailcirrus/pycontrails-cirium.git"

.. _accf-install:

ACCF
~~~~

Interface to DLR / UMadrid `ACCF model <https://gmd.copernicus.org/preprints/gmd-2022-203/>`__.

.. code-block:: bash

    pip install "climaccf @ git+ssh://git@github.com/dlr-pa/climaccf"
