
Install
=======

Requires
--------

- Python (3.9 or later)

Core
----

Install the latest release using ``pip``:

.. code-block:: bash

    # core installation
    $ pip install pycontrails

    # install with all optional dependencies
    $ pip install "pycontrails[complete]"


Install the latest development version directly from GitHub:

.. code-block:: bash

    $ pip install git+https://github.com/contrailcirrus/pycontrails.git


Optional Dependencies
---------------------

The ``pycontrails`` package uses optional dependencies for specific features:

.. code-block:: bash

    # install all non-development optional dependencies
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


Extensions
----------

Some features of ``pycontrails`` are written as extensions that can be added manually:

.. _bada-install:

BADA
~~~~

    This extension is private due to license restrictions

`pycontrails-BADA <https://github.com/contrailcirrus/pycontrails-bada>`__ is an extension to
interface with `BADA <https://www.eurocontrol.int/model/bada>`__ aircraft performance data.

.. code-block:: bash

    pip install "pycontrails-bada @ git+ssh://git@github.com/contrailcirrus/pycontrails-bada.git"

Cirium
~~~~~~

    This extension is private due to license restrictions

`pycontrails-cirium <https://github.com/contrailcirrus/pycontrails-cirium>`__ is an extension
to a `Cirium <https://www.cirium.com/>`__ database of jet engines.

.. code-block:: bash

    pip install "pycontrails-cirium @ git+ssh://git@github.com/contrailcirrus/pycontrails-cirium.git"

.. _accf-install:

ACCF
~~~~

Interface to the DLR / UMadrid `ACCF model <https://gmd.copernicus.org/preprints/gmd-2022-203/>`__
using a forked version of the `climaccf repository <https://github.com/dlr-pa/climaccf>`__.

.. code-block:: bash

    pip install "climaccf @ git+ssh://git@github.com/contrailcirrus/climaccf.git"
