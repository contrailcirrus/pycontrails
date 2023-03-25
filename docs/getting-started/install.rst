
Installation
============

Requires
--------

- Python (3.9 or later)

Install
-------

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
~~~~~~~~~~~~~~~~~~~~~

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
    # They are excluded from the "complete" set
    $ pip install "pycontrails[open3d]"   # Polyhedra contruction methods

See ``[project.optional-dependencies]`` `pyproject.toml <https://github.com/contrailcirrus/pycontrails/blob/main/pyproject.toml>`_
for the latest optional dependencies.


Extensions
~~~~~~~~~~

``pycontrails`` includes extensions that can be added manually

BADA
""""

    This extension is private due to license restrictions

`pycontrails-BADA <https://github.com/contrailcirrus/pycontrails-bada>`_ is an extension to
interface with `BADA <https://www.eurocontrol.int/model/bada>`_ aircraft performance models.

.. code-block:: bash

    pip install "pycontrails-bada @ git+ssh://git@github.com/contrailcirrus/pycontrails-bada.git"

Cirium
""""""

    This extension is private due to license restrictions

`pycontrails-cirium <https://github.com/contrailcirrus/pycontrails-cirium>`_ is an extension
to include a `Cirium <https://www.cirium.com/>`_ database of jet engines.

.. code-block:: bash

    pip install "pycontrails-cirium @ git+ssh://git@github.com/contrailcirrus/pycontrails-cirium.git"

ACCF
""""

``pycontrails`` includes an interface to the DLR / UMadrid
`ACCF model <https://gmd.copernicus.org/preprints/gmd-2022-203/>`_
using a forked version of the `climaccf repository <https://github.com/dlr-pa/climaccf>`_.

To run :ref:`pycontrails.models.accf` model, you must install the `climaccf` dependency:

.. code-block:: bash

    pip install "climaccf @ git+ssh://git@github.com/contrailcirrus/climaccf.git"


Develop
-------

Requires
~~~~~~~~

- `git <https://git-scm.com/>`_
- `Make <https://www.gnu.org/software/make/>`_. See `Makefile <https://github.com/contrailcirrus/pycontrails/blob/main/Makefile>`_ for a list of ``make`` commands.

The documentation requires the additional dependencies:

- `pandoc <https://pandoc.org/installing.html>`_ for interpreting Jupyter notebooks
- `LaTeX <https://www.latex-project.org/get/>`_ for pdf outputs.
  If you are using a Mac, `MacTeX <https://www.tug.org/mactex/index.html>`_ is the best option.
  Note that LaTeX can be fairly large to install (~6GB).

Environment
~~~~~~~~~~~

Create a dedicated virtual environment for development:

.. code-block:: bash

    # create environment in <DIR>
    $ python3 -m venv <DIR>

    # activate environment (Unix-like)
    $ source <DIR>/bin/activate

If using `Anaconda <https://www.anaconda.com/>`_ / `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
Python, create a dedicated Anaconda environment:

.. code-block:: bash

    # create conda environment
    $ conda create -n contrails python=3.10

    # activate environment
    $ conda activate contrails


Development Install
~~~~~~~~~~~~~~~~~~~

After activating the virtual environment, clone the `pycontrails repository <https://github.com/contrailcirrus/pycontrails>`_:

.. code-block:: bash

    $ cd <install-path>
    $ git clone git@github.com:contrailcirrus/pycontrails.git
    $ cd pycontrails

Install the development verison of ``pycontrails`` using ``make``:

.. code-block:: bash

    $ make dev-install

Install dependencies manually using ``pip`` in editable mode:

.. code-block:: bash

    # core development installation
    $ pip install -e ".[docs,dev]"

    # install optional dependencies as above
    $ pip install -e ".[ecmwf,gfs]"

    # make sure to add the pre-commit hooks if installing manually
    $ pre-commit install


Test
~~~~

Run all code quality checks and unit tests:

.. code-block:: bash

    $ make test

Lint the repository with ``flake8``:

.. code-block:: bash

    $ make flake8

Autoformat the repository with ``black``:

.. code-block:: bash

    $ make black

Run type checking with ``mypy``:

.. code-block:: bash

    $ make mypy

Run unit tests with ``pytest``:

.. code-block:: bash

    $ make pytest


Documentation
~~~~~~~~~~~~~

Documentation is written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`_
and built with `Sphinx <https://www.sphinx-doc.org/en/master/>`_.

Sphinx configuration is stored in `conf.py <https://github.com/contrailcirrus/pycontrails/blob/main/docs/conf.py>`_.
The full list of configuration options is in the `Sphinx configuration docs <https://www.sphinx-doc.org/en/master/usage/configuration.html>`_.

Build HTML documentation:

.. code-block:: bash

    # docs build to directory docs/_build/html
    $ make docs-build

    # automatically build docs on changes
    # docs will be served at http://127.0.0.1:8000
    $ make docs-serve

    # cleanup all built documentation
    $ make docs-clean

Build manually with ``sphinx-build``:

.. code-block:: bash

    $ sphinx-build -b html docs docs/_build/html      # HTML output

Sphinx caches builds between changes, which can lead to certain pages not updating.
To force the whole site to rebuild, use the options ``-aE``:

.. code-block:: bash

    $ sphinx-build -aE -b html docs docs/_build/html  # rebuild all output

See `sphinx-build <https://www.sphinx-doc.org/en/master/man/sphinx-build.html#cmdoption-sphinx-build-b>`_
for a list of all the possible output builders.


PDF Output
""""""""""

    Building PDF output requires a `LaTeX distribution https://www.latex-project.org/get/`_.

Build pdf documentation:

.. code-block:: bash

    $ make docs-pdf

A single pdf output (i.e. ``pycontrails.pdf``) will be built within ``docs/_build/latex``.

To build manually, run:


.. code-block:: bash

    $ sphinx-build -b latex docs docs/_build/latex
    $ cd docs/_build/latex
    $ make

References
""""""""""

Bibliography references managed in a `Zotero library <https://www.zotero.org/groups/4730892/pycontrails/library>`_.

To automatically sync this library with the
`docs/_static/pycontrails.bib <https://github.com/contrailcirrus/pycontrails/blob/main/docs/_static/pycontrails.bib>`_ Bibtex file:

- Install `Zotero <https://www.zotero.org/>`_ and add the `pycontrails collection <https://www.zotero.org/groups/4730892/pycontrails/library>`_.
- Install the `Zotero Better Bibtex extension <https://retorque.re/zotero-better-bibtex/installation/>`_. Leave defaults during setup.
- Right click on the **pycontrails** library and select *Export Library*
- Export as *Better Bibtex*. You can optionally check *Keep Updated* if you want
  this file to update every time you make a change to the library.
- Select the file ``_static/pycontrails.bib`` and press *Save* to overwrite the file.
- Commit the updated ``_static/pycontrails.bib``
