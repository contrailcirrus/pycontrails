
Develop
=======

Requires
--------

- `git <https://git-scm.com/>`__
- `Make <https://www.gnu.org/software/make/>`__. See `Makefile <https://github.com/contrailcirrus/pycontrails/blob/main/Makefile>`__ for a list of ``make`` commands.

Developing documentation requires:

- `pandoc <https://pandoc.org/installing.html>`__ for interpreting Jupyter notebooks
- `LaTeX <https://www.latex-project.org/get/>`__ for pdf outputs.
  If you are using a Mac, `MacTeX <https://www.tug.org/mactex/index.html>`__ is the best option.
  Note that LaTeX can be fairly large to install (~6GB).

Environment
-----------

Create a dedicated virtual environment for development:

.. code-block:: bash

    # create environment in <DIR>
    $ python3 -m venv <DIR>

    # activate environment (Unix-like)
    $ source <DIR>/bin/activate

If using `Anaconda <https://www.anaconda.com/>`__ / `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__
Python, create a dedicated Anaconda environment:

.. code-block:: bash

    # create conda environment
    $ conda create -n contrails

    # activate environment
    $ conda activate contrails


Install
-------

After activating the virtual environment, clone the `pycontrails repository <https://github.com/contrailcirrus/pycontrails>`__:

.. code-block:: bash

    $ cd <install-path>
    $ git clone git@github.com:contrailcirrus/pycontrails.git
    $ cd pycontrails

These commands clone via SSH and may require `adding an SSH key to your GitHub account <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>`__.
Alternatively, you can clone via HTTPS by running

.. code-block:: bash

    $ cd <install-path>
    $ git clone https://github.com/contrailcirrus/pycontrails.git
    $ cd pycontrails

Install the development verison of ``pycontrails`` using ``make``:

.. code-block:: bash

    $ make dev-install

or install dependencies manually using ``pip`` in editable mode:

.. code-block:: bash

    # core development installation
    $ pip install -e ".[docs,dev]"

    # install optional dependencies as above
    $ pip install -e ".[ecmwf,gfs]"

    # make sure to add pre-commit hooks if installing manually
    $ pre-commit install


Test
----

Run all code quality checks and unit tests.
This is run in the `test workflow <https://github.com/contrailcirrus/pycontrails/blob/main/.github/workflows/test.yaml>`__,
but should also be run locally before submitting PRs:

.. code-block:: bash

    $ make test

Lint the repository with `ruff <https://docs.astral.sh/ruff/>`__:

.. code-block:: bash

    $ make lint

Autoformat the repository with `ruff <https://docs.astral.sh/ruff/formatter/>`__:

.. code-block:: bash

    $ make format

Run type checking with `mypy <https://www.mypy-lang.org/>`__:

.. code-block:: bash

    $ make mypy

Run unit tests with `pytest <https://docs.pytest.org/en/latest/>`__:

.. code-block:: bash

    $ make pytest

Run notebook validation with `nbval <https://github.com/computationalmodelling/nbval>`__:

.. code-block:: bash

    $ make nb-test

Run doctests with `pytest <https://docs.pytest.org/en/latest/>`__:

.. code-block:: bash

    $ make doctest

Notebook validation and doctests require `Copernicus Climate Data Store (CDS) credentials <api/pycontrails.datalib.ecmwf.ERA5.html>`__, and doctests additionally require `Google application credentials <https://cloud.google.com/docs/authentication/application-default-credentials>`__. If either are missing, the test suite will issue a warning and exit.

Documentation
-------------

Documentation is written in `reStructuredText (rst) <https://docutils.sourceforge.io/rst.html>`__
and built with `Sphinx <https://www.sphinx-doc.org/en/master/>`__. The `quick reStructuredText
reference <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`__
provides a decent rst syntax overview.

Sphinx includes many additional
`roles <https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html>`__,
`directives <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html>`__,
and
`extensions <https://www.sphinx-doc.org/en/master/usage/extensions/index.html>`__
to enhance documentation.

Sphinx configuration is written in `docs/conf.py <https://github.com/contrailcirrus/pycontrails/blob/main/docs/conf.py>`__.
See the `Sphinx configuration docs <https://www.sphinx-doc.org/en/master/usage/configuration.html>`__ for the full list of configuration options.

Build HTML documentation:

.. code-block:: bash

    # docs build to directory docs/_build/html
    $ make docs-build

    # automatically build docs on changes
    # docs will be served at http://127.0.0.1:8000
    $ make docs-serve

    # clean up built documentation
    $ make docs-clean

Build manually with ``sphinx-build``:

.. code-block:: bash

    $ sphinx-build -b html docs docs/_build/html      # HTML output

Sphinx caches builds between changes.
To force the whole site to rebuild, use the options ``-aE``:

.. code-block:: bash

    $ sphinx-build -aE -b html docs docs/_build/html  # rebuild all output

See `sphinx-build <https://www.sphinx-doc.org/en/master/man/sphinx-build.html#cmdoption-sphinx-build-b>`__
for a list of all the possible output builders.

Notebooks
~~~~~~~~~

Examples and tutorials should be written as isolated executable `Jupyter
Notebooks <https://jupyter.org/>`__. The
`nbsphinx <https://nbsphinx.readthedocs.io/en/0.9.1/>`__ extension
includes notebooks in the static documentation.

Notebooks will be automatically evaluated during tests, unless
explicitly ignored. To exclude a notebook cell from evaluation during
testing or automatic execution, `add the
tags <https://jupyterbook.org/en/stable/content/metadata.html#adding-tags-using-notebook-interfaces>`__
``nbval-skip`` and ``skip-execution`` to cell metadata.

To test notebooks locally, run:

.. code:: bash

   $ make nb-test

To re-execute all notebooks, run:

.. code:: bash

   $ make nb-execute

PDF Output
~~~~~~~~~~

    Building PDF output requires a `LaTeX distribution <https://www.latex-project.org/get/>`__.

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
~~~~~~~~~~

Literature references managed in the `pycontrails Zotero library <https://www.zotero.org/groups/4730892/pycontrails/library>`__.

The documentation uses
`sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html>`__
to include citations and a bibliography.

All references should be cited through documentation and docstrings
using the `:cite: role <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#role-cite>`__.


To automatically sync the Zotero library with the
`docs/_static/pycontrails.bib <https://github.com/contrailcirrus/pycontrails/blob/main/docs/_static/pycontrails.bib>`__ Bibtex file:

- Install `Zotero <https://www.zotero.org/>`__ and add the `pycontrails library <https://www.zotero.org/groups/4730892/pycontrails/library>`__.
- Install the `Zotero Better Bibtex extension <https://retorque.re/zotero-better-bibtex/installation/>`__. Leave defaults during setup.
- Right click on the **pycontrails** library and select *Export Library*
- Export as *Better Bibtex*. You can optionally check *Keep Updated* if you want
  this file to update every time you make a change to the library.
- Select the file ``_static/pycontrails.bib`` and press *Save* to overwrite the file.
- Commit the updated ``_static/pycontrails.bib``

Test
~~~~

    All doc tests first ensure ERA5 data is cached locally:

    .. code-block:: bash

        $ make ensure-era5-cached

Run rst linter with `doc8 <https://doc8.readthedocs.io/en/latest/readme.html>`__:

.. code-block:: bash

    $ make doc8

Run docstring example tests with `doctest <https://docs.python.org/3/library/doctest.html>`__:

.. code-block:: bash

    $ make doctest

Test notebook examples with `nbval pytest plugin <https://github.com/computationalmodelling/nbval>`__:

.. code:: bash

   $ make nb-test


Conventions
-----------

Code
~~~~

``pycontrails`` aims to implement clear, consistent, performant data
structures and models.

The project uses `mypy <http://mypy-lang.org/>`__ for static type
checking. All code should have specific, clear type annotations.

The project uses `Black <https://black.readthedocs.io/en/stable/>`__,
`ruff <https://github.com/charliermarsh/ruff>`__ and
`doc8 <https://doc8.readthedocs.io/en/latest/readme.html>`__ to
standardize code-formatting. These tools are run automatically in a
pre-commit hook.

The project uses `pytest <https://docs.pytest.org/en/latest/>`__ to run
unit tests. New code should include clear unit tests for implementation
and output values. New test files should be included in the
`/tests/unit/ directory <https://github.com/contrailcirrus/pycontrails/tree/main/tests/unit>`__.

The project uses `Github
Actions <https://github.com/contrailcirrus/pycontrails/actions>`__ to
run code quality and unit tests on each pull request. Test locally
before pushing using:

.. code:: bash

   $ make test

Docstrings
~~~~~~~~~~

Wherever possible, we adhere to the `NumPy docstring
conventions <https://numpydoc.readthedocs.io/en/latest/format.html>`__.

The following links are good references for writing *numpy* docstrings:

-  `numpydoc docstring
   guide <https://numpydoc.readthedocs.io/en/latest/format.html>`__
-  `pandas docstring
   guide <https://pandas.pydata.org/docs/development/contributing_docstring.html>`__
-  `scipy docstring
   guideline <https://docs.scipy.org/doc//scipy/dev/contributor/rendering_documentation.html#documentation-guidelines>`__

General guidelines:

.. code:: rst

   Use *italics*, **bold** and ``monospace`` if needed in any explanations
   (but not for variable names and doctest code or multi-line code).
   Variable, module, function, and class names
   should be written between single back-ticks (`numpy`).

When specifying types in **Parameters** or **See Also**, Sphinx will
automatically replace the text with the ``napolean_type_aliases``
specified in
`conf.py <https://github.com/contrailcirrus/pycontrails/blob/main/docs/conf.py>`__,
e.g.

.. code:: python

   """
   Parameters
   ----------
   x : np.ndarray
       Sphinx will automatically replace
       "np.ndarray" with the napolean type alias "numpy.ndarray"
   """

The **See Also** section is *not a list*. All of the following work:

.. code:: python

   """
   See Also
   --------
   :func:`segment_lengths`
   segment_lengths
   :class:`numpy.datetime64`
   np.datetime64
   """

When you specify a type outside of **Parameters**, you have to use the
`sphinx cross-referencing
syntax <https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects>`__
with the full module name:

.. code:: python

   """
   This is a :func:`pd.to_datetime`    # NO
   and :func:`pandas.to_datetime`      # YES

   This is a :class:`np.datetime64`    # NO
   and :class:`numpy.datetime64`       # YES
   """
