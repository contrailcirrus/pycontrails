
# Contributing

Contributions (bug reports, fixes, documentation,
enhancements, ideas, ...) are welcome and appreciated.

To get started, find the best path for your contribution:

- Ask questions, discuss models, and present ideas in [Discussions](https://github.com/contrailcirrus/pycontrails/discussions).
- Report bugs or suggest changes as [Issues](https://github.com/contrailcirrus/pycontrails/issues).
- Contribute fixes or improvements as [Pull Requests](https://github.com/contrailcirrus/pycontrails/pulls).

Please follow the [Github Community Guidelines](https://docs.github.com/en/site-policy/github-terms/github-community-guidelines) when participating in any of these forums.

The following emulates the [xarray contributing guidelines](https://docs.xarray.dev/en/stable/contributing.html).

## Contributing to documentation

Documentation is written in [reStructuredText](http://docutils.sourceforge.net/rst.html) and synthesized with [Sphinx](https://www.sphinx-doc.org/en/master/).

For small changes, [fork and edit](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files) files directly in the Github interface.

For larger changes:

- Set up a [local development environment](https://py.contrails.org/develop.html).
  Make sure to install the `docs` optional dependencies:

```bash
$ pip install -e ".[docs]"
```

- Edit documents and notebooks following [existing conventions](#conventions).
- Build and review the documentation locally:

```bash
# docs build to directory docs/_build/html
$ make docs-build

# automatically build docs on changes
# docs will be served at http://127.0.0.1:8000
$ make docs-serve
```

- Submit changes as a [Pull Request](https://github.com/contrailcirrus/pycontrails/pulls).

## Contributing to the code base

If you are new to development, see xarray's [Working with the code](https://docs.xarray.dev/en/stable/contributing.html#working-with-the-code).
This reference provides an introduction to version control, [git](http://git-scm.com/), [Github](https://github.com/contrailcirrus/pycontrails),
[Forking](https://docs.github.com/en/get-started/quickstart/fork-a-repo), and [creating branches](https://docs.xarray.dev/en/stable/contributing.html#creating-a-branch).

For more involved changes, create a [Github Issue](https://github.com/contrailcirrus/pycontrails/issues) describing the intended changes first.

Once you're ready to develop:

- Set up a [local development environment](https://py.contrails.org/develop.html).
  Install the `dev` optional dependencies and [pre-commit](https://pre-commit.com/) hooks:

```bash
# set up dev install automatically
$ make dev-install

# install dependencies and pre-commit hooks manually
$ pip install -e ".[dev]"
$ pre-commit install
```

- Implement updates.
  Make sure code is documented using [existing conventions](#conventions).
- Ensure tests pass locally:

```bash
$ make test
```

- Submit changes as a [Pull Request](https://github.com/contrailcirrus/pycontrails/pulls).

## Conventions

### Code

`pycontrails` aims to implement clear, consistent, performant data structures and models.

The project uses [mypy](http://mypy-lang.org/) for static type checking.
All code should have specific, clear type annotations.

The project uses [Black](https://black.readthedocs.io/en/stable/), [ruff](https://github.com/charliermarsh/ruff) and [doc8](https://doc8.readthedocs.io/en/latest/readme.html) to standardize code-formatting.
These tools are run automatically in a pre-commit hook.

The project uses [pytest](https://docs.pytest.org/en/7.2.x/) to run unit tests.
New code should include clear unit tests for implementation and output values.
New test files should be included in the [`/tests/unit/` directory](https://github.com/contrailcirrus/pycontrails/tree/main/tests/unit).

The project uses [Github Actions](https://github.com/contrailcirrus/pycontrails/actions) to run code quality and unit tests on each pull request.
Test locally before pushing using:

```bash
$ make test
```

### Documentation

Sphinx uses [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) to synthesize documentation.
The [quick reStructuredText reference](https://docutils.sourceforge.io/docs/user/rst/quickref.html) provides a basic overview.

Sphinx includes many additional [roles](https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html), [directives](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html), and [extensions](https://www.sphinx-doc.org/en/master/usage/extensions/index.html) to enhance documentation.

Sphinx configuration is stored in [conf.py](https://github.com/contrailcirrus/pycontrails/blob/main/docs/conf.py).
The full list of Sphinx configuration options is in the [Sphinx configuration docs](https://www.sphinx-doc.org/en/master/usage/configuration.html).

#### Tutorials

Tutorials should be written as isolated executable [Jupyter Notebooks](https://jupyter.org/).
The [nbsphinx](https://nbsphinx.readthedocs.io/en/0.9.1/) extension includes notebooks in the static documentation.

Notebooks will be automatically evaluated during tests.
To exclude a notebook cell from evaluation during testing or documentation generation,
[add the tags](https://jupyterbook.org/en/stable/content/metadata.html#adding-tags-using-notebook-interfaces) `nbval-skip` and `skip-execution` to cell metadata.
See *[Avoiding output comparison](https://nbval.readthedocs.io/en/latest/index.html#Avoid-output-comparison-for-specific-cells) in the `nbval` documentation for more information.

#### Literature References

The documentation uses [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html) to include citations and a bibliography.

All references should be cited through documentation and docstrings using the [`:cite:` directive](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#role-cite).

Bibliography references are managed in the [Zotero `pycontrails` library](https://www.zotero.org/groups/4730892/pycontrails/library).
The [Zotero Better Bibtex extension](https://retorque.re/zotero-better-bibtex/installation/) automatically syncs this library
to the [docs/_static/pycontrails.bib](https://github.com/contrailcirrus/pycontrails/blob/main/docs/_static/pycontrails.bib) Bibtex file.

See the [References](https://py.contrails.org/develop.html#references) section of the docs for setup.

#### Docstrings

Wherever possible, we adhere to the [NumPy docstring conventions](https://numpydoc.readthedocs.io/en/latest/format.html).

The following links are good references for writing *numpy* docstrings:

- [numpydoc docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [pandas docstring guide](https://pandas.pydata.org/docs/development/contributing_docstring.html)
- [scipy docstring guideline](https://docs.scipy.org/doc//scipy/dev/contributor/rendering_documentation.html#documentation-guidelines)

General guidelines:

```rst
Use *italics*, **bold** and ``monospace`` if needed in any explanations 
(but not for variable names and doctest code or multi-line code). 
Variable, module, function, and class names
should be written between single back-ticks (`numpy`).
```

When you specify a type in **Parameters** or **See Also**, Sphinx will automatically replace the text with the `napolean_type_aliases` specified in [conf.py](https://github.com/contrailcirrus/pycontrails/blob/main/docs/conf.py), e.g.

```python
"""
Parameters
----------
x : np.ndarray
    Sphinx will automatically replace
    "np.ndarray" with the napolean type alias "numpy.ndarray"
"""
```

The **See Also** section is *not a list*. All of the following work:

```python
"""
See Also
--------
:func:`segment_lengths`
segment_lengths
:class:`numpy.datetime64`
np.datetime64
"""
```

When you specify a type outside of **Parameters**, you have to use the
[sphinx cross-referencing syntax](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects):

```python
"""
This is a :func:`pd.to_datetime`    # NO
and :func:`pandas.to_datetime`      # YES

This is a :class:`np.datetime64`    # NO
and :class:`numpy.datetime64`       # YES
"""
```
