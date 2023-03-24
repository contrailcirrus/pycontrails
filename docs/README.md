# Documentation

The `pycontrails` package uses [Sphinx](https://www.sphinx-doc.org/en/master/) for documentation.

## Requirements

- See [README.md](../README.md) for development setup instructions
- [pandoc](https://pandoc.org/installing.html) - Used for interpreting notebooks
- *For PDF Output*: Install a LaTeX distribution.
  If you are using a Mac, [MacTeX](https://www.tug.org/mactex/index.html) is the best option.
  Note that LaTeX is fairly large to install (~6GB).

## Configuration

See [conf.py](source/conf.py) for all Sphinx configuration.
The full list of Sphinx configuration options is in the [Sphinx configuration docs](https://www.sphinx-doc.org/en/master/usage/configuration.html)

## Conventions

Wherever possible, we adhere to the [NumPy](https://numpydoc.readthedocs.io/en/latest/format.html) docstring conventions.

Resources:

- <https://numpydoc.readthedocs.io/en/latest/format.html#sections>
- <https://numpydoc.readthedocs.io/en/latest/format.html#common-rest-concepts>
- <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>
- <https://pandas.pydata.org/docs/development/contributing_docstring.html#about-docstrings-and-standards>
- <https://docs.scipy.org/doc/numpy/docs/howto_document.html#example-source>

### General guidelines

Numpy says

```rst
Use *italics*, **bold** and ``monospace`` if needed in any explanations 
(but not for variable names and doctest code or multi-line code). 
Variable, module, function, and class names should be written between single back-ticks (`numpy`).
```

- When you specify a type, it will automatically use the napolean aliases specified in `conf.py`. i.e.

```python
"""
Parameters
----------
x : np.ndarray
    Description of parameter `x`.
"""
```

- When you specify a type outside of the Parameters,
  then you have to use the [sphinx cross-referencing syntax](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects)

```python
"""

This is a :func:`pd.to_datetime`    # NO
and :func:`pandas.to_datetime`      # YES

This is a :class:`np.datetime64`    # NO
and :class:`numpy.datetime64`       # YES

```

- The *See Also* section is *not a list* and will automatically use
  the aliases specified in `conf.py`. All of the following work

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

## HTML Output

### Build

From the root of the repository, run:

```bash
$ make docs-build
```

The docs will be built as a static website in `docs/_build/html`.

To build manually, use:

```bash
$ sphinx-build -b html docs docs/_build/html      # website output
```

Sphinx caches builds between changes, which can sometimes result in the table of contents not updated.
To force the whole site to rebuild, use the options `-aE`:

```bash
$ sphinx-build -aE -b html docs docs/_build/html  # rebuild all website output
```

### Serve

To automatically build the website on changes, run:

```bash
$ make docs-serve
```

The documentation will be served at `http://127.0.0.1:8000`

To serve manually with custom command line arguments, run:

```bash
$ sphinx-autobuild -aE -b html docs docs/_build/html
```

## PDF Output

> Building PDF output requires a [LaTeX distribution](https://www.latex-project.org/get/).

From the root of the repository, run:

```bash
$ make docs-pdf
```

A single pdf output (i.e. `pycontrails.pdf`) will be built within `docs/_build/latex`.

To build manually, run:

```bash
$ sphinx-build -b latex docs docs/_build/latex    # pdf output
$ cd docs/_build/latex
$ make
```

See [sphinx-build](https://www.sphinx-doc.org/en/master/man/sphinx-build.html#cmdoption-sphinx-build-b) documentation for a list of all the possible builders.

## References

Bibliography references managed in the [Zotero `pycontrails` collection](https://www.zotero.org/groups/4730892/pycontrails/library).

To update the bibliography (`_static/pycontrails.bib`):

- Install the [Zotero Better Bibtex extension](https://retorque.re/zotero-better-bibtex/installation/). Leave defaults during setup.
- Right click on the `pycontrails` library -> Export Library
- Export as *Better Bibtex*. You can optionally check *Keep Updated* if you want
  this file to update every time you make a change to the collection.
- Select the file `_static/pycontrails.bib` and press *Save* to overwrite the file.
- Commit the updated `_static/pycontrails.bib`
