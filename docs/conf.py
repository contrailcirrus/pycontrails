"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import pathlib

import pycontrails

# -- Project information -----------------------------------------------------

project = "pycontrails"
copyright = f"{datetime.datetime.now().year}, Breakthrough Energy"

author = "Breakthrough Energy"
version = pycontrails.__version__
release = pycontrails.__version__

# -- General configuration ---------------------------------------------------

# parsed files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    # https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html
    "sphinxcontrib.bibtex",
    # https://nbsphinx.readthedocs.io/en/0.8.5/
    "nbsphinx",
    # https://sphinx-copybutton.readthedocs.io/en/latest/
    "sphinx_copybutton",
    # Markdown parsing, https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
    "myst_parser",
    # https://github.com/jbms/sphinx-immaterial/issues/38#issuecomment-1055785564
    "IPython.sphinxext.ipython_console_highlighting",
]

extlinks = {"issue": ("https://github.com/contrailcirrus/pycontrails/issues/%s", "issue %s")}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "README.md",
    "examples/README.md",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

# suppress warnings during build
suppress_warnings = ["myst.header"]

# Set up mapping for other projects' docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
}

# Display todos by setting to True
todo_include_todos = True

# Napoleon configuration - https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

# if convenient, set mapping that maps docstring words to
# richer text in sphinx output
# Note the ~ prefix removes the module name in presentation
napoleon_type_aliases = {
    # general terms
    "sequence": ":term:`sequence`",
    "iterable": ":term:`iterable`",
    "callable": ":py:func:`callable`",
    "dict_like": ":term:`dict-like <mapping>`",
    "dict-like": ":term:`dict-like <mapping>`",
    "path-like": ":term:`path-like <path-like object>`",
    "mapping": ":term:`mapping`",
    # pycontrails
    "VectorDataset": "~pycontrails.VectorDataset",
    "GeoVectorDataset": "~pycontrails.GeoVectorDataset",
    "Flight": "~pycontrails.Flight",
    "Aircraft": "~pycontrails.Aircraft",
    "Fuel": "~pycontrails.Fuel",
    "JetA": "~pycontrails.JetA",
    "GCPCacheStore": "~pycontrails.GCPCacheStore",
    "DiskCacheStore": "~pycontrails.DiskCacheStore",
    "MetDataset": "~pycontrails.MetDataset",
    "MetDataArray": "~pycontrails.MetDataArray",
    "MetDataSource": "~pycontrails.MetDataSource",
    "MetDataArray": "~pycontrails.MetDataArray",
    "Model": "~pycontrails.Model",
    "ModelParams": "~pycontrails.ModelParams",
    # pycontrails.models
    "BADA3": "~pycontrails.ext.bada.BADA3",
    "BADA4": "~pycontrails.ext.bada.BADA4",
    "BADAFlight": "~pycontrails.ext.bada.BADAFlight",
    "BADAFlightParams": "~pycontrails.ext.bada.BADAFlightParams",
    "BADAGrid": "~pycontrails.ext.bada.BADAGrid",
    "BADAGridParams": "~pycontrails.ext.bada.BADAGridParams",
    "Cocip": "~pycontrails.models.cocip.Cocip",
    "CocipParams": "~pycontrails.models.cocip.CocipParams",
    "CocipUncertaintyParams": "~pycontrails.models.cocip.CocipUncertaintyParams",
    "CocipFlightParams": "~pycontrails.models.cocip.CocipFlightParams",
    "CocipGrid": "~pycontrails.models.cocip.CocipGrid",
    "CocipGridParams": "~pycontrails.models.cocip.CocipGridParams",
    "Emissions": "~pycontrails.models.emissions.Emissions",
    "ISSR": "~pycontrails.models.issr.ISSR",
    "ISSRParams": "~pycontrails.models.issr.ISSRParams",
    "PCR": "~pycontrails.models.pcr.PCR",
    "PCRParams": "~pycontrails.models.pcr.PCRParams",
    "SAC": "~pycontrails.models.sac.SAC",
    "SACParams": "~pycontrails.models.sac.SACParams",
    "PCC": "~pycontrails.models.pcc.PCC",
    "PCCParams": "~pycontrails.models.pcc.PCCParams",
    # pycontrails.utils
    "ArrayScalarLike": "~pycontrails.utils.types.ArrayScalarLike",
    "ArrayLike": "~pycontrails.utils.types.ArrayLike",
    # numpy
    "np.ndarray": "numpy.ndarray",
    "np.datetime64": "numpy.datetime64",
    "np.timedelta64": "numpy.timedelta64",
    # xarray
    "xr.DataArray": "xarray.DataArray",
    "xr.Dataset": "xarray.Dataset",
    # pandas
    "pd.DataFrame": "pandas.DataFrame",
    "pd.Series": "pandas.Series",
    "pd.Timestamp": "pandas.Timestamp",
}


# generate autosummary files into the :toctree: directory
#   See http://www.sphinx-doc.org/en/master/ext/autosummary.html
autosummary_generate = True
autodoc_typehints = "none"

# autodoc options
autoclass_content = "class"  # only include docstring from Class (not __init__ method)
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": None,  # means yes/true/on
    "undoc-members": None,
    "show-inheritance": None,
}

# Add references in bibtex format here
# use with :cite:`perez2011python` etc
# See the References section of the README for instructions to add new references
bibtex_bibfiles = ["_static/pycontrails.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

nbsphinx_timeout = 600
nbsphinx_execute = "never"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = (
    "sphinx_book_theme"  # https://sphinx-book-theme.readthedocs.io/en/latest/configure.html
)
html_title = ""

html_context = {
    "doc_path": "docs",
}
html_last_updated_fmt = ""

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "repository_url": "https://github.com/contrailcirrus/pycontrails",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_fullscreen_button": False,
    "use_issues_button": True,
    "home_page_in_toc": False,
    "use_edit_page_button": False,
    "show_prev_next": False,
}


# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/img/logo.jpg"
html_favicon = "_static/img/favicon.png"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/style.css"]
# html_js_files = []

html_sourcelink_suffix = ""

# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-pygments_style
# https://pygments.org/styles/
pygments_style = "monokai"
