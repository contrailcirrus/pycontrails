"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from __future__ import annotations

import datetime

import pycontrails

# -- Project information -----------------------------------------------------

project = "pycontrails"
copyright = f"2021-{datetime.datetime.now().year}, Breakthrough Energy"

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
    # https://github.com/wpilibsuite/sphinxext-opengraph
    "sphinxext.opengraph",
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
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev/", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
    "traffic": ("https://traffic-viz.github.io/", None),
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
    "Fuel": "~pycontrails.Fuel",
    "JetA": "~pycontrails.JetA",
    "GCPCacheStore": "~pycontrails.GCPCacheStore",
    "DiskCacheStore": "~pycontrails.DiskCacheStore",
    "MetDataset": "~pycontrails.MetDataset",
    "MetDataArray": "~pycontrails.MetDataArray",
    "MetDataSource": "~pycontrails.MetDataSource",
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
bibtex_reference_style = "author_year"
bibtex_default_style = "unsrt"
bibtex_cite_id = "cite-{key}"

nbsphinx_timeout = 600
nbsphinx_execute = "never"

# Allow headers to be linkable to level 3.
myst_heading_anchors = 3

# Disable myst translations
myst_disable_syntax: list[str] = []

# Optional MyST Syntaxes
myst_enable_extensions: list[str] = []

# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-pygments_style
# https://pygments.org/styles/
pygments_style = "default"
pygments_dark_style = "monokai"  # furo-specific


# -- Options for HTML output -------------------------------------------------

# https://github.com/pradyunsg/furo
html_theme = "furo"
html_title = f"{project} v{release}"

# Adds a "last updated on" to the bottom of each page
# Set to None to disable
html_last_updated_fmt = ""

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "source_repository": "https://github.com/contrailcirrus/pycontrails",
    "source_branch": "main",
    "source_directory": "docs/",
    # "sidebar_hide_name": False,   # default
    # "top_of_page_button": "edit", # default
    # this adds a github icon to the footer. Ugly, but useful.
    # See https://pradyunsg.me/furo/customisation/footer/#using-embedded-svgs
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/contrailcirrus/pycontrails",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "light_css_variables": {
        "color-brand-primary": "#0F6F8A",
        "color-brand-content": "#0F6F8A",
    },
    "dark_css_variables": {
        "color-brand-primary": "#34C3EB",
        "color-brand-content": "#34C3EB",
    },
    # Note these paths must be relative to `_static/`
    # "light_logo": "img/logo.png",
    # "dark_logo": "img/logo-dark.png",
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "_static/img/logo.jpg"
html_favicon = "_static/img/favicon.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/style.css"]
# html_js_files = []

html_sourcelink_suffix = ""

# Add plausible script to track usage
html_js_files = [
    ("https://plausible.io/js/script.js", {"data-domain": "py.contrails.org", "defer": "defer"}),
]
