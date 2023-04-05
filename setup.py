"""Support for building pycontrails Cython extension modules.

See https://stackoverflow.com/q/66157987/ for details.
"""

import numpy
import setuptools
from Cython.Build import cythonize

rgi_ext = setuptools.Extension(
    "pycontrails.core.rgi_cython",
    ["pycontrails/core/rgi_cython.pyx"],
    language="c",
    include_dirs=[numpy.get_include()],
)


setuptools.setup(ext_modules=cythonize([rgi_ext]))
