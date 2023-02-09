"""Support for building pycontrails Cython extension modules.

See https://stackoverflow.com/q/66157987/ for details.
"""

import numpy
import setuptools
from Cython.Build import cythonize

setuptools.setup(
    ext_modules=cythonize("pycontrails/core/rgi_cython.pyx"),
    include_dirs=[numpy.get_include()],
)
