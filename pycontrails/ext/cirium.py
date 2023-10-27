"""Interface to `pycontrails-cirium` extension."""

from __future__ import annotations

try:
    from pycontrails_cirium import Cirium

except ImportError as e:
    raise ImportError(
        "Failed to import `pycontrails-cirium` extension. Install with `pip install"
        ' "pycontrails-cirium @ git+ssh://git@github.com/contrailcirrus/pycontrails-cirium.git"`'
    ) from e
else:
    __all__ = ["Cirium"]
