"""
Interface to ``pycontrails-bada`` extension.

Requires data files obtained with a
`BADA License <https://www.eurocontrol.int/model/bada>`_ from Eurocontrol.
"""

from __future__ import annotations

try:
    from pycontrails_bada import bada3, bada4, bada_model
    from pycontrails_bada.bada3 import BADA3
    from pycontrails_bada.bada4 import BADA4
    from pycontrails_bada.bada_interface import BADA
    from pycontrails_bada.bada_model import (
        BADAFlight,
        BADAFlightParams,
        BADAGrid,
        BADAGridParams,
        BADAParams,
    )

except ImportError as e:
    raise ImportError(
        'Failed to import `pycontrails-bada` extension. Install with `pip install "pycontrails-bada'
        ' @ git+ssh://git@github.com/contrailcirrus/pycontrails-bada.git"`'
    ) from e
else:
    __all__ = [
        "BADA",
        "BADA3",
        "BADA4",
        "BADAFlight",
        "BADAFlightParams",
        "BADAGrid",
        "BADAGridParams",
        "BADAParams",
        "bada3",
        "bada4",
        "bada_model",
    ]
