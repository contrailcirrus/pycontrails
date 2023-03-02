"""Test ACCF model"""


from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pycontrails import Flight, MetDataset
from pycontrails.models.accf import ACCF


def _is_climaccf_available():
    try:
        import climaccf  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(not _is_climaccf_available(), reason="climaccf package not available")
def test_accf_default(met_accf_pl: MetDataset, met_accf_sl: MetDataset) -> None:
    """Test Default accf algorithm."""

    # little hack to avoid warning
    del met_accf_pl.attrs["history"]
    met_accf_pl.attrs["met_source"] = "not_ecmwf"
    del met_accf_sl.attrs["history"]
    met_accf_sl.attrs["met_source"] = "not_ecmwf"

    accf = ACCF(met_accf_pl, met_accf_sl)

    n = 10000
    longitude = np.linspace(45, 75, n) + np.linspace(0, 1, n)
    latitude = np.linspace(50, 60, n) - np.linspace(0, 1, n)
    level = np.full_like(longitude, 225)

    start = np.datetime64("2022-11-11")
    time = pd.date_range(start, start + np.timedelta64(90, "m"), periods=n)
    fl = Flight(
        longitude=longitude,
        latitude=latitude,
        level=level,
        time=time,
        aircraft_type="B737",
        flight_id=17,
    )

    out = accf.eval(fl)

    assert np.all(np.isfinite(out["aCCF_NOx"]))
