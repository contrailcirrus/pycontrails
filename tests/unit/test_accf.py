"""Test ACCF model"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pycontrails import Flight, MetDataset
from pycontrails.models.accf import ACCF


@pytest.fixture()
def fl() -> Flight:
    """Create a flight for testing."""

    n = 10000
    longitude = np.linspace(45, 75, n) + np.linspace(0, 1, n)
    latitude = np.linspace(50, 60, n) - np.linspace(0, 1, n)
    level = np.full_like(longitude, 225)

    start = np.datetime64("2022-11-11")
    time = pd.date_range(start, start + np.timedelta64(90, "m"), periods=n)
    return Flight(
        longitude=longitude,
        latitude=latitude,
        level=level,
        time=time,
        aircraft_type="B737",
        flight_id=17,
    )


@pytest.mark.parametrize("use_watts", [True, False])
def test_accf_default(
    met_accf_pl: MetDataset,
    met_accf_sl: MetDataset,
    fl: Flight,
    use_watts: bool,
) -> None:
    """Test Default accf algorithm."""

    pytest.importorskip("climaccf", reason="climaccf package not available")

    for name, da in met_accf_sl.data.items():
        assert da.attrs["units"] == "J m**-2"
        if use_watts:
            da = da / 3600.0
            da.attrs["units"] = "W m**-2"
            met_accf_sl.data[name] = da

    accf = ACCF(met=met_accf_pl, surface=met_accf_sl)
    out = accf.eval(fl)

    # Pin some values
    rel = 0.01
    assert np.mean(out["olr"]) == pytest.approx(-35.23, rel=rel)
    assert np.mean(out["aCCF_CO2"]) == pytest.approx(6.94e-16, rel=rel)
    assert np.mean(out["aCCF_NOx"]) == pytest.approx(1.19e-12, rel=rel)
    assert np.mean(out["aCCF_Cont"]) == pytest.approx(9.91e-15, rel=rel)
