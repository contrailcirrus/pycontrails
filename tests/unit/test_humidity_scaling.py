"""Test functions in the `Humidity_scaling` module."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from overrides import overrides

from pycontrails import GeoVectorDataset, MetDataArray, MetDataset, VectorDataset
from pycontrails.core import models
from pycontrails.models import humidity_scaling as hs
from pycontrails.physics import constants, thermo, units
from pycontrails.utils.json import NumpyEncoder
from pycontrails.utils.types import ArrayLike

cls_list = [
    cls
    for cls in hs.__dict__.values()
    if isinstance(cls, type)
    and issubclass(cls, hs.HumidityScaling)
    and cls is not hs.HumidityScaling
    and cls is not hs.HistogramMatchingWithEckel
]


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Get random number generator for module."""
    return np.random.default_rng(1234)


def assign_random_t_and_q(vector: GeoVectorDataset, rng: np.random.Generator) -> GeoVectorDataset:
    """Assign random temperature and humidity to vector."""
    T = vector["air_temperature"] = units.m_to_T_isa(vector.altitude)

    rhi = rng.triangular(0.3, 0.7, 1.3, vector.size)
    rhi_on_q = vector.air_pressure * (constants.R_v / constants.R_d) / thermo.e_sat_ice(T)
    vector["specific_humidity"] = rhi / rhi_on_q

    return vector


@pytest.fixture
def vector(rng: np.random.Generator) -> GeoVectorDataset:
    """Create vector on which to run humidity scaling."""
    n = 100
    longitude = rng.uniform(-180, 180, n)
    latitude = rng.uniform(-90, 90, n)
    altitude = rng.uniform(5000, 12000, n)
    time = pd.date_range("2019-01-01", "2019-02-01", n)  # never used

    vector = GeoVectorDataset(longitude=longitude, latitude=latitude, altitude=altitude, time=time)
    return assign_random_t_and_q(vector, rng)


class DefaultHumidityScaling(hs.HumidityScaling):
    """Compute uncorrected RHi without any scaling.

    This is simply a wrapper around :func:`thermo.rhi` to ensure a consistent
    humidity scaling interface.
    """

    name = "no_scaling"
    long_name = "No humidity scaling"
    formula = "rhi -> rhi"

    @overrides
    def scale(
        self,
        specific_humidity: ArrayLike,
        air_temperature: ArrayLike,
        air_pressure: ArrayLike,
        **kwargs: Any,
    ) -> tuple[ArrayLike, ArrayLike]:
        rhi = thermo.rhi(specific_humidity, air_temperature, air_pressure)
        return specific_humidity, rhi


@pytest.mark.parametrize("scaler_cls", cls_list)
def test_all_scalers(vector: GeoVectorDataset, scaler_cls: type) -> None:
    """Check basic usage of humidity scaler instances."""
    scaler = scaler_cls(copy_source=False)
    q1 = vector["specific_humidity"]
    assert "rhi" not in vector

    # Call the scaler and check how vector has been mutated
    scaler.eval(vector)
    assert "rhi" in vector
    q2 = vector["specific_humidity"]

    rhi1 = thermo.rhi(vector["specific_humidity"], vector["air_temperature"], vector.air_pressure)
    rhi2 = vector["rhi"]
    np.testing.assert_allclose(rhi1, rhi2, rtol=1e-15, atol=1e-15)

    # Humidity adjusted from the original
    assert np.any(q1 != q2)

    # But not by too much
    np.testing.assert_allclose(q1, q2, rtol=0.26)


def test_rhi_adj_override(vector: GeoVectorDataset) -> None:
    """Confirm vector variables override scaler params."""
    rhi0 = thermo.rhi(vector["specific_humidity"], vector["air_temperature"], vector.air_pressure)

    # instantiate scaler
    scaler = hs.ConstantHumidityScaling(rhi_adj=0.9)

    # use scaler rhi_adj
    vector1 = scaler.eval(vector)
    rhi1 = vector1["rhi"]

    # override with vector data
    vector2 = vector.copy()
    vector2["rhi_adj"] = np.full(len(vector2), 0.8)
    vector2 = scaler.eval(vector2)
    rhi2 = vector2["rhi"]

    # override with vector attrs
    vector3 = vector.copy()
    vector3.attrs["rhi_adj"] = 0.7
    vector3 = scaler.eval(vector3)
    rhi3 = vector3["rhi"]

    # data takes priority over attrs
    vector4 = vector.copy()
    vector4.attrs["rhi_adj"] = 0.666
    vector4["rhi_adj"] = np.full(len(vector4), 0.6)
    vector4 = scaler.eval(vector4)
    rhi4 = vector4["rhi"]

    np.testing.assert_allclose(rhi0, 0.9 * rhi1, atol=1e-15)
    np.testing.assert_allclose(rhi0, 0.8 * rhi2, atol=1e-15)
    np.testing.assert_allclose(rhi0, 0.7 * rhi3, atol=1e-15)
    np.testing.assert_allclose(rhi0, 0.6 * rhi4, atol=1e-15)


def test_rhi_already_exists_warning(vector: GeoVectorDataset) -> None:
    """Confirm a warning is issued if humidity appears to have been scaled twice."""
    scaler = hs.HumidityScalingByLevel(copy_source=False)
    scaler.eval(vector)
    assert "rhi" in vector
    assert "air_pressure" in vector

    scaler = hs.ExponentialBoostLatitudeCorrectionHumidityScaling()
    with pytest.warns(UserWarning, match="Variable 'rhi' already found on source to be scaled."):
        scaler.eval(vector)


@pytest.mark.parametrize("scaler_cls", cls_list)
def test_description(scaler_cls: type) -> None:
    """Check description for scalers and ensure JSON serialization via NumpyEncoder."""
    scaler = scaler_cls()
    description = scaler.description
    assert "name" in description
    assert "formula" in description

    # other fields are numeric corresponding to parameters
    for k, v in description.items():
        if k in ["name", "formula"]:
            assert isinstance(v, str)
        else:
            assert isinstance(v, float)

    with pytest.raises(TypeError, match="serializable"):
        json.dumps(scaler)

    out = json.dumps(scaler, cls=NumpyEncoder)
    assert out == json.dumps(description)


def test_pin_specific_description() -> None:
    """Pin description for HumidityscalementConstantExponentialBoost (default for models)."""
    scaler = hs.ExponentialBoostHumidityScaling()
    assert scaler.description == {
        "name": "exponential_boost",
        "formula": "rhi -> (rhi / rhi_adj) ^ rhi_boost_exponent",
        "rhi_adj": 0.97,
        "rhi_boost_exponent": 1.7,
        "clip_upper": 1.7,
    }


@pytest.mark.parametrize("exp", [1.1, 1.2, 1.9, 2.0, 3.1])
def test_rhi_boost_exponential(vector: GeoVectorDataset, exp: float) -> None:
    """Test exponential boosting patterns."""
    base = DefaultHumidityScaling()
    _, rhi1 = base.scale(
        vector["specific_humidity"], vector["air_temperature"], vector.air_pressure
    )

    assert "rhi" not in vector
    scaler = hs.ExponentialBoostHumidityScaling(
        rhi_adj=1, rhi_boost_exponent=exp, copy_source=False
    )
    scaler.eval(vector)
    rhi2 = vector["rhi"]
    atol = 1e-15
    assert np.all(rhi2 + atol >= rhi1)

    # confirm some values above 1.0 and some below 1.0
    assert np.any(rhi2 > 1)
    assert np.any(rhi2 < 1)
    assert np.all(rhi2 <= scaler.params["clip_upper"])

    # no boosting if rhi < 1
    filt = rhi2 < 1
    np.testing.assert_allclose(rhi1[filt], rhi2[filt], atol=atol)

    # and boosting if rhi >= 1
    expected = np.minimum(rhi1[~filt] ** exp, scaler.params["clip_upper"])
    np.testing.assert_allclose(expected, rhi2[~filt], atol=atol)


@pytest.mark.parametrize("scaler_cls", cls_list)
def test_scalers_pass_nan_through(vector: GeoVectorDataset, scaler_cls: type) -> None:
    """Confirm each scaler pass nan values through when computing rhi."""
    vector["specific_humidity"][55] = np.nan
    vector["air_temperature"][66] = np.nan
    scaler = scaler_cls(copy_source=False)
    scaler.eval(vector)
    assert np.flatnonzero(np.isnan(vector["rhi"])).tolist() == [55, 66]

    # And, because some scalers convert from rhi back to q, we now get a nan
    # value there arising from the nan value in air temperature
    boosters = (
        hs.ExponentialBoostHumidityScaling,
        hs.ExponentialBoostLatitudeCorrectionHumidityScaling,
        hs.HistogramMatching,
    )
    if isinstance(scaler, boosters):
        assert np.isnan(vector["specific_humidity"][66])
    else:
        assert np.isfinite(vector["specific_humidity"][66])


@pytest.fixture
def custom() -> VectorDataset:
    """Create custom VectorDataset for testing ExponentialBoostLatitudeCorrectionHumidityScaling."""
    # Data from Roger: uncorrected and correct RHi values
    latitude = [2.4970, 14.4288, 48.8538, 61.7205, 69.2504, -33.4201, 45.0802, 0.6897]
    air_temperature = [200.0, 210.0, 220.0, 225.0, 230.0, 235.0, 240.0, 245.0]
    rhi_uncor = [1.070010, 1.101309, 1.084645, 1.061042, 1.379778, 1.145034, 0.394136, 0.576414]
    rhi_cor = [1.165983, 1.248463, 1.203655, 1.167633, 1.474444, 1.195842, 0.412155, 0.567973]

    vector = VectorDataset()
    vector["latitude"] = latitude
    vector["rhi_uncor"] = rhi_uncor
    vector["rhi_cor"] = rhi_cor
    vector["T"] = air_temperature

    # Get into form expected by humidity scaler: calculate q
    vector["p"] = np.full(len(vector), 25000.0)
    rhi_on_q = vector["p"] * (constants.R_v / constants.R_d) / thermo.e_sat_ice(vector["T"])
    vector["q"] = vector["rhi_uncor"] / rhi_on_q

    return vector


def test_global_rhi_correction(custom: VectorDataset) -> None:
    """Check `ExponentialBoostLatitudeCorrectionHumidityScaling` implementation."""
    scaler = hs.ExponentialBoostLatitudeCorrectionHumidityScaling()

    with pytest.raises(KeyError, match="latitude"):
        scaler.scale(custom["q"], custom["T"], custom["p"], **scaler.params)

    _, rhi_cor = scaler.scale(
        custom["q"],
        custom["T"],
        custom["p"],
        latitude=custom["latitude"],
        **scaler.params,
    )
    np.testing.assert_allclose(rhi_cor, custom["rhi_cor"], atol=1e-5)


@pytest.fixture(scope="module")
def ensemble_q(met_issr: MetDataset) -> list[MetDataArray]:
    """Perturb specific humidity to mock ensemble members."""

    rng = np.random.default_rng(424242)

    da = met_issr["specific_humidity"].data

    q_ens = []
    for _ in range(10):
        q = da * rng.uniform(0.3, 0.4, size=da.shape)
        q_ens.append(MetDataArray(q))
    return q_ens


@pytest.fixture
def ensemble_vector(met_issr: MetDataset) -> GeoVectorDataset:
    """Construct a vector in bounds of the ensemble."""

    rng = np.random.default_rng(242424)
    n = 1000

    x0, x1 = met_issr.data.longitude[[0, -1]].values
    longitude = rng.uniform(x0, x1, size=n)

    y0, y1 = met_issr.data.latitude[[0, -1]].values
    latitude = rng.uniform(y0, y1, size=n)

    z0, z1 = met_issr.data.level[[0, -1]].values
    level = rng.uniform(z0, z1, size=n)

    t0, t1 = met_issr.data.time.values
    time = rng.uniform(t0, t1, size=n).astype("datetime64[ns]")

    return GeoVectorDataset(
        longitude=longitude,
        latitude=latitude,
        level=level,
        time=time,
    )


@pytest.mark.filterwarnings("ignore:No variable key 'log_specific_humidity'")
@pytest.mark.parametrize("q_method", [None, "cubic-spline", "log-q-log-p"])
def test_histogram_matching_reanalysis(
    met_issr: MetDataset,
    ensemble_vector: GeoVectorDataset,
    q_method: str,
) -> None:
    """Test the HistogramMatching model."""

    models.interpolate_met(met_issr, ensemble_vector, "air_temperature")
    q0 = models.interpolate_met(met_issr, ensemble_vector, "specific_humidity", q_method=q_method)

    model = hs.HistogramMatching(product_type="reanalysis", interpolation_q_method=q_method)
    vector = model.eval(ensemble_vector)
    q1 = vector["specific_humidity"]

    # Check that the two methods give different results
    diff = np.abs(np.log(q1) - np.log(q0))
    assert diff.min() >= 1e-6
    assert diff.max() <= 2
    assert 0.02 < diff.mean() < 0.1


@pytest.mark.filterwarnings("ignore:No variable key 'log_specific_humidity'")
@pytest.mark.parametrize("q_method", [None, "cubic-spline", "log-q-log-p"])
@pytest.mark.parametrize("member", range(10))
def test_histogram_matching_members(
    met_issr: MetDataset,
    ensemble_q: list[MetDataArray],
    ensemble_vector: GeoVectorDataset,
    member: int,
    q_method: str,
) -> None:
    """Test the HistogramMatchingWithEckel scaling."""

    models.interpolate_met(met_issr, ensemble_vector, "air_temperature")
    mds_q = MetDataset(ensemble_q[member].data.to_dataset(name="specific_humidity"))
    q0 = models.interpolate_met(mds_q, ensemble_vector, "specific_humidity", q_method=q_method)

    rhi0 = thermo.rhi(
        ensemble_vector["specific_humidity"],
        ensemble_vector["air_temperature"],
        ensemble_vector.air_pressure,
    )
    model = hs.HistogramMatching(
        product_type="ensemble_members",
        member=member,
        interpolation_q_method=q_method,
    )

    assert "rhi" not in ensemble_vector
    out = model.eval(ensemble_vector)
    assert "rhi" not in ensemble_vector

    q1 = out["specific_humidity"]
    rhi1 = out["rhi"]

    if q_method == "log-q-log-p":
        ensemble_q = [MetDataArray(np.log(mda.data)) for mda in ensemble_q]
        log_applied = True
    else:
        log_applied = False

    model = hs.HistogramMatchingWithEckel(
        ensemble_specific_humidity=ensemble_q,
        member=member,
        interpolation_q_method=q_method,
        log_applied=log_applied,
    )
    out = model.eval(ensemble_vector)
    assert "rhi" not in ensemble_vector

    q2 = out["specific_humidity"]
    rhi2 = out["rhi"]

    assert np.all(rhi0 >= 0)
    assert np.all(rhi1 >= 0)
    assert np.all(rhi2 >= 0)

    assert np.all(q0 >= 0)
    assert np.all(q1 >= 0)
    assert np.all(q2 >= 0)

    rhi0_mean = rhi0.mean()
    rhi1_mean = rhi1.mean()
    rhi2_mean = rhi2.mean()

    # Crude checks that work for all methods
    assert rhi0_mean == pytest.approx(rhi1_mean, abs=0.02)
    assert rhi0_mean == pytest.approx(rhi2_mean, abs=0.04)

    if q_method is None:
        assert rhi2_mean == pytest.approx(0.209, abs=0.01)
    elif q_method == "cubic-spline":
        assert rhi2_mean == pytest.approx(0.204, abs=0.01)
    elif q_method == "log-q-log-p":
        assert rhi2_mean == pytest.approx(0.179, abs=0.01)
