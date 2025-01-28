"""Test the ``ValidateTrajectoryHandler`` from the ``spire`` module."""

from __future__ import annotations

import pandas as pd
import pytest

from pycontrails.datalib.spire import ValidateTrajectoryHandler
from pycontrails.datalib.spire.exceptions import (
    BadTrajectoryException,
    FlightDuplicateTimestamps,
    FlightInvariantFieldViolation,
    FlightTooFastError,
    OrderingError,
    ROCDError,
    SchemaError,
)
from tests.unit import get_static_path


@pytest.fixture()
def df() -> pd.DataFrame:
    """Return some Spire data for testing.

    This data appears to contain several distinct flight concatenated together.
    """
    return pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))


@pytest.fixture()
def df_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return some Spire data with additional fields for schema compatibility."""
    return df.assign(
        flight_id="123",
        arrival_airport_icao="abc",
        arrival_scheduled_time=df["timestamp"].iloc[-1],
        departure_airport_icao="abc",
        departure_scheduled_time=pd.to_datetime(df["departure_scheduled_time"].iloc[0]),
        flight_number="abc",
        ingestion_time=pd.Timestamp.now(),
    )


@pytest.fixture()
def df_single_callsign(df_schema: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with a single callsign."""
    callsign = df_schema["callsign"].iloc[0]  # noqa: F841
    return df_schema.query("callsign == @callsign")


class TestValidateTrajectorySet:
    """Test the different exceptions raised by the ``ValidateTrajectoryHandler``."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.vth = ValidateTrajectoryHandler()

    def test_set_empty(self) -> None:
        df = pd.DataFrame()
        with pytest.raises(BadTrajectoryException, match="empty"):
            self.vth.set(df)

    def test_set_flight_id_missing(self) -> None:
        df = pd.DataFrame({"longitude": [1, 2, 3]})
        with pytest.raises(BadTrajectoryException, match="flight_id"):
            self.vth.set(df)

    def test_set_flight_id_not_unique(self) -> None:
        df = pd.DataFrame({"flight_id": [1, 2, 3]})
        with pytest.raises(BadTrajectoryException, match="flight_id"):
            self.vth.set(df)

    def test_set_correct(self) -> None:
        df = pd.DataFrame({"flight_id": [1, 1, 1]})

        assert self.vth._df is None
        self.vth.set(df)
        assert isinstance(self.vth._df, pd.DataFrame)

        assert self.vth._df is not df
        pd.testing.assert_frame_equal(self.vth._df, df)

    def test_value_error_when_no_set(self) -> None:
        vth = ValidateTrajectoryHandler()

        with pytest.raises(ValueError, match="No trajectory DataFrame has been set"):
            vth.evaluate()

        with pytest.raises(ValueError, match="No trajectory DataFrame has been set"):
            _ = vth.validation_df


def test_missing_fields(df: pd.DataFrame) -> None:
    """Confirm a SchemaError is raised when the DataFrame is missing fields."""
    df = df.assign(flight_id=1)

    vth = ValidateTrajectoryHandler()
    vth.set(df)
    violations = vth.evaluate()

    assert len(violations) == 1
    (exc,) = violations
    assert isinstance(exc, SchemaError)

    msg = exc.args[0]
    assert msg.startswith("Trajectory DataFrame is missing expected fields")


def test_wrong_dtypes(df: pd.DataFrame) -> None:
    """Confirm a SchemaError is raised when the DataFrame has wrong dtypes."""
    df = df.assign(
        flight_id=1,
        arrival_airport_icao="abc",
        arrival_scheduled_time="2023-01-01",
        departure_airport_icao="abc",
        flight_number="abc",
        ingestion_time="2023-01-01",
    )

    vth = ValidateTrajectoryHandler()
    vth.set(df)
    violations = vth.evaluate()

    assert len(violations) == 1
    (exc,) = violations
    assert isinstance(exc, SchemaError)

    msg = exc.args[0]
    assert msg.startswith("Trajectory DataFrame has columns with invalid data types")


def test_round2_violations(df_schema: pd.DataFrame) -> None:
    """Confirm some of the round 2 violations are raised."""
    vth = ValidateTrajectoryHandler()
    vth.set(df_schema)
    violations = vth.evaluate()
    assert len(violations) == 2

    exc0, exc1 = violations
    assert isinstance(exc0, FlightDuplicateTimestamps)
    assert isinstance(exc1, FlightInvariantFieldViolation)


def test_timestamp_ordering(df_schema: pd.DataFrame) -> None:
    """Confirm an OrderingError is raised when the timestamps are not in order."""
    df = df_schema.iloc[:100]
    df.iloc[1], df.iloc[2] = df.iloc[2], df.iloc[1]

    vth = ValidateTrajectoryHandler()
    vth.set(df)
    violations = vth.evaluate()
    assert len(violations) == 1

    (exc,) = violations
    assert isinstance(exc, OrderingError)


def test_round3_violations(df_single_callsign: pd.DataFrame) -> None:
    """Confirm some of the round 3 violations are raised.

    This test is not particularly comprehensive, but it assures that at least
    some of the round 3 violations are checked for.
    """
    vth = ValidateTrajectoryHandler()
    vth.set(df_single_callsign)
    violations = vth.evaluate()
    assert len(violations) == 2

    exc0, exc1 = violations
    assert isinstance(exc0, FlightTooFastError)
    assert isinstance(exc1, ROCDError)


def test_violation_df(df_single_callsign: pd.DataFrame) -> None:
    """Confirm the ``validation_df`` property returns a DataFrame that extends the input."""
    df_in = df_single_callsign
    vth = ValidateTrajectoryHandler()
    vth.set(df_in)

    df_out = vth.validation_df
    assert isinstance(df_out, pd.DataFrame)
    for k, v in df_in.items():
        assert k in df_out
        pd.testing.assert_series_equal(df_out[k], v)

    additional_cols = set(df_out) - set(df_in)
    assert additional_cols == {
        "arrival_airport_alt_ft",
        "arrival_airport_dist_m",
        "arrival_airport_lat",
        "arrival_airport_lon",
        "departure_airport_alt_ft",
        "departure_airport_dist_m",
        "departure_airport_lat",
        "departure_airport_lon",
        "elapsed_distance_m",
        "elapsed_seconds",
        "ground_speed_m_s",
        "rocd_fps",
    }
