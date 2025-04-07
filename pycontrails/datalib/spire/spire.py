"""Support for `Spire Aviation <https://spire.com/aviation/>`_ data validation."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pandas.api.types as pdtypes

from pycontrails.core import airports
from pycontrails.datalib.spire.exceptions import (
    BadTrajectoryException,
    BaseSpireError,
    DestinationAirportError,
    FlightAltitudeProfileError,
    FlightDuplicateTimestamps,
    FlightInvariantFieldViolation,
    FlightTooFastError,
    FlightTooLongError,
    FlightTooShortError,
    FlightTooSlowError,
    OrderingError,
    OriginAirportError,
    ROCDError,
    SchemaError,
)
from pycontrails.physics import geo, units


def _segment_haversine_3d(
    longitude: np.ndarray,
    latitude: np.ndarray,
    altitude_ft: np.ndarray,
) -> np.ndarray:
    """Calculate a 3D haversine distance between waypoints.

    Returns the distance between each waypoint in meters.
    """
    horizontal_distance = geo.segment_haversine(longitude, latitude)

    altitude_m = units.ft_to_m(altitude_ft)
    alt0 = altitude_m[:-1]
    alt1 = altitude_m[1:]
    vertical_displacement = np.empty_like(altitude_m)
    vertical_displacement[:-1] = alt1 - alt0
    vertical_displacement[-1] = np.nan

    distance = (horizontal_distance**2 + vertical_displacement**2) ** 0.5

    # Roll the array to match usual pandas conventions.
    # This moves the nan from the -1st index to the 0th index
    return np.roll(distance, 1)


def _pointed_haversine_3d(
    longitude: np.ndarray,
    latitude: np.ndarray,
    altitude_ft: np.ndarray,
    lon0: float,
    lat0: float,
    alt0_ft: float,
) -> np.ndarray:
    horizontal_dinstance = geo.haversine(longitude, latitude, lon0, lat0)  # type: ignore[type-var]
    altitude_m = units.ft_to_m(altitude_ft)
    alt0_m = units.ft_to_m(alt0_ft)
    vertical_displacement = altitude_m - alt0_m
    return (horizontal_dinstance**2 + vertical_displacement**2) ** 0.5  # type: ignore[operator]


class ValidateTrajectoryHandler:
    """
    Evaluates a trajectory and identifies if it violates any verification rules.

    <LINK HERE TO HOSTED REFERENCE EXAMPLE(S)>.
    """

    ROCD_THRESHOLD_FPS = 83.25  # 83.25 ft/sec ~= 5000 ft/min
    CRUISE_LOW_ALTITUDE_THRESHOLD_FT = 15000.0  # lowest expected cruise altitude
    INSTANTANEOUS_HIGH_GROUND_SPEED_THRESHOLD_MPS = 350.0  # 350m/sec ~= 780mph ~= 1260kph
    INSTANTANEOUS_LOW_GROUND_SPEED_THRESHOLD_MPS = 45.0  # 45m/sec ~= 100mph ~= 160kph
    AVG_LOW_GROUND_SPEED_THRESHOLD_MPS = 100.0  # 120m/sec ~= 223mph ~= 360 kph
    AVG_LOW_GROUND_SPEED_ROLLING_WINDOW_PERIOD_MIN = 30.0  # rolling period for avg speed comparison
    AIRPORT_DISTANCE_THRESHOLD_KM = 200.0
    MIN_FLIGHT_LENGTH_HR = 0.4
    MAX_FLIGHT_LENGTH_HR = 19.0

    # expected schema of pandas dataframe passed on initialization
    SCHEMA: ClassVar = {
        "icao_address": pdtypes.is_string_dtype,
        "flight_id": pdtypes.is_string_dtype,
        "callsign": pdtypes.is_string_dtype,
        "tail_number": pdtypes.is_string_dtype,
        "flight_number": pdtypes.is_string_dtype,
        "aircraft_type_icao": pdtypes.is_string_dtype,
        "airline_iata": pdtypes.is_string_dtype,
        "departure_airport_icao": pdtypes.is_string_dtype,
        "departure_scheduled_time": pdtypes.is_datetime64_any_dtype,
        "arrival_airport_icao": pdtypes.is_string_dtype,
        "arrival_scheduled_time": pdtypes.is_datetime64_any_dtype,
        "ingestion_time": pdtypes.is_datetime64_any_dtype,
        "timestamp": pdtypes.is_datetime64_any_dtype,
        "latitude": pdtypes.is_numeric_dtype,
        "longitude": pdtypes.is_numeric_dtype,
        "collection_type": pdtypes.is_string_dtype,
        "altitude_baro": pdtypes.is_numeric_dtype,
    }

    airports_db: pd.DataFrame | None = None

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None

    def set(self, trajectory: pd.DataFrame) -> None:
        """
        Set a single flight trajectory into handler state.

        Parameters
        ----------
        trajectory
            A dataframe representing a single flight trajectory.
            Must include those columns itemized in :attr:`SCHEMA`.
        """
        if trajectory.empty:
            msg = "The trajectory DataFrame is empty."
            raise BadTrajectoryException(msg)

        if "flight_id" not in trajectory:
            msg = "The trajectory DataFrame must have a 'flight_id' column."
            raise BadTrajectoryException(msg)

        n_unique = trajectory["flight_id"].nunique()
        if n_unique > 1:
            msg = f"The trajectory DataFrame must have a unique flight_id. Found {n_unique}."
            raise BadTrajectoryException(msg)

        self._df = trajectory.copy()

    def unset(self) -> None:
        """Pop _df from handler state."""
        self._df = None

    @classmethod
    def _find_airport_coords(cls, airport_icao: str | None) -> tuple[float, float, float]:
        """
        Find the latitude and longitude for a given airport.

        Parameters
        ----------
        airport_icao : str | None
            string representation of the airport's icao code

        Returns
        -------
        tuple[float, float, float]
            ``(longitude, latitude, alt_ft)`` of the airport.
            Returns ``(np.nan, np.nan, np.nan)`` if it cannot be found.
        """
        if airport_icao is None:
            return np.nan, np.nan, np.nan

        if cls.airports_db is None:
            cls.airports_db = airports.global_airport_database()

        matches = cls.airports_db[cls.airports_db["icao_code"] == airport_icao]
        if len(matches) == 0:
            return np.nan, np.nan, np.nan
        if len(matches) > 1:
            msg = f"Found multiple matches for aiport icao {airport_icao} in airports database."
            raise ValueError(msg)

        lon = matches["longitude"].iloc[0].item()
        lat = matches["latitude"].iloc[0].item()
        alt_ft = matches["elevation_ft"].iloc[0].item()

        return lon, lat, alt_ft

    def _calculate_additional_fields(self) -> None:
        """
        Add additional columns to the provided dataframe.

        These additional fields are needed to apply the validation ruleset.

        The following fields are added:

        - elapsed_seconds: time elapsed between two consecutive waypoints
        - elapsed_distance_m: distance travelled between two consecutive waypoints
        - ground_speed_m_s: ground speed in meters per second
        - rocd_fps: rate of climb/descent in feet per second
        - departure_airport_lat: latitude of the departure airport
        - departure_airport_lon: longitude of the departure airport
        - departure_airport_alt_ft: altitude of the departure airport
        - arrival_airport_lat: latitude of the arrival airport
        - arrival_airport_lon: longitude of the arrival airport
        - arrival_airport_alt_ft: altitude of the arrival airport
        - departure_airport_dist_m: distance to the departure airport
        - arrival_airport_dist_m: distance to the arrival airport
        """
        if self._df is None:
            msg = "No trajectory DataFrame has been set. Call set() before calling this method."
            raise ValueError(msg)

        elapsed_seconds = self._df["timestamp"].diff().dt.total_seconds()
        self._df["elapsed_seconds"] = elapsed_seconds

        elapsed_distance_m = _segment_haversine_3d(
            self._df["longitude"].to_numpy(),
            self._df["latitude"].to_numpy(),
            self._df["altitude_baro"].to_numpy(),
        )
        self._df["elapsed_distance_m"] = elapsed_distance_m

        ground_speed_m_s = self._df["elapsed_distance_m"] / self._df["elapsed_seconds"]
        self._df["ground_speed_m_s"] = ground_speed_m_s.replace(np.inf, np.nan)

        rocd_fps = self._df["altitude_baro"].diff() / self._df["elapsed_seconds"]
        self._df["rocd_fps"] = rocd_fps

        if self._df["departure_airport_icao"].nunique() > 1:  # This has already been checked
            raise ValueError("expected only one airport icao for flight departure airport.")
        departure_airport_icao = self._df["departure_airport_icao"].iloc[0]

        if self._df["arrival_airport_icao"].nunique() > 1:  # This has already been checked
            raise ValueError("expected only one airport icao for flight arrival airport.")
        arrival_airport_icao = self._df["arrival_airport_icao"].iloc[0]

        dep_lon, dep_lat, dep_alt_ft = self._find_airport_coords(departure_airport_icao)
        arr_lon, arr_lat, arr_alt_ft = self._find_airport_coords(arrival_airport_icao)

        self._df["departure_airport_lon"] = dep_lon
        self._df["departure_airport_lat"] = dep_lat
        self._df["departure_airport_alt_ft"] = dep_alt_ft
        self._df["arrival_airport_lon"] = arr_lon
        self._df["arrival_airport_lat"] = arr_lat
        self._df["arrival_airport_alt_ft"] = arr_alt_ft

        departure_airport_dist_m = _pointed_haversine_3d(
            self._df["longitude"].to_numpy(),
            self._df["latitude"].to_numpy(),
            self._df["altitude_baro"].to_numpy(),
            dep_lon,
            dep_lat,
            dep_alt_ft,
        )
        self._df["departure_airport_dist_m"] = departure_airport_dist_m

        arrival_airport_dist_m = _pointed_haversine_3d(
            self._df["longitude"].to_numpy(),
            self._df["latitude"].to_numpy(),
            self._df["altitude_baro"].to_numpy(),
            arr_lon,
            arr_lat,
            arr_alt_ft,
        )
        self._df["arrival_airport_dist_m"] = arrival_airport_dist_m

    def _is_valid_schema(self) -> SchemaError | None:
        """Verify that a pandas dataframe has required cols, and that they are of required type."""
        if self._df is None:
            msg = "No trajectory DataFrame has been set. Call set() before calling this method."
            raise ValueError(msg)

        missing_cols = set(self.SCHEMA).difference(self._df)
        if missing_cols:
            msg = f"Trajectory DataFrame is missing expected fields: {sorted(missing_cols)}"
            return SchemaError(msg)

        col_types = self._df.dtypes
        col_w_bad_dtypes = []
        for col, check_fn in self.SCHEMA.items():
            is_valid = check_fn(col_types[col])
            if not is_valid:
                col_w_bad_dtypes.append(f"{col} failed check {check_fn.__name__}")

        if col_w_bad_dtypes:
            msg = f"Trajectory DataFrame has columns with invalid data types: {col_w_bad_dtypes}"
            return SchemaError(msg)

        return None

    def _is_timestamp_sorted_and_unique(self) -> list[OrderingError | FlightDuplicateTimestamps]:
        """Verify that the data is sorted by waypoint timestamp in ascending order."""
        if self._df is None:
            msg = "No trajectory DataFrame has been set. Call set() before calling this method."
            raise ValueError(msg)

        violations: list[OrderingError | FlightDuplicateTimestamps] = []

        ts_index = pd.Index(self._df["timestamp"])
        if not ts_index.is_monotonic_increasing:
            msg = "Trajectory DataFrame must be sorted by timestamp in ascending order."
            violations.append(OrderingError(msg))

        if ts_index.has_duplicates:
            n_duplicates = ts_index.duplicated().sum()
            msg = f"Trajectory DataFrame has {n_duplicates} duplicate timestamps."
            violations.append(FlightDuplicateTimestamps(msg))

        return violations

    def _is_valid_invariant_fields(self) -> FlightInvariantFieldViolation | None:
        """
        Verify that fields expected to be invariant are indeed invariant.

        Presence of null values does not constitute an invariance violation.
        """
        if self._df is None:
            msg = "No trajectory DataFrame has been set. Call set() before calling this method."
            raise ValueError(msg)

        invariant_fields = (
            "icao_address",
            "flight_id",
            "callsign",
            "tail_number",
            "aircraft_type_icao",
            "airline_iata",
            "departure_airport_icao",
            "departure_scheduled_time",
            "arrival_airport_icao",
            "arrival_scheduled_time",
        )

        fields = []
        for k in invariant_fields:
            if self._df[k].nunique(dropna=True) > 1:
                fields.append(k)

        if fields:
            msg = f"The following fields have multiple values for this trajectory: {fields}"
            return FlightInvariantFieldViolation(msg)

        return None

    def _is_valid_flight_length(self) -> FlightTooShortError | FlightTooLongError | None:
        """Verify that the flight is of a reasonable length."""
        if self._df is None:
            msg = "No trajectory DataFrame has been set. Call set() before calling this method."
            raise ValueError(msg)

        flight_duration_sec = np.ptp(self._df["timestamp"]).seconds
        flight_duration_hours = flight_duration_sec / 60.0 / 60.0

        if flight_duration_hours > self.MAX_FLIGHT_LENGTH_HR:
            return FlightTooLongError(
                f"flight exceeds max duration of {self.MAX_FLIGHT_LENGTH_HR} hours."
                f"this trajectory spans {flight_duration_hours:.2f} hours."
            )

        if flight_duration_hours < self.MIN_FLIGHT_LENGTH_HR:
            return FlightTooShortError(
                f"flight less than min duration of {self.MIN_FLIGHT_LENGTH_HR} hours. "
                f"this trajectory spans {flight_duration_hours:.2f} hours."
            )

        return None

    def _is_from_origin_airport(self) -> OriginAirportError | None:
        """Verify the trajectory origin is a reasonable distance from the origin airport."""
        if self._df is None:
            msg = "No trajectory DataFrame has been set. Call set() before calling this method."
            raise ValueError(msg)

        first_waypoint = self._df.iloc[0]
        first_waypoint_dist_km = first_waypoint["departure_airport_dist_m"] / 1000.0
        if first_waypoint_dist_km > self.AIRPORT_DISTANCE_THRESHOLD_KM:
            return OriginAirportError(
                "First waypoint in trajectory too far from departure airport icao: "
                f"{first_waypoint['departure_airport_icao']}. "
                f"Distance {first_waypoint_dist_km:.3f}km is greater than "
                f"threshold of {self.AIRPORT_DISTANCE_THRESHOLD_KM}km."
            )

        return None

    def _is_to_destination_airport(self) -> DestinationAirportError | None:
        """Verify the trajectory destination is reasonable distance from the destination airport."""
        if self._df is None:
            msg = "No trajectory DataFrame has been set. Call set() before calling this method."
            raise ValueError(msg)

        last_waypoint = self._df.iloc[-1]
        last_waypoint_dist_km = last_waypoint["arrival_airport_dist_m"] / 1000.0
        if last_waypoint_dist_km > self.AIRPORT_DISTANCE_THRESHOLD_KM:
            return DestinationAirportError(
                "Last waypoint in trajectory too far from arrival airport icao: "
                f"{last_waypoint['arrival_airport_icao']}. "
                f"Distance {last_waypoint_dist_km:.3f}km is greater than "
                f"threshold of {self.AIRPORT_DISTANCE_THRESHOLD_KM:.3f}km."
            )

        return None

    def _is_too_slow(self) -> list[FlightTooSlowError]:
        """
        Evaluate the flight trajectory for unreasonably slow speed.

        This is evaluated both for instantaneous discrete steps in the trajectory
        (between consecutive waypoints), and on a rolling average basis.

        For instantaneous speed, we don't consider the first or last 10 minutes of the flight.
        """
        if self._df is None:
            msg = "No trajectory DataFrame has been set. Call set() before calling this method."
            raise ValueError(msg)

        violations: list[FlightTooSlowError] = []

        # NOTE: When we get here, we have already checked that the timestamps are sorted and unique.
        gs = self._df.set_index("timestamp")["ground_speed_m_s"]

        t0 = self._df["timestamp"].iloc[0]
        t1 = self._df["timestamp"].iloc[-1]
        cropped_gs = gs[t0 + pd.Timedelta(minutes=10) : t1 - pd.Timedelta(minutes=10)]

        cond = cropped_gs <= self.INSTANTANEOUS_LOW_GROUND_SPEED_THRESHOLD_MPS
        if cond.any():
            below_inst_thresh = cropped_gs[cond]
            violations.append(
                FlightTooSlowError(
                    f"Found {len(below_inst_thresh)} instances where speed between waypoints is "
                    "below threshold of "
                    f"{self.INSTANTANEOUS_LOW_GROUND_SPEED_THRESHOLD_MPS:.2f} m/s. "
                    f"max value: {below_inst_thresh.max():.2f}, "
                    f"min value: {below_inst_thresh.min():.2f},"
                )
            )

        # Consider averages occurring at least window minutes after the flight origination
        window = pd.Timedelta(minutes=self.AVG_LOW_GROUND_SPEED_ROLLING_WINDOW_PERIOD_MIN)
        rolling_gs = gs.rolling(window).mean().loc[t0 + window :]

        cond = rolling_gs <= self.AVG_LOW_GROUND_SPEED_THRESHOLD_MPS
        if cond.any():
            below_avg_thresh = rolling_gs[cond]
            violations.append(
                FlightTooSlowError(
                    f"Found {len(below_avg_thresh)} instances where rolling average speed is "
                    f"below threshold of {self.AVG_LOW_GROUND_SPEED_THRESHOLD_MPS} m/s "
                    f"(rolling window of "
                    f"{self.AVG_LOW_GROUND_SPEED_ROLLING_WINDOW_PERIOD_MIN} minutes). "
                    f"max value: {below_avg_thresh.max()}, "
                    f"min value: {below_avg_thresh.min()},"
                )
            )

        return violations

    def _is_too_fast(self) -> FlightTooFastError | None:
        """
        Evaluate the flight trajectory for reasonably high speed.

        This is evaluated on discrete steps between consecutive waypoints.
        """
        if self._df is None:
            msg = "No trajectory DataFrame has been set. Call set() before calling this method."
            raise ValueError(msg)

        cond = self._df["ground_speed_m_s"] >= self.INSTANTANEOUS_HIGH_GROUND_SPEED_THRESHOLD_MPS
        if cond.any():
            above_inst_thresh = self._df[cond]
            return FlightTooFastError(
                f"Found {len(above_inst_thresh)} instances where speed between waypoints is "
                f"above threshold of {self.INSTANTANEOUS_HIGH_GROUND_SPEED_THRESHOLD_MPS:.2f} m/s. "
                f"max value: {above_inst_thresh['ground_speed_m_s'].max():.2f}, "
                f"min value: {above_inst_thresh['ground_speed_m_s'].min():.2f}"
            )

        return None

    def _is_expected_altitude_profile(self) -> list[FlightAltitudeProfileError | ROCDError]:
        """
        Evaluate flight altitude profile.

        Failure modes include:
        FlightAltitudeProfileError
        1) flight climbs above alt threshold,
            then descends below that threshold one or more times,
            before making final descent to land.

        RocdError
        2) rate of instantaneous (between consecutive waypoint) climb or descent is above threshold,
           while aircraft is above the cruise altitude.
        """
        if self._df is None:
            msg = "No trajectory DataFrame has been set. Call set() before calling this method."
            raise ValueError(msg)

        violations: list[FlightAltitudeProfileError | ROCDError] = []

        # evaluate ROCD
        rocd_above_thres = self._df["rocd_fps"].abs() >= self.ROCD_THRESHOLD_FPS
        if rocd_above_thres.any():
            msg = (
                "Flight trajectory has rate of climb/descent values "
                "between consecutive waypoints that exceed threshold "
                f"of {self.ROCD_THRESHOLD_FPS:.3f}ft/sec. "
                f"Max value found: {self._df['rocd_fps'].abs().max():.3f}ft/sec"
            )
            violations.append(ROCDError(msg))

        alt_below_thresh = self._df["altitude_baro"] <= self.CRUISE_LOW_ALTITUDE_THRESHOLD_FT
        alt_thresh_transitions = alt_below_thresh.rolling(window=2).sum()
        cond = alt_thresh_transitions == 1
        if cond.sum() > 2:
            msg = (
                "Flight trajectory dropped below altitude threshold "
                f"of {self.CRUISE_LOW_ALTITUDE_THRESHOLD_FT}ft while in-flight."
            )
            violations.append(FlightAltitudeProfileError(msg))

        return violations

    @property
    def validation_df(self) -> pd.DataFrame:
        """
        Return an augmented trajectory dataframe.

        Returns
        -------
        dataframe mirroring that provided to the handler,
        but including the additional computed columns that are used in verification.
        e.g. elapsed_sec, ground_speed_m_s, etc.
        """
        if self._df is None:
            msg = "No trajectory DataFrame has been set. Call set() before calling this method."
            raise ValueError(msg)

        violations = self.evaluate()

        FatalException = (
            SchemaError | OrderingError | FlightDuplicateTimestamps | FlightInvariantFieldViolation
        )
        if any(isinstance(v, FatalException) for v in violations):
            msg = f"Validation DataFrame has fatal violation(s): {violations}"
            raise BadTrajectoryException(msg)

        # safeguard to ensure this call follows the addition of the columns
        # assumes calculate_additional_fields is idempotent
        self._calculate_additional_fields()
        return self._df

    def evaluate(self) -> list[BaseSpireError]:
        """Evaluate the flight trajectory for one or more violations.

        This method performs 3 rounds of checks:

        1. Schema checks
        2. Timestamp ordering and invariant field checks
        3. Flight profile and motion checks

        If any violations are found at the end of a round, the method returns the
        current list of violations and does not proceed to the next round.
        """
        if self._df is None:
            msg = "No trajectory DataFrame has been set. Call set() before calling this method."
            raise ValueError(msg)

        all_violations: list[BaseSpireError] = []

        # Round 1 checks
        schema_check = self._is_valid_schema()
        if schema_check:
            all_violations.append(schema_check)
            return all_violations

        # Round 2 checks: We're assuming the schema is valid
        timestamp_check = self._is_timestamp_sorted_and_unique()
        all_violations.extend(timestamp_check)

        invariant_fields_check = self._is_valid_invariant_fields()
        if invariant_fields_check:
            all_violations.append(invariant_fields_check)

        if all_violations:
            return all_violations

        # Round 3 checks: We're assuming the schema and timestamps are valid
        # and no invariant field violations
        self._calculate_additional_fields()

        flight_length_check = self._is_valid_flight_length()
        if flight_length_check:
            all_violations.append(flight_length_check)

        origin_airport_check = self._is_from_origin_airport()
        if origin_airport_check:
            all_violations.append(origin_airport_check)

        destination_airport_check = self._is_to_destination_airport()
        if destination_airport_check:
            all_violations.append(destination_airport_check)

        slow_speed_check = self._is_too_slow()
        all_violations.extend(slow_speed_check)

        fast_speed_check = self._is_too_fast()
        if fast_speed_check:
            all_violations.append(fast_speed_check)

        altitude_profile_check = self._is_expected_altitude_profile()
        all_violations.extend(altitude_profile_check)

        return all_violations
