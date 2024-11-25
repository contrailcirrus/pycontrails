"""Custom exceptions."""


class BadTrajectoryException(Exception):
    """Exception indicating a trajectory (flight instance) is invalid."""


class SchemaError(Exception):
    """Data object is inconsistent with required schema."""


class OrderingError(Exception):
    """Data object has incorrect ordering."""


class OriginAirportError(Exception):
    """
    Trajectory is not originating at expected location.

    We do not assume that the departure airports are invariant in the dataframe,
    thus we handle the case of multiple airports listed.
    """


class DestinationAirportError(Exception):
    """Trajectory is not terminating at expected location."""


class FlightTooShortError(Exception):
    """Trajectory is unreasonably short in flight time."""


class FlightTooLongError(Exception):
    """Trajectory is unreasonably long in flight time."""


class FlightTooSlowError(Exception):
    """Trajectory has period(s) of unrealistically slow speed."""


class FlightTooFastError(Exception):
    """Trajectory has period(s) of unrealistically high speed."""


class ROCDError(Exception):
    """Trajectory has an unrealistic rate of climb or descent."""


class FlightAltitudeProfileError(Exception):
    """Trajectory has an unrealistic rate of climb or descent."""


class FlightDuplicateTimestamps(Exception):
    """Trajectory contains waypoints with the same timestamp."""


class FlightInvariantFieldViolation(Exception):
    """Trajectory has multiple values for field(s) that should be invariant."""
