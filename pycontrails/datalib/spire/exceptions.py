"""Custom exceptions used for spire data validation."""


class BaseSpireError(Exception):
    """Base class for all spire exceptions."""


class BadTrajectoryException(BaseSpireError):
    """A generic exception indicating a trajectory (flight instance) is invalid."""


class SchemaError(BaseSpireError):
    """Data object is inconsistent with required schema."""


class OrderingError(BaseSpireError):
    """Data object has incorrect ordering."""


class OriginAirportError(BaseSpireError):
    """
    Trajectory is not originating at expected location.

    We do not assume that the departure airports are invariant in the dataframe,
    thus we handle the case of multiple airports listed.
    """


class DestinationAirportError(BaseSpireError):
    """Trajectory is not terminating at expected location."""


class FlightTooShortError(BaseSpireError):
    """Trajectory is unreasonably short in flight time."""


class FlightTooLongError(BaseSpireError):
    """Trajectory is unreasonably long in flight time."""


class FlightTooSlowError(BaseSpireError):
    """Trajectory has period(s) of unrealistically slow speed."""


class FlightTooFastError(BaseSpireError):
    """Trajectory has period(s) of unrealistically high speed."""


class ROCDError(BaseSpireError):
    """Trajectory has an unrealistic rate of climb or descent."""


class FlightAltitudeProfileError(BaseSpireError):
    """Trajectory has an unrealistic rate of climb or descent."""


class FlightDuplicateTimestamps(BaseSpireError):
    """Trajectory contains waypoints with the same timestamp."""


class FlightInvariantFieldViolation(BaseSpireError):
    """Trajectory has multiple values for field(s) that should be invariant."""
