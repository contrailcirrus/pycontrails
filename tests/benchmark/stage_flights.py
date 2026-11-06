"""Stage GAIA flight trajectories for the CoCiP benchmark.

This script selects 2020-01-30 for the benchmark day based on
``tests/benchmark/day-selection.csv``.

Take every flight departing on 2020-01-30 that stays inside the selection box
(25W-45E, 30N-72N).

    python tests/benchmark/stage_flights.py

Write ``waypoints.pq`` and ``metadata.pq`` to ``OUTPUT``.
"""

from __future__ import annotations

import pandas as pd

PREFIX = "gs://contrails-301217-gaia-trajectories/2019-21-Spire/enhanced/2020"
OUTPUT = "gs://contrails-301217-benchmark-data/inputs"
DAY = pd.Timestamp("2020-01-30")
LONGITUDE = (-25.0, 45.0)
LATITUDE = (30.0, 72.0)


def main() -> None:
    """Select flights and stage their metadata and waypoints."""
    path = f"{PREFIX}/metadata/{DAY:%Y-%m-%d}-metadata.pq"
    filters = [
        ("first_waypoint_time", ">=", DAY),
        ("first_waypoint_time", "<", DAY + pd.Timedelta(days=1)),
        ("longitude_min", ">=", LONGITUDE[0]),
        ("longitude_max", "<=", LONGITUDE[1]),
        ("latitude_min", ">=", LATITUDE[0]),
        ("latitude_max", "<=", LATITUDE[1]),
    ]
    metadata = pd.read_parquet(path, filters=filters)

    paths = [
        f"{PREFIX}/waypoints/{day:%Y-%m-%d}-waypoints.pq"
        for day in (DAY, DAY + pd.Timedelta(days=1))
    ]
    columns = ["flight_id", "longitude", "latitude", "altitude_ft", "time"]
    filters = [("flight_id", "in", metadata["flight_id"].tolist())]
    waypoints = pd.concat(pd.read_parquet(p, columns=columns, filters=filters) for p in paths)
    waypoints = waypoints.sort_values(["flight_id", "time"])

    metadata.to_parquet(f"{OUTPUT}/metadata.pq", index=False)
    waypoints.to_parquet(f"{OUTPUT}/waypoints.pq", index=False)


if __name__ == "__main__":
    main()
