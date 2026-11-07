"""Stage GAIA flight trajectories for the CoCiP benchmark.

This script selects 2020-01-30 for the benchmark day based on
``tests/benchmark/day-selection.csv``.

Take every flight departing on 2020-01-30 that stays inside the selection box
(25W-45E, 30N-72N). Drop duplicated times, skip flights shorter than ``MIN_DURATION``,
resample to ``RESAMPLE_FREQ`` and attach ch-aviation aircraft properties.

    python tests/benchmark/stage_flights.py

Write ``waypoints.pq`` and ``metadata.pq`` to ``OUTPUT``.
"""

from __future__ import annotations

import pandas as pd
import tqdm

import pycontrails
from pycontrails.datalib import ch_aviation

PREFIX = "gs://contrails-301217-gaia-trajectories/2019-21-Spire/enhanced/2020"
OUTPUT = "gs://contrails-301217-benchmark-data/inputs"
DAY = pd.Timestamp("2020-01-30")
LONGITUDE = (-25.0, 45.0)
LATITUDE = (30.0, 72.0)
RESAMPLE_FREQ = "60s"
MIN_DURATION = pd.Timedelta(minutes=20)


def main() -> None:
    """Select, resample and stage flights with their ch-aviation properties."""
    path = f"{PREFIX}/metadata/{DAY:%Y-%m-%d}-metadata.pq"
    filters = [
        ("first_waypoint_time", ">=", DAY),
        ("first_waypoint_time", "<", DAY + pd.Timedelta(days=1)),
        ("longitude_min", ">=", LONGITUDE[0]),
        ("longitude_max", "<=", LONGITUDE[1]),
        ("latitude_min", ">=", LATITUDE[0]),
        ("latitude_max", "<=", LATITUDE[1]),
    ]
    metadata = pd.read_parquet(path, filters=filters).set_index("flight_id")

    paths = [
        f"{PREFIX}/waypoints/{day:%Y-%m-%d}-waypoints.pq"
        for day in (DAY, DAY + pd.Timedelta(days=1))
    ]
    columns = ["flight_id", "longitude", "latitude", "altitude_ft", "time"]
    filters = [("flight_id", "in", metadata.index.tolist())]
    waypoints = pd.concat(pd.read_parquet(p, columns=columns, filters=filters) for p in paths)
    waypoints = waypoints.sort_values(["flight_id", "time"])

    cha = ch_aviation.ChAviation(copy_source=False)
    dfs = []
    attrs = []
    for flight_id, df in tqdm.tqdm(waypoints.groupby("flight_id")):
        df = df.drop_duplicates(subset="time")
        row = metadata.loc[flight_id]
        fl = pycontrails.Flight(
            df,
            flight_id=flight_id,
            aircraft_type=row["aircraft_type_icao"],
            tail_number=row["tail_number"],
            icao_address=row["icao_address"],
            origin_airport_icao=row["origin_airport"],
            destination_airport_icao=row["destination_airport"],
        )
        if fl.duration < MIN_DURATION:
            continue

        fl = fl.resample_and_fill(RESAMPLE_FREQ)
        cha.eval(fl)
        dfs.append(fl.dataframe.assign(flight_id=flight_id))
        attrs.append(fl.attrs)

    pd.DataFrame(attrs).to_parquet(f"{OUTPUT}/metadata.pq", index=False)
    pd.concat(dfs).to_parquet(f"{OUTPUT}/waypoints.pq", index=False)


if __name__ == "__main__":
    main()
