import pandas as pd
import numpy as np
from tqdm import tqdm
# from pycontrails.datalib.ads_b.common import unique_flight_waypoints

WAYPOINTS_PREFIX = "F:/Spire/data-hi-res"
# TODO: log?


def load_waypoints(t_start: pd.Timestamp, t_end: pd.Timestamp, *, dt: int = "300") -> pd.DataFrame:
    df_waypoints = list()
    t_slices = pd.date_range(start=t_start, end=t_end, freq=f"{dt}S")

    for t in tqdm(t_slices):
        t_str = t.strftime("%Y%m%d-%H%M%S")
        rpath = f"{WAYPOINTS_PREFIX}/{t_str}.pq"

        try:
            df_wypts_t = pd.read_parquet(rpath)
        except FileNotFoundError:
            continue
        else:
            df_waypoints.append(df_wypts_t)

    df_waypoints = pd.concat(df_waypoints)

    #: Remove waypoints without aircraft type and tail number metadata
    #: Satellites waypoints should contain these information
    tn = df_waypoints["tail_number"].unique().astype(str)
    tn = tn[(tn != "None") & (tn != "VARIOUS")]
    df_waypoints = df_waypoints[df_waypoints["tail_number"].isin(tn)]

    atyps = df_waypoints["aircraft_type_icao"].unique().astype(str)
    atyps = atyps[(atyps != "N/A") & (atyps != "None")]
    df_waypoints = df_waypoints[df_waypoints["aircraft_type_icao"].isin(atyps)]
    return df_waypoints


def unique_flights_in_icao_address(df_flt_wypts: pd.DataFrame):
    flights = adsb.unique_flight_waypoints(df_flt_wypts, column="tail_number")
    # TODO: Separate by aircraft type
    # TODO: Separate by callsign, backfill satellites


    # TODO: For each ICAO address, identify unique flights
    return


t_start = pd.to_datetime("2022-02-01 00:00:00")
t_end = pd.to_datetime("2022-02-01 06:00:00")

df_wypts = load_waypoints(t_start, t_end)

# Separate flights by ICAO address
flts_icao_address = list(df_wypts.groupby("icao_address"))

print(" ")


"""
Omitted codes

# TODO: Filter valid waypoints
    # Identify unique flights by callsign
    callsigns = df_waypoints["callsign"].unique().astype(str)
    is_valid = (np.char.str_len(callsigns) > 1) & (~np.char.isspace(callsigns))
    callsigns = callsigns[is_valid]

for icao_add, df_flt_wypts in tqdm(flts_icao_address):

    print(" ")
"""


