import pandas as pd
import numpy as np
from tqdm import tqdm
from pycontrails.datalib.spire.separate_flights import identify_and_categorise_unique_flights

WAYPOINTS_PREFIX = "F:/Spire/data-hi-res"
COLUMNS_REQ = [
    'icao_address', 'timestamp', 'latitude', 'longitude', 'altitude_baro', 'heading', 'speed',  'on_ground',
    'callsign', 'tail_number', 'collection_type', 'aircraft_type_icao', 'aircraft_type_name',
    'airline_iata', 'airline_name', 'departure_utc_offset', 'departure_scheduled_time'
]
# TODO: log?


def load_waypoints(t_start: pd.Timestamp, t_end: pd.Timestamp, *, dt: int = "300") -> pd.DataFrame:
    df_waypoints = list()
    t_slices = pd.date_range(start=t_start, end=t_end, freq=f"{dt}S")

    for t in tqdm(t_slices):
        t_str = t.strftime("%Y%m%d-%H%M%S")
        rpath = f"{WAYPOINTS_PREFIX}/{t_str}.pq"

        try:
            df_waypoints_t = pd.read_parquet(rpath, columns=COLUMNS_REQ)
        except FileNotFoundError:
            continue
        else:
            df_waypoints.append(df_waypoints_t)

    df_waypoints = pd.concat(df_waypoints)
    df_waypoints["timestamp"] = pd.to_datetime(df_waypoints["timestamp"])
    df_waypoints.sort_values(by=["timestamp"], ascending=True, inplace=True)

    #: Remove waypoints without altitude data
    df_waypoints = df_waypoints[df_waypoints["altitude_baro"].notna()]

    #: Remove waypoints without aircraft type and tail number metadata
    #: Satellites waypoints should contain these data
    tn = df_waypoints["tail_number"].unique()
    tn = tn[(tn != "None") & (tn != "VARIOUS")]
    df_waypoints = df_waypoints[df_waypoints["tail_number"].isin(tn)]

    atyps = df_waypoints["aircraft_type_icao"].unique()
    atyps = atyps[(atyps != "N/A") & (atyps != "None")]
    df_waypoints = df_waypoints[df_waypoints["aircraft_type_icao"].isin(atyps)]

    # Remove terrestrial waypoints without callsign


    df_waypoints.reset_index(inplace=True, drop=True)



    # TODO: Assert "on_ground" is bool?
    # TODO: Remove aircraft types not covered by BADA 3
    return df_waypoints


t_start = pd.to_datetime("2022-02-01 00:00:00")
t_end = pd.to_datetime("2022-02-01 06:00:00")

df_wypts = load_waypoints(t_start, t_end)

# Separate flights by ICAO address
flts_icao_address = list(df_wypts.groupby("icao_address"))
# df_flt_wypts = flts_icao_address[101][1]
# TODO: UNIT TEST!
df_flt_wypts = flts_icao_address[1020][1]
len(df_flt_wypts)
test = identify_and_categorise_unique_flights(df_flt_wypts, t_cut_off=df_wypts["timestamp"].max())
print(" ")


