import pandas as pd
import numpy as np
import pycontrails.datalib.spire.data_cleaning as spire

from tqdm import tqdm

WAYPOINTS_PREFIX = "F:/Spire/data-hi-res"

# TODO: log?


t_start = pd.to_datetime("2022-02-01 00:00:00")
t_end = pd.to_datetime("2022-02-01 06:00:00")

df_waypoints = list()
t_slices = pd.date_range(start=t_start, end=t_end, freq="300S")

# TODO: Can be parallelised
for t in tqdm(t_slices):
    t_str = t.strftime("%Y%m%d-%H%M%S")
    rpath = f"{WAYPOINTS_PREFIX}/{t_str}.pq"
    df_waypoints_t = spire.load_raw_ads_b_file(rpath)
    df_waypoints.append(df_waypoints_t)

df_waypoints = pd.concat(df_waypoints)
df_waypoints = spire.clean_raw_ads_b_file(df_waypoints)

# Separate flights by ICAO address
flts_icao_address = list(df_waypoints.groupby("icao_address"))
# df_flt_wypts = flts_icao_address[101][1]

df_flt_wypts = flts_icao_address[1020][1]
len(df_flt_wypts)
test = spire.identify_and_categorise_unique_flights(df_flt_wypts, t_cut_off=df_waypoints["timestamp"].max())
print(" ")

# TODO: UNIT TEST!
