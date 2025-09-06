SELECT {columns} FROM `{table}` WHERE 
   sensing_time >= "{start_time}" AND
   sensing_time <= "{end_time}" AND
   ST_INTERSECTSBOX(ST_GEOGFROMGEOJSON("{geojson_str}"), west_lon, south_lat, east_lon, north_lat)
   {extra_filters}
   ORDER BY sensing_time ASC
