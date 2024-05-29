SELECT {columns} FROM `bigquery-public-data.cloud_storage_geo_index.sentinel_2_index` WHERE 
   sensing_time >= "{start_time}" AND
   sensing_time <= "{end_time}" AND
   ST_INTERSECTSBOX(ST_GEOGFROMGEOJSON("{geojson_str}"), west_lon, south_lat, east_lon, north_lat)
   ORDER BY sensing_time ASC
