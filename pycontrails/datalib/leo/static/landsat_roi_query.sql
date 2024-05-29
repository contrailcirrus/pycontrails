SELECT {columns} FROM `bigquery-public-data.cloud_storage_geo_index.landsat_index` WHERE 
   sensing_time >= "{start_time}" AND
   sensing_time <= "{end_time}" AND
   spacecraft_id IN ("LANDSAT_8", "LANDSAT_9") AND
   ST_INTERSECTSBOX(ST_GEOGFROMGEOJSON("{geojson_str}"), west_lon, south_lat, east_lon, north_lat)
   ORDER BY sensing_time ASC
