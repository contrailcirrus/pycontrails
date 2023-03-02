# Data

Cocip output data for 100 global flights on 2020-01-01.


## Inputs

### Flight waypoint data

- `flight_id` (str): Flight identifier
- `longitude` (float64): Longitude of waypoint (EPSG:4326)
- `latitude` (float64): Latitude of waypoint (EPSG:4326)
- `altitude_ft` (float64): Altitude of waypoint (ft)
- `time` (datetime64[ns], or str): Waypoint timestamp. Formatted as a str in CSV.
- `time_unix` (int64): Waypoint timestamp, as unix epoch (seconds)

### Flight metadata

- `callsign` (str): Callsign
- `icao_address` (str): ICAO identifier
- `flight_number` (str): Flight number
- `tail_number` (str): Tail number
- `aircraft_type_icao` (str): ICAO aircraft identifier
- `aircraft_engine_type` (str): Engine type
- `origin_airport` (str): Origin airport identifier
- `origin_airport_name` (str): Origin airport long name
- `origin_country` (str): Origin country identifier
- `destination_airport` (str): Destination airport identifier
- `destination_airport_name` (str): Destination airport long name
- `destination_country` (str): Destination country identifier
- `time_first_waypoint` (datetime64[ns], or str): First waypoint timestamp. Formatted as a str in CSV.
- `time_last_waypoint` (datetime64[ns], or str): Last waypoint timestamp. Formatted as a str in CSV.
- `duration_hours` (float64): Flight duration (hrs)
- `total_distance_km` (float64): Total distance (km)
- `longitude_first_waypoint` (float64): Longitude of the first waypoint (EPSG:4326)
- `longitude_last_waypoint` (float64): Longitude of the last waypoint (EPSG:4326)
- `longitude_min` (float64): Minimum longitude along flight (EPSG:4326)
- `longitude_max` (float64): Maximum longitude along flight (EPSG:4326)
- `latitude_first_waypoint` (float64): Latitude of the first waypoint (EPSG:4326)
- `latitude_last_waypoint` (float64): Latitude of the last waypoint (EPSG:4326)
- `latitude_min` (float64): Minimum latitude along flight (EPSG:4326)
- `latitude_max` (float64): Maximum latitude along flight (EPSG:4326)
- `altitude_first_waypoint` (float64): Altitude of the first waypoint (ft)
- `altitude_last_waypoint` (float64): Altitude of the last waypoint (ft)
- `altitude_min` (float64): Minimum altitude along the flight (ft)
- `altitude_max` (float64): Maximum altitude along the flight (ft)
- `n_waypoints` (int64): Total number of waypoints
- `aircraft_type_bada` (str): BADA Aircraft identifier
- `engine_type_edb` (object): ICAO EDB engine identifier
- `engine_uid` (object): Engine uid
- `n_engine` (int64): Number of engines
- `assumed_load_factor` (float64): Assumed aircraft load factor
- `aircraft_mass_initial_kg` (float64): Estimated initial aircraft mass


### Meteorology and Radiation Data

Meteorology and radiation data is sourced from [ECMWF ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) and downloaded from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/#!/home).

See the [ECMWF Parameter Database](https://apps.ecmwf.int/codes/grib/param-db) for more information on each parameter.

Files are separated by time, one for each hour.

**met/**

Data is provided for pressure levels [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100] hPa.

The full dataset has dimensions and shape `(time: 33, level: 27, latitude: 721, longitude: 1440)`.


```
t        (time, level, latitude, longitude) float32>
q        (time, level, latitude, longitude) float32>
u        (time, level, latitude, longitude) float32>
v        (time, level, latitude, longitude) float32>
w        (time, level, latitude, longitude) float32>
ciwc     (time, level, latitude, longitude) float32>
z        (time, level, latitude, longitude) float32>
```

**rad/**

The full dataset has dimensions and shape `(time: 33, latitude: 721, longitude: 1440)`.

```
tsr      (time, latitude, longitude) float32>
ttr      (time, latitude, longitude) float32>
```

### Model Parameters

All [Cocip parameters set to defaults](https://py.contrails.earth/api/pycontrails.models.cocip.CocipParams.html#pycontrails.models.cocip.CocipParams) except as overridden in [benchmark.py](benchmark.py).
See `run_cocip()` for model parameter overrides.


## Outputs

Values are set to `NaN` when there is no data.

This happens when the flight waypoint is out of the meteorology domain, or for model intermediates that were not calculated.

### Flight waypoint data

**Position**

- `flight_id` (str): Flight identifier
- `waypoint` (int64): Waypoint id
- `time` (datetime64[ns]): Waypoint timestamp (UTC).
- `time_unix` (int64): Waypoint timestamp (UTC), as unix epoch (seconds)
- `longitude` (float64): Longitude of waypoint (EPSG:4326)
- `latitude` (float64): Latitude of waypoint (EPSG:4326)
- `level` (float64): Pressure level of waypoint (hPa)
- `altitude` (float64): Altitude of waypoint (m)
- `altitude_ft` (float64): Altitude of waypoint (ft)
- `air_pressure` (float64): Air pressure at waypoint (Pa)
- `segment_length` (float64): Haversine distance between current flight waypoint and next (m). The last waypoint in a flight has a segment length of 0.
- `sin_a` (float64): Sine of the segment angle with the longitudinal axis
- `cos_a` (float64): Cosine the Segment angle with the longitudinal axis

**Aircraft performance**

- `wingspan` (float32): Wingspan (m)
- `true_airspeed` (float64): True airspeed (m s^-1)
- `mach_number` (float64): Mach number
- `aircraft_mass` (float64): Estimated aircraft mass (kg)
- `fuel_flow` (float64): Fuel mass flowrate (kg s^-1)
- `fuel_burn` (float64): Total fuel burn between two waypoints (kg)
- `thrust` (float64): Total thrust force (N)
- `engine_efficiency` (float64): Overall propulsion efficiency
- `rocd` (float64): Rate of climb or descent (ft min^-1)

**Emissions**

- `nvpm_ei_n` (int64): Non-volatile number emissions index (kg^-1)
- `co` (float64): Carbon monoxide emissions (kg)
- `co2` (float64): Carbon dioxide emissions (kg)
- `h2o` (float64): Water vapor emissions (kg)
- `nox` (float64): NOx emissions (kg)
- `oc` (float64): Organic carbon emissions (kg)
- `so2` (float64): Sulfur dioxide emissions (kg)
- `sulphates` (float64): Sulphate emissions (kg)

**Meteorology**

- `air_temperature` (float64): Ambient air temperature at waypoint (K)
- `specific_humidity` (float64): Specific humidity at waypoint (kg_{H20v} kg_{moist air}^1)
- `u_wind` (float64): Eastward wind at waypoint (m s^1)
- `v_wind` (float64): Northward wind at waypoint (m s^1)
- `specific_cloud_ice_water_content` (float64): Cloud ice water content at waypoint (kg_{ice} kg_{moist air}^1)
- `rho_air` (float64): Air density at waypoint (kg m**-3)
- `tau_cirrus` (float64): Estimated optical depth of existing cirrus at flight waypoint

**Initial contrail formation**

- `G` (float64): Slope of the mixing line in the temperature-humidity diagram
- `T_sat_liquid` (float64): Temperature where the tangent line to the liquid saturation curve is slope G, (K). In  literature, this quantity is `T_LM`.
- `rh` (float64): Relative humidity with respect to liquid water at the flight waypoint
- `rhi` (float64): Relative humidity with respect to ice at the flight waypoint
- `rh_critical_sac` (float64): Relative humidity threshold of SAC criteria
- `T_critical_sac` (float64): Temperature threshold of SAC criteria (K)
- `sac` (float64): 1 if flight waypoint satisfies the Schmidt-Appleman criteria, 0 if not
- `width` (float64): Initial contrail width (m)
- `depth` (float64): Initial contrail depth (m)
- `dT_dz` (float64): Potential temperature gradient between original altitude and initial contrail altitude  (K m^-1)
- `ds_dz` (float64): Wind shear (m s^-1)
- `dz_max` (float64): Maximum contrail downward displacement after the wake vortex phase (m)
- `rhi_1` (float64): Relative humidity over ice after wake vortex downwash
- `air_temperature_1` (float64): Air temperature after wake vortex downwash (K)
- `specific_humidity_1` (float64): Specific humidity after wake vortex downwash (kg_{H20v} kg_{ moist air}^1)
- `altitude_1` (float64): Altitude of contrail waypoint after wake vortex downwash
- `rho_air_1` (float64): Air density of contrail waypoint after wake vortex downwash (kg m^-3)
- `n_ice_per_m_1` (float64): Number of ice particles per distance flight after wake vortex downwash (m^-1)
- `iwc_1` (float64): Ice water content after wake vortex downwash (kg_{ice} kg_{moist air}^1)
- `persistent_1` (float64): 1 if contrail waypoint is persistent after wake vortex downwash, 0 if not persistent

**Radiation**

- `olr_mean` (float64): Mean outgoing longwave radiation (OLR) at contrail waypoints resulting from flight waypoint (W m^-2)
- `olr_min` (float64): Min OLR at contrail waypoints resulting from flight waypoint (W m^-2)
- `olr_max` (float64): Max OLR at contrail waypoints resulting from flight waypoint (W m^-2)
- `sdr_mean` (float64): Mean shortwave direct radiation (SDR) at contrail waypoints resulting from flight waypoint (W m^-2)
- `sdr_min` (float64): Min SDR at contrail waypoints resulting from flight waypoint (W m^-2)
- `sdr_max` (float64): Max SDR at contrail waypoints resulting from flight waypoint (W m^-2)
- `rsr_mean` (float64): Mean reflected solar radiation (RSR) at contrail waypoints resulting from flight waypoint (W m^-2)
- `rsr_min` (float64): Min RSR at contrail waypoints resulting from flight waypoint (W m^-2)
- `rsr_max` (float64): Max RSR at contrail waypoints resulting from flight waypoint (W m^-2)
- `rf_sw_mean` (float64): Mean shortwave contrail radiative forcing resulting from flight waypoint (W m^-2)
- `rf_sw_min` (float64): Min shortwave contrail radiative forcing resulting from flight waypoint (W m^-2)
- `rf_sw_max` (float64): Max shortwave contrail radiative forcing resulting from flight waypoint (W m^-2)
- `rf_lw_mean` (float64): Mean longwave contrail radiative forcing resulting from flight waypoint (W m^-2)
- `rf_lw_min` (float64): Min longwave contrail radiative forcing resulting from flight waypoint (W m^-2)
- `rf_lw_max` (float64): Max longwave contrail radiative forcing resulting from flight waypoint (W m^-2)
- `rf_net_mean` (float64): Mean net contrail radiative forcing resulting from flight waypoint (W m^-2)
- `rf_net_min` (float64): Min net contrail radiative forcing resulting from flight waypoint (W m^-2)
- `rf_net_max` (float64): Max net contrail radiative forcing resulting from flight waypoint (W m^-2)

**Outputs**

- `cocip` (float64): -1 if flight waypoint is net cooling, 1 if net warming, 0 if no impact.
- `ef` (float64): Sum of contrail energy forcing resulting from flight waypoint (J)
- `contrail_age` (float64): Max age of the contrail resulting from flight waypoint (s)


### Contrail data

Contrail waypoints correspond to the start and end of contrail segments at a point in time.
Contrail waypoints are indexed by the original flight waypoint (`waypoint`) and the timestep of evolution (`timestep`).

Contrail segment properties (e.g. `segment_length`, `ef`) are attached to the *first* contrail waypoint in the segment.
The *last* waypoint in a continuous contrail segment will have segment properties set to 0.

**Position & Geometry**

- `flight_id` (str): Flight ID of the contrail forming flight
- `waypoint` (int64): 0-based index of the original flight waypoint associated with contrail waypoint
- `time` (datetime64[ns]): Contrail waypoint timestamp (UTC).
- `time_unix` (int64): Contrail waypoint timestamp (UTC), as unix epoch (seconds)
- `formation_time` (datetime64[ns]): Flight waypoint time (UTC) of initial contrail formation
- `formation_time_unix` (int64): Flight waypoint time (UTC) of initial contrail formation, as unix epoch (seconds)
- `longitude` (float64): Longitude of contrail waypoint (EPSG:4326)
- `latitude` (float64):  Latitude of contrail waypoint (EPSG:4326)
- `altitude` (float64): Altitude of contrail waypoint (m)
- `level` (float64): Pressure level of contrail waypoint (hPa)
- `continuous` (int64): 1 if the contrail waypoint creates a segment with the next contrail waypoint at the current timestep, 0 if waypoint does not form a continuous segment.
- `segment_length` (float64): Haversine distance between current contrail waypoint and next (m). The last contrail waypoint in a continuous set has a segment length of 0.
- `sin_a` (float64): Sine of the segment angle with the longitudinal axis
- `cos_a` (float64): Cosine the segment angle with the longitudinal axis
- `width` (float64): Contrail width of segment (m)
- `depth` (float64):  Contrail depth of segment (m)

**Meteorology**

- `air_temperature` (float64): Ambient air temperature at contrail waypoint (K)
- `air_pressure` (float64): Air pressure at contrail waypoint (Pa)
- `specific_humidity` (float64): Specific humidity at contrail waypoint (kg_{H20v} kg_{moist air}^1)
- `rho_air` (float64): Air density at contrail waypoint (kg m^-3)
- `q_sat` (float64): Saturation specific humidity over ice at contrail waypoint (kg_{H20v} kg_{moist air}^1)
- `rhi` (float64): Relative humidity over ice at contrail waypoint
- `iwc` (float64): Ice water content after wake vortex downwash (kg_{ice} kg_{moist air}^1)
- `u_wind` (float64): Eastward wind at contrail waypoint (m s^-1)
- `v_wind` (float64): Northward wind at contrail waypoint (m s^-1)
- `vertical_velocity` (float64): Vertical wind velocity at contrail waypoint (Pa s^-1)

**Evolution & Persistence**

- `sigma_yz` (float64): The covariance of the ice concentration field in the `yz` plane
- `air_temperature_lower` (float64): Air temperature 200m below contrail waypoint (m)
- `u_wind_lower` (float64): Eastward wind 200m below contrail waypoint (m s^-1)
- `v_wind_lower` (float64): Northward wind 200m below contrail waypoint (m s^-1)
- `dT_dz` (float64): Potential temperature gradient between contrail waypoint and 200m below contrail waypoint (K m^-1)
- `ds_dz` (float64): Wind shear (m s^-1)
- `dsn_dz` (float64): Wind shear normal to the contrail orientation (m s^-1)
- `diffuse_h` (float64): Contrail horizontal diffusivity (m^2 s^-1)
- `diffuse_v` (float64): Contrail vertical diffusivity (m^2 s^-1)
- `n_ice_per_vol` (float64): Number of contrail ice particles per volume of plume (m^-3)
- `n_ice_per_m` (float64): Number of ice particles per contrail length (m^-1)
- `terminal_fall_speed` (float64): Terminal fall speed of contrail ice particles (m s^-1)
- `area_eff` (float64): Effective cross-sectional area of the contrail plume (m^2)
- `plume_mass_per_m` (float64): Contrail plume mass per contrail length (kg m^-1)
- `r_ice_vol` (float64): Ice particle volume mean radius (m)
- `dn_dt_agg` (float64): Rate of contrail ice particle losses due to sedimentation-induced aggregation (s^-1)
- `dn_dt_turb` (float64): Rate of contrail ice particle losses due to plume-internal turbulence (s^-1)
- `heat_rate` (float64): Radiative heating rate affecting the contrail plume (K s^-1)
- `d_heat_rate` (float64): Differential heating rate between the upper and lower half of the contrail plume (K s^-1)

**Radiative Forcing**

- `tau_cirrus` (float64):  Optical depth of the natural cirrus above the contrail
- `tau_contrail` (float64): Contrail optical depth
- `sdr` (float64): Incident solar (shortwave) radiation (W m^-2). Values calculated theoretically.
- `top_net_solar_radiation` (float64): Net incoming solar (shortwave) radiation (W m^-2). Values interpolated from ECMWF ERA5.
- `rsr` (float64): Time-average reflected shortwave radiation (W m^-2)
- `top_net_thermal_radiation` (float64): Net thermal (longwave) radiation (W m^-2). Values interpolated from ECMWF ERA5.
- `olr` (float64): Time-average outgoing longwave radiation (W m^-2)
- `rf_sw` (float64): Local shortwave radiative forcing (W m^-2)
- `rf_lw` (float64): Local longwave radiative forcing (W m^-2)
- `rf_net` (float64): Local net radiative forcing of the contrail segment (W m^-2)

**Outputs**

- `timestep` (int64): Model integration timestep, as an integer index
- `persistent` (int64): 1 if the contrail waypoint is persistent until the next timestep, 0 if not
- `dt_integration` (int64): Integration time between last timestep and current timestep (seconds)
- `age` (float64): Contrail age (seconds)
- `ef` (float64): Energy forcing of the contrail segment integrated over the contrail area (width x length) and integration time (J)

