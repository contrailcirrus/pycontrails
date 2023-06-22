"""CoCiP output formats.

This module includes functions to produce additional output formats,
including grid and time-slice outputs, and flight summary outputs.
"""
from __future__ import annotations

import warnings
import pandas as pd
import xarray as xr
import numpy as np
import numpy.typing as npt

from pycontrails.physics import geo
from pycontrails.core import Flight
from pycontrails import MetDataArray, MetDataset, GeoVectorDataset
from pycontrails.core.vector import vector_to_lon_lat_grid
from pycontrails.physics.units import m_to_pl
from pycontrails.physics.thermo import rho_d
from pycontrails.models.cocip.contrail_properties import plume_mass_per_distance
from pycontrails.models.cocip.radiative_forcing import albedo
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.models.tau_cirrus import tau_cirrus

# TODO: @Zeb would it be able to make Flight automatically attach `segment_length` in dataframe?


# -------------------------------------------------
# Main functions for different CoCiP output formats
# -------------------------------------------------


def flight_waypoint_outputs(
        flight_waypoints: GeoVectorDataset, contrails: GeoVectorDataset
) -> GeoVectorDataset:
    """
    Calculate the contrail summary statistics at each flight waypoint.

    Parameters
    ----------
    flight_waypoints : GeoVectorDataset
        Flight waypoints that were used in `cocip.eval` to produce `contrails`.
    contrails : GeoVectorDataset
        Contrail waypoint outputs from CoCiP, `cocip.contrail`.

    Returns
    -------
    GeoVectorDataset
        Contrail summary statistics attached to each flight waypoint.
    """
    # Aggregation map
    agg_map_contrails_to_flight_waypoints = {
        # Location, ambient meteorology and properties
        "altitude": "mean",
        "rhi": ["mean", "std"],
        "n_ice_per_m": ["mean", "std"],
        "r_ice_vol": "mean",
        "width": "mean",
        "depth": "mean",
        "tau_contrail": "mean",
        "tau_cirrus": "mean",
        "age_h": "max",

        # Radiative properties
        "rf_sw": "mean",
        "rf_lw": "mean",
        "rf_net": "mean",
        "ef": "sum",
        "olr": "mean",
        "sdr": "mean",
        "rsr": "mean",
    }

    # Check and pre-process `flights` variable
    flight_waypoints.ensure_vars(["flight_id", "waypoint"])
    flight_waypoints = flight_waypoints.dataframe
    flight_waypoints.set_index(["flight_id", "waypoint"], inplace=True)

    # Check and pre-process `contrails` variable
    contrail_vars = (
            ["flight_id", "waypoint", "formation_time"]
            + list(agg_map_contrails_to_flight_waypoints.keys())
    )
    contrail_vars.remove("age_h")
    contrails.ensure_vars(contrail_vars)
    contrails['age_h'] = (contrails['time'] - contrails['formation_time']) / np.timedelta64(1, 'h')

    # Calculate contrail statistics at each flight waypoint
    contrails = contrails.dataframe.copy()
    contrails = contrails.groupby(["flight_id", "waypoint"]).agg(
        agg_map_contrails_to_flight_waypoints
    )
    contrails.columns = (
            contrails.columns.get_level_values(1) + "_" + contrails.columns.get_level_values(0)
    )
    rename_cols = {
        "mean_altitude": "mean_contrail_altitude",
        "max_age_h": "contrail_age",
        "sum_ef": "ef"
    }
    contrails.rename(columns=rename_cols, inplace=True)

    # Concatenate to flight-waypoint outputs
    flight_waypoints = flight_waypoints.join(contrails, how="left")
    flight_waypoints.reset_index(inplace=True)
    return GeoVectorDataset(flight_waypoints, copy=True)


def flight_summary_outputs(
        flight_waypoints: GeoVectorDataset,
) -> pd.DataFrame:
    # TODO: Documentation

    # Aggregation map
    agg_map_flight_waypoints_to_summary = {
        # Trajectory and aircraft performance
        "altitude": ["min", "max"],
        "time": ["first", "last"],
        "segment_length": "sum",
        "aircraft_mass": "mean",
        "engine_efficiency": "mean",

        # Contrail properties and ambient meteorology
        "contrails_km": "sum",
        "persistent_contrails_km": "sum",
        "mean_contrail_altitude": "mean",
        "mean_rhi": "mean",
        "mean_n_ice_per_m": "mean",
        "mean_r_ice_vol": "mean",
        "mean_width": "mean",
        "mean_depth": "mean",
        "mean_tau_contrail": "mean",
        "mean_tau_cirrus": "mean",
        "max_age_h": ["mean", "max"],

        # Radiative properties
        "mean_rf_sw": "mean",
        "mean_rf_lw": "mean",
        "mean_rf_net": "mean",
        "ef": "sum",
        "mean_olr": "mean",
        "mean_sdr": "mean",
        "mean_rsr": "mean",
    }

    return


def longitude_latitude_grid(
        t_start: np.datetime64 | pd.Timestamp,
        t_end: np.datetime64 | pd.Timestamp,
        flight_waypoints: GeoVectorDataset,
        contrails: GeoVectorDataset, *,
        met: MetDataset,
        spatial_bbox: list[float] = [-180, -90, 180, 90],
        spatial_grid_res: float = 0.5,
) -> xr.Dataset:
    """
    Aggregate air traffic and contrail outputs to a longitude-latitude grid

    Parameters
    ----------
    t_start : np.datetime64 | pd.Timestamp
        UTC time at beginning of time step.
    t_end : np.datetime64 | pd.Timestamp
        UTC time at end of time step.
    flight_waypoints : GeoVectorDataset
        Flight waypoint outputs with contrail summary statistics attached.
        See :func:`flight_waypoint_outputs`.
    contrails : GeoVectorDataset
        Contrail waypoint outputs from CoCiP, `cocip.contrail`.
    met : MetDataset
        Pressure level dataset containing 'air_temperature', 'specific_humidity',
        'specific_cloud_ice_water_content', and 'geopotential'.
    spatial_bbox : list[float]
        Spatial bounding box, [lon_min, lat_min, lon_max, lat_max], [:math:`\deg`]
    spatial_grid_res : float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    xr.Dataset
        Air traffic and contrail outputs at a longitude-latitude grid.
    """
    # Ensure the required columns are included in `flight_waypoints`, `contrails` and `met`
    flight_waypoints.ensure_vars(('segment_length', 'ef'))
    contrails.ensure_vars(
        (
            'formation_time',
            'segment_length',
            'width',
            'tau_contrail',
            'rf_sw',
            'rf_lw',
            'rf_net',
            'ef'
        )
    )
    met.ensure_vars(
        (
            'air_temperature',
            'specific_humidity',
            'specific_cloud_ice_water_content',
            'geopotential'
        )
    )

    # Downselect `met` to specified spatial bounding box
    met = met.downselect(spatial_bbox)

    # Ensure that `flight_waypoints` and `contrails` are within `t_start` and `t_end`
    is_in_time = flight_waypoints.dataframe["time"].between(t_start, t_end, inclusive="right")
    if ~np.all(is_in_time):
        warnings.warn(
            "Flight waypoints have times that are outside the range of `t_start` and `t_end`. "
            "Waypoints outside the defined time bounds are removed. "
        )
        flight_waypoints = flight_waypoints.filter(is_in_time)

    is_in_time = contrails.dataframe["time"].between(t_start, t_end, inclusive="right")
    if ~np.all(is_in_time):
        warnings.warn(
            "Contrail waypoints have times that are outside the range of `t_start` and `t_end`."
            "Waypoints outside the defined time bounds are removed. "
        )
        contrails = contrails.filter(is_in_time)

    # Calculate additional variables
    dt_integration_sec = np.diff(np.unique(contrails["time"]))[0] / np.timedelta64(1, 's')

    da_area = geo.grid_surface_area(met["longitude"].values, met["latitude"].values)

    flight_waypoints["persistent_contrails"] = np.where(
        np.isnan(flight_waypoints["ef"]),
        0,
        flight_waypoints["segment_length"]
    )

    # ----------------
    # Grid aggregation
    # ----------------
    # (1) Waypoint properties between `t_start` and `t_end`
    is_between_time = flight_waypoints.dataframe["time"].between(t_start, t_end, inclusive="right")
    ds_wypts_t = vector_to_lon_lat_grid(
        flight_waypoints.filter(is_between_time, copy=True),
        agg={"segment_length": "sum", "persistent_contrails": "sum", "ef": "sum"},
        spatial_bbox=spatial_bbox,
        spatial_grid_res=spatial_grid_res
    )

    # (2) Contrail properties at `t_end`
    contrails_t_end = contrails.filter(contrails["time"] == t_end)

    contrails_t_end["tau_contrail_area"] = (
        contrails_t_end["tau_contrail"]
        * contrails_t_end["segment_length"]
        * contrails_t_end["width"]
    )

    contrails_t_end['age_h'] = (
        (contrails_t_end['time'] - contrails_t_end['formation_time'])
        / np.timedelta64(1, 'h')
    )

    ds_contrails_t_end = vector_to_lon_lat_grid(
        contrails_t_end,
        agg={"segment_length": "sum", "tau_contrail_area": "sum", "age_h": "mean"},
        spatial_bbox=spatial_bbox,
        spatial_grid_res=spatial_grid_res
    )
    ds_contrails_t_end["tau_contrail"] = ds_contrails_t_end["tau_contrail_area"] / da_area

    # (3) Contrail and natural cirrus coverage area at `t_end`
    ds_cirrus_coverage = cirrus_coverage_single_level(t_end, met, contrails)
    ds_cirrus_coverage = ds_cirrus_coverage.data.squeeze(dim=["level", "time"])

    # (4) Contrail climate forcing between `t_start` and `t_end`
    contrails["ef_sw"] = np.where(
        contrails["ef"] == 0,
        0,
        contrails["rf_sw"] * contrails["segment_length"] * contrails["width"] * dt_integration_sec
    )
    contrails["ef_lw"] = np.where(
        contrails["ef"] == 0,
        0,
        contrails["rf_lw"] * contrails["segment_length"] * contrails["width"] * dt_integration_sec
    )

    ds_forcing = vector_to_lon_lat_grid(
        contrails,
        agg={"ef_sw": "sum", "ef_lw": "sum", "ef": "sum"},
        spatial_bbox=spatial_bbox,
        spatial_grid_res=spatial_grid_res
    )
    ds_forcing["rf_sw"] = ds_forcing["ef_sw"] / (da_area * dt_integration_sec)
    ds_forcing["rf_lw"] = ds_forcing["ef_lw"] / (da_area * dt_integration_sec)
    ds_forcing["rf_net"] = ds_forcing["ef"] / (da_area * dt_integration_sec)

    # -----------------------
    # Package gridded outputs
    # -----------------------
    ds = xr.Dataset(
        data_vars=dict(
            flight_distance_flown=ds_wypts_t["segment_length"] / 1000,
            persistent_contrails_formed=ds_wypts_t["persistent_contrails"] / 1000,
            persistent_contrails=ds_contrails_t_end["segment_length"] / 1000,
            tau_contrail=ds_contrails_t_end["tau_contrail"],
            contrail_age=ds_contrails_t_end["age_h"],
            cc_natural_cirrus=ds_cirrus_coverage["natural_cirrus"],
            cc_contrails=ds_cirrus_coverage["contrails"],
            cc_contrails_clear_sky=ds_cirrus_coverage["contrails_clear_sky"],
            rf_sw=ds_forcing["rf_sw"] * 1000,
            rf_lw=ds_forcing["rf_lw"] * 1000,
            rf_net=ds_forcing["rf_net"] * 1000,
            contrail_ef=ds_forcing["ef"],
            contrail_ef_initial_loc=ds_wypts_t["ef"],
        ),
        coords=ds_wypts_t.coords
    )
    ds = ds.fillna(0)
    ds.expand_dims().assign_coords({"time": t_end})

    # Assign attributes
    ds["flight_distance_flown"].attrs = {
        "units": "km", "long_name": f"Total flight distance flown between t_start and t_end"
    }
    ds["persistent_contrails_formed"].attrs = {
        "units": "km", "long_name": "Persistent contrails formed between t_start and t_end"
    }
    ds["persistent_contrails"].attrs = {
        "units": "km", "long_name": "Persistent contrails at t_end"}
    ds["tau_contrail"].attrs = {
        "units": " ", "long_name": "Area-normalised mean contrail optical depth at t_end"
    }
    ds["contrail_age"].attrs = {"units": "h", "long_name": "Mean contrail age at t_end"}
    ds["cc_natural_cirrus"].attrs = {"units": " ", "long_name": "Natural cirrus cover at t_end"}
    ds["cc_contrails"].attrs = {"units": " ", "long_name": "Contrail cirrus cover at t_end"}
    ds["cc_contrails_clear_sky"].attrs = {
        "units": " ", "long_name": "Contrail cirrus cover under clear sky conditions at t_end"
    }
    ds["rf_sw"].attrs = {
        "units": "mW/m**2", "long_name": "Mean contrail cirrus shortwave radiative forcing at t_end"
    }
    ds["rf_lw"].attrs = {
        "units": "mW/m**2", "long_name": "Mean contrail cirrus longwave radiative forcing at t_end"
    }
    ds["rf_net"].attrs = {
        "units": "mW/m**2", "long_name": "Mean contrail cirrus net radiative forcing at t_end"}
    ds["ef"].attrs = {
        "units": "J", "long_name": "Total contrail energy forcing between t_start and t_end"}
    ds["ef_initial_loc"].attrs = {
        "units": "J",
        "long_name": "Total contrail energy forcing attributed back to the flight waypoint."
    }
    return ds


def time_slice_statistics(
        t_start: np.datetime64 | pd.Timestamp,
        t_end: np.datetime64 | pd.Timestamp,
        flight_waypoints: GeoVectorDataset,
        contrails: GeoVectorDataset, *,
        met: MetDataset | None = None,
        rad: MetDataset | None = None,
        humidity_scaling: HumidityScaling | None = None,
        spatial_bbox: list[float] = [-180, -90, 180, 90],
) -> pd.Series:
    """
    Calculate the flight and contrail summary statistics between `t_start` and `t_end`.

    Parameters
    ----------
    t_start : np.datetime64 | pd.Timestamp
        UTC time at beginning of time step.
    t_end : np.datetime64 | pd.Timestamp
        UTC time at end of time step.
    flight_waypoints : GeoVectorDataset
        Flight waypoint outputs.
    contrails : GeoVectorDataset
        Contrail waypoint outputs from CoCiP, `cocip.contrail`.
    met : MetDataset | None
        Pressure level dataset containing 'air_temperature', 'specific_humidity',
        'specific_cloud_ice_water_content', and 'geopotential'.
        Meteorological statistics will not be computed if `None` is provided.
    rad : MetDataset | None
        Single level dataset containing the `sdr`, `rsr` and `olr`.Radiation statistics
        will not be computed if `None` is provided.
    humidity_scaling : HumidityScaling
        Humidity scaling methodology.
        See :attr:`CocipParams.humidity_scaling`
    spatial_bbox : list[float]
        Spatial bounding box, [lon_min, lat_min, lon_max, lat_max], [:math:`\deg`]

    Returns
    -------
    pd.Series
        Flight and contrail summary statistics. Contrail statistics are provided at `t_end`.
        The units for each output are outlined in `Notes`.

    Notes
    -----
    Description of the outputs and units of provided by this function:
    - ``n_flights``, [dimensionless]
    - ``n_flights_forming_contrails``, [dimensionless]
    - ``n_flights_forming_persistent_contrails``, [dimensionless]
    - ``n_flights_with_persistent_contrails_at_t_end``, [dimensionless]

    - ``n_waypoints``, [dimensionless]
    - ``n_waypoints_forming_contrails``, [dimensionless]
    - ``n_waypoints_forming_persistent_contrails``, [dimensionless]
    - ``n_waypoints_with_persistent_contrails_at_t_end``, [dimensionless]
    - ``n_contrail_waypoints_at_night``, [dimensionless]
    - ``pct_contrail_waypoints_at_night``, [%]

    - ``total_flight_distance``, [:math:`km`]
    - ``total_contrails_formed``, [:math:`km`]
    - ``total_persistent_contrails_formed``, [:math:`km`]
    - ``total_persistent_contrails_at_t_end``, [:math:`km`]

    - ``total_fuel_burn``, [:math:`kg`]
    - ``mean_propulsion_efficiency_all_flights``, [dimensionless]
    - ``mean_propulsion_efficiency_flights_with_persistent_contrails``, [dimensionless]
    - ``mean_nvpm_ei_n_all_flights``, [:math:`kg^{-1}`]
    - ``mean_nvpm_ei_n_flights_with_persistent_contrails``, [:math:`kg^{-1}`]

    - ``mean_contrail_age``, [:math:`h`]
    - ``max_contrail_age``, [:math:`h`]
    - ``mean_n_ice_per_m``, [:math:`m^{-1}`]
    - ``mean_contrail_ice_water_path``, [:math:`kg m^{-2}`]
    - ``area_mean_contrail_ice_radius``, [:math:`\mu m`]
    - ``volume_mean_contrail_ice_radius``, [:math:`\mu m`]
    - ``mean_contrail_ice_effective_radius``, [:math:`\mu m`]
    - ``mean_tau_contrail``, [dimensionless]
    - ``mean_tau_cirrus``, [dimensionless]

    - ``mean_rf_sw``, [:math:`W m^{-2}`]
    - ``mean_rf_lw``, [:math:`W m^{-2}`]
    - ``mean_rf_net``, [:math:`W m^{-2}`]
    - ``total_contrail_ef``, [:math:`J`]

    - ``issr_percentage_coverage``, [%]
    - ``mean_rhi_in_issr``, [dimensionless]
    - ``contrail_cirrus_percentage_coverage``, [%]
    - ``contrail_cirrus_clear_sky_percentage_coverage``, [%]
    - ``natural_cirrus_percentage_coverage``, [%]
    - ``cloud_contrail_overlap_percentage``, [%]

    - ``mean_sdr_domain``, [:math:`W m^{-2}`]
    - ``mean_sdr_at_contrail_wypts``, [:math:`W m^{-2}`]
    - ``mean_rsr_domain``, [:math:`W m^{-2}`]
    - ``mean_rsr_at_contrail_wypts``, [:math:`W m^{-2}`]
    - ``mean_olr_domain``, [:math:`W m^{-2}`]
    - ``mean_olr_at_contrail_wypts``, [:math:`W m^{-2}`]
    - ``mean_albedo_at_contrail_wypts``, [dimensionless]
    """
    # Ensure the required columns are included in `flight_waypoints`, `contrails`, `met` and `rad`
    flight_waypoints.ensure_vars(('flight_id', 'segment_length', 'true_airspeed', 'fuel_flow'))
    contrails.ensure_vars(
        (
            'flight_id', 'segment_length', 'air_temperature', 'iwc', 'r_ice_vol',
            'n_ice_per_m', 'tau_contrail', 'tau_cirrus', 'width', 'area_eff',
            'sdr', 'rsr', 'olr', 'rf_sw', 'rf_lw', 'rf_net', 'ef',
        )
    )
    met.ensure_vars(
        (
            'air_temperature',
            'specific_humidity',
            'specific_cloud_ice_water_content',
            'geopotential'
        )
    )
    rad.ensure_vars(('sdr', 'rsr', 'olr'))

    # Downselect `met` and `rad` to specified spatial bounding box
    met = met.downselect(spatial_bbox)
    rad = rad.downselect(spatial_bbox)

    # Ensure that the waypoints are within `t_start` and `t_end`
    is_in_time = flight_waypoints.dataframe["time"].between(t_start, t_end, inclusive="right")
    if ~np.all(is_in_time):
        warnings.warn(
            "Flight waypoints have times that are outside the range of `t_start` and `t_end`. "
            "Waypoints outside the defined time bounds are removed. "
        )
        flight_waypoints = flight_waypoints.filter(is_in_time)

    is_in_time = contrails.dataframe["time"].between(t_start, t_end, inclusive="right")
    if ~np.all(is_in_time):
        warnings.warn(
            "Contrail waypoints have times that are outside the range of `t_start` and `t_end`."
            "Waypoints outside the defined time bounds are removed. "
        )
        contrails = contrails.filter(is_in_time)

    # Additional variables
    flight_waypoints['fuel_burn'] = (
            flight_waypoints['fuel_flow']
            * (1 / flight_waypoints['true_airspeed'])
            * flight_waypoints['segment_length']
    )
    contrails['pressure'] = m_to_pl(contrails['altitude'])
    contrails['rho_air'] = rho_d(
        contrails['air_temperature'],
        contrails['pressure']
    )
    contrails['plume_mass_per_m'] = plume_mass_per_distance(
        contrails['area_eff'], contrails['rho_air']
    )
    contrails['age_h'] = (contrails["time"] - contrails["formation_time"]) / np.timedelta64(1, 'h')

    # Meteorology domain statistics
    if met is not None:
        met_stats = meteorological_statistics(t_end, contrails, met, humidity_scaling)

    # Radiation domain statistics
    if rad is not None:
        rad_stats = radiation_statistics(rad, t_end)

    # Calculate time-slice statistics
    is_sac = flight_waypoints['sac'] == 1
    is_persistent = flight_waypoints['persistent_1'] == 1
    is_at_t_end = contrails['time'] == t_end
    is_night_time = contrails['sdr'] < 0.1
    domain_area = geo.domain_surface_area(spatial_bbox)

    stats_t = {
        'time_start': t_start,
        'time_end': t_end,

        # Flight statistics
        'n_flights': len(flight_waypoints.dataframe['flight_id'].unique()),
        'n_flights_forming_contrails': len(
            flight_waypoints.filter(is_sac).dataframe['flight_id'].unique()
        ),
        'n_flights_forming_persistent_contrails': len(
            flight_waypoints.filter(is_persistent).dataframe['flight_id'].unique()
        ),
        'n_flights_with_persistent_contrails_at_t_end': len(
            contrails.filter(is_at_t_end).dataframe['flight_id'].unique()
        ),

        # Waypoint statistics
        'n_waypoints': len(flight_waypoints),
        'n_waypoints_forming_contrails': len(flight_waypoints.filter(is_sac)),
        'n_waypoints_forming_persistent_contrails': len(flight_waypoints.filter(is_persistent)),
        'n_waypoints_with_persistent_contrails_at_t_end': len(contrails.filter(is_at_t_end)),
        'n_contrail_waypoints_at_night': (
            len(contrails.filter(is_at_t_end))
        ),
        'pct_contrail_waypoints_at_night': (
                len(contrails.filter(is_night_time)) / len(contrails) * 100
        ),

        # Distance statistics
        'total_flight_distance': np.nansum(flight_waypoints['segment_length']) / 1000,
        'total_contrails_formed': (
                np.nansum(flight_waypoints.filter(is_sac)['segment_length']) / 1000
        ),
        'total_persistent_contrails_formed': (
                np.nansum(flight_waypoints.filter(is_persistent)['segment_length']) / 1000
        ),
        'total_persistent_contrails_at_t_end': (
                np.nansum(contrails.filter(is_at_t_end)['segment_length']) / 1000
        ),

        # Aircraft performance statistics
        'total_fuel_burn': np.nansum(flight_waypoints['fuel_burn']),
        'mean_propulsion_efficiency_all_flights': np.nanmean(
            flight_waypoints['engine_efficiency']
        ),
        'mean_propulsion_efficiency_flights_with_persistent_contrails': np.nanmean(
            flight_waypoints.filter(is_persistent)['engine_efficiency']
        ) if np.any(is_persistent) else np.nan,

        'mean_nvpm_ei_n_all_flights': np.nanmean(flight_waypoints['nvpm_ei_n']),
        'mean_nvpm_ei_n_flights_with_persistent_contrails': np.nanmean(
            flight_waypoints.filter(is_persistent)['nvpm_ei_n']
        ) if np.any(is_persistent) else np.nan,

        # Contrail properties at `time_end`
        'mean_contrail_age': np.nanmean(
            contrails.filter(is_at_t_end)['age_h']
        ) if np.any(is_at_t_end) else np.nan,

        'max_contrail_age': np.nanmax(
            contrails.filter(is_at_t_end)['age_h']
        ) if np.any(is_at_t_end) else np.nan,

        'mean_n_ice_per_m': np.nanmean(
            contrails.filter(is_at_t_end)['n_ice_per_m']
        ) if np.any(is_at_t_end) else np.nan,

        'mean_contrail_ice_water_path': area_mean_ice_water_path(
            contrails.filter(is_at_t_end)['iwc'],
            contrails.filter(is_at_t_end)['plume_mass_per_m'],
            contrails.filter(is_at_t_end)['segment_length'],
            domain_area
        ) if np.any(is_at_t_end) else np.nan,

        'area_mean_contrail_ice_radius': area_mean_ice_particle_radius(
            contrails.filter(is_at_t_end)['r_ice_vol'],
            contrails.filter(is_at_t_end)['n_ice_per_m'],
            contrails.filter(is_at_t_end)['segment_length'],
        ) if np.any(is_at_t_end) else np.nan,

        'volume_mean_contrail_ice_radius': volume_mean_ice_particle_radius(
            contrails.filter(is_at_t_end)['r_ice_vol'],
            contrails.filter(is_at_t_end)['n_ice_per_m'],
            contrails.filter(is_at_t_end)['segment_length'],
        ) if np.any(is_at_t_end) else np.nan,

        'mean_contrail_ice_effective_radius': mean_ice_particle_effective_radius(
            contrails.filter(is_at_t_end)['r_ice_vol'],
            contrails.filter(is_at_t_end)['n_ice_per_m'],
            contrails.filter(is_at_t_end)['segment_length'],
        ) if np.any(is_at_t_end) else np.nan,

        'mean_tau_contrail': area_mean_contrail_property(
            contrails.filter(is_at_t_end)['tau_contrail'],
            contrails.filter(is_at_t_end)['segment_length'],
            contrails.filter(is_at_t_end)['width'],
            domain_area
        ) if np.any(is_at_t_end) else np.nan,

        'mean_tau_cirrus': area_mean_contrail_property(
            contrails.filter(is_at_t_end)['tau_cirrus'],
            contrails.filter(is_at_t_end)['segment_length'],
            contrails.filter(is_at_t_end)['width'],
            domain_area
        ) if np.any(is_at_t_end) else np.nan,

        # Contrail climate forcing
        'mean_rf_sw': area_mean_contrail_property(
            contrails.filter(is_at_t_end)['rf_sw'],
            contrails.filter(is_at_t_end)['segment_length'],
            contrails.filter(is_at_t_end)['width'],
            domain_area
        ) if np.any(is_at_t_end) else np.nan,

        'mean_rf_lw': area_mean_contrail_property(
            contrails.filter(is_at_t_end)['rf_lw'],
            contrails.filter(is_at_t_end)['segment_length'],
            contrails.filter(is_at_t_end)['width'],
            domain_area
        ) if np.any(is_at_t_end) else np.nan,

        'mean_rf_net': area_mean_contrail_property(
            contrails.filter(is_at_t_end)['rf_net'],
            contrails.filter(is_at_t_end)['segment_length'],
            contrails.filter(is_at_t_end)['width'],
            domain_area
        ) if np.any(is_at_t_end) else np.nan,

        'total_contrail_ef': np.nansum(contrails['ef']) if np.any(is_at_t_end) else np.nan,

        # Meteorology statistics
        "issr_percentage_coverage": (
            met_stats["issr_percentage_coverage"]
        ) if met is not None else np.nan,

        "mean_rhi_in_issr": met_stats["mean_rhi_in_issr"] if met is not None else np.nan,

        "contrail_cirrus_percentage_coverage": (
            met_stats["contrail_cirrus_percentage_coverage"]
        ) if met is not None else np.nan,

        "contrail_cirrus_clear_sky_percentage_coverage": (
            met_stats["contrail_cirrus_clear_sky_percentage_coverage"]
        ) if met is not None else np.nan,

        "natural_cirrus_percentage_coverage": (
            met_stats["natural_cirrus_percentage_coverage"]
        ) if met is not None else np.nan,

        "cloud_contrail_overlap_percentage": percentage_cloud_contrail_overlap(
            met_stats["contrail_cirrus_percentage_coverage"],
            met_stats["contrail_cirrus_clear_sky_percentage_coverage"]
        ) if met is not None else np.nan,

        # Radiation statistics
        'mean_sdr_domain': rad_stats["mean_sdr_domain"] if rad is not None else np.nan,

        'mean_sdr_at_contrail_wypts': np.nanmean(
            contrails.filter(is_at_t_end)['sdr']
        ) if np.any(is_at_t_end) else np.nan,

        'mean_rsr_domain': rad_stats["mean_rsr_domain"] if rad is not None else np.nan,

        'mean_rsr_at_contrail_wypts': np.nanmean(
            contrails.filter(is_at_t_end)['rsr']
        ) if np.any(is_at_t_end) else np.nan,

        'mean_olr_domain': rad_stats["mean_olr_domain"] if rad is not None else np.nan,

        'mean_olr_at_contrail_wypts': np.nanmean(
            contrails.filter(is_at_t_end)['olr']
        ) if np.any(is_at_t_end) else np.nan,

        "mean_albedo_at_contrail_wypts": np.nanmean(
            albedo(contrails.filter(is_at_t_end)['sdr'], contrails.filter(is_at_t_end)['rsr'])
        ) if np.any(is_at_t_end) else np.nan,
    }
    return pd.Series(stats_t)


# --------------
# Grid helpers
# --------------

def cirrus_coverage_single_level(
        time: np.datetime64 | pd.Timestamp,
        met: MetDataset,
        contrails: GeoVectorDataset, *,
        optical_depth_threshold: float = 0.1
) -> MetDataset:
    """
    Identify presence of contrail and natural cirrus in a longitude-latitude grid.

    Parameters
    ----------
    met : MetDataset
        Pressure level dataset containing 'air_temperature', 'specific_cloud_ice_water_content',
        and 'geopotential'
    contrails : GeoVectorDataset
        Contrail waypoints containing `tau_contrail`.
    time : np.datetime64 | pd.Timestamp
        Time when the cirrus statistics is computed.
    optical_depth_threshold : float
        Sensitivity of cirrus detection, set at 0.1 to match the capability of satellites.

    Returns
    -------
    MetDataset
        Single level dataset containing the contrail and natural cirrus coverage.
    """
    # Ensure `met` and `contrails` contains the required variables
    met.ensure_vars(('air_temperature', 'specific_cloud_ice_water_content', 'geopotential'))
    contrails.ensure_vars('tau_contrail')

    # Spatial bounding box and resolution of `met`
    spatial_bbox = [
        np.min(met["longitude"].values),
        np.min(met["latitude"].values),
        np.max(met["longitude"].values),
        np.max(met["latitude"].values),
    ]
    spatial_grid_res = np.diff(met["longitude"].values)[0]

    # Contrail cirrus optical depth in a longitude-latitude grid
    tau_contrail = vector_to_lon_lat_grid(
        contrails.filter(contrails["time"] == time),
        agg={"tau_contrail": "sum"},
        spatial_bbox=spatial_bbox,
        spatial_grid_res=spatial_grid_res
    )['tau_contrail']
    tau_contrail = tau_contrail.expand_dims({"level": np.array([-1])})
    tau_contrail = tau_contrail.expand_dims({"time": np.array([time])})
    tau_contrail = MetDataArray(tau_contrail)

    # Natural cirrus optical depth in a longitude-latitude grid
    met["tau_cirrus"] = tau_cirrus(met)
    tau_cirrus_max = met['tau_cirrus'].data.sel(level=met["level"].data[-1], time=time)
    tau_cirrus_max = tau_cirrus_max.expand_dims({"level": np.array([-1])})
    tau_cirrus_max = tau_cirrus_max.expand_dims({"time": np.array([time])})
    tau_cirrus_max = MetDataArray(tau_cirrus_max)

    tau_all = tau_contrail.data + tau_cirrus_max.data
    tau_all = MetDataArray(tau_all)

    # Contrail and natural cirrus coverage in a longitude-latitude grid
    cc_contrails_clear_sky = optical_depth_to_cirrus_coverage(
        tau_contrail, threshold=optical_depth_threshold
    )
    cc_natural_cirrus = optical_depth_to_cirrus_coverage(
        tau_cirrus_max, threshold=optical_depth_threshold
    )
    cc_total = optical_depth_to_cirrus_coverage(
        tau_all, threshold=optical_depth_threshold
    )
    cc_contrails = cc_total.data - cc_natural_cirrus.data
    cc_contrails = MetDataArray(cc_contrails)

    # Concatenate data
    ds = xr.Dataset(
        data_vars=dict(
            contrails_clear_sky=cc_contrails_clear_sky.data,
            natural_cirrus=cc_natural_cirrus.data,
            contrails=cc_contrails.data,
        ),
        coords=cc_contrails_clear_sky.coords
    )

    # Update attributes
    ds["contrails_clear_sky"].attrs = {
        "units": " ", "long_name": "Contrail cirrus cover in clear sky conditions."
    }
    ds["natural_cirrus"].attrs = {
        "units": " ", "long_name": "Natural cirrus cover."
    }
    ds["contrails"].attrs = {
        "units": " ", "long_name": "Contrail cirrus cover without overlap with natural cirrus."
    }
    return MetDataset(ds)


def optical_depth_to_cirrus_coverage(
        optical_depth: MetDataArray, *,
        threshold: float = 0.1,
) -> MetDataArray:
    """
    Calculate contrail or natural cirrus coverage in a longitude-latitude grid.

    A grid cell is assumed to be covered by cirrus if the optical depth is above `threshold`

    Parameters
    ----------
    optical_depth : MetDataArray
        Contrail or natural cirrus optical depth in a longitude-latitude grid
    threshold : float
        Sensitivity of cirrus detection, set at 0.1 to match the capability of satellites.

    Returns
    -------
    MetDataArray
        Contrail or natural cirrus coverage in a longitude-latitude grid
    """
    cirrus_cover = xr.where(optical_depth.data > threshold, 1, 0)
    return MetDataArray(cirrus_cover)


# -------------------
# Time-slice helpers
# -------------------


def meteorological_statistics(
        time: np.datetime64 | pd.Timestamp,
        contrails: GeoVectorDataset,
        met: MetDataset,
        humidity_scaling: HumidityScaling | None = None,
        cirrus_coverage: MetDataset | None = None,
) -> pd.Series:
    """
    Calculate meteorological statistics in the domain provided.

    Parameters
    ----------
    time : np.datetime64 | pd.Timestamp
        Time when the meteorological statistics is computed.
    contrails : GeoVectorDataset
        Contrail waypoints containing `tau_contrail`.
    met : MetDataset
        Pressure level dataset containing 'air_temperature', 'specific_humidity',
        'specific_cloud_ice_water_content', and 'geopotential'
    humidity_scaling : HumidityScaling
        Humidity scaling methodology.
        See :attr:`CocipParams.humidity_scaling`
    cirrus_coverage : MetDataset
        Single level dataset containing the contrail and natural cirrus coverage, including
        `cc_contrails_clear_sky`, `cc_natural_cirrus`, `cc_contrails`

    Returns
    -------
    pd.Series
        Mean ISSR characteristics, and the percentage of contrail and natural cirrus coverage in
        domain area.
    """
    # Ensure vars
    met.ensure_vars(
        (
            'air_temperature',
            'specific_humidity',
            'specific_cloud_ice_water_content',
            'geopotential'
        )
    )

    # ISSR: Volume of airspace with RHi > 100% between FL300 and FL450
    met = humidity_scaling.eval(met)
    rhi = met["rhi"].data.sel(level=slice(150, 300))
    is_issr = (rhi > 1)

    # Cirrus in a longitude-latitude grid
    if cirrus_coverage is None:
        cirrus_coverage = cirrus_coverage_single_level(time, met, contrails)

    # Calculate statistics
    area = geo.grid_surface_area(met["longitude"].values, met["latitude"].values)
    weights = area / np.nansum(area)

    stats = {
        'issr_percentage_coverage': (
                np.nansum((is_issr * weights)) / (np.nansum(weights) * len(rhi.level))
        ) * 100,
        'mean_rhi_in_issr': np.nanmean(rhi.values[is_issr.values]),
        'contrail_cirrus_percentage_coverage': (
                np.nansum((area * cirrus_coverage["contrails"].data)) / np.nansum(area)
        ) * 100,
        'contrail_cirrus_clear_sky_percentage_coverage': (
                np.nansum((area * cirrus_coverage["contrails_clear_sky"].data)) / np.nansum(area)
        ) * 100,
        'natural_cirrus_percentage_coverage': (
                np.nansum((area * cirrus_coverage["natural_cirrus"].data)) / np.nansum(area)
        ) * 100,
    }
    return pd.Series(stats)


def radiation_statistics(
        rad: MetDataset,
        time: np.datetime64 | pd.Timestamp
) -> pd.Series:
    """
    Calculate radiation statistics in the domain provided.

    Parameters
    ----------
    rad : MetDataset
        Single level dataset containing the `sdr`, `rsr` and `olr`.
    time : np.datetime64 | pd.Timestamp
        Time when the radiation statistics is computed.

    Returns
    -------
    pd.Series
        Mean SDR, RSR and OLR in domain area.
    """
    rad.ensure_vars(('sdr', 'rsr', 'olr'))
    surface_area = geo.grid_surface_area(rad["longitude"].values, rad["latitude"].values)
    weights = surface_area.values / np.nansum(surface_area)
    stats = {
        'mean_sdr_domain': np.nansum(rad["sdr"].data.sel(level=-1, time=time).values * weights),
        'mean_rsr_domain': np.nansum(rad["rsr"].data.sel(level=-1, time=time).values * weights),
        'mean_olr_domain': np.nansum(rad["olr"].data.sel(level=-1, time=time).values * weights)
    }
    return pd.Series(stats)


def area_mean_ice_water_path(
        iwc: npt.NDArray[np.float_],
        plume_mass_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_],
        domain_area: float
) -> float:
    """
    Calculate area-mean contrail ice water path

    Ice water path (IWC) is the contrail ice mass divided by the domain area of interest.

    Parameters
    ----------
    iwc : npt.NDArray[np.float_]
        Contrail ice water content, i.e., contrail ice mass per kg of
        air, [:math:`kg_{H_{2}O}/kg_{air}`]
    plume_mass_per_m : npt.NDArray[np.float_]
        Contrail plume mass per unit length, [:math:`kg m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]
    domain_area : float
        Domain surface area, [:math:`m^{2}`]
    Returns
    -------
    float
        Mean contrail ice water path, [:math:`kg m^{-2}`]
    """
    return np.nansum(iwc * plume_mass_per_m * segment_length) / domain_area


def area_mean_ice_particle_radius(
        r_ice_vol: npt.NDArray[np.float_],
        n_ice_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_]
) -> float:
    """
    Calculate the area-mean contrail ice particle radius.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.float_]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Area-mean contrail ice particle radius `r_area`, [:math:`\mu m`]

    Notes
    -----
    - Re-arranged from `tot_ice_cross_sec_area` = `tot_n_ice_particles` * (np.pi * `r_ice_vol`**2)
    - Assumes that the contrail ice crystals are spherical.
    """
    tot_ice_cross_sec_area = _total_ice_particle_cross_sectional_area(
        r_ice_vol, n_ice_per_m, segment_length
    )
    tot_n_ice_particles = _total_ice_particle_number(n_ice_per_m, segment_length)
    return (tot_ice_cross_sec_area / (np.pi * tot_n_ice_particles)) ** (1 / 2) * 10**6


def volume_mean_ice_particle_radius(
        r_ice_vol: npt.NDArray[np.float_],
        n_ice_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_]
) -> float:
    """
    Calculate the volume-mean contrail ice particle radius.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.float_]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Volume-mean contrail ice particle radius `r_vol`, [:math:`\mu m`]

    Notes
    -----
    - Re-arranged from `tot_ice_vol` = `tot_n_ice_particles` * (4 / 3 * np.pi * `r_ice_vol`**3)
    - Assumes that the contrail ice crystals are spherical.
    """
    tot_ice_vol = _total_ice_particle_volume(r_ice_vol, n_ice_per_m, segment_length)
    tot_n_ice_particles = _total_ice_particle_number(n_ice_per_m, segment_length)
    return (tot_ice_vol / ((4 / 3) * np.pi * tot_n_ice_particles)) ** (1 / 3) * 10**6


def mean_ice_particle_effective_radius(
        r_ice_vol: npt.NDArray[np.float_],
        n_ice_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_]
) -> float:
    """
    Calculate the mean contrail ice particle effective radius.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.float_]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Mean contrail ice particle effective radius `r_eff`, [:math:`\mu m`]

    Notes
    -----
    - `r_eff` is the ratio of the particle volume to particle projected area.
    - `r_eff` = (3 / 4) * (`tot_ice_vol` / `tot_ice_cross_sec_area`)
    - See Eq. (62) of :cite:`schumannContrailCirrusPrediction2012`.
    """
    tot_ice_vol = _total_ice_particle_volume(r_ice_vol, n_ice_per_m, segment_length)
    tot_ice_cross_sec_area = _total_ice_particle_cross_sectional_area(
        r_ice_vol, n_ice_per_m, segment_length
    )
    return (3 / 4) * (tot_ice_vol / tot_ice_cross_sec_area) * 10**6


def _total_ice_particle_cross_sectional_area(
        r_ice_vol: npt.NDArray[np.float_],
        n_ice_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_]
) -> float:
    """
    Calculate total contrail ice particle cross-sectional area.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.float_]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Total ice particle cross-sectional area from all contrail waypoints, [:math:`m^{2}`]
    """
    ice_cross_sec_area = 0.9 * np.pi * r_ice_vol**2
    return np.nansum(ice_cross_sec_area * n_ice_per_m * segment_length)


def _total_ice_particle_volume(
        r_ice_vol: npt.NDArray[np.float_],
        n_ice_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_]
) -> float:
    """
    Calculate total contrail ice particle volume.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.float_]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Total ice particle volume from all contrail waypoints, [:math:`m^{2}`]
    """
    ice_vol = (4 / 3) * np.pi * r_ice_vol**3
    return np.nansum(ice_vol * n_ice_per_m * segment_length)


def _total_ice_particle_number(
        n_ice_per_m: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_]
) -> float:
    """
    Calculate total number of contrail ice particles.

    Parameters
    ----------
    n_ice_per_m : npt.NDArray[np.float_]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Total number of ice particles from all contrail waypoints.
    """
    return np.nansum(n_ice_per_m * segment_length)


def area_mean_contrail_property(
        contrail_property: npt.NDArray[np.float_],
        segment_length: npt.NDArray[np.float_],
        width: npt.NDArray[np.float_],
        domain_area: float
) -> float:
    """
    Calculate area mean contrail property.

    Used to calculate the area mean `tau_contrail`, `tau_cirrus`, `sdr`, `rsr`, `olr`, `rf_sw`,
    `rf_lw` and `rf_net`.

    Parameters
    ----------
    contrail_property : npt.NDArray[np.float_]
        Selected contrail property for each waypoint
    segment_length : npt.NDArray[np.float_]
        Contrail segment length for each waypoint, [:math:`m`]
    width : npt.NDArray[np.float_]
        Contrail width for each waypoint, [:math:`m`]
    domain_area : float
        Domain surface area, [:math:`m^{2}`]

    Returns
    -------
    float
        Area mean contrail property
    """
    return np.nansum(contrail_property * segment_length * width) / domain_area


def percentage_cloud_contrail_overlap(
        contrail_cover: float | np.ndarray,
        contrail_cover_clear_sky: float | np.ndarray
) -> float | np.ndarray:
    """
    Calculate the percentage area of cloud-contrail overlap.

    Parameters
    ----------
    contrail_cover : float | np.ndarray
        Percentage of contrail cirrus cover without overlap with natural cirrus.
        See `cirrus_coverage_single_level` function.
    contrail_cover_clear_sky : float | np.ndarray
        Percentage of contrail cirrus cover in clear sky conditions.
        See `cirrus_coverage_single_level` function.

    Returns
    -------
    float | np.ndarray
        Percentage of cloud-contrail overlap
    """
    return np.where(
        contrail_cover_clear_sky > 0,
        100 - (contrail_cover / contrail_cover_clear_sky * 100),
        0,
    )
