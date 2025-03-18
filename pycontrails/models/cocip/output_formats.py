"""CoCiP output formats.

This module includes functions to produce additional output formats, including the:
    (1) Flight waypoint outputs.
            See :func:`flight_waypoint_summary_statistics`.
    (2) Contrail flight summary outputs.
            See :func:`contrail_flight_summary_statistics`.
    (3) Gridded outputs.
            See :func:`longitude_latitude_grid`.
    (4) Time-slice statistics.
            See :func:`time_slice_statistics`.
    (5) Aggregate contrail segment optical depth/RF to a high-resolution longitude-latitude grid.
            See :func:`contrails_to_hi_res_grid`.
    (6) Increase spatial resolution of natural cirrus properties, required to estimate the
        high-resolution contrail cirrus coverage for (5).
            See :func:`natural_cirrus_properties_to_hi_res_grid`.
    (7) Comparing simulated contrails from CoCiP with GOES satellite imagery.
            See :func:`compare_cocip_with_goes`.
"""

from __future__ import annotations

import pathlib
import warnings
from collections.abc import Hashable

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.core.vector import GeoVectorDataset, vector_to_lon_lat_grid
from pycontrails.models.cocip.contrail_properties import contrail_edges, plume_mass_per_distance
from pycontrails.models.cocip.radiative_forcing import albedo
from pycontrails.models.humidity_scaling import HumidityScaling
from pycontrails.models.tau_cirrus import tau_cirrus
from pycontrails.physics import geo, thermo, units
from pycontrails.utils import dependencies

# -----------------------
# Flight waypoint outputs
# -----------------------


def flight_waypoint_summary_statistics(
    flight_waypoints: GeoVectorDataset | pd.DataFrame,
    contrails: GeoVectorDataset | pd.DataFrame,
) -> GeoVectorDataset:
    """
    Calculate the contrail summary statistics at each flight waypoint.

    Parameters
    ----------
    flight_waypoints : GeoVectorDataset | pd.DataFrame
        Flight waypoints that were used in :meth:`Cocip.eval` to produce ``contrails``.
    contrails : GeoVectorDataset | pd.DataFrame
        Contrail evolution outputs from CoCiP, :attr:`Cocip.contrail`

    Returns
    -------
    GeoVectorDataset
        Contrail summary statistics attached to each flight waypoint.

    Notes
    -----
    Outputs and units:
    - ``mean_contrail_altitude``, [:math:`m`]
    - ``mean_rhi``, [dimensionless]
    - ``mean_n_ice_per_m``, [:math:`m^{-1}`]
    - ``mean_r_ice_vol``, [:math:`m`]
    - ``mean_width``, [:math:`m`]
    - ``mean_depth``, [:math:`m`]
    - ``mean_tau_contrail``, [dimensionless]
    - ``mean_tau_cirrus``, [dimensionless]
    - ``max_age``, [:math:`h`]
    - ``mean_rf_sw``, [:math:`W m^{-2}`]
    - ``mean_rf_lw``, [:math:`W m^{-2}`]
    - ``mean_rf_net``, [:math:`W m^{-2}`]
    - ``ef``, [:math:`J`]
    - ``mean_olr``, [:math:`W m^{-2}`]
    - ``mean_sdr``, [:math:`W m^{-2}`]
    - ``mean_rsr``, [:math:`W m^{-2}`]
    """
    # Aggregation map
    agg_map = {
        # Location, ambient meteorology and properties
        "altitude": "mean",
        "rhi": ["mean", "std"],
        "n_ice_per_m": ["mean", "std"],
        "r_ice_vol": "mean",
        "width": "mean",
        "depth": "mean",
        "tau_contrail": "mean",
        "tau_cirrus": "mean",
        "age": "max",
        # Radiative properties
        "rf_sw": "mean",
        "rf_lw": "mean",
        "rf_net": "mean",
        "olr": "mean",
        "sdr": "mean",
        "rsr": "mean",
    }
    if "ef" not in flight_waypoints:
        agg_map["ef"] = "sum"

    # Check and pre-process `flights`
    if isinstance(flight_waypoints, GeoVectorDataset):
        flight_waypoints.ensure_vars(["flight_id", "waypoint"])
        flight_waypoints = flight_waypoints.dataframe

    flight_waypoints = flight_waypoints.set_index(["flight_id", "waypoint"])

    # Check and pre-process `contrails`
    if isinstance(contrails, GeoVectorDataset):
        contrail_vars = ["flight_id", "waypoint", "formation_time", *agg_map]
        contrail_vars.remove("age")
        contrails.ensure_vars(contrail_vars)
        contrails = contrails.dataframe

    contrails["age"] = (contrails["time"] - contrails["formation_time"]) / np.timedelta64(1, "h")

    # Calculate contrail statistics at each flight waypoint
    contrails = contrails.groupby(["flight_id", "waypoint"]).agg(agg_map)
    contrails.columns = (
        contrails.columns.get_level_values(1) + "_" + contrails.columns.get_level_values(0)
    )
    rename_cols = {"mean_altitude": "mean_contrail_altitude", "sum_ef": "ef"}
    contrails = contrails.rename(columns=rename_cols)

    # Concatenate to flight-waypoint outputs
    out = flight_waypoints.join(contrails, how="left")
    out = out.reset_index()
    return GeoVectorDataset(out)


# -------------------------------
# Contrail flight summary outputs
# -------------------------------


def contrail_flight_summary_statistics(flight_waypoints: GeoVectorDataset) -> pd.DataFrame:
    """
    Calculate contrail summary statistics for each flight.

    Parameters
    ----------
    flight_waypoints : GeoVectorDataset
        Flight waypoint outputs with contrail summary statistics attached.
        See :func:`flight_waypoint_summary_statistics`.

    Returns
    -------
    pd.DataFrame
        Contrail summary statistics for each flight

    Notes
    -----
    Outputs and units:
    - ``total_flight_distance_flown``, [:math:`m`]
    - ``total_contrails_formed``, [:math:`m`]
    - ``total_persistent_contrails_formed``, [:math:`m`]
    - ``mean_lifetime_contrail_altitude``, [:math:`m`]
    - ``mean_lifetime_rhi``, [dimensionless]
    - ``mean_lifetime_n_ice_per_m``, [:math:`m^{-1}`]
    - ``mean_lifetime_r_ice_vol``, [:math:`m`]
    - ``mean_lifetime_contrail_width``, [:math:`m`]
    - ``mean_lifetime_contrail_depth``, [:math:`m`]
    - ``mean_lifetime_tau_contrail``, [dimensionless]
    - ``mean_lifetime_tau_cirrus``, [dimensionless]
    - ``mean_contrail_lifetime``, [:math:`h`]
    - ``max_contrail_lifetime``, [:math:`h`]
    - ``mean_lifetime_rf_sw``, [:math:`W m^{-2}`]
    - ``mean_lifetime_rf_lw``, [:math:`W m^{-2}`]
    - ``mean_lifetime_rf_net``, [:math:`W m^{-2}`]
    - ``total_energy_forcing``, [:math:`J`]
    - ``mean_lifetime_olr``, [:math:`W m^{-2}`]
    - ``mean_lifetime_sdr``, [:math:`W m^{-2}`]
    - ``mean_lifetime_rsr``, [:math:`W m^{-2}`]
    """
    # Aggregation map
    agg_map = {
        # Contrail properties and ambient meteorology
        "segment_length": "sum",
        "contrail_length": "sum",
        "persistent_contrail_length": "sum",
        "mean_contrail_altitude": "mean",
        "mean_rhi": "mean",
        "mean_n_ice_per_m": "mean",
        "mean_r_ice_vol": "mean",
        "mean_width": "mean",
        "mean_depth": "mean",
        "mean_tau_contrail": "mean",
        "mean_tau_cirrus": "mean",
        "max_age": ["mean", "max"],
        # Radiative properties
        "mean_rf_sw": "mean",
        "mean_rf_lw": "mean",
        "mean_rf_net": "mean",
        "ef": "sum",
        "mean_olr": "mean",
        "mean_sdr": "mean",
        "mean_rsr": "mean",
    }

    # Check and pre-process `flight_waypoints`
    vars_required = ["flight_id", "sac", *agg_map]
    vars_required.remove("contrail_length")
    vars_required.remove("persistent_contrail_length")
    flight_waypoints.ensure_vars(vars_required)

    flight_waypoints["contrail_length"] = np.where(
        flight_waypoints["sac"] == 1.0, flight_waypoints["segment_length"], 0.0
    )

    flight_waypoints["persistent_contrail_length"] = np.where(
        np.nan_to_num(flight_waypoints["ef"]) == 0.0, 0.0, flight_waypoints["segment_length"]
    )

    # Calculate contrail statistics for each flight
    flight_summary = flight_waypoints.dataframe.groupby(["flight_id"]).agg(agg_map)
    flight_summary.columns = (
        flight_summary.columns.get_level_values(1)
        + "_"
        + flight_summary.columns.get_level_values(0)
    )

    rename_flight_summary_cols = {
        "sum_segment_length": "total_flight_distance_flown",
        "sum_contrail_length": "total_contrails_formed",
        "sum_persistent_contrail_length": "total_persistent_contrails_formed",
        "mean_mean_contrail_altitude": "mean_lifetime_contrail_altitude",
        "mean_mean_rhi": "mean_lifetime_rhi",
        "mean_mean_n_ice_per_m": "mean_lifetime_n_ice_per_m",
        "mean_mean_r_ice_vol": "mean_lifetime_r_ice_vol",
        "mean_mean_width": "mean_lifetime_contrail_width",
        "mean_mean_depth": "mean_lifetime_contrail_depth",
        "mean_mean_tau_contrail": "mean_lifetime_tau_contrail",
        "mean_mean_tau_cirrus": "mean_lifetime_tau_cirrus",
        "mean_max_age": "mean_contrail_lifetime",
        "max_max_age": "max_contrail_lifetime",
        "mean_mean_rf_sw": "mean_lifetime_rf_sw",
        "mean_mean_rf_lw": "mean_lifetime_rf_lw",
        "mean_mean_rf_net": "mean_lifetime_rf_net",
        "sum_ef": "total_energy_forcing",
        "mean_mean_olr": "mean_lifetime_olr",
        "mean_mean_sdr": "mean_lifetime_sdr",
        "mean_mean_rsr": "mean_lifetime_rsr",
    }

    return flight_summary.rename(columns=rename_flight_summary_cols).reset_index(["flight_id"])


# ---------------
# Gridded outputs
# ---------------


def longitude_latitude_grid(
    t_start: np.datetime64 | pd.Timestamp,
    t_end: np.datetime64 | pd.Timestamp,
    flight_waypoints: GeoVectorDataset,
    contrails: GeoVectorDataset,
    *,
    met: MetDataset,
    spatial_bbox: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    spatial_grid_res: float = 0.5,
) -> xr.Dataset:
    r"""
    Aggregate air traffic and contrail outputs to a longitude-latitude grid.

    Parameters
    ----------
    t_start : np.datetime64 | pd.Timestamp
        UTC time at beginning of time step.
    t_end : np.datetime64 | pd.Timestamp
        UTC time at end of time step.
    flight_waypoints : GeoVectorDataset
        Flight waypoint outputs with contrail summary statistics attached.
        See :func:`flight_waypoint_summary_statistics`.
    contrails : GeoVectorDataset
        Contrail evolution outputs from CoCiP, :attr:`Cocip.contrail`.
    met : MetDataset
        Pressure level dataset containing 'air_temperature', 'specific_humidity',
        'specific_cloud_ice_water_content', and 'geopotential'.
    spatial_bbox : tuple[float, float, float, float]
        Spatial bounding box, ``(lon_min, lat_min, lon_max, lat_max)``, [:math:`\deg`]
    spatial_grid_res : float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    xr.Dataset
        Air traffic and contrail outputs at a longitude-latitude grid.
    """
    # Ensure the required columns are included in `flight_waypoints`, `contrails` and `met`
    flight_waypoints.ensure_vars(("segment_length", "ef"))
    contrails.ensure_vars(
        (
            "formation_time",
            "segment_length",
            "width",
            "tau_contrail",
            "rf_sw",
            "rf_lw",
            "rf_net",
            "ef",
        )
    )
    met.ensure_vars(
        ("air_temperature", "specific_humidity", "specific_cloud_ice_water_content", "geopotential")
    )

    # Downselect `met` to specified spatial bounding box
    met = met.downselect(spatial_bbox)

    # Ensure that `flight_waypoints` and `contrails` are within `t_start` and `t_end`
    is_in_time = flight_waypoints.dataframe["time"].between(t_start, t_end, inclusive="right")
    if not np.all(is_in_time):
        warnings.warn(
            "Flight waypoints have times that are outside the range of `t_start` and `t_end`. "
            "Waypoints outside the defined time bounds are removed. "
        )
        flight_waypoints = flight_waypoints.filter(is_in_time)

    is_in_time = contrails.dataframe["time"].between(t_start, t_end, inclusive="right")

    if not np.all(is_in_time):
        warnings.warn(
            "Contrail waypoints have times that are outside the range of `t_start` and `t_end`."
            "Waypoints outside the defined time bounds are removed. "
        )
        contrails = contrails.filter(is_in_time)

    # Calculate additional variables
    t_slices = np.unique(contrails["time"])
    dt_integration_sec = (t_slices[1] - t_slices[0]) / np.timedelta64(1, "s")

    da_area = geo.grid_surface_area(met["longitude"].values, met["latitude"].values)

    flight_waypoints["persistent_contrails"] = np.where(
        np.isnan(flight_waypoints["ef"]), 0.0, flight_waypoints["segment_length"]
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
        spatial_grid_res=spatial_grid_res,
    )

    # (2) Contrail properties at `t_end`
    contrails_t_end = contrails.filter(contrails["time"] == t_end)

    contrails_t_end["tau_contrail_area"] = (
        contrails_t_end["tau_contrail"]
        * contrails_t_end["segment_length"]
        * contrails_t_end["width"]
    )

    contrails_t_end["age"] = (
        contrails_t_end["time"] - contrails_t_end["formation_time"]
    ) / np.timedelta64(1, "h")

    ds_contrails_t_end = vector_to_lon_lat_grid(
        contrails_t_end,
        agg={"segment_length": "sum", "tau_contrail_area": "sum", "age": "mean"},
        spatial_bbox=spatial_bbox,
        spatial_grid_res=spatial_grid_res,
    )
    ds_contrails_t_end["tau_contrail"] = ds_contrails_t_end["tau_contrail_area"] / da_area

    # (3) Contrail and natural cirrus coverage area at `t_end`
    mds_cirrus_coverage = cirrus_coverage_single_level(t_end, met, contrails)
    ds_cirrus_coverage = mds_cirrus_coverage.data.squeeze(dim=["level", "time"])

    # (4) Contrail climate forcing between `t_start` and `t_end`
    contrails["ef_sw"] = np.where(
        contrails["ef"] == 0.0,
        0.0,
        contrails["rf_sw"] * contrails["segment_length"] * contrails["width"] * dt_integration_sec,
    )
    contrails["ef_lw"] = np.where(
        contrails["ef"] == 0.0,
        0.0,
        contrails["rf_lw"] * contrails["segment_length"] * contrails["width"] * dt_integration_sec,
    )

    ds_forcing = vector_to_lon_lat_grid(
        contrails,
        agg={"ef_sw": "sum", "ef_lw": "sum", "ef": "sum"},
        spatial_bbox=spatial_bbox,
        spatial_grid_res=spatial_grid_res,
    )
    ds_forcing["rf_sw"] = ds_forcing["ef_sw"] / (da_area * dt_integration_sec)
    ds_forcing["rf_lw"] = ds_forcing["ef_lw"] / (da_area * dt_integration_sec)
    ds_forcing["rf_net"] = ds_forcing["ef"] / (da_area * dt_integration_sec)

    # -----------------------
    # Package gridded outputs
    # -----------------------
    ds = xr.Dataset(
        data_vars=dict(
            flight_distance_flown=ds_wypts_t["segment_length"] / 1000.0,
            persistent_contrails_formed=ds_wypts_t["persistent_contrails"] / 1000.0,
            persistent_contrails=ds_contrails_t_end["segment_length"] / 1000.0,
            tau_contrail=ds_contrails_t_end["tau_contrail"],
            contrail_age=ds_contrails_t_end["age"],
            cc_natural_cirrus=ds_cirrus_coverage["natural_cirrus"],
            cc_contrails=ds_cirrus_coverage["contrails"],
            cc_contrails_clear_sky=ds_cirrus_coverage["contrails_clear_sky"],
            rf_sw=ds_forcing["rf_sw"] * 1000.0,
            rf_lw=ds_forcing["rf_lw"] * 1000.0,
            rf_net=ds_forcing["rf_net"] * 1000.0,
            ef=ds_forcing["ef"],
            ef_initial_loc=ds_wypts_t["ef"],
        ),
        coords=ds_wypts_t.coords,
    )
    ds = ds.fillna(0.0)
    ds = ds.expand_dims({"time": np.array([t_end])})

    # Assign attributes
    attrs = _create_attributes()

    for name in ds.data_vars:
        ds[name].attrs = attrs[name]

    return ds


def _create_attributes() -> dict[Hashable, dict[str, str]]:
    return {
        "flight_distance_flown": {
            "long_name": "Total flight distance flown between t_start and t_end",
            "units": "km",
        },
        "persistent_contrails_formed": {
            "long_name": "Persistent contrails formed between t_start and t_end",
            "units": "km",
        },
        "persistent_contrails": {
            "long_name": "Persistent contrails at t_end",
            "units": "km",
        },
        "tau_contrail": {
            "long_name": "Area-normalised mean contrail optical depth at t_end",
            "units": " ",
        },
        "contrail_age": {
            "long_name": "Mean contrail age at t_end",
            "units": "h",
        },
        "cc_natural_cirrus": {
            "long_name": "Natural cirrus cover at t_end",
            "units": " ",
        },
        "cc_contrails": {
            "long_name": "Contrail cirrus cover at t_end",
            "units": " ",
        },
        "cc_contrails_clear_sky": {
            "long_name": "Contrail cirrus cover under clear sky conditions at t_end",
            "units": " ",
        },
        "rf_sw": {
            "long_name": "Mean contrail cirrus shortwave radiative forcing at t_end",
            "units": "mW/m**2",
        },
        "rf_lw": {
            "long_name": "Mean contrail cirrus longwave radiative forcing at t_end",
            "units": "mW/m**2",
        },
        "rf_net": {
            "long_name": "Mean contrail cirrus net radiative forcing at t_end",
            "units": "mW/m**2",
        },
        "ef": {
            "long_name": "Total contrail energy forcing between t_start and t_end",
            "units": "J",
        },
        "ef_initial_loc": {
            "long_name": "Total contrail energy forcing attributed back to the flight waypoint.",
            "units": "J",
        },
        "contrails_clear_sky": {
            "long_name": "Contrail cirrus cover in clear sky conditions.",
            "units": " ",
        },
        "natural_cirrus": {
            "long_name": "Natural cirrus cover.",
            "units": " ",
        },
        "contrails": {
            "long_name": "Contrail cirrus cover without overlap with natural cirrus.",
            "units": " ",
        },
    }


def cirrus_coverage_single_level(
    time: np.datetime64 | pd.Timestamp,
    met: MetDataset,
    contrails: GeoVectorDataset,
    *,
    optical_depth_threshold: float = 0.1,
) -> MetDataset:
    """
    Identify presence of contrail and natural cirrus in a longitude-latitude grid.

    Parameters
    ----------
    met : MetDataset
        Pressure level dataset containing 'air_temperature', 'specific_cloud_ice_water_content',
        and 'geopotential' fields.
    contrails : GeoVectorDataset
        Contrail waypoints containing 'tau_contrail' field.
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
    met.ensure_vars(("air_temperature", "specific_cloud_ice_water_content", "geopotential"))
    contrails.ensure_vars("tau_contrail")

    # Spatial bounding box and resolution of `met`
    spatial_bbox = (
        np.min(met["longitude"].values),
        np.min(met["latitude"].values),
        np.max(met["longitude"].values),
        np.max(met["latitude"].values),
    )
    spatial_grid_res = np.diff(met["longitude"].values)[0]

    # Contrail cirrus optical depth in a longitude-latitude grid
    tau_contrail = vector_to_lon_lat_grid(
        contrails.filter(contrails["time"] == time),
        agg={"tau_contrail": "sum"},
        spatial_bbox=spatial_bbox,
        spatial_grid_res=spatial_grid_res,
    )["tau_contrail"]
    tau_contrail = tau_contrail.expand_dims({"level": np.array([-1])})
    tau_contrail = tau_contrail.expand_dims({"time": np.array([time])})
    mda_tau_contrail = MetDataArray(tau_contrail)

    # Natural cirrus optical depth in a longitude-latitude grid
    met["tau_cirrus"] = tau_cirrus(met)
    tau_cirrus_max = met["tau_cirrus"].data.sel(level=met["level"].data[-1], time=time)
    tau_cirrus_max = tau_cirrus_max.expand_dims({"level": np.array([-1])})
    tau_cirrus_max = tau_cirrus_max.expand_dims({"time": np.array([time])})
    mda_tau_cirrus_max = MetDataArray(tau_cirrus_max)
    mda_tau_all = MetDataArray(mda_tau_contrail.data + mda_tau_cirrus_max.data)

    # Contrail and natural cirrus coverage in a longitude-latitude grid
    mda_cc_contrails_clear_sky = optical_depth_to_cirrus_coverage(
        mda_tau_contrail, threshold=optical_depth_threshold
    )
    mda_cc_natural_cirrus = optical_depth_to_cirrus_coverage(
        mda_tau_cirrus_max, threshold=optical_depth_threshold
    )
    mda_cc_total = optical_depth_to_cirrus_coverage(mda_tau_all, threshold=optical_depth_threshold)
    mda_cc_contrails = MetDataArray(mda_cc_total.data - mda_cc_natural_cirrus.data)

    # Concatenate data
    ds = xr.Dataset(
        data_vars=dict(
            contrails_clear_sky=mda_cc_contrails_clear_sky.data,
            natural_cirrus=mda_cc_natural_cirrus.data,
            contrails=mda_cc_contrails.data,
        ),
        coords=mda_cc_contrails_clear_sky.coords,
    )

    # Update attributes
    attrs = _create_attributes()

    for name in ds.data_vars:
        ds[name].attrs = attrs[name]

    return MetDataset(ds)


def optical_depth_to_cirrus_coverage(
    optical_depth: MetDataArray,
    *,
    threshold: float = 0.1,
) -> MetDataArray:
    """
    Calculate contrail or natural cirrus coverage in a longitude-latitude grid.

    A grid cell is assumed to be covered by cirrus if the optical depth is above ``threshold``.

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
    cirrus_cover = (optical_depth.data > threshold).astype(int)
    return MetDataArray(cirrus_cover)


def regional_statistics(da_var: xr.DataArray, *, agg: str) -> pd.Series:
    """
    Calculate regional statistics from longitude-latitude grid.

    Parameters
    ----------
    da_var : xr.DataArray
        Air traffic or contrail variable in a longitude-latitude grid.
    agg : str
        Function selected for aggregation, (i.e., "sum" and "mean").

    Returns
    -------
    pd.Series
        Regional statistics

    Notes
    -----
    - The spatial bounding box for each region is defined in Teoh et al. (2023)
    - Teoh, R., Engberg, Z., Shapiro, M., Dray, L., and Stettler, M.: A high-resolution Global
        Aviation emissions Inventory based on ADS-B (GAIA) for 2019-2021, EGUsphere [preprint],
        https://doi.org/10.5194/egusphere-2023-724, 2023.
    """
    if (agg == "mean") and (len(da_var.time) > 1):
        da_var = da_var.mean(dim=["time"])
        da_var = da_var.fillna(0.0)

    # Get regional domain
    vars_regional = _regional_data_arrays(da_var)

    if agg == "sum":
        vals = {
            "World": np.nansum(vars_regional["world"].values),
            "USA": np.nansum(vars_regional["usa"].values),
            "Europe": np.nansum(vars_regional["europe"].values),
            "East Asia": np.nansum(vars_regional["east_asia"].values),
            "SEA": np.nansum(vars_regional["sea"].values),
            "Latin America": np.nansum(vars_regional["latin_america"].values),
            "Africa": np.nansum(vars_regional["africa"].values),
            "China": np.nansum(vars_regional["china"].values),
            "India": np.nansum(vars_regional["india"].values),
            "North Atlantic": np.nansum(vars_regional["n_atlantic"].values),
            "North Pacific": np.nansum(vars_regional["n_pacific_1"].values)
            + np.nansum(vars_regional["n_pacific_2"].values),
            "Arctic": np.nansum(vars_regional["arctic"].values),
        }
    elif agg == "mean":
        area_world = geo.grid_surface_area(da_var["longitude"].values, da_var["latitude"].values)
        area_regional = _regional_data_arrays(area_world)

        vals = {
            "World": _area_mean_properties(vars_regional["world"], area_regional["world"]),
            "USA": _area_mean_properties(vars_regional["usa"], area_regional["usa"]),
            "Europe": _area_mean_properties(vars_regional["europe"], area_regional["europe"]),
            "East Asia": _area_mean_properties(
                vars_regional["east_asia"], area_regional["east_asia"]
            ),
            "SEA": _area_mean_properties(vars_regional["sea"], area_regional["sea"]),
            "Latin America": _area_mean_properties(
                vars_regional["latin_america"], area_regional["latin_america"]
            ),
            "Africa": _area_mean_properties(vars_regional["africa"], area_regional["africa"]),
            "China": _area_mean_properties(vars_regional["china"], area_regional["china"]),
            "India": _area_mean_properties(vars_regional["india"], area_regional["india"]),
            "North Atlantic": _area_mean_properties(
                vars_regional["n_atlantic"], area_regional["n_atlantic"]
            ),
            "North Pacific": 0.4
            * _area_mean_properties(vars_regional["n_pacific_1"], area_regional["n_pacific_1"])
            + 0.6
            * _area_mean_properties(vars_regional["n_pacific_2"], area_regional["n_pacific_2"]),
            "Arctic": _area_mean_properties(vars_regional["arctic"], area_regional["arctic"]),
        }
    else:
        raise NotImplementedError('Aggregation only accepts operations of "mean" or "sum".')

    return pd.Series(vals)


def _regional_data_arrays(da_global: xr.DataArray) -> dict[str, xr.DataArray]:
    """
    Extract regional data arrays from global data array.

    Parameters
    ----------
    da_global : xr.DataArray
        Global air traffic or contrail variable in a longitude-latitude grid.

    Returns
    -------
    dict[str, xr.DataArray]
        Regional data arrays.

    Notes
    -----
    - The spatial bounding box for each region is defined in Teoh et al. (2023)
    - Teoh, R., Engberg, Z., Shapiro, M., Dray, L., and Stettler, M.: A high-resolution Global
        Aviation emissions Inventory based on ADS-B (GAIA) for 2019-2021, EGUsphere [preprint],
        https://doi.org/10.5194/egusphere-2023-724, 2023.
    """
    return {
        "world": da_global.copy(),
        "usa": da_global.sel(longitude=slice(-126.0, -66.0), latitude=slice(23.0, 50.0)),
        "europe": da_global.sel(longitude=slice(-12.0, 20.0), latitude=slice(35.0, 60.0)),
        "east_asia": da_global.sel(longitude=slice(103.0, 150.0), latitude=slice(15.0, 48.0)),
        "sea": da_global.sel(longitude=slice(87.5, 130.0), latitude=slice(-10.0, 20.0)),
        "latin_america": da_global.sel(longitude=slice(-85.0, -35.0), latitude=slice(-60.0, 15.0)),
        "africa": da_global.sel(longitude=slice(-20.0, 50.0), latitude=slice(-35.0, 40.0)),
        "china": da_global.sel(longitude=slice(73.5, 135.0), latitude=slice(18.0, 53.5)),
        "india": da_global.sel(longitude=slice(68.0, 97.5), latitude=slice(8.0, 35.5)),
        "n_atlantic": da_global.sel(longitude=slice(-70.0, -5.0), latitude=slice(40.0, 63.0)),
        "n_pacific_1": da_global.sel(longitude=slice(-180.0, -140.0), latitude=slice(35.0, 65.0)),
        "n_pacific_2": da_global.sel(longitude=slice(120.0, 180.0), latitude=slice(35.0, 65.0)),
        "arctic": da_global.sel(latitude=slice(66.5, 90.0)),
    }


def _area_mean_properties(da_var_region: xr.DataArray, da_area_region: xr.DataArray) -> float:
    """
    Calculate area-mean properties.

    Parameters
    ----------
    da_var_region : xr.DataArray
        Regional air traffic or contrail variable in a longitude-latitude grid.
    da_area_region : xr.DataArray
        Regional surface area in a longitude-latitude grid.

    Returns
    -------
    float
        Area-mean properties
    """
    return np.nansum(da_var_region.values * da_area_region.values) / np.nansum(
        da_area_region.values
    )


# ---------------------
# Time-slice statistics
# ---------------------


def time_slice_statistics(
    t_start: np.datetime64 | pd.Timestamp,
    t_end: np.datetime64 | pd.Timestamp,
    flight_waypoints: GeoVectorDataset,
    contrails: GeoVectorDataset,
    *,
    humidity_scaling: HumidityScaling,
    met: MetDataset | None = None,
    rad: MetDataset | None = None,
    spatial_bbox: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
) -> pd.Series:
    r"""
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
        Contrail evolution outputs from CoCiP, `cocip.contrail`.
    humidity_scaling : HumidityScaling
        Humidity scaling methodology.
        See :attr:`CocipParams.humidity_scaling`
    met : MetDataset | None
        Pressure level dataset containing 'air_temperature', 'specific_humidity',
        'specific_cloud_ice_water_content', and 'geopotential'.
        Meteorological statistics will not be computed if `None` is provided.
    rad : MetDataset | None
        Single level dataset containing the `sdr`, `rsr` and `olr`.Radiation statistics
        will not be computed if `None` is provided.

    spatial_bbox : tuple[float, float, float, float]
        Spatial bounding box, `(lon_min, lat_min, lon_max, lat_max)`, [:math:`\deg`]

    Returns
    -------
    pd.Series
        Flight and contrail summary statistics. Contrail statistics are provided at `t_end`.
        The units for each output are outlined in `Notes`.

    Notes
    -----
    Outputs and units:
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
    flight_waypoints.ensure_vars(
        (
            "flight_id",
            "segment_length",
            "true_airspeed",
            "fuel_flow",
            "engine_efficiency",
            "nvpm_ei_n",
            "sac",
            "persistent_1",
        )
    )
    contrails.ensure_vars(
        (
            "flight_id",
            "segment_length",
            "air_temperature",
            "iwc",
            "r_ice_vol",
            "n_ice_per_m",
            "tau_contrail",
            "tau_cirrus",
            "width",
            "area_eff",
            "sdr",
            "rsr",
            "olr",
            "rf_sw",
            "rf_lw",
            "rf_net",
            "ef",
        )
    )

    # Ensure that the waypoints are within `t_start` and `t_end`
    is_in_time = flight_waypoints.dataframe["time"].between(t_start, t_end, inclusive="right")

    if not np.all(is_in_time):
        warnings.warn(
            "Flight waypoints have times that are outside the range of `t_start` and `t_end`. "
            "Waypoints outside the defined time bounds are removed. "
        )
        flight_waypoints = flight_waypoints.filter(is_in_time)

    is_in_time = contrails.dataframe["time"].between(t_start, t_end, inclusive="right")
    if not np.all(is_in_time):
        warnings.warn(
            "Contrail waypoints have times that are outside the range of `t_start` and `t_end`."
            "Waypoints outside the defined time bounds are removed. "
        )
        contrails = contrails.filter(is_in_time)

    # Additional variables
    flight_waypoints["fuel_burn"] = (
        flight_waypoints["fuel_flow"]
        * (1 / flight_waypoints["true_airspeed"])
        * flight_waypoints["segment_length"]
    )
    contrails["pressure"] = units.m_to_pl(contrails["altitude"])
    contrails["rho_air"] = thermo.rho_d(contrails["air_temperature"], contrails["pressure"])
    contrails["plume_mass_per_m"] = plume_mass_per_distance(
        contrails["area_eff"], contrails["rho_air"]
    )
    contrails["age"] = (contrails["time"] - contrails["formation_time"]) / np.timedelta64(1, "h")

    # Meteorology domain statistics
    if met is not None:
        met.ensure_vars(
            (
                "air_temperature",
                "specific_humidity",
                "specific_cloud_ice_water_content",
                "geopotential",
            )
        )
        met = met.downselect(spatial_bbox)
        met_stats = meteorological_time_slice_statistics(t_end, contrails, met, humidity_scaling)

    # Radiation domain statistics
    if rad is not None:
        rad.ensure_vars(("sdr", "rsr", "olr"))
        rad = rad.downselect(spatial_bbox)
        rad_stats = radiation_time_slice_statistics(rad, t_end)

    # Calculate time-slice statistics
    is_sac = flight_waypoints["sac"] == 1.0
    is_persistent = flight_waypoints["persistent_1"] == 1.0
    is_at_t_end = contrails["time"] == t_end
    is_night_time = contrails["sdr"] < 0.1
    domain_area = geo.domain_surface_area(spatial_bbox)

    stats_t = {
        "time_start": t_start,
        "time_end": t_end,
        # Flight statistics
        "n_flights": len(flight_waypoints.dataframe["flight_id"].unique()),
        "n_flights_forming_contrails": len(
            flight_waypoints.filter(is_sac).dataframe["flight_id"].unique()
        ),
        "n_flights_forming_persistent_contrails": len(
            flight_waypoints.filter(is_persistent).dataframe["flight_id"].unique()
        ),
        "n_flights_with_persistent_contrails_at_t_end": len(
            contrails.filter(is_at_t_end).dataframe["flight_id"].unique()
        ),
        # Waypoint statistics
        "n_waypoints": len(flight_waypoints),
        "n_waypoints_forming_contrails": len(flight_waypoints.filter(is_sac)),
        "n_waypoints_forming_persistent_contrails": len(flight_waypoints.filter(is_persistent)),
        "n_waypoints_with_persistent_contrails_at_t_end": len(contrails.filter(is_at_t_end)),
        "n_contrail_waypoints_at_night": len(contrails.filter(is_at_t_end)),
        "pct_contrail_waypoints_at_night": (
            len(contrails.filter(is_night_time)) / len(contrails) * 100
        ),
        # Distance statistics
        "total_flight_distance": np.nansum(flight_waypoints["segment_length"]) / 1000,
        "total_contrails_formed": (
            np.nansum(flight_waypoints.filter(is_sac)["segment_length"]) / 1000
        ),
        "total_persistent_contrails_formed": (
            np.nansum(flight_waypoints.filter(is_persistent)["segment_length"]) / 1000
        ),
        "total_persistent_contrails_at_t_end": (
            np.nansum(contrails.filter(is_at_t_end)["segment_length"]) / 1000
        ),
        # Aircraft performance statistics
        "total_fuel_burn": np.nansum(flight_waypoints["fuel_burn"]),
        "mean_propulsion_efficiency_all_flights": np.nanmean(flight_waypoints["engine_efficiency"]),
        "mean_propulsion_efficiency_flights_with_persistent_contrails": (
            np.nanmean(flight_waypoints.filter(is_persistent)["engine_efficiency"])
            if np.any(is_persistent)
            else np.nan
        ),
        "mean_nvpm_ei_n_all_flights": np.nanmean(flight_waypoints["nvpm_ei_n"]),
        "mean_nvpm_ei_n_flights_with_persistent_contrails": (
            np.nanmean(flight_waypoints.filter(is_persistent)["nvpm_ei_n"])
            if np.any(is_persistent)
            else np.nan
        ),
        # Contrail properties at `time_end`
        "mean_contrail_age": (
            np.nanmean(contrails.filter(is_at_t_end)["age"]) if np.any(is_at_t_end) else np.nan
        ),
        "max_contrail_age": (
            np.nanmax(contrails.filter(is_at_t_end)["age"]) if np.any(is_at_t_end) else np.nan
        ),
        "mean_n_ice_per_m": (
            np.nanmean(contrails.filter(is_at_t_end)["n_ice_per_m"])
            if np.any(is_at_t_end)
            else np.nan
        ),
        "mean_contrail_ice_water_path": (
            area_mean_ice_water_path(
                contrails.filter(is_at_t_end)["iwc"],
                contrails.filter(is_at_t_end)["plume_mass_per_m"],
                contrails.filter(is_at_t_end)["segment_length"],
                domain_area,
            )
            if np.any(is_at_t_end)
            else np.nan
        ),
        "area_mean_contrail_ice_radius": (
            area_mean_ice_particle_radius(
                contrails.filter(is_at_t_end)["r_ice_vol"],
                contrails.filter(is_at_t_end)["n_ice_per_m"],
                contrails.filter(is_at_t_end)["segment_length"],
            )
            if np.any(is_at_t_end)
            else np.nan
        ),
        "volume_mean_contrail_ice_radius": (
            volume_mean_ice_particle_radius(
                contrails.filter(is_at_t_end)["r_ice_vol"],
                contrails.filter(is_at_t_end)["n_ice_per_m"],
                contrails.filter(is_at_t_end)["segment_length"],
            )
            if np.any(is_at_t_end)
            else np.nan
        ),
        "mean_contrail_ice_effective_radius": (
            mean_ice_particle_effective_radius(
                contrails.filter(is_at_t_end)["r_ice_vol"],
                contrails.filter(is_at_t_end)["n_ice_per_m"],
                contrails.filter(is_at_t_end)["segment_length"],
            )
            if np.any(is_at_t_end)
            else np.nan
        ),
        "mean_tau_contrail": (
            area_mean_contrail_property(
                contrails.filter(is_at_t_end)["tau_contrail"],
                contrails.filter(is_at_t_end)["segment_length"],
                contrails.filter(is_at_t_end)["width"],
                domain_area,
            )
            if np.any(is_at_t_end)
            else np.nan
        ),
        "mean_tau_cirrus": (
            area_mean_contrail_property(
                contrails.filter(is_at_t_end)["tau_cirrus"],
                contrails.filter(is_at_t_end)["segment_length"],
                contrails.filter(is_at_t_end)["width"],
                domain_area,
            )
            if np.any(is_at_t_end)
            else np.nan
        ),
        # Contrail climate forcing
        "mean_rf_sw": (
            area_mean_contrail_property(
                contrails.filter(is_at_t_end)["rf_sw"],
                contrails.filter(is_at_t_end)["segment_length"],
                contrails.filter(is_at_t_end)["width"],
                domain_area,
            )
            if np.any(is_at_t_end)
            else np.nan
        ),
        "mean_rf_lw": (
            area_mean_contrail_property(
                contrails.filter(is_at_t_end)["rf_lw"],
                contrails.filter(is_at_t_end)["segment_length"],
                contrails.filter(is_at_t_end)["width"],
                domain_area,
            )
            if np.any(is_at_t_end)
            else np.nan
        ),
        "mean_rf_net": (
            area_mean_contrail_property(
                contrails.filter(is_at_t_end)["rf_net"],
                contrails.filter(is_at_t_end)["segment_length"],
                contrails.filter(is_at_t_end)["width"],
                domain_area,
            )
            if np.any(is_at_t_end)
            else np.nan
        ),
        "total_contrail_ef": np.nansum(contrails["ef"]) if np.any(is_at_t_end) else np.nan,
        # Meteorology statistics
        "issr_percentage_coverage": (
            (met_stats["issr_percentage_coverage"]) if met is not None else np.nan
        ),
        "mean_rhi_in_issr": met_stats["mean_rhi_in_issr"] if met is not None else np.nan,
        "contrail_cirrus_percentage_coverage": (
            (met_stats["contrail_cirrus_percentage_coverage"]) if met is not None else np.nan
        ),
        "contrail_cirrus_clear_sky_percentage_coverage": (
            (met_stats["contrail_cirrus_clear_sky_percentage_coverage"])
            if met is not None
            else np.nan
        ),
        "natural_cirrus_percentage_coverage": (
            (met_stats["natural_cirrus_percentage_coverage"]) if met is not None else np.nan
        ),
        "cloud_contrail_overlap_percentage": (
            percentage_cloud_contrail_overlap(
                met_stats["contrail_cirrus_percentage_coverage"],
                met_stats["contrail_cirrus_clear_sky_percentage_coverage"],
            )
            if met is not None
            else np.nan
        ),
        # Radiation statistics
        "mean_sdr_domain": rad_stats["mean_sdr_domain"] if rad is not None else np.nan,
        "mean_sdr_at_contrail_wypts": (
            np.nanmean(contrails.filter(is_at_t_end)["sdr"]) if np.any(is_at_t_end) else np.nan
        ),
        "mean_rsr_domain": rad_stats["mean_rsr_domain"] if rad is not None else np.nan,
        "mean_rsr_at_contrail_wypts": (
            np.nanmean(contrails.filter(is_at_t_end)["rsr"]) if np.any(is_at_t_end) else np.nan
        ),
        "mean_olr_domain": rad_stats["mean_olr_domain"] if rad is not None else np.nan,
        "mean_olr_at_contrail_wypts": (
            np.nanmean(contrails.filter(is_at_t_end)["olr"]) if np.any(is_at_t_end) else np.nan
        ),
        "mean_albedo_at_contrail_wypts": (
            np.nanmean(
                albedo(contrails.filter(is_at_t_end)["sdr"], contrails.filter(is_at_t_end)["rsr"])
            )
            if np.any(is_at_t_end)
            else np.nan
        ),
    }
    return pd.Series(stats_t)


def meteorological_time_slice_statistics(
    time: np.datetime64 | pd.Timestamp,
    contrails: GeoVectorDataset,
    met: MetDataset,
    humidity_scaling: HumidityScaling,
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
        ("air_temperature", "specific_humidity", "specific_cloud_ice_water_content", "geopotential")
    )

    # ISSR: Volume of airspace with RHi > 100% between FL300 and FL450
    met_cruise = MetDataset(met.data.sel(level=slice(150, 300)))
    rhi = humidity_scaling.eval(met_cruise)["rhi"].data

    try:
        # If the given time is already in the dataset, select the time slice
        i = rhi.get_index("time").get_loc(time)
    except KeyError:
        rhi = rhi.interp(time=time)
    else:
        rhi = rhi.isel(time=i)

    is_issr = rhi > 1.0

    # Cirrus in a longitude-latitude grid
    if cirrus_coverage is None:
        cirrus_coverage = cirrus_coverage_single_level(time, met, contrails)

    # Calculate statistics
    area = geo.grid_surface_area(met["longitude"].values, met["latitude"].values)
    weights = area / np.nansum(area)

    stats = {
        "issr_percentage_coverage": (
            np.nansum(is_issr * weights) / (np.nansum(weights) * len(rhi.level))
        )
        * 100,
        "mean_rhi_in_issr": np.nanmean(rhi.values[is_issr.values]),
        "contrail_cirrus_percentage_coverage": (
            np.nansum(area * cirrus_coverage["contrails"].data) / np.nansum(area)
        )
        * 100,
        "contrail_cirrus_clear_sky_percentage_coverage": (
            np.nansum(area * cirrus_coverage["contrails_clear_sky"].data) / np.nansum(area)
        )
        * 100,
        "natural_cirrus_percentage_coverage": (
            np.nansum(area * cirrus_coverage["natural_cirrus"].data) / np.nansum(area)
        )
        * 100,
    }
    return pd.Series(stats)


def radiation_time_slice_statistics(
    rad: MetDataset, time: np.datetime64 | pd.Timestamp
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
    rad.ensure_vars(("sdr", "rsr", "olr"))
    surface_area = geo.grid_surface_area(rad["longitude"].values, rad["latitude"].values)
    weights = surface_area.values / np.nansum(surface_area)
    stats = {
        "mean_sdr_domain": np.nansum(
            np.squeeze(rad["sdr"].data.interp(time=time).values) * weights
        ),
        "mean_rsr_domain": np.nansum(
            np.squeeze(rad["rsr"].data.interp(time=time).values) * weights
        ),
        "mean_olr_domain": np.nansum(
            np.squeeze(rad["olr"].data.interp(time=time).values) * weights
        ),
    }
    return pd.Series(stats)


def area_mean_ice_water_path(
    iwc: npt.NDArray[np.floating],
    plume_mass_per_m: npt.NDArray[np.floating],
    segment_length: npt.NDArray[np.floating],
    domain_area: float,
) -> float:
    """
    Calculate area-mean contrail ice water path.

    Ice water path (IWC) is the contrail ice mass divided by the domain area of interest.

    Parameters
    ----------
    iwc : npt.NDArray[np.floating]
        Contrail ice water content, i.e., contrail ice mass per kg of
        air, [:math:`kg_{H_{2}O}/kg_{air}`]
    plume_mass_per_m : npt.NDArray[np.floating]
        Contrail plume mass per unit length, [:math:`kg m^{-1}`]
    segment_length : npt.NDArray[np.floating]
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
    r_ice_vol: npt.NDArray[np.floating],
    n_ice_per_m: npt.NDArray[np.floating],
    segment_length: npt.NDArray[np.floating],
) -> float:
    r"""
    Calculate the area-mean contrail ice particle radius.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.floating]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.floating]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.floating]
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
    r_ice_vol: npt.NDArray[np.floating],
    n_ice_per_m: npt.NDArray[np.floating],
    segment_length: npt.NDArray[np.floating],
) -> float:
    r"""
    Calculate the volume-mean contrail ice particle radius.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.floating]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.floating]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.floating]
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
    r_ice_vol: npt.NDArray[np.floating],
    n_ice_per_m: npt.NDArray[np.floating],
    segment_length: npt.NDArray[np.floating],
) -> float:
    r"""
    Calculate the mean contrail ice particle effective radius.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.floating]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.floating]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.floating]
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
    r_ice_vol: npt.NDArray[np.floating],
    n_ice_per_m: npt.NDArray[np.floating],
    segment_length: npt.NDArray[np.floating],
) -> float:
    """
    Calculate total contrail ice particle cross-sectional area.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.floating]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.floating]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.floating]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Total ice particle cross-sectional area from all contrail waypoints, [:math:`m^{2}`]
    """
    ice_cross_sec_area = 0.9 * np.pi * r_ice_vol**2
    return np.nansum(ice_cross_sec_area * n_ice_per_m * segment_length)


def _total_ice_particle_volume(
    r_ice_vol: npt.NDArray[np.floating],
    n_ice_per_m: npt.NDArray[np.floating],
    segment_length: npt.NDArray[np.floating],
) -> float:
    """
    Calculate total contrail ice particle volume.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.floating]
        Ice particle volume mean radius for each waypoint, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.floating]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.floating]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Total ice particle volume from all contrail waypoints, [:math:`m^{2}`]
    """
    ice_vol = (4 / 3) * np.pi * r_ice_vol**3
    return np.nansum(ice_vol * n_ice_per_m * segment_length)


def _total_ice_particle_number(
    n_ice_per_m: npt.NDArray[np.floating], segment_length: npt.NDArray[np.floating]
) -> float:
    """
    Calculate total number of contrail ice particles.

    Parameters
    ----------
    n_ice_per_m : npt.NDArray[np.floating]
        Number of ice particles per distance for each waypoint, [:math:`m^{-1}`]
    segment_length : npt.NDArray[np.floating]
        Contrail segment length for each waypoint, [:math:`m`]

    Returns
    -------
    float
        Total number of ice particles from all contrail waypoints.
    """
    return np.nansum(n_ice_per_m * segment_length)


def area_mean_contrail_property(
    contrail_property: npt.NDArray[np.floating],
    segment_length: npt.NDArray[np.floating],
    width: npt.NDArray[np.floating],
    domain_area: float,
) -> float:
    """
    Calculate area mean contrail property.

    Used to calculate the area mean `tau_contrail`, `tau_cirrus`, `sdr`, `rsr`, `olr`, `rf_sw`,
    `rf_lw` and `rf_net`.

    Parameters
    ----------
    contrail_property : npt.NDArray[np.floating]
        Selected contrail property for each waypoint
    segment_length : npt.NDArray[np.floating]
        Contrail segment length for each waypoint, [:math:`m`]
    width : npt.NDArray[np.floating]
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
    contrail_cover: float | np.ndarray, contrail_cover_clear_sky: float | np.ndarray
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


# ---------------------------------------
# High resolution grid: contrail segments
# ---------------------------------------


def contrails_to_hi_res_grid(
    time: pd.Timestamp | np.datetime64,
    contrails_t: GeoVectorDataset,
    *,
    var_name: str,
    spatial_bbox: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    spatial_grid_res: float = 0.05,
) -> xr.DataArray:
    r"""
    Aggregate contrail segments to a high-resolution longitude-latitude grid.

    Parameters
    ----------
    time : pd.Timestamp | np.datetime64
        UTC time of interest.
    contrails_t : GeoVectorDataset
        All contrail waypoint outputs at `time`.
    var_name : str
        Contrail property for aggregation, where `var_name` must be included in `contrail_segment`.
        For example, `tau_contrail`, `rf_sw`, `rf_lw`, and `rf_net`
    spatial_bbox : tuple[float, float, float, float]
        Spatial bounding box, `(lon_min, lat_min, lon_max, lat_max)`, [:math:`\deg`]
    spatial_grid_res : float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    xr.DataArray
        Contrail segments and their properties aggregated to a longitude-latitude grid.
    """
    # Ensure the required columns are included in `contrails_t`
    cols_req = [
        "flight_id",
        "waypoint",
        "longitude",
        "latitude",
        "altitude",
        "time",
        "sin_a",
        "cos_a",
        "width",
        var_name,
    ]
    contrails_t.ensure_vars(cols_req)

    # Ensure that the times in `contrails_t` are the same.
    is_in_time = contrails_t["time"] == time
    if not np.all(is_in_time):
        warnings.warn(
            f"Contrails have inconsistent times. Waypoints that are not in {time} are removed."
        )
        contrails_t = contrails_t.filter(is_in_time)

    main_grid = _initialise_longitude_latitude_grid(spatial_bbox, spatial_grid_res)

    # Contrail head and tails: continuous segments only
    heads_t = contrails_t.dataframe
    heads_t = heads_t.sort_values(["flight_id", "waypoint"])
    tails_t = heads_t.shift(periods=-1)

    is_continuous = heads_t["continuous"]
    heads_t = heads_t[is_continuous].copy()
    tails_t = tails_t[is_continuous].copy()
    tails_t["waypoint"] = tails_t["waypoint"].astype("int")

    heads_t = heads_t.set_index(["flight_id", "waypoint"], drop=False)
    tails_t.index = heads_t.index

    # Aggregate contrail segments to a high resolution longitude-latitude grid
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            name="contrails_to_hi_res_grid function",
            package_name="tqdm",
            module_not_found_error=exc,
        )

    for i in tqdm(heads_t.index):
        contrail_segment = GeoVectorDataset(
            pd.concat([heads_t[cols_req].loc[i], tails_t[cols_req].loc[i]], axis=1).T, copy=True
        )

        segment_grid = segment_property_to_hi_res_grid(
            contrail_segment, var_name=var_name, spatial_grid_res=spatial_grid_res
        )
        main_grid = _add_segment_to_main_grid(main_grid, segment_grid)

    return main_grid


def _initialise_longitude_latitude_grid(
    spatial_bbox: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    spatial_grid_res: float = 0.05,
) -> xr.DataArray:
    r"""
    Create longitude-latitude grid of specified coordinates and spatial resolution.

    Parameters
    ----------
    spatial_bbox : tuple[float, float, float, float]
        Spatial bounding box, `(lon_min, lat_min, lon_max, lat_max)`, [:math:`\deg`]
    spatial_grid_res : float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    xr.DataArray
        Longitude-latitude grid of specified coordinates and spatial resolution, filled with zeros.

    Notes
    -----
    This empty grid is used to store the aggregated contrail properties of the individual
    contrail segments, such as the gridded contrail optical depth and radiative forcing.
    """
    lon_coords = np.arange(spatial_bbox[0], spatial_bbox[2] + spatial_grid_res, spatial_grid_res)
    lat_coords = np.arange(spatial_bbox[1], spatial_bbox[3] + spatial_grid_res, spatial_grid_res)
    return xr.DataArray(
        np.zeros((len(lon_coords), len(lat_coords))),
        dims=["longitude", "latitude"],
        coords={"longitude": lon_coords, "latitude": lat_coords},
    )


def segment_property_to_hi_res_grid(
    contrail_segment: GeoVectorDataset,
    *,
    var_name: str,
    spatial_grid_res: float = 0.05,
) -> xr.DataArray:
    r"""
    Convert the contrail segment property to a high-resolution longitude-latitude grid.

    Parameters
    ----------
    contrail_segment : GeoVectorDataset
        Contrail segment waypoints (head and tail).
    var_name : str
        Contrail property of interest, where `var_name` must be included in `contrail_segment`.
        For example, `tau_contrail`, `rf_sw`, `rf_lw`, and `rf_net`
    spatial_grid_res : float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    xr.DataArray
        Contrail segment dimension and property projected to a longitude-latitude grid.

    Notes
    -----
    - See Appendix A11 and A12 of :cite:`schumannContrailCirrusPrediction2012`.
    """
    # Ensure that `contrail_segment` contains the required variables
    contrail_segment.ensure_vars(("sin_a", "cos_a", "width", var_name))

    # Ensure that `contrail_segment` only contains two waypoints and have the same time.
    assert len(contrail_segment) == 2
    assert contrail_segment["time"][0] == contrail_segment["time"][1]

    # Calculate contrail edges
    (
        contrail_segment["lon_edge_l"],
        contrail_segment["lat_edge_l"],
        contrail_segment["lon_edge_r"],
        contrail_segment["lat_edge_r"],
    ) = contrail_edges(
        contrail_segment["longitude"],
        contrail_segment["latitude"],
        contrail_segment["sin_a"],
        contrail_segment["cos_a"],
        contrail_segment["width"],
    )

    # Initialise contrail segment grid with spatial domain that covers the contrail area.
    lon_edges = np.concatenate(
        [contrail_segment["lon_edge_l"], contrail_segment["lon_edge_r"]], axis=0
    )
    lat_edges = np.concatenate(
        [contrail_segment["lat_edge_l"], contrail_segment["lat_edge_r"]], axis=0
    )
    spatial_bbox = geo.spatial_bounding_box(lon_edges, lat_edges, buffer=0.5)
    segment_grid = _initialise_longitude_latitude_grid(spatial_bbox, spatial_grid_res)

    # Calculate gridded contrail segment properties
    weights = _pixel_weights(contrail_segment, segment_grid)
    dist_perpendicular = _segment_perpendicular_distance_to_pixels(contrail_segment, weights)
    plume_concentration = _gaussian_plume_concentration(
        contrail_segment, weights, dist_perpendicular
    )

    # Distribute selected contrail property to grid
    return plume_concentration * (
        weights * xr.ones_like(weights) * contrail_segment[var_name][1]
        + (1 - weights) * xr.ones_like(weights) * contrail_segment[var_name][0]
    )


def _pixel_weights(contrail_segment: GeoVectorDataset, segment_grid: xr.DataArray) -> xr.DataArray:
    """
    Calculate the pixel weights for `segment_grid`.

    Parameters
    ----------
    contrail_segment : GeoVectorDataset
        Contrail segment waypoints (head and tail).
    segment_grid : xr.DataArray
        Contrail segment grid with spatial domain that covers the contrail area.

    Returns
    -------
    xr.DataArray
        Pixel weights for `segment_grid`

    Notes
    -----
    - See Appendix A12 of :cite:`schumannContrailCirrusPrediction2012`.
    - This is the weights (from the beginning of the contrail segment) to the nearest longitude and
        latitude pixel in the `segment_grid`.
    - The contrail segment do not contribute to the pixel if weight < 0 or > 1.
    """
    head = contrail_segment.dataframe.iloc[0]
    tail = contrail_segment.dataframe.iloc[1]

    # Calculate determinant
    dx = units.longitude_distance_to_m(
        (tail["longitude"] - head["longitude"]),
        0.5 * (head["latitude"] + tail["latitude"]),
    )
    dy = units.latitude_distance_to_m(tail["latitude"] - head["latitude"])
    det = dx**2 + dy**2

    # Calculate pixel weights
    lon_grid, lat_grid = np.meshgrid(
        segment_grid["longitude"].values, segment_grid["latitude"].values
    )
    dx_grid = units.longitude_distance_to_m(
        (lon_grid - head["longitude"]),
        0.5 * (head["latitude"] + lat_grid),
    )
    dy_grid = units.latitude_distance_to_m(lat_grid - head["latitude"])
    weights = (dx * dx_grid + dy * dy_grid) / det
    return xr.DataArray(
        data=weights.T,
        dims=["longitude", "latitude"],
        coords={"longitude": segment_grid["longitude"], "latitude": segment_grid["latitude"]},
    )


def _segment_perpendicular_distance_to_pixels(
    contrail_segment: GeoVectorDataset, weights: xr.DataArray
) -> xr.DataArray:
    """
    Calculate perpendicular distance from contrail segment to each segment grid pixel.

    Parameters
    ----------
    contrail_segment : GeoVectorDataset
        Contrail segment waypoints (head and tail).
    weights : xr.DataArray
        Pixel weights for `segment_grid`.
        See `_pixel_weights` function.

    Returns
    -------
    xr.DataArray
        Perpendicular distance from contrail segment to each segment grid pixel, [:math:`m`]

    Notes
    -----
    - See Figure A7 of :cite:`schumannContrailCirrusPrediction2012`.
    """
    head = contrail_segment.dataframe.iloc[0]
    tail = contrail_segment.dataframe.iloc[1]

    # Longitude and latitude along contrail segment
    lon_grid, lat_grid = np.meshgrid(weights["longitude"].values, weights["latitude"].values)

    lon_s = head["longitude"] + weights.T.values * (tail["longitude"] - head["longitude"])
    lat_s = head["latitude"] + weights.T.values * (tail["latitude"] - head["latitude"])

    lon_dist = units.longitude_distance_to_m(np.abs(lon_grid - lon_s), 0.5 * (lat_s + lat_grid))

    lat_dist = units.latitude_distance_to_m(np.abs(lat_grid - lat_s))
    dist_perp = (lon_dist**2 + lat_dist**2) ** 0.5
    return xr.DataArray(dist_perp.T, coords=weights.coords)


def _gaussian_plume_concentration(
    contrail_segment: GeoVectorDataset,
    weights: xr.DataArray,
    dist_perpendicular: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate relative gaussian plume concentration along the contrail width.

    Parameters
    ----------
    contrail_segment : GeoVectorDataset
        Contrail segment waypoints (head and tail).
    weights : xr.DataArray
        Pixel weights for `segment_grid`.
        See `_pixel_weights` function.
    dist_perpendicular : xr.DataArray
        Perpendicular distance from contrail segment to each segment grid pixel, [:math:`m`]
        See `_segment_perpendicular_distance_to_pixels` function.

    Returns
    -------
    xr.DataArray
        Relative gaussian plume concentration along the contrail width

    Notes
    -----
    - Assume a one-dimensional Gaussian plume.
    - See Appendix A11 of :cite:`schumannContrailCirrusPrediction2012`.
    """
    head = contrail_segment.dataframe.iloc[0]
    tail = contrail_segment.dataframe.iloc[1]

    width = weights.values * tail["width"] + (1 - weights.values) * head["width"]
    sigma_yy = 0.125 * width**2

    concentration = np.where(
        (weights.values < 0) | (weights.values > 1),
        0,
        (4 / np.pi) ** 0.5 * np.exp(-0.5 * dist_perpendicular.values**2 / sigma_yy),
    )
    return xr.DataArray(concentration, coords=weights.coords)


def _add_segment_to_main_grid(main_grid: xr.DataArray, segment_grid: xr.DataArray) -> xr.DataArray:
    """
    Add the gridded contrail segment to the main grid.

    Parameters
    ----------
    main_grid : xr.DataArray
        Aggregated contrail segment properties in a longitude-latitude grid.
    segment_grid : xr.DataArray
        Contrail segment dimension and property projected to a longitude-latitude grid.

    Returns
    -------
    xr.DataArray
        Aggregated contrail segment properties, including `segment_grid`.

    Notes
    -----
    - The spatial domain of `segment_grid` only covers the contrail segment, which is added to
        the `main_grid` which is expected to have a larger spatial domain than the `segment_grid`.
    - This architecture is used to reduce the computational resources.
    """
    lon_main = main_grid["longitude"].values
    lat_main = main_grid["latitude"].values

    lon_segment_grid = np.round(segment_grid["longitude"].values, decimals=2)
    lat_segment_grid = np.round(segment_grid["latitude"].values, decimals=2)

    main_grid_arr = main_grid.values
    subgrid_arr = segment_grid.values

    try:
        ix_ = np.searchsorted(lon_main, lon_segment_grid[0])
        ix = np.searchsorted(lon_main, lon_segment_grid[-1]) + 1
        iy_ = np.searchsorted(lat_main, lat_segment_grid[0])
        iy = np.searchsorted(lat_main, lat_segment_grid[-1]) + 1
    except IndexError:
        warnings.warn(
            "Contrail segment ignored as it is outside spatial bounding box of the main grid. "
        )
    else:
        main_grid_arr[ix_:ix, iy_:iy] = main_grid_arr[ix_:ix, iy_:iy] + subgrid_arr

    return xr.DataArray(main_grid_arr, coords=main_grid.coords)


# ------------------------------------
# High resolution grid: natural cirrus
# ------------------------------------


def natural_cirrus_properties_to_hi_res_grid(
    met: MetDataset,
    *,
    spatial_grid_res: float = 0.05,
    optical_depth_threshold: float = 0.1,
    random_state: np.random.Generator | int | None = None,
) -> MetDataset:
    r"""
    Increase the longitude-latitude resolution of natural cirrus cover and optical depth.

    Parameters
    ----------
    met : MetDataset
        Pressure level dataset for one time step containing 'air_temperature', 'specific_humidity',
        'specific_cloud_ice_water_content', 'geopotential',and `fraction_of_cloud_cover`
    spatial_grid_res : float
        Spatial grid resolution for the output, [:math:`\deg`]
    optical_depth_threshold : float
        Sensitivity of cirrus detection, set at 0.1 to match the capability of satellites.
    random_state : np.random.Generator | int | None
        A number used to initialize a pseudorandom number generator.

    Returns
    -------
    MetDataset
        Single-level dataset containing the high resolution natural cirrus properties.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`

    Notes
    -----
    - The high-resolution natural cirrus coverage and optical depth is distributed randomly,
        ensuring that the mean value is equal to the value of the original grid.
    - Enhancing the spatial resolution is necessary because the existing spatial resolution of
        numerical weather prediction (NWP) models are too coarse to resolve the coverage area of
        relatively narrow contrails.
    """
    # Ensure the required columns are included in `met`
    met.ensure_vars(
        (
            "air_temperature",
            "specific_humidity",
            "specific_cloud_ice_water_content",
            "geopotential",
            "fraction_of_cloud_cover",
        )
    )

    # Ensure `met` only contains one time step, constraint can be relaxed in the future.
    if len(met["time"].data) > 1:
        raise AssertionError(
            "`met` contains more than one time step, but function only accepts one time step. "
        )

    # Calculate tau_cirrus as observed by satellites
    met["tau_cirrus"] = tau_cirrus(met)
    tau_cirrus_max = met["tau_cirrus"].data.sel(level=met["level"].data[-1])

    # Calculate cirrus coverage as observed by satellites, cc_max(x,y,t) = max[cc(x,y,z,t)]
    cirrus_cover_max = met["fraction_of_cloud_cover"].data.max(dim="level")

    # Increase resolution of longitude and latitude dimensions
    lon_coords_hi_res, lat_coords_hi_res = _hi_res_grid_coordinates(
        met["longitude"].values, met["latitude"].values, spatial_grid_res=spatial_grid_res
    )

    # Increase spatial resolution by repeating existing values (temporarily)
    n_reps = int(
        np.round(np.diff(met["longitude"].values)[0], decimals=2)
        / np.round(np.diff(lon_coords_hi_res)[0], decimals=2)
    )
    cc_rep = _repeat_rows_and_columns(cirrus_cover_max.values, n_reps=n_reps)
    tau_cirrus_rep = _repeat_rows_and_columns(tau_cirrus_max.values, n_reps=n_reps)

    # Enhance resolution of `tau_cirrus`
    rng = np.random.default_rng(random_state)
    rand_number = rng.uniform(0, 1, np.shape(tau_cirrus_rep))
    dx = 0.03  # Prevent division of small values: calibrated to match the original cirrus cover
    has_cirrus = rand_number > (1 + dx - cc_rep)

    tau_cirrus_hi_res = np.zeros_like(tau_cirrus_rep)
    tau_cirrus_hi_res[has_cirrus] = tau_cirrus_rep[has_cirrus] / cc_rep[has_cirrus]

    # Enhance resolution of `cirrus coverage`
    cirrus_cover_hi_res = np.where(tau_cirrus_hi_res > optical_depth_threshold, 1, 0)

    # Package outputs
    ds_hi_res = xr.Dataset(
        data_vars=dict(
            tau_cirrus=(["longitude", "latitude"], tau_cirrus_hi_res),
            cc_natural_cirrus=(["longitude", "latitude"], cirrus_cover_hi_res),
        ),
        coords=dict(longitude=lon_coords_hi_res, latitude=lat_coords_hi_res),
    )
    ds_hi_res = ds_hi_res.expand_dims({"level": np.array([-1])})
    ds_hi_res = ds_hi_res.expand_dims({"time": met["time"].values})
    return MetDataset(ds_hi_res)


def _hi_res_grid_coordinates(
    lon_coords: npt.NDArray[np.floating],
    lat_coords: npt.NDArray[np.floating],
    *,
    spatial_grid_res: float = 0.05,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""
    Calculate longitude and latitude coordinates for the high resolution grid.

    Parameters
    ----------
    lon_coords : npt.NDArray[np.floating]
        Longitude coordinates provided by the original `MetDataset`.
    lat_coords : npt.NDArray[np.floating]
        Latitude coordinates provided by the original `MetDataset`.
    spatial_grid_res : float
        Spatial grid resolution for the output, [:math:`\deg`]

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]
        Longitude and latitude coordinates for the high resolution grid.
    """
    d_lon = np.abs(np.diff(lon_coords)[0])
    d_lat = np.abs(np.diff(lat_coords)[0])
    is_whole_number = (d_lon / spatial_grid_res) - int(d_lon / spatial_grid_res) == 0

    if (d_lon <= spatial_grid_res) | (d_lat <= spatial_grid_res):
        raise ArithmeticError(
            "Spatial resolution of `met` is already higher than `spatial_grid_res`"
        )

    if not is_whole_number:
        raise ArithmeticError(
            "Select a spatial grid resolution where `spatial_grid_res / existing_grid_res` is "
            "a whole number. "
        )

    lon_coords_hi_res = np.arange(
        lon_coords[0], lon_coords[-1] + spatial_grid_res, spatial_grid_res, dtype=float
    )

    lat_coords_hi_res = np.arange(
        lat_coords[0], lat_coords[-1] + spatial_grid_res, spatial_grid_res, dtype=float
    )

    return (np.round(lon_coords_hi_res, decimals=3), np.round(lat_coords_hi_res, decimals=3))


def _repeat_rows_and_columns(
    array_2d: npt.NDArray[np.floating], *, n_reps: int
) -> npt.NDArray[np.floating]:
    """
    Repeat the elements in `array_2d` along each row and column.

    Parameters
    ----------
    array_2d : npt.NDArray[np.float64, np.float64]
        2D array containing `tau_cirrus` or `cirrus_coverage` across longitude and latitude.
    n_reps : int
        Number of repetitions.

    Returns
    -------
    npt.NDArray[np.float64, np.float64]
        2D array containing `tau_cirrus` or `cirrus_coverage` at a higher spatial resolution.
        See :func:`_hi_res_grid_coordinates`.
    """
    dimension = np.shape(array_2d)

    # Repeating elements along axis=1
    array_1d_rep = [np.repeat(array_2d[i, :], n_reps) for i in np.arange(dimension[0])]
    stacked = np.vstack(array_1d_rep)

    # Repeating elements along axis=0
    array_2d_rep = np.repeat(stacked, n_reps, axis=0)

    # Do not repeat final row and column as they are on the edge
    return array_2d_rep[: -(n_reps - 1), : -(n_reps - 1)]


# -----------------------------------------
# Compare CoCiP outputs with GOES satellite
# -----------------------------------------


def compare_cocip_with_goes(
    time: np.timedelta64 | pd.Timestamp,
    flight: GeoVectorDataset | pd.DataFrame,
    contrail: GeoVectorDataset | pd.DataFrame,
    *,
    spatial_bbox: tuple[float, float, float, float] = (-160.0, -80.0, 10.0, 80.0),
    region: str = "F",
    path_write_img: pathlib.Path | None = None,
) -> None | pathlib.Path:
    r"""
    Compare simulated persistent contrails from CoCiP with GOES satellite imagery.

    Parameters
    ----------
    time : np.timedelta64 | pd.Timestamp
        Time of GOES satellite image.
    flight : GeoVectorDataset | pd.DataFrame
        Flight waypoints.
        Best to use the returned output :class:`Flight` from
        :meth:`pycontrails.models.cocip.Cocip.eval`.
    contrail : GeoVectorDataset | pd.DataFrame,
        Contrail evolution outputs (:attr:`pycontrails.models.cocip.Cocip.contrail`)
        set during :meth:`pycontrails.models.cocip.Cocip.eval`.
    spatial_bbox : tuple[float, float, float, float]
        Spatial bounding box, ``(lon_min, lat_min, lon_max, lat_max)``, [:math:`\deg`]
    region : str
        'F' for full disk (image provided every 10 m), and 'C' for CONUS (image provided every 5 m)
    path_write_img : None | pathlib.Path
        File path to save the CoCiP-GOES image.

    Returns
    -------
    None | pathlib.Path
        File path of saved CoCiP-GOES image if ``path_write_img`` is provided.
    """

    from pycontrails.datalib.goes import GOES, extract_goes_visualization

    try:
        import cartopy.crs as ccrs
        from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    except ModuleNotFoundError as e:
        dependencies.raise_module_not_found_error(
            name="compare_cocip_with_goes function",
            package_name="cartopy",
            module_not_found_error=e,
            pycontrails_optional_package="sat",
        )

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        dependencies.raise_module_not_found_error(
            name="compare_cocip_with_goes function",
            package_name="matplotlib",
            module_not_found_error=e,
            pycontrails_optional_package="vis",
        )

    # Round `time` to nearest GOES image time slice
    if isinstance(time, np.timedelta64):
        time = pd.to_datetime(time)

    if region == "F":
        time = time.round("10min")
    elif region == "C":
        time = time.round("5min")
    else:
        raise AssertionError("`region` only accepts inputs of `F` (full disk) or `C` (CONUS)")

    _flight = GeoVectorDataset(flight)
    _contrail = GeoVectorDataset(contrail)

    # Ensure the required columns are included in `flight_waypoints` and `contrails`
    _flight.ensure_vars(["flight_id", "waypoint"])
    _contrail.ensure_vars(
        ["flight_id", "waypoint", "sin_a", "cos_a", "width", "tau_contrail", "age_hours"]
    )

    # Downselect `_flight` only to spatial domain covered by GOES full disk
    is_in_lon = _flight.dataframe["longitude"].between(spatial_bbox[0], spatial_bbox[2])
    is_in_lat = _flight.dataframe["latitude"].between(spatial_bbox[1], spatial_bbox[3])
    is_in_lon_lat = is_in_lon & is_in_lat

    if not np.any(is_in_lon_lat):
        warnings.warn(
            "Flight trajectory does not intersect with the defined spatial bounding box or spatial "
            "domain covered by GOES."
        )

    _flight = _flight.filter(is_in_lon_lat)

    # Filter `_flight` if time bounds were previously defined.
    is_before_time = _flight["time"] < time

    if not np.any(is_before_time):
        warnings.warn("No flight waypoints were recorded before the specified `time`.")

    _flight = _flight.filter(is_before_time)

    # Downselect `_contrail` only to include the filtered flight waypoints
    is_in_domain = _contrail.dataframe["waypoint"].isin(_flight["waypoint"])

    if not np.any(is_in_domain):
        warnings.warn(
            "No persistent contrails were formed within the defined spatial bounding box."
        )

    _contrail = _contrail.filter(is_in_domain)

    # Download GOES image at `time`
    goes = GOES(region=region)
    da = goes.get(time)
    rgb, transform, extent = extract_goes_visualization(da)
    bbox = spatial_bbox[0], spatial_bbox[2], spatial_bbox[1], spatial_bbox[3]

    # Calculate optimal figure dimensions
    d_lon = spatial_bbox[2] - spatial_bbox[0]
    d_lat = spatial_bbox[3] - spatial_bbox[1]
    x_dim = 9.99
    y_dim = x_dim * (d_lat / d_lon)

    # Plot data
    fig = plt.figure(figsize=(1.2 * x_dim, y_dim))
    pc = ccrs.PlateCarree()
    ax = fig.add_subplot(projection=pc, extent=bbox)
    ax.coastlines()  # type: ignore[attr-defined]
    ax.imshow(rgb, extent=extent, transform=transform)

    ax.set_xticks([spatial_bbox[0], spatial_bbox[2]], crs=ccrs.PlateCarree())
    ax.set_yticks([spatial_bbox[1], spatial_bbox[3]], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    # Plot flight trajectory up to `time`
    ax.plot(_flight["longitude"], _flight["latitude"], c="k", linewidth=2.5)
    plt.legend(["Flight trajectory"])

    # Plot persistent contrails at `time`
    is_time = (_contrail["time"] == time) & (~np.isnan(_contrail["age_hours"]))
    im = ax.scatter(
        _contrail["longitude"][is_time],
        _contrail["latitude"][is_time],
        c=_contrail["tau_contrail"][is_time],
        s=4,
        cmap="YlOrRd_r",
        vmin=0,
        vmax=0.2,
    )
    cbar = plt.colorbar(im)
    cbar.set_label(r"$\tau_{\rm contrail}$")
    ax.set_title(f"{time}")
    plt.tight_layout()

    # return output path if `path_write_img` is not None
    if path_write_img is not None:
        t_str = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"goes_{t_str}.png"
        output_path = path_write_img.joinpath(file_name)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path
    return None
