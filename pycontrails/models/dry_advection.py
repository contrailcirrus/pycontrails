"""Simulate dry advection."""

import dataclasses
from typing import Any, NoReturn, overload

import numpy as np

from pycontrails.core import models
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import AirTemperature, EastwardWind, NorthwardWind, VerticalVelocity
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.models.cocip import contrail_properties, wind_shear
from pycontrails.physics import geo, thermo


@dataclasses.dataclass
class DryAdvectionParams(models.ModelParams):
    """Parameters for the :class:`DryAdvection` model."""

    #: Apply Euler's method with a fixed step size of ``dt_integration``. Advected waypoints
    #: are interpolated against met data once each ``dt_integration``.
    dt_integration: np.timedelta64 = np.timedelta64(30, "m")

    #: Max age of contrail evolution.
    max_age: np.timedelta64 = np.timedelta64(20, "h")

    #: Difference in altitude between top and bottom layer for stratification calculations,
    #: [:math:`m`]. Used to approximate derivative of "lagrangian_tendency_of_air_pressure" layer.
    dz_m: float = 200.0

    #: Upper bound for evolved plume depth, constraining it to realistic values.
    max_depth: float | None = 1500.0

    #: Initial plume width, [:math:`m`]. Overridden by "width" key on :attr:`source`.
    # If None, only pointwise advection is simulated without wind shear effects.
    initial_width: float | None = 100.0

    #: Initial plume depth, [:math:`m`]. Overridden by "depth" key on :attr:`source`.
    # If None, only pointwise advection is simulated without wind shear effects.
    initial_depth: float | None = 100.0

    #: Initial plume direction, [:math:`m`]. Only used if "cos_a" and "sin_a" keys are
    #: not included on :attr:`source`.
    # If None, only pointwise advection is simulated without wind shear effects.
    initial_azimuth: float | None = 0.0


class DryAdvection(models.Model):
    """Simulate "dry advection" of a plume with an elliptical cross section."""

    name = "dry_advection"
    long_name = "Advection without sedimentation"
    met_variables = AirTemperature, EastwardWind, NorthwardWind, VerticalVelocity
    default_params = DryAdvectionParams

    met: MetDataset
    met_required = True

    @overload
    def eval(self, source: None = None, **params: Any) -> NoReturn:
        ...

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset:
        ...

    def eval(self, source: GeoVectorDataset | None = None, **params: Any) -> GeoVectorDataset:
        """Simulate dry advection (no sedimentation) of arbitrary points.

        Parameters
        ----------
        source : GeoVectorDataset
            Arbitrary points to advect.
        params : Any
            Overwrite model parameters defined in ``__init__``.

        Returns
        -------
        GeoVectorDataset
            Advected points.
        """
        self.update_params(params)
        self.set_source(source)
        self.source = self.require_source_type(GeoVectorDataset)

        vector = _prepare_source(self.source, self.params)
        interp_kwargs = self.interp_kwargs

        dt = self.params["dt_integration"]
        n_steps = self.params["max_age"] // dt
        dz_m = self.params["dz_m"]
        max_depth = self.params["max_depth"]

        evolved = []
        for _ in range(n_steps):
            vector = _evolve_one_step(
                self.met,
                vector,
                dz_m=dz_m,
                dt=dt,
                max_depth=max_depth,
                **interp_kwargs,
            )
            evolved.append(vector)
            if not np.any(vector.coords_intersect_met(self.met)):
                break

        return GeoVectorDataset.sum(evolved, fill_value=np.nan)


def _prepare_source(vector: GeoVectorDataset, params: dict[str, Any]) -> GeoVectorDataset:
    """Prepare vector for advection."""

    # TODO: head, tail
    if "sigma_yz" not in vector:
        vector["sigma_yz"] = np.zeros_like(vector["longitude"])

    if "width" not in vector:
        try:
            vector.broadcast_attrs("width")
        except KeyError:
            vector["width"] = np.full_like(vector["longitude"], params["initial_width"])

    if "depth" not in vector:
        try:
            vector.broadcast_attrs("depth")
        except KeyError:
            vector["depth"] = np.full_like(vector["longitude"], params["initial_depth"])

    return vector


def _evolve_one_step(
    met: MetDataset,
    vector: GeoVectorDataset,
    *,
    dz_m: float,
    max_depth: float | None,
    dt: np.timedelta64,
    **interp_kwargs: Any,
) -> GeoVectorDataset:
    air_temperature = models.interpolate_met(met, vector, "air_temperature", **interp_kwargs)
    v_wind = models.interpolate_met(met, vector, "northward_wind", "v_wind", **interp_kwargs)
    u_wind = models.interpolate_met(met, vector, "eastward_wind", "u_wind", **interp_kwargs)
    vertical_velocity = models.interpolate_met(
        met,
        vector,
        "lagrangian_tendency_of_air_pressure",
        "vertical_velocity",
        **interp_kwargs,
    )

    level = vector.level
    air_pressure = vector.air_pressure

    air_pressure_lower = thermo.p_dz(air_temperature, air_pressure, dz_m)
    level_lower = air_pressure_lower / 100.0

    u_wind_lower = models.interpolate_met(
        met,
        vector,
        "eastward_wind",
        "u_wind_lower",
        level=level_lower,
        **interp_kwargs,
    )
    v_wind_lower = models.interpolate_met(
        met,
        vector,
        "northward_wind",
        "v_wind_lower",
        level=level_lower,
        **interp_kwargs,
    )
    air_temperature_lower = models.interpolate_met(
        met,
        vector,
        "air_temperature",
        level=level_lower,
        **interp_kwargs,
    )

    latitude = vector["latitude"]
    longitude = vector["longitude"]

    # FIXME: azimuth
    cos_a = 1.0
    sin_a = 0.0

    width = vector["width"]
    depth = vector["depth"]
    sigma_yz = vector["sigma_yz"]

    ds_dz = wind_shear.wind_shear(u_wind, u_wind_lower, v_wind, v_wind_lower, dz_m)

    # wind shear normal
    dsn_dz = wind_shear.wind_shear_normal(
        u_wind_top=u_wind,
        u_wind_btm=u_wind_lower,
        v_wind_top=v_wind,
        v_wind_btm=v_wind_lower,
        cos_a=cos_a,
        sin_a=sin_a,
        dz=dz_m,
    )

    # shear_enhancement = wind_shear.wind_shear_enhancement_factor(
    #     contrail_depth=depth,
    #     effective_vertical_resolution=effective_vertical_resolution,
    #     wind_shear_enhancement_exponent=wind_shear_enhancement_exponent,
    # )
    # ds_dz = ds_dz * shear_enhancement
    # dsn_dz = dsn_dz * shear_enhancement

    dT_dz = thermo.T_potential_gradient(
        air_temperature,
        air_pressure,
        air_temperature_lower,
        air_pressure_lower,
        dz_m,
    )

    area_eff = contrail_properties.plume_effective_cross_sectional_area(width, depth, sigma_yz)
    depth_eff = contrail_properties.plume_effective_depth(width, area_eff)

    diffuse_h = contrail_properties.horizontal_diffusivity(ds_dz, depth)
    diffuse_v = contrail_properties.vertical_diffusivity(
        air_pressure,
        air_temperature,
        dT_dz,
        depth_eff,
        terminal_fall_speed=0.0,
        sedimentation_impact_factor=0.0,
        eff_heat_rate=None,
    )

    sigma_yy_2, sigma_zz_2, sigma_yz_2 = contrail_properties.plume_temporal_evolution(
        width,
        depth,
        sigma_yz,
        dsn_dz,
        diffuse_h,
        diffuse_v,
        seg_ratio=1.0,
        dt=dt,
        max_depth=max_depth,
    )
    width_2, depth_2 = contrail_properties.new_contrail_dimensions(sigma_yy_2, sigma_zz_2)

    longitude_2 = geo.advect_longitude(longitude, latitude, u_wind, dt)
    latitude_2 = geo.advect_latitude(latitude, v_wind, dt)
    level_2 = geo.advect_level(level, vertical_velocity, 0.0, 0.0, dt=dt)

    return GeoVectorDataset(
        {
            "longitude": longitude_2,
            "latitude": latitude_2,
            "level": level_2,
            "time": vector["time"] + dt,
            "sigma_yz": sigma_yz_2,
            "width": width_2,
            "depth": depth_2,
        },
        copy=False,
    )
