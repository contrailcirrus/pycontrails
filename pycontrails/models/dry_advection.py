"""Simulate dry advection."""

from __future__ import annotations

import dataclasses
import sys
from typing import Any, NoReturn, overload

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import numpy.typing as npt
import pandas as pd

from pycontrails.core import models
from pycontrails.core.met import MetDataset, maybe_downselect_mds
from pycontrails.core.met_var import (
    AirTemperature,
    EastwardWind,
    MetVariable,
    NorthwardWind,
    VerticalVelocity,
)
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.models.cocip import contrail_properties, wind_shear
from pycontrails.physics import geo, thermo


@dataclasses.dataclass
class DryAdvectionParams(models.AdvectionBuffers):
    """Parameters for the :class:`DryAdvection` model."""

    #: Apply Euler's method with a fixed step size of ``dt_integration``. Advected waypoints
    #: are interpolated against met data once each ``dt_integration``.
    dt_integration: np.timedelta64 = np.timedelta64(30, "m")

    #: Max age of plume evolution.
    max_age: np.timedelta64 = np.timedelta64(20, "h")

    #: Rate of change of pressure due to sedimentation [:math:`Pa/s`]
    sedimentation_rate: float = 0.0

    #: Difference in altitude between top and bottom layer for stratification calculations,
    #: [:math:`m`]. Used to approximate derivative of "lagrangian_tendency_of_air_pressure"
    #: (upward component of air velocity) with respect to altitude.
    dz_m: float = 200.0

    #: Upper bound for evolved plume depth, constraining it to realistic values.
    max_depth: float | None = 1500.0

    #: Initial plume width, [:math:`m`]. Overridden by "width" key on :attr:`source`.
    # If None, only pointwise advection is simulated without wind shear effects.
    width: float | None = 100.0

    #: Initial plume depth, [:math:`m`]. Overridden by "depth" key on :attr:`source`.
    # If None, only pointwise advection is simulated without wind shear effects.
    depth: float | None = 100.0

    #: Initial plume direction, [:math:`m`]. Overridden by "azimuth" key on :attr:`source`.
    # If None, only pointwise advection is simulated without wind shear effects.
    azimuth: float | None = 0.0

    #: Add additional intermediate variables to the output vector.
    #: This includes interpolated met variables and wind-shear-derived geometry.
    verbose_outputs: bool = False

    #: Whether to include ``source`` points in the output vector. Enabling allows
    #: the user to view additional data (e.g., interpolated met variables) for
    #: source points as well as evolved points.
    include_source_in_output: bool = False


class DryAdvection(models.Model):
    """Simulate "dry advection" of an emissions plume with an elliptical cross section.

    The model simulates both horizontal and vertical advection of a weightless
    plume without any sedimentation effects. Unlike :class:`Cocip`, humidity is
    not considered, and radiative forcing is not simulated. The model is
    therefore useful simulating plume advection and dispersion itself.

    .. versionadded:: 0.46.0

    This model has two distinct modes of operation:

    - **Pointwise only**: If ``azimuth`` is None, then the model will only advect
        points without any wind shear effects. This mode is useful for testing
        the advection algorithm itself, and for simulating the evolution of
        a single point.
    - **Wind shear effects**: If ``azimuth`` is not None, then the model will
        advect points with wind shear effects. At each time step, the model
        will evolve the plume geometry according to diffusion and wind shear
        effects. This mode is also used in :class:`CocipGrid` and :class:`Cocip`.

    Parameters
    ----------
    met : MetDataset
        Meteorological data.
    params : dict[str, Any]
        Model parameters. See :class:`DryAdvectionParams` for details.
    **kwargs : Any
        Additional parameters passed as keyword arguments.
    """

    name = "dry_advection"
    long_name = "Emission plume advection without sedimentation"
    met_variables: tuple[MetVariable, ...] = (
        AirTemperature,
        EastwardWind,
        NorthwardWind,
        VerticalVelocity,
    )
    default_params = DryAdvectionParams

    met: MetDataset
    met_required = True
    source: GeoVectorDataset

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset: ...

    @overload
    def eval(self, source: None = ..., **params: Any) -> NoReturn: ...

    def eval(self, source: GeoVectorDataset | None = None, **params: Any) -> GeoVectorDataset:
        """Simulate dry advection (no sedimentation) of arbitrary points.

        Like :class:`Cocip`, this model adds a "waypoint" column to the :attr:`source`.

        Parameters
        ----------
        source : GeoVectorDataset
            Arbitrary points to advect. A :class:`Flight` instance is not treated any
            differently than a :class:`GeoVectorDataset`. In particular, the user must
            explicitly set ``flight["azimuth"] = flight.segment_azimuth()`` if they
            want to use wind shear effects for a flight.
            In the current implementation, any existing meteorological variables in the ``source``
            are ignored. The ``source`` will be interpolated against the :attr:`met` dataset.
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
        self.downselect_met()
        if not self.source.coords_intersect_met(self.met).any():
            msg = "No source coordinates intersect met data."
            raise ValueError(msg)

        self.source = self._prepare_source()

        interp_kwargs = self.interp_kwargs

        dt_integration = self.params["dt_integration"]
        max_age = self.params["max_age"]
        sedimentation_rate = self.params["sedimentation_rate"]
        dz_m = self.params["dz_m"]
        max_depth = self.params["max_depth"]
        verbose_outputs = self.params["verbose_outputs"]

        source_time = self.source["time"]
        t0 = pd.Timestamp(source_time.min()).floor(pd.Timedelta(dt_integration)).to_numpy()
        t1 = source_time.max()
        timesteps = np.arange(t0 + dt_integration, t1 + dt_integration + max_age, dt_integration)

        vector2 = GeoVectorDataset()
        met = None

        evolved = []
        for t in timesteps:
            filt = (source_time < t) & (source_time >= t - dt_integration)
            vector1 = vector2 + self.source.filter(filt, copy=False)

            t0 = vector1["time"].min()
            t1 = vector1["time"].max()
            met = maybe_downselect_mds(self.met, met, t0, t1)

            vector2 = _evolve_one_step(
                met,
                vector1,
                t,
                sedimentation_rate=sedimentation_rate,
                dz_m=dz_m,
                max_depth=max_depth,
                verbose_outputs=verbose_outputs,
                **interp_kwargs,
            )
            evolved.append(vector1)

            filt = (vector2["age"] <= max_age) & vector2.coords_intersect_met(self.met)
            vector2 = vector2.filter(filt)

            if not vector2 and np.all(source_time < t):
                break

        evolved.append(vector2)
        out = GeoVectorDataset.sum(evolved, fill_value=np.nan)

        if self.params["include_source_in_output"]:
            return out

        filt = out["age"] > np.timedelta64(0, "ns")
        return out.filter(filt)

    def _prepare_source(self) -> GeoVectorDataset:
        r"""Prepare :attr:`source` vector for advection by wind-shear-derived variables.

        The following variables are always guaranteed to be present in :attr:`source`:

        - ``age``: Age of plume.
        - ``waypoint``: Identifier for each waypoint.

        If ``flight_id`` is present in :attr:`source`, it is retained.

        If `"azimuth"` is present in :attr:`source`, `source.attrs`, or :attr:`params`,
        the following variables will also be added:

        - ``azimuth``: Initial plume direction, measured in clockwise direction from
            true north, [:math:`\deg`].
        - ``width``: Initial plume width, [:math:`m`].
        - ``depth``: Initial plume depth, [:math:`m`].
        - ``sigma_yz``: All zeros for cross-term term in covariance matrix of plume.

        Returns
        -------
        GeoVectorDataset
            A filtered version of the source with only the required columns.
        """
        self.source.setdefault("level", self.source.level)
        self.source["age"] = np.full(self.source.size, np.timedelta64(0, "ns"))
        self.source.setdefault("waypoint", np.arange(self.source.size))

        columns = ["longitude", "latitude", "level", "time", "age", "waypoint"]
        if "flight_id" in self.source:
            columns.append("flight_id")

        azimuth = self.get_source_param("azimuth", set_attr=False)
        if azimuth is None:
            # Early exit for pointwise only simulation
            if self.params["width"] is not None or self.params["depth"] is not None:
                raise ValueError(
                    "If 'azimuth' is None, then 'width' and 'depth' must also be None."
                )
            return GeoVectorDataset._from_fastpath(self.source.select(columns, copy=False).data)

        if "azimuth" not in self.source:
            self.source["azimuth"] = np.full_like(self.source["longitude"], azimuth)

        for key in ("width", "depth"):
            if key in self.source:
                continue
            if key in self.source.attrs:
                self.source.broadcast_attrs(key)
                continue

            val = self.params[key]
            if val is None:
                raise ValueError(f"If '{key}' is None, then 'azimuth' must also be None.")

            self.source[key] = np.full_like(self.source["longitude"], val)

        columns.extend(["azimuth", "width", "depth", "sigma_yz", "area_eff"])
        self.source["sigma_yz"] = np.zeros_like(self.source["longitude"])
        width = self.source["width"]
        depth = self.source["depth"]
        self.source["area_eff"] = contrail_properties.plume_effective_cross_sectional_area(
            width, depth, sigma_yz=0.0
        )

        return GeoVectorDataset._from_fastpath(self.source.select(columns, copy=False).data)

    @override
    def downselect_met(self) -> None:
        if not self.params["downselect_met"]:
            return

        buffers = {
            f"{coord}_buffer": self.params[f"met_{coord}_buffer"]
            for coord in ("longitude", "latitude", "level")
        }
        buffers["time_buffer"] = (np.timedelta64(0, "ns"), self.params["max_age"])
        self.met = self.source.downselect_met(self.met, **buffers)


def _perform_interp_for_step(
    met: MetDataset,
    vector: GeoVectorDataset,
    dz_m: float,
    **interp_kwargs: Any,
) -> None:
    """Perform all interpolation required for one step of advection."""

    vector.setdefault("level", vector.level)
    air_pressure = vector.setdefault("air_pressure", vector.air_pressure)

    models.interpolate_met(met, vector, "northward_wind", "v_wind", **interp_kwargs)
    models.interpolate_met(met, vector, "eastward_wind", "u_wind", **interp_kwargs)
    models.interpolate_met(
        met,
        vector,
        "lagrangian_tendency_of_air_pressure",
        "vertical_velocity",
        **interp_kwargs,
    )

    az = vector.get("azimuth")
    if az is None:
        # Early exit for pointwise only simulation
        return

    air_temperature = models.interpolate_met(met, vector, "air_temperature", **interp_kwargs)
    air_pressure_lower = thermo.pressure_dz(air_temperature, air_pressure, dz_m)
    vector["air_pressure_lower"] = air_pressure_lower
    level_lower = air_pressure_lower / 100.0

    models.interpolate_met(
        met,
        vector,
        "eastward_wind",
        "u_wind_lower",
        level=level_lower,
        **interp_kwargs,
    )
    models.interpolate_met(
        met,
        vector,
        "northward_wind",
        "v_wind_lower",
        level=level_lower,
        **interp_kwargs,
    )
    models.interpolate_met(
        met,
        vector,
        "air_temperature",
        "air_temperature_lower",
        level=level_lower,
        **interp_kwargs,
    )

    lons = vector["longitude"]
    lats = vector["latitude"]
    dist = 1000.0

    # These should probably not be included in model input ... so
    # we'll get a warning if they get overwritten
    longitude_head, latitude_head = geo.forward_azimuth(lons=lons, lats=lats, az=az, dist=dist)
    longitude_tail, latitude_tail = geo.forward_azimuth(lons=lons, lats=lats, az=az, dist=-dist)
    vector["longitude_head"] = longitude_head
    vector["latitude_head"] = latitude_head
    vector["longitude_tail"] = longitude_tail
    vector["latitude_tail"] = latitude_tail

    for met_key in ("eastward_wind", "northward_wind"):
        vector_key = f"{met_key}_head"
        models.interpolate_met(
            met,
            vector,
            met_key,
            vector_key,
            **interp_kwargs,
            longitude=longitude_head,
            latitude=latitude_head,
        )

        vector_key = f"{met_key}_tail"
        models.interpolate_met(
            met,
            vector,
            met_key,
            vector_key,
            **interp_kwargs,
            longitude=longitude_tail,
            latitude=latitude_tail,
        )


def _calc_geometry(
    vector: GeoVectorDataset,
    dz_m: float,
    dt: npt.NDArray[np.timedelta64] | np.timedelta64,
    max_depth: float | None,
    verbose_outputs: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate wind-shear-derived geometry of evolved plume.

    This method mutates the input ``vector`` in place.
    """

    u_wind = vector["u_wind"]
    v_wind = vector["v_wind"]
    u_wind_lower = vector.data.pop("u_wind_lower")
    v_wind_lower = vector.data.pop("v_wind_lower")

    air_temperature = vector["air_temperature"]
    air_temperature_lower = vector.data.pop("air_temperature_lower")
    air_pressure = vector["air_pressure"]
    air_pressure_lower = vector.data.pop("air_pressure_lower")

    ds_dz = wind_shear.wind_shear(u_wind, u_wind_lower, v_wind, v_wind_lower, dz_m)

    azimuth = vector["azimuth"]
    latitude = vector["latitude"]
    cos_a, sin_a = geo.azimuth_to_direction(azimuth, latitude)

    width = vector["width"]
    depth = vector["depth"]
    sigma_yz = vector["sigma_yz"]
    area_eff = vector["area_eff"]

    dsn_dz = wind_shear.wind_shear_normal(
        u_wind_top=u_wind,
        u_wind_btm=u_wind_lower,
        v_wind_top=v_wind,
        v_wind_btm=v_wind_lower,
        cos_a=cos_a,
        sin_a=sin_a,
        dz=dz_m,
    )

    dT_dz = thermo.T_potential_gradient(
        air_temperature,
        air_pressure,
        air_temperature_lower,
        air_pressure_lower,
        dz_m,
    )

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

    if verbose_outputs:
        vector["ds_dz"] = ds_dz
        vector["dsn_dz"] = dsn_dz
        vector["dT_dz"] = dT_dz

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
    area_eff_2 = contrail_properties.plume_effective_cross_sectional_area(
        width_2, depth_2, sigma_yz_2
    )

    longitude_head = vector.data.pop("longitude_head")
    latitude_head = vector.data.pop("latitude_head")
    longitude_tail = vector.data.pop("longitude_tail")
    latitude_tail = vector.data.pop("latitude_tail")
    u_wind_head = vector.data.pop("eastward_wind_head")
    v_wind_head = vector.data.pop("northward_wind_head")
    u_wind_tail = vector.data.pop("eastward_wind_tail")
    v_wind_tail = vector.data.pop("northward_wind_tail")

    longitude_head_t2, latitude_head_t2 = geo.advect_horizontal(
        longitude=longitude_head,
        latitude=latitude_head,
        u_wind=u_wind_head,
        v_wind=v_wind_head,
        dt=dt,
    )
    longitude_tail_t2, latitude_tail_t2 = geo.advect_horizontal(
        longitude=longitude_tail,
        latitude=latitude_tail,
        u_wind=u_wind_tail,
        v_wind=v_wind_tail,
        dt=dt,
    )

    azimuth_2 = geo.azimuth(
        lons0=longitude_tail_t2,
        lats0=latitude_tail_t2,
        lons1=longitude_head_t2,
        lats1=latitude_head_t2,
    )

    return azimuth_2, width_2, depth_2, sigma_yz_2, area_eff_2


def _evolve_one_step(
    met: MetDataset,
    vector: GeoVectorDataset,
    t: np.datetime64,
    *,
    sedimentation_rate: float,
    dz_m: float,
    max_depth: float | None,
    verbose_outputs: bool,
    **interp_kwargs: Any,
) -> GeoVectorDataset:
    """Evolve plume geometry by one step.

    This method mutates the input ``vector`` in place.
    """

    _perform_interp_for_step(met, vector, dz_m, **interp_kwargs)
    u_wind = vector["u_wind"]
    v_wind = vector["v_wind"]
    vertical_velocity = vector["vertical_velocity"] + sedimentation_rate

    latitude = vector["latitude"]
    longitude = vector["longitude"]

    dt = t - vector["time"]
    longitude_2, latitude_2 = geo.advect_horizontal(longitude, latitude, u_wind, v_wind, dt)  # type: ignore[arg-type]
    level_2 = geo.advect_level(
        vector.level,
        vertical_velocity,
        rho_air=0.0,
        terminal_fall_speed=0.0,
        dt=dt,  # type: ignore[arg-type]
    )

    out = GeoVectorDataset._from_fastpath(
        {
            "longitude": longitude_2,
            "latitude": latitude_2,
            "level": level_2,
            "time": np.full(longitude_2.shape, t),
            "age": vector["age"] + dt,
            "waypoint": vector["waypoint"],
        }
    )

    flight_id = vector.get("flight_id")
    if flight_id is not None:
        out["flight_id"] = flight_id

    azimuth = vector.get("azimuth")
    if azimuth is None:
        # Early exit for "pointwise only" simulation
        return out

    # Attach wind-shear-derived geometry to output vector
    azimuth_2, width_2, depth_2, sigma_yz_2, area_eff_2 = _calc_geometry(
        vector,
        dz_m=dz_m,
        dt=dt,  # type: ignore[arg-type]
        max_depth=max_depth,  # type: ignore[arg-type]
        verbose_outputs=verbose_outputs,
    )
    out["azimuth"] = azimuth_2
    out["width"] = width_2
    out["depth"] = depth_2
    out["sigma_yz"] = sigma_yz_2
    out["area_eff"] = area_eff_2

    return out
