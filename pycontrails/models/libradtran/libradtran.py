"""Interface to libRadTran radiative transfer model."""

from __future__ import annotations

import copy
import dataclasses
import logging
import multiprocessing
import os
import warnings
from collections.abc import Callable, Iterable
from typing import Any, NoReturn, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pycontrails.core import GeoVectorDataset, MetDataset, cache, met_var, models
from pycontrails.core.models import interpolate_met
from pycontrails.datalib.ecmwf import variables as ecmwf
from pycontrails.models.libradtran import options, subcolumns, utils
from pycontrails.models.libradtran.cocip_input import CocipInput
from pycontrails.physics import constants
from pycontrails.utils.types import ArrayScalarLike

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LibRadtranParams(models.ModelParams):
    """Default parameters for the pycontrails :class:`LibRadTran` interface."""

    #: If True, ignore background cloud
    clearsky: bool = False

    #: Number of subcolumns for background cloud
    subcolumns: int | Iterable[float] = 0

    #: Postprocessing function
    postprocess: Callable[[LibRadtran], GeoVectorDataset] | None = None

    #: CO2 volume mixing ratio :math:`[ppmv]`
    co2_ppmv: float = 400.0

    #: Liquid cloud droplet number concentration [:math:`m^{-3}`]
    nd_liq: float = 50e6

    #: Liquid cloud droplet size spectrum parameter
    k_liq: float = 0.77

    #: Threshold snow depth, in m water equivalent, to treat
    #: pixel as snow-covered.
    threshold_snow_depth: float = 0.1

    #: Threshols sea ice concentration to treat pixel as sea ice
    threshold_sea_ice_concentration: float = 0.5

    #: Raise error if met data does not cover full vertical grid.
    #: Otherwise, exclude levels where met data is missing.
    missing_met_error: bool = False

    #: Number of parallel workers.
    #: Each worker is responsible for an independent libRadtran calculation.
    num_workers: int = 1


class LibRadtran(models.Model):
    r"""Interface to libRadtran radiative transfer model.

    libRadtran paper: https://gmd.copernicus.org/articles/9/1647/2016/.

    Parameters
    ----------
    met : MetDataset
        Pressure level dataset containing :attr:`met_variables` variables.
        See *Notes* for information about required variables.
    sfc : MetDataset
        Single level dataset containing :attr:`sfc_variables` surface variables.
        See *Notes* for information about required variables.
    params : dict[str, Any], optional
        Override Cocip model parameters with dictionary.
        See :class:`LibRadTranParams` for model parameters.
    **params_kwargs : Any
        Override Cocip model parameters with keyword arguments.
        See :class:`LibRadTranParams` for model parameters.

    Notes
    -----
    **Meteorology**

    TODO

    **Acquiring and compiling LibRadTran**

    TODO
    """

    __slots__ = (
        "cachestore",
        "cocip",
        "lrt_options",
        "paths",
        "sfc",
    )

    name = "libRadtran"
    long_name = "libRadtran interface"
    default_params = LibRadtranParams
    met_variables = (
        met_var.AirTemperature,
        met_var.Geopotential,
        met_var.SpecificHumidity,
        ecmwf.OzoneMassMixingRatio,
        ecmwf.CloudAreaFractionInLayer,
        ecmwf.SpecificCloudLiquidWaterContent,
        ecmwf.SpecificCloudIceWaterContent,
    )

    sfc_variables = (
        ecmwf.SurfaceSkinTemperature,
        ecmwf.SurfaceGeopotential,
        ecmwf.SnowDepth,
        ecmwf.SeaIceConcentration,
    )

    #: Met data is not optional
    met: MetDataset
    met_required = True

    #: Surface data
    sfc: MetDataset

    #: Cocip contrail input
    cocip: CocipInput | None

    #: Output paths and metadata
    paths: pd.DataFrame | None

    #: Options provided directly to libRadtran.
    #: See libRadtran documentation for details.
    lrt_options: dict[str, str]

    #: Cachestore where input and output files are stored
    cachestore: cache.CacheStore

    #: Last set of points where model was run
    source: GeoVectorDataset

    def __init__(
        self,
        met: MetDataset,
        sfc: MetDataset,
        cocip: CocipInput | None = None,
        lrt_options: dict[str, str] | None = None,
        cachestore: cache.CacheStore | None = None,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ) -> None:
        # call Model init
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="\nMet data appears")
            super().__init__(met, params=params, **params_kwargs)

        sfc.ensure_vars(self.sfc_variables)
        self.sfc = sfc

        self.cocip = cocip

        self.lrt_options = lrt_options or options.get_default_options("thermal radiance")

        if cachestore is None:
            cache_root = cache._get_user_cache_dir()
            cache_dir = f"{cache_root}/libRadtran"
            cachestore = cache.DiskCacheStore(cache_dir=cache_dir)
        self.cachestore = cachestore

        self.paths = None

    # ----------
    # Public API
    # ----------

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset: ...

    @overload
    def eval(self, source: None = ..., **params: Any) -> NoReturn: ...

    def eval(self, source: GeoVectorDataset | None = None, **params: Any) -> GeoVectorDataset:
        """Run libRadtran.

        Parameters
        ----------
        source : GeoVectorDataset
            Locations where libRadtran should be run.

        Returns
        -------
        TODO
        """
        self.update_params(params)
        self.set_source(source)
        self.source = self.require_source_type(GeoVectorDataset)

        self.paths = self.prepare_input()
        rundirs = self.paths["path"]

        if self.params["num_workers"] == 1:
            for job in rundirs:
                utils.run(job)
        else:
            with multiprocessing.Pool(self.params["num_workers"]) as pool:
                pool.map(utils.run, rundirs)

        postprocess = self.params["postprocess"]
        if postprocess is not None:
            return postprocess(self)
        return self.source

    def prepare_input(self) -> pd.DataFrame:
        """Prepare libRadtran input files."""

        # Downselect and load met data
        met = self.source.downselect_met(
            self.met,
            level_buffer=(np.inf, np.inf),
            copy=False,
        )
        logger.debug(f"Loading {met.data.nbytes/1e6:.2f} MB of met data")
        met.data.load()

        # Downselect and load surface data
        sfc = self.source.downselect_met(
            self.sfc,
            copy=False,
        )
        logger.debug(f"Loading {sfc.data.nbytes/1e6:.2f} MB of surface data")
        sfc.data.load()

        paths = []
        for scene, row in self.source.dataframe.iterrows():
            lon = row["longitude"]
            lat = row["latitude"]
            time = row["time"].to_numpy()
            paths.append(self.write_input(scene, met, sfc, lon, lat, time))

        return pd.concat(paths)

    def write_input(
        self,
        scene: int,
        met: MetDataset,
        sfc: MetDataset,
        lon: float,
        lat: float,
        time: np.datetime64,
    ) -> pd.DataFrame:
        """Write input files for a single point."""

        rootdir = self.cachestore.path(str(scene))
        try:
            os.makedirs(rootdir, exist_ok=False)
        except FileExistsError as exc:
            msg = f"Run directory {rootdir} already exists"
            raise FileExistsError(msg) from exc

        # Avoid overwriting instance copy of static options
        options = copy.copy(self.lrt_options)

        interp_kwargs = self.interp_kwargs

        # Basic options
        target = GeoVectorDataset(longitude=[lon], latitude=[lat], altitude=[-1], time=[time])
        zs = (
            interpolate_met(sfc, target, "geopotential_at_surface", **interp_kwargs).item()
            / constants.g
        )
        ts = interpolate_met(sfc, target, "skin_temperature", **interp_kwargs).item()
        snow = interpolate_met(sfc, target, "sea_ice_cover", **interp_kwargs).item()
        sea_ice = interpolate_met(sfc, target, "snow_depth", **interp_kwargs).item()
        new_options = {
            "time": f"{_format_time(time)}",
            "latitude": f"{_format_lat(lat)}",
            "longitude": f"{_format_lon(lon)}",
            "altitude": f"{zs/1e3:.8f}",
            "sur_temperature": f"{ts:.8f}",
            "albedo_library": "IGBP",
        }
        if not (snow or sea_ice):
            new_options["surface_type_map"] = "IGBP"
        elif snow:
            new_options["brdf_rpv_type"] = "19"
        else:
            new_options["brdf_rpv_type"] = "20"
        options = utils.check_merge(options, new_options)

        # Atmosphere profiles
        z, p, t, n, n_o3, n_o2, n_v, n_co2 = self._interpolate_atmosphere(met, lon, lat, time)
        path = os.path.join(rootdir, "atmosphere")
        utils.write_atmosphere_file(path, z, p, t, n, n_o3, n_o2, n_v, n_co2)
        options = utils.check_set(options, "atmosphere_file", path)

        # Contrail profiles
        if self.cocip is not None:
            profiles = self.cocip.get_profiles(lon, lat, time)
            for name, data in profiles.items():
                path = os.path.join(rootdir, f"cocip-{name}")
                utils.write_cloud_file(path, data["z"], data["cwc"], data["re"])
                utils.check_set(options, f"profile_file cocip-{name}", f"1D {path}")
                for opt in data["options"]:
                    words = opt.split()
                    key = " ".join([words[0], f"cocip-{name}"])
                    value = " ".join(words[1:])
                    utils.check_set(options, key, value)

        # Done if ignoring background cloud
        if self.params["clearsky"]:
            path = os.path.join(rootdir, "stdin")
            utils.write_input(path, options)
            return pd.DataFrame(data={"scene": [scene], "path": rootdir}).set_index("scene")

        # Interpolate background cloud
        z, t, lwc, iwc, cf = self._interpolate_cloud(met, sfc, lon, lat, time)

        # Set paths
        cloud_subcolumns = self.params["subcolumns"]
        if isinstance(cloud_subcolumns, Iterable):
            columns = np.clip(np.asarray(cloud_subcolumns), 0.0, 1.0)
            lwc_subcol, iwc_subcol, weights = subcolumns.generate_subcolumns(lwc, iwc, cf, columns)
            ncol = len(lwc_subcol)
            rundirs = [os.path.join(rootdir, "subcolumns", str(i)) for i in range(ncol)]
            paths = pd.DataFrame(
                data={
                    "scene": [scene] * ncol,
                    "subcolumn": list(range(ncol)),
                    "path": rundirs,
                    "weight": weights,
                }
            ).set_index(["scene", "subcolumn"])
        elif cloud_subcolumns >= 1:
            col_start = 0.5 / self.params["subcolumns"]
            col_end = 1.0 - col_start
            columns = np.linspace(col_start, col_end, self.params["subcolumns"])
            lwc_subcol, iwc_subcol, weights = subcolumns.generate_subcolumns(lwc, iwc, cf, columns)
            ncol = len(lwc_subcol)
            rundirs = [os.path.join(rootdir, "subcolumns", str(i)) for i in range(ncol)]
            paths = pd.DataFrame(
                data={
                    "scene": [scene] * ncol,
                    "subcolumn": list(range(ncol)),
                    "path": rundirs,
                    "weight": weights,
                }
            ).set_index(["scene", "subcolumn"])
        else:
            lwc_subcol = [lwc]
            iwc_subcol = [iwc]
            weights = [1.0]
            rundirs = [rootdir]
            paths = pd.DataFrame(data={"scene": [scene], "path": rundirs}).set_index("scene")

        # Write background cloud profiles
        for lwc, iwc, rundir in zip(lwc_subcol, iwc_subcol, rundirs, strict=True):
            local_options = copy.copy(options)
            os.makedirs(rundir, exist_ok=True)

            if np.any(lwc > 0):
                rel = np.zeros_like(lwc)
                rel[lwc > 0] = _reff_liquid(
                    lwc[lwc > 0], self.params["nd_liq"], self.params["k_liq"]
                )
                start = max(0, np.flatnonzero(lwc > 0).min() - 1)
                if lwc[start] != 0:
                    msg = "Nonzero liquid water content in top layer. Consider extending domain."
                    warnings.warn(msg)
                path = os.path.join(rundir, "background-liquid")
                utils.write_cloud_file(path, z[start:], lwc[start:], rel[start:])
                utils.check_set(local_options, "profile_file background-liquid", f"1D {path}")
                utils.check_set(
                    local_options,
                    "profile_properties background-liquid",
                    "mie interpolate",
                )

            if np.any(iwc > 0):
                rei = np.zeros_like(iwc)
                rei[iwc > 0] = _reff_ice(iwc[iwc > 0], t[iwc > 0], lat)
                start = max(0, np.flatnonzero(iwc > 0).min() - 1)
                if iwc[start] != 0:
                    msg = "Nonzero ice water content in top layer. Consider extending domain."
                    warnings.warn(msg)
                path = os.path.join(rundir, "background-ice")
                utils.write_cloud_file(path, z[start:], iwc[start:], rei[start:])
                utils.check_set(local_options, "profile_file background-ice", f"1D {path}")
                utils.check_set(
                    local_options,
                    "profile_properties background-ice",
                    "baum_v36 interpolate",
                )
                utils.check_set(local_options, "profile_habit background-ice", "ghm")

            path = os.path.join(rundir, "stdin")
            utils.write_input(path, local_options)

        return paths

    def _interpolate_atmosphere(
        self,
        met: MetDataset,
        lon: float,
        lat: float,
        time: np.datetime64,
    ) -> tuple[npt.NDArray[np.float64], ...]:
        """Interpolate meteorology data to get input profiles."""

        interp_kwargs = self.interp_kwargs

        level = met["level"].data.to_numpy()[1:]  # interpolation produces nan at top level
        target = GeoVectorDataset(
            time=np.full(level.shape, time),
            level=level,
            latitude=np.full(level.shape, lat),
            longitude=np.full(level.shape, lon),
        )

        z = interpolate_met(met, target, "geopotential", **interp_kwargs) / constants.g
        p = met["air_pressure"].data.to_numpy()[1:]
        t = interpolate_met(met, target, "air_temperature", **interp_kwargs)
        n = p / (constants.k_boltzmann * t)
        q_o3 = interpolate_met(met, target, "mass_fraction_of_ozone_in_air", **interp_kwargs)
        q_o3 = np.maximum(0, q_o3)
        n_o3 = n * q_o3 * constants.M_d / constants.M_o3
        n_o2 = n * constants.eta_o2
        q_v = interpolate_met(met, target, "specific_humidity", **interp_kwargs)
        q_v = np.maximum(0, q_v)
        n_v = n * q_v * constants.M_d / constants.M_v
        n_co2 = n * self.params["co2_ppmv"] / 1e6

        # Mask missing values at lowest levels
        mask = np.isnan(t)
        for v in [n, n_o3, n_o2, n_v, n_co2]:
            mask = mask | np.isnan(v)
        end = np.flatnonzero(~mask).max() + 1
        z = z[:end]
        p = p[:end]
        t = t[:end]
        n = n[:end]
        n_o3 = n_o3[:end]
        n_o2 = n_o2[:end]
        n_v = n_v[:end]
        n_co2 = n_co2[:end]

        # Check for missing values elsewhere
        missing = np.isnan(z)
        for v in [p, t, n, n_o3, n_o2, n_v, n_co2]:
            missing = missing | np.isnan(v)
        if np.any(missing):
            missing_z = z[missing] / 1e3
            missing_p = p[missing] / 1e2
            msg = (
                f"Met data missing at {missing.sum()} levels "
                f"(z = {missing_z.tolist()} km, "
                f"p = {missing_p.tolist()} hPa)."
            )
            if self.params["missing_met_error"]:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
        if np.all(missing):
            msg = "All levels missing in interpolated met data."
            raise ValueError(msg)

        return (
            z[~missing],
            p[~missing],
            t[~missing],
            n[~missing],
            n_o3[~missing],
            n_o2[~missing],
            n_v[~missing],
            n_co2[~missing],
        )

    def _interpolate_cloud(
        self,
        met: MetDataset,
        sfc: MetDataset,
        lon: float,
        lat: float,
        time: np.datetime64,
    ) -> tuple[npt.NDArray[np.float64], ...]:
        """Interpolate meteorology data to get background cloud profiles."""

        interp_kwargs = self.interp_kwargs

        # Get target points on profile
        level = met.data["level"].to_numpy()[1:]
        target = GeoVectorDataset(
            time=np.full(level.shape, time),
            level=level,
            latitude=np.full(level.shape, lat),
            longitude=np.full(level.shape, lon),
        )
        sfc_target = GeoVectorDataset(
            time=[time],
            level=[-1],
            latitude=[lat],
            longitude=[lon],
        )

        # Interpolate
        z = interpolate_met(met, target, "geopotential", **interp_kwargs) / constants.g
        zs = (
            interpolate_met(sfc, sfc_target, "geopotential_at_surface", **interp_kwargs).item()
            / constants.g
        )
        p = met.data["air_pressure"].to_numpy()[1:]
        t = interpolate_met(met, target, "air_temperature", **interp_kwargs)
        rho = p / (constants.R_d * t)
        lwc = rho * interpolate_met(
            met, target, "specific_cloud_liquid_water_content", **interp_kwargs
        )
        iwc = rho * interpolate_met(
            met, target, "specific_cloud_ice_water_content", **interp_kwargs
        )
        cf = interpolate_met(met, target, "fraction_of_cloud_cover", **interp_kwargs)
        lwc = np.maximum(0.0, lwc)
        iwc = np.maximum(0.0, iwc)
        cf = np.clip(cf, 0.0, 1.0)

        # Mask missing values at lowest levels
        mask = np.isnan(z)
        for v in [t, lwc, iwc, cf]:
            mask = mask | np.isnan(v)
        end = np.flatnonzero(~mask).max() + 1
        z = z[:end]
        t = t[:end]
        lwc = lwc[:end]
        iwc = iwc[:end]
        cf = cf[:end]

        # Compute altitudes of below-layer interfaces
        zb = 0.5 * (zs + z[-1])
        zi = np.append(0.5 * (z[1:] + z[:-1]), zb)
        return zi, t, lwc, iwc, cf


def _format_time(time: np.datetime64) -> str:
    return pd.Timestamp(time).strftime("%Y %m %d %H %M %S")


def _format_lat(lat: float) -> str:
    if lat >= 0:
        return f"N {lat:.6f}"
    return f"S {-lat:.6f}"


def _format_lon(lon: float) -> str:
    if lon >= 0:
        return f"E {lon:.6f}"
    return f"W {-lon:.6f}"


def _reff_liquid(
    lwc: ArrayScalarLike, nd_liq: ArrayScalarLike, k_liq: ArrayScalarLike
) -> ArrayScalarLike:
    """Compute effective radius of liquid cloud.

    Based on IFS formulation: https://www.ecmwf.int/sites/default/files/2023-06/Part-IV-Physical-Processes.pdf
    """
    reff = (3.0 * lwc / (4.0 * np.pi * constants.rho_liq * k_liq * nd_liq)) ** (1.0 / 3.0)
    return np.clip(reff, 1e-6, 25e-6)


def _reff_ice(iwc: ArrayScalarLike, t: ArrayScalarLike, lat: ArrayScalarLike) -> ArrayScalarLike:
    """Compute effective radius of ice cloud.

    Based on IFS formulation: https://www.ecmwf.int/sites/default/files/2023-06/Part-IV-Physical-Processes.pdf
    """
    iwc = iwc * 1e3
    a = 45.8966 * iwc**0.2214
    b = 0.7957 * iwc**0.2535
    deff = (1.2351 + 0.0105 * (t - 273.15)) * (a + b * (t - 83.15))
    dmin = 20.0 + 40.0 * np.cos(np.deg2rad(lat))
    dmax = 155.0
    deff = np.clip(deff, dmin, dmax)
    reff = 0.64952 * deff
    return np.clip(1e-6 * reff, 5e-6, 60e-6)
