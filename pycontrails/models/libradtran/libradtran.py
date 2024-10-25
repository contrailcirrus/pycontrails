"""Interface to libRadTran radiative transfer model."""

from __future__ import annotations

import dataclasses
import functools
import logging
import multiprocessing
import warnings
from typing import Any, NoReturn, overload

import numpy as np

from pycontrails.core import GeoVectorDataset, MetDataset, cache, met_var, models
from pycontrails.core.models import interpolate_met
from pycontrails.datalib.ecmwf import variables as ecmwf
from pycontrails.models.libradtran import options, utils
from pycontrails.models.libradtran.clouds import LRTClouds
from pycontrails.physics import constants

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LibRadtranParams(models.ModelParams):
    """Default parameters for the pycontrails :class:`LibRadTran` interface."""

    #: CO2 volume mixing ratio :math:`[ppmv]`
    co2_ppmv: float = 400.0

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
        "clouds",
        "lrt_options",
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

    #: List of cloud inputs for radiative transfer calculation
    clouds: list[LRTClouds]

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
        clouds: list[LRTClouds] | None = None,
        lrt_options: dict[str, str] | None = None,
        cachestore: cache.CacheStore | None = None,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ) -> None:
        # call Model init
        super().__init__(met, params=params, **params_kwargs)

        sfc.ensure_vars(self.sfc_variables)
        self.sfc = sfc

        self.clouds = clouds or []

        self.lrt_options = lrt_options or options.get_default_options("thermal radiance")

        if cachestore is None:
            cache_root = cache._get_user_cache_dir()
            cache_dir = f"{cache_root}/libRadtran"
            cachestore = cache.DiskCacheStore(cache_dir=cache_dir)
        self.cachestore = cachestore

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

        scene_locations = self.get_locations(self.source)
        scene_met = self.get_met_profiles(self.source)
        scene_sfc = self.get_surface_options(self.source)
        scene_clouds = self.get_cloud_profiles(self.source)

        output_dirs = [self.cachestore.path(str(i)) for i in range(len(scene_locations))]
        jobs = zip(scene_locations, scene_met, scene_sfc, scene_clouds, output_dirs, strict=True)

        run = functools.partial(utils.run, options=self.lrt_options)

        if self.params["num_workers"] == 1:
            output_paths = []
            for job in jobs:
                output_paths.append(run(*job))

        else:
            with multiprocessing.Pool(self.params["num_workers"]) as pool:
                output_paths = pool.starmap(run, jobs)

        self.source["output_location"] = output_paths
        return self.source

    def get_locations(self, source: GeoVectorDataset) -> list[dict[str, Any]]:
        """Get scene locations.

        Parameters
        ----------
        source : GeoVectorDataset
            Dataset with scene locations.

        Returns
        -------
        list[dict[str, Any]]
            Input options specifying the location of each scene.
        """

        def _format_lat(lat: float) -> str:
            if lat >= 0:
                return f"N {lat:.6f}"
            return f"S {-lat:.6f}"

        def _format_lon(lon: float) -> str:
            if lon >= 0:
                return f"E {lon:.6f}"
            return f"W {-lon:.6f}"

        return [
            {
                "time": point["time"].strftime("%Y %m %d %H %M %S"),
                "latitude": f"{_format_lat(point['latitude'])}",
                "longitude": f"{_format_lon(point['longitude'])}",
            }
            for _, point in source.dataframe.iterrows()
        ]

    def get_surface_options(self, source: GeoVectorDataset) -> list[dict[str, Any]]:
        """Get input options related to surface properties.

        Parameters
        ----------
        source : GeoVectorDataset
            Locations where surface properties are required.

        Returns
        -------
        list[dict[str, Any]]
            Surface properties at required locations
        """

        # Downselect surface data
        sfc = source.downselect_met(
            self.sfc,
            copy=False,
        )
        logger.debug(f"Loading {sfc.data.nbytes/1e6:.2f} MB of surface data")
        sfc.data.load()

        # Interpolate to target locations
        interp_kwargs = self.interp_kwargs
        interpolate_met(sfc, source, "geopotential_at_surface", **interp_kwargs)
        interpolate_met(sfc, source, "skin_temperature", **interp_kwargs)
        interpolate_met(sfc, source, "sea_ice_cover", **interp_kwargs)
        interpolate_met(sfc, source, "snow_depth", **interp_kwargs)

        options = []
        for _, point in source.dataframe.iterrows():
            z = point["geopotential_at_surface"] / constants.g / 1e3
            ts = point["skin_temperature"]
            snow = point["snow_depth"] > self.params["threshold_snow_depth"]
            sea_ice = point["sea_ice_cover"] > self.params["threshold_sea_ice_concentration"]

            opt = {
                "altitude": f"{z:.8f}",
                "sur_temperature": f"{ts:.8f}",
                "albedo_library": "IGBP",
            }
            if not (snow or sea_ice):
                opt["surface_type_map"] = "IGBP"
            elif snow:
                opt["brdf_rpv_type"] = "19"
            else:
                opt["brdf_rpv_type"] = "20"

            options.append(opt)

        return options

    def get_met_profiles(self, source: GeoVectorDataset) -> list[dict[str, Any]]:
        """Get atmospheric profiles from met data.

        Parameters
        ----------
        source : GeoVectorDataset
            Locations where atmospheric profiles are required

        Returns
        -------
        list[dict[str, Any]]
            Required atmospheric profiles.
        """

        # Downselect meteorology
        met = source.downselect_met(
            self.met,
            level_buffer=(np.inf, np.inf),
            copy=False,
        )
        logger.debug(f"Loading {met.data.nbytes/1e6:.2f} MB of met data")
        met.data.load()

        # Interpolate to target profiles
        profiles = []
        interp_kwargs = self.interp_kwargs
        for _, point in source.dataframe.iterrows():
            # Interpolate met data
            level = met["level"].data.to_numpy()
            target = GeoVectorDataset(
                time=np.full(level.shape, point["time"]),
                level=level,
                latitude=np.full(level.shape, point["latitude"]),
                longitude=np.full(level.shape, point["longitude"]),
            )
            z = interpolate_met(met, target, "geopotential", **interp_kwargs) / constants.g
            p = met["air_pressure"].data.to_numpy()
            t = interpolate_met(met, target, "air_temperature", **interp_kwargs)
            n = p / (constants.k_boltzmann * t)
            q_o3 = interpolate_met(met, target, "mass_fraction_of_ozone_in_air", **interp_kwargs)
            n_o3 = n * q_o3 * constants.M_d / constants.M_o3
            n_o2 = n * constants.eta_o2
            q_v = interpolate_met(met, target, "specific_humidity", **interp_kwargs)
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
            missing = np.isnan(t)
            for v in [n, n_o3, n_o2, n_v, n_co2]:
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

            profiles.append(
                {
                    "z": z[~missing] / 1e3,
                    "p": p[~missing] / 1e2,
                    "t": t[~missing],
                    "n": n[~missing] / 1e6,
                    "n_o3": n_o3[~missing] / 1e6,
                    "n_o2": n_o2[~missing] / 1e6,
                    "n_v": n_v[~missing] / 1e6,
                    "n_co2": n_co2[~missing] / 1e6,
                }
            )

        return profiles

    def get_cloud_profiles(self, source: GeoVectorDataset) -> list[list[dict[str, Any]]]:
        """Get cloud profiles.

        Parameters
        ----------
        source : GeoVectorDataset
            Locations where cloud profiles are required

        Returns
        -------
        list[list[dict[str, Any]]]
            Required cloud profiles.
        """
        if len(self.clouds) == 0:
            return [[]] * len(source.dataframe)

        profiles = [cloud.get_profiles(source) for cloud in self.clouds]
        return [sum(p, start=[]) for p in zip(*profiles, strict=True)]
