"""Support for Himawari-8/9 satellite data."""

from __future__ import annotations

import bz2
import collections
import datetime
import enum
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pycontrails.core import cache
from pycontrails.datalib import geo_utils
from pycontrails.datalib.himawari import header_struct
from pycontrails.utils import dependencies

if TYPE_CHECKING:
    import cartopy.crs

try:
    import s3fs
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="goes module",
        package_name="s3fs",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )


#: Default bands to use if none are specified. These are the channels
#: required by the SEVIRI (MIT) ash color scheme.
DEFAULT_BANDS = "B11", "B14", "B15"

#: The date at which Himawari-9 was declared operational, replacing Himawari-8.
#: This is used to determine which S3 bucket to use if none is specified.
#: See the `documentation <https://www.data.jma.go.jp/mscweb/en/oper/switchover.html>`_
HIMAWARI_8_9_SWITCH_DATE = datetime.datetime(2022, 12, 13, 5, 0)


#: The S3 bucket for Himawari-8 data.
HIMAWARI_8_BUCKET = "noaa-himawari8"

#: The S3 bucket for Himawari-9 data.
HIMAWARI_9_BUCKET = "noaa-himawari9"


class HimawariRegion(enum.Enum):
    """Himawari-8/9 regions."""

    FLDK = enum.auto()  # Full Disk
    Japan = enum.auto()
    Target = enum.auto()


def _check_time_resolution(
    t: datetime.datetime,
    region: HimawariRegion,
) -> tuple[datetime.datetime, str]:
    """Check that the time is at a valid Himawari time resolution.

    Return the time and the scan type (FLDK, or JP01, JP02, JP03, JP04).
    """
    if t.microsecond:
        raise ValueError("Microseconds are not supported in Himawari time.")

    total_seconds = t.minute * 60 + t.second

    if region == HimawariRegion.FLDK:
        if total_seconds % 600:
            raise ValueError("Himawari FLDK data is only available at 10-minute intervals.")
        return t, "FLDK"

    if total_seconds % 150:
        raise ValueError("Himawari Japan or Target data is only available at 2.5-minute intervals.")

    offset = (total_seconds // 150) % 4
    t_floor = t - datetime.timedelta(minutes=t.minute % 10, seconds=t.second)

    prefix = "JP0" if region == HimawariRegion.Japan else "R30"
    scan_type = f"{prefix}{offset + 1}"
    return t_floor, scan_type


def _parse_bands(bands: str | Iterable[str] | None) -> set[str]:
    """Check that the bands are valid and return as a set.

    This function is nearly identical to the GOES _parse_channels function.
    """
    if bands is None:
        return set(DEFAULT_BANDS)

    if isinstance(bands, str):
        bands = (bands,)

    available = {f"B{i:02d}" for i in range(1, 17)}
    bands = {b.upper() for b in bands}
    if not bands.issubset(available):
        raise ValueError(f"bands must be in {sorted(available)}")
    return bands


def _check_band_resolution(bands: Iterable[str]) -> None:
    # https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html
    res = {
        "B01": 1.0,
        "B02": 1.0,
        "B03": 1.0,  # XXX: this actually has a resolution of 0.5 km, but we coarsen it to 1 km
        "B04": 1.0,
        "B05": 2.0,
        "B06": 2.0,
        "B07": 2.0,
        "B08": 2.0,
        "B09": 2.0,
        "B10": 2.0,
        "B11": 2.0,
        "B12": 2.0,
        "B13": 2.0,
        "B14": 2.0,
        "B15": 2.0,
        "B16": 2.0,
    }

    found_res = {b: res[b] for b in bands}
    unique_res = set(found_res.values())
    if len(unique_res) > 1:
        b0, r0 = found_res.popitem()
        b1, r1 = next((b, r) for b, r in found_res.items() if r != r0)
        raise ValueError(
            "Bands must have a common horizontal resolution. "
            f"Band {b0} has resolution {r0} km and band {b1} has resolution {r1} km."
        )


def _parse_region(region: HimawariRegion | str) -> HimawariRegion:
    """Parse region from string."""
    if isinstance(region, HimawariRegion):
        return region

    region = region.upper().replace(" ", "").replace("_", "")

    if region in ("F", "FLDK", "FULL", "FULLDISK"):
        return HimawariRegion.FLDK
    if region in ("J", "JAPAN"):
        return HimawariRegion.Japan
    if region in ("T", "TARGET", "MESOSCALE"):
        return HimawariRegion.Target
    raise ValueError(f"Region must be one of {HimawariRegion._member_names_}")


def _extract_band_from_rpath(rpath: str) -> str:
    sep = "_B"
    suffix = rpath.split(sep, maxsplit=1)[1]
    return f"B{suffix[:2]}"  # B??


def _mask_invalid(
    data: npt.NDArray[np.uint16],
    calib_info: dict[str, Any],
) -> npt.NDArray[np.float32]:
    """Mask invalid data."""
    error_pixel = calib_info["count_error_pixels"]
    outside_pixel = calib_info["count_outside_scan_area"]

    mask = (data == error_pixel) | (data == outside_pixel)
    return np.where(mask, np.float32(np.nan), data.astype(np.float32))


def _radiance_to_brightness_temperature(
    radiance: npt.NDArray[np.float32],
    calib_info: dict[str, Any],
) -> npt.NDArray[np.float32]:
    """Convert radiance to brightness temperature."""
    radiance = np.where(radiance <= 0.0, np.float32(np.nan), radiance)  # remove invalid
    radiance_m = radiance * 1e6  # W/m^2/sr/um -> W/m^2/sr/m

    lmbda = calib_info["central_wavelength"] * 1e-6  # um -> m
    h = calib_info["planck_constant"]
    c = calib_info["speed_of_light"]
    k = calib_info["boltzmann_constant"]

    term = (2 * h * c**2) / (lmbda**5 * radiance_m)
    effective_bt = (h * c) / (k * lmbda * np.log1p(term))

    c0 = calib_info["c0"]
    c1 = calib_info["c1"]
    c2 = calib_info["c2"]
    return c0 + c1 * effective_bt + c2 * effective_bt**2


def _radiance_to_reflectance(
    radiance: npt.NDArray[np.float32],
    calib_info: dict[str, Any],
) -> npt.NDArray[np.float32]:
    """Convert radiance to reflectance."""
    coeff = calib_info["coeff_c_prime"]
    return radiance * coeff


def _load_raw_counts(content: bytes, metadata: dict[str, Any]) -> npt.NDArray[np.uint16]:
    """Load raw counts from Himawari data."""
    offset = metadata["basic_information"]["total_header_length"]
    n_columns = metadata["data_information"]["num_columns"]
    n_lines = metadata["data_information"]["num_lines"]
    return np.frombuffer(content, dtype=np.uint16, offset=offset).reshape((n_lines, n_columns))


def _counts_to_radiance(
    counts: npt.NDArray[np.float32],
    calib_info: dict[str, Any],
) -> npt.NDArray[np.float32]:
    """Convert raw counts to radiance."""
    gain = calib_info["gain"]
    const = calib_info["constant"]
    return counts * gain + const


def _load_image_data(
    content: bytes, metadata: dict[str, dict[str, Any]]
) -> npt.NDArray[np.float32]:
    counts = _load_raw_counts(content, metadata)

    calib_info = metadata["calibration_information"]
    masked_counts = _mask_invalid(counts, calib_info)
    radiance = _counts_to_radiance(masked_counts, calib_info)

    if calib_info["band_number"] <= 6:  # visible/NIR
        return _radiance_to_reflectance(radiance, calib_info)
    return _radiance_to_brightness_temperature(radiance, calib_info)


def _ahi_fixed_grid(
    proj_info: dict[str, Any],
    arr: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray]:
    n_lines, n_columns = arr.shape

    i = np.arange(n_columns, dtype=np.float32)
    j = np.arange(n_lines, dtype=np.float32)

    # See section 4.4.4 (scaling functions) of the CGMS LRIT/HRIT specification
    # https://www.cgms-info.org/wp-content/uploads/2021/10/cgms-lrit-hrit-global-specification-(v2-8-of-30-oct-2013).pdf
    x_deg = (i - proj_info["coff"]) / proj_info["cfac"] * 2**16
    y_deg = -(j - proj_info["loff"]) / proj_info["lfac"] * 2**16  # positive y is north

    x_rad = np.deg2rad(x_deg)
    y_rad = np.deg2rad(y_deg)

    x = xr.DataArray(
        x_rad,
        dims=("x",),
        attrs={
            "units": "rad",
            "axis": "X",
            "long_name": "AHI fixed grid projection x-coordinate",
            "standard_name": "projection_x_coordinate",
        },
    )
    y = xr.DataArray(
        y_rad,
        dims=("y",),
        attrs={
            "units": "rad",
            "axis": "Y",
            "long_name": "AHI fixed grid projection y-coordinate",
            "standard_name": "projection_y_coordinate",
        },
    )

    return x, y


def _himawari_proj4_string(proj_info: dict[str, Any]) -> str:
    H = proj_info["dist_from_earth_center"] * 1000.0  # km -> m
    a = proj_info["equatorial_radius"] * 1000.0  # km -> m
    b = proj_info["polar_radius"] * 1000.0  # km -> m
    lon = proj_info["sub_lon"]
    h = H - a  # height above surface
    return f"+proj=geos +h={h} +a={a} +b={b} +lon_0={lon} +sweep=x +units=m +no_defs"


def _earth_disk_mask(
    proj_info: dict[str, Any],
    x: xr.DataArray,
    y: xr.DataArray,
) -> npt.NDArray[np.bool_]:
    """Return a boolean mask where True indicates pixels over the Earth disk."""
    a = proj_info["equatorial_radius"] * 1000.0  # km -> m
    b = proj_info["polar_radius"] * 1000.0  # km -> m
    h = proj_info["dist_from_earth_center"] * 1000.0  # km -> m

    # Precompute trig terms
    cosx = np.cos(x.values[np.newaxis, :])  # shape (1, nx)
    cosy = np.cos(y.values[:, np.newaxis])  # shape (ny, 1)
    siny = np.sin(y.values[:, np.newaxis])  # shape (ny, 1)

    # Form a ray from the satellite to each pixel (in the scan angle space). Compute the
    # intersection of the ray with the ellipsoid gives a quadratic equation.
    A = cosy**2 / a**2 + siny**2 / b**2
    B = -2 * h * cosy * cosx / a**2
    C = h**2 / a**2 - 1.0

    discriminant = B**2 - 4 * A * C

    # A positive discriminant indicates the ray from satellite intersects ellipsoid
    # within Earth disk. Return True for valid Earth pixels.
    return discriminant >= 0.0


def _parse_start_time(metadata: dict[str, dict[str, Any]]) -> datetime.datetime:
    """Parse the start time from the metadata."""
    mjd_value = metadata["basic_information"]["obs_start_time"]
    mjd_epoch = datetime.datetime(1858, 11, 17)
    return mjd_epoch + datetime.timedelta(days=mjd_value)


def _parse_s3_raw_data(raw_data: list[bytes]) -> xr.DataArray:
    """Decode a list of Himawari bz2-compressed bytes to an xarray DataArray."""
    arrays = []
    proj_info = None
    start_time = None

    for data in raw_data:
        content = bz2.decompress(data)
        metadata = header_struct.parse_himawari_header(content)
        proj_info = proj_info or metadata["projection_information"]
        start_time = start_time or _parse_start_time(metadata)

        arr = _load_image_data(content, metadata)

        segment_number = metadata["segment_information"]["segment_seq_number"]
        arrays.append((segment_number, arr))

    # (This sorting isn't really necessary since s3fs.glob returns sorted results)
    sorted_arrays = [arr for _, arr in sorted(arrays, key=lambda x: x[0])]
    combined = np.vstack(sorted_arrays)

    assert proj_info is not None
    x, y = _ahi_fixed_grid(proj_info, combined)

    mask = _earth_disk_mask(proj_info, x, y)  # mask values outside Earth disk
    combined[~mask] = np.float32(np.nan)

    crs = _himawari_proj4_string(proj_info)
    band = metadata["calibration_information"]["band_number"]
    if band > 6:
        long_name = "Advanced Himawari Imager (AHI) brightness temperature"
        standard_name = "toa_brightness_temperature"
        units = "K"
    else:
        long_name = "Advanced Himawari Imager (AHI) reflectance"
        standard_name = "toa_reflectance"
        units = ""

    return xr.DataArray(
        combined,
        dims=("y", "x"),
        coords={"x": x, "y": y, "t": start_time},
        attrs={"crs": crs, "long_name": long_name, "standard_name": standard_name, "units": units},
    ).expand_dims(band_id=np.array([band], dtype=np.int32))  # use band_id to match GOES


class Himawari:
    """Support for Himawari-8/9 satellite data access via AWS S3.

    This interface requires the ``s3fs`` package to download data from the
    `AWS Public Dataset <https://registry.opendata.aws/himawari/>`_.

    Parameters
    ----------
    region : HimawariRegion | str, optional
        The Himawari-8/9 area to download. By default, :attr:`HimawariRegion.FLDK` (Full Disk).
    bands : str | Iterable[str] | None, optional
        The bands to download. The 16 possible bands are ``B01`` to ``B16``. For the SEVIRI
        ash color scheme, bands ``B11``, ``B14``, and ``B15`` are required (default). For
        the true color scheme, bands ``B01``, ``B02``, and ``B03`` are required.
        See `here <https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html#band>`_
        for more information on the bands.
    bucket : str | None, optional
        The S3 bucket to use. By default, the bucket is chosen based on the time
        (Himawari-8 before 2022-12-13, Himawari-9 after).
    cachestore : cache.CacheStore | None, optional
        The cache store to use. By default, a disk cache in the user cache directory
        is used. If None, data is downloaded directly into memory from S3.

    See Also
    --------
    pycontrails.datalib.goes.GOES
    HimawariRegion
    """

    __marker = object()

    def __init__(
        self,
        region: HimawariRegion | str = HimawariRegion.FLDK,
        bands: str | Iterable[str] | None = None,
        *,
        bucket: str | None = None,
        cachestore: cache.CacheStore | None = __marker,  # type: ignore[assignment]
    ) -> None:
        self.region = _parse_region(region)
        self.bands = _parse_bands(bands)
        _check_band_resolution(self.bands)

        self.bucket = bucket
        self.fs = s3fs.S3FileSystem(anon=True)

        if cachestore is self.__marker:
            cache_root = cache._get_user_cache_dir()
            cache_dir = f"{cache_root}/himawari"
            cachestore = cache.DiskCacheStore(cache_dir=cache_dir)
        self.cachestore = cachestore

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Himawari(region='{self.region.name}', bands={sorted(self.bands)}, "
            f"bucket={self.bucket})"
        )

    def s3_rpaths(self, time: datetime.datetime) -> dict[str, list[str]]:
        """Return S3 remote paths for a given time."""
        t, scan_type = _check_time_resolution(time, self.region)

        if self.bucket is None:
            bucket = HIMAWARI_8_BUCKET if t < HIMAWARI_8_9_SWITCH_DATE else HIMAWARI_9_BUCKET
        else:
            bucket = self.bucket

        sat_number = bucket.removeprefix("noaa-himawari")  # Will not work for custom buckets

        # Get all bands for the time
        prefix = f"{bucket}/AHI-L1b-{self.region.name}/{t:%Y/%m/%d/%H%M}/HS_H0{sat_number}_{t:%Y%m%d_%H%M}_B??_{scan_type}"  # noqa: E501
        rpaths = self.fs.glob(f"{prefix}*")

        out = collections.defaultdict(list)
        for rpath in rpaths:
            band = _extract_band_from_rpath(rpath)
            if band in self.bands:
                out[band].append(rpath)

        return out

    def _lpaths(self, time: datetime.datetime) -> dict[str, str]:
        """Construct names for local netcdf files using the :attr:`cachestore`.

        Returns dictionary of the form ``{band: local_path}``.

        Implementation is copied directly from :meth:`GOES._lpaths`.
        """
        if not self.cachestore:
            raise ValueError("cachestore must be set to use _lpaths")

        t_str = time.strftime("%Y%m%d%H%M%S")

        out = {}
        for band in self.bands:
            if self.bucket:
                name = f"{self.bucket}_{self.region.name}_{t_str}_{band}.nc"
            else:
                name = f"{self.region.name}_{t_str}_{band}.nc"

            lpath = self.cachestore.path(name)
            out[band] = lpath

        return out

    def get(self, time: datetime.datetime | str) -> xr.DataArray:
        """Get Himawari-8/9 data for a given time."""
        t = pd.Timestamp(time).to_pydatetime()

        if self.cachestore is not None:
            return self._get_with_cache(t)
        return self._get_without_cache(t)

    def _get_with_cache(self, time: datetime.datetime) -> xr.DataArray:
        """Get Himawari-8/9 data for a given time, using the cache if available."""
        if self.cachestore is None:
            raise ValueError("cachestore must be set to use get_with_cache")

        lpaths = self._lpaths(time)

        missing_bands = [b for b, p in lpaths.items() if not self.cachestore.exists(p)]
        if missing_bands:
            rpaths_all_bands = self.s3_rpaths(time)
            for band in missing_bands:
                rpaths = rpaths_all_bands[band]
                if not rpaths:
                    raise ValueError(f"No data found for band {band} at time {time}")
                raw_data = list(self.fs.cat(rpaths).values())
                da = _parse_s3_raw_data(raw_data)
                da.to_dataset(name="CMI").to_netcdf(lpaths[band])  # only using CMI to match GOES

        kwargs = {
            "concat_dim": "band_id",
            "combine": "nested",
            "combine_attrs": "override",
            "coords": "minimal",
            "compat": "override",
        }
        if len(lpaths) == 1 or "B03" not in lpaths:
            return xr.open_mfdataset(lpaths.values(), **kwargs)["CMI"].sortby("band_id")  # type: ignore[arg-type]

        lpath03 = lpaths.pop("B03")
        da1 = xr.open_mfdataset(lpaths.values(), **kwargs)["CMI"]  # type: ignore[arg-type]
        da03 = xr.open_dataset(lpath03)["CMI"]
        return geo_utils._coarsen_then_concat(da1, da03).sortby("band_id")

    def _get_without_cache(self, time: datetime.datetime) -> xr.DataArray:
        """Get Himawari-8/9 data for a given time, without using the cache."""
        all_rpaths = self.s3_rpaths(time)

        da_dict = {}
        for band, rpaths in all_rpaths.items():
            if len(rpaths) == 0:
                raise ValueError(f"No data found for band {band} at time {time}")

            raw_data = list(self.fs.cat(rpaths).values())
            da = _parse_s3_raw_data(raw_data).rename("CMI")  # only using CMI to match GOES
            da_dict[band] = da

        kwargs = {
            "dim": "band_id",
            "coords": "minimal",
            "compat": "override",
        }
        if len(da_dict) == 1 or "B03" not in da_dict:
            return xr.concat(da_dict.values(), **kwargs).sortby("band_id")  # type: ignore[call-overload]
        da03 = da_dict.pop("B03")
        da1 = xr.concat(da_dict.values(), **kwargs)  # type: ignore[call-overload]
        return geo_utils._coarsen_then_concat(da1, da03).sortby("band_id")


def _cartopy_crs(proj4_string: str) -> cartopy.crs.Geostationary:
    try:
        import pyproj
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            name="Himawari visualization",
            package_name="pyproj",
            module_not_found_error=exc,
            pycontrails_optional_package="sat",
        )
    try:
        from cartopy import crs as ccrs
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            name="Himawari visualization",
            package_name="cartopy",
            module_not_found_error=exc,
            pycontrails_optional_package="sat",
        )

    crs_obj = pyproj.CRS(proj4_string)

    with warnings.catch_warnings():
        # pyproj warns that to_dict is lossy, but we built it ourselves so it's fine
        warnings.filterwarnings("ignore", category=UserWarning)
        crs_dict = crs_obj.to_dict()

    globe = ccrs.Globe(
        semimajor_axis=crs_dict["a"],
        semiminor_axis=crs_dict["b"],
    )
    return ccrs.Geostationary(
        central_longitude=crs_dict["lon_0"],
        satellite_height=crs_dict["h"],
        sweep_axis=crs_dict["sweep"],
        globe=globe,
    )


def extract_visualization(
    da: xr.DataArray,
    color_scheme: str = "ash",
    ash_convention: str = "SEVIRI",
    gamma: float = 2.2,
) -> tuple[npt.NDArray[np.float32], cartopy.crs.Geostationary, tuple[float, float, float, float]]:
    """Extract artifacts for visualizing Himawari data with the given color scheme.

    Parameters
    ----------
    da : xr.DataArray
        DataArray of Himawari data as returned by :meth:`Himawari.get`. Must have the channels
        required by :func:`to_ash`.
    color_scheme : str
        Color scheme to use for visualization. Must be one of {"true", "ash"}.
        If "true", the ``da`` must contain channels B01, B02, and B03.
        If "ash", the ``da`` must contain channels B11, B14, and B15 (SEVIRI convention)
        or channels B11, B13, B14, and B15 (standard convention).
    ash_convention : str
        Passed into :func:`to_ash`. Only used if ``color_scheme="ash"``. Must be one
        of {"SEVIRI", "standard"}. By default, "SEVIRI" is used.
    gamma : float
        Passed into :func:`to_true_color`. Only used if ``color_scheme="true"``. By
        default, 2.2 is used.

    Returns
    -------
    rgb : npt.NDArray[np.float32]
        3D RGB array of shape ``(height, width, 3)``. Any nan values are replaced with 0.
    src_crs : cartopy.crs.Geostationary
        The Geostationary projection built from the Himawari metadata.
    src_extent : tuple[float, float, float, float]
        Extent of Himawari data in the Geostationary projection
    """
    proj4_string = da.attrs["crs"]
    src_crs = _cartopy_crs(proj4_string)

    if color_scheme == "true":
        rgb = to_true_color(da, gamma)
    elif color_scheme == "ash":
        rgb = geo_utils.to_ash(da, ash_convention)
    else:
        raise ValueError(f"Color scheme must be 'true' or 'ash', not '{color_scheme}'")

    np.nan_to_num(rgb, copy=False)

    x = da["x"].values
    y = da["y"].values

    # Multiply extremes by the satellite height
    h = src_crs.proj4_params["h"]
    src_extent = h * x.min(), h * x.max(), h * y.min(), h * y.max()

    return rgb, src_crs, src_extent


def to_true_color(da: xr.DataArray, gamma: float = 2.2) -> npt.NDArray[np.floating]:
    """Compute 3d RGB array for the true color scheme.

    Parameters
    ----------
    da : xr.DataArray
        DataArray of GOES data with channels B01, B02, B03.
    gamma : float, optional
        Gamma correction for the RGB channels.

    Returns
    -------
    npt.NDArray[np.floating]
        3d RGB array with true color scheme.

    References
    ----------
    - https://www.jma.go.jp/jma/jma-eng/satellite/VLab/QG/RGB_QG_TrueColor_en.pdf
    """
    if not np.all(np.isin([1, 2, 3], da["band_id"])):
        msg = "DataArray must contain bands 1, 2, and 3 for true color"
        raise ValueError(msg)

    red = da.sel(band_id=3).values
    green = da.sel(band_id=2).values
    blue = da.sel(band_id=1).values

    red = geo_utils._clip_and_scale(red, 0.0, 1.0)
    green = geo_utils._clip_and_scale(green, 0.0, 1.0)
    blue = geo_utils._clip_and_scale(blue, 0.0, 1.0)

    return np.dstack((red, green, blue)) ** (1 / gamma)
