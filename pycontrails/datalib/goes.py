"""Support for GOES access and analysis.

Resources
---------

- `GOES 16/18 on GCP notes <https://console.cloud.google.com/marketplace/product/noaa-public/goes>`_
- `GOES on AWS notes <https://docs.opendata.aws/noaa-goes16/cics-readme.html>`_
- `Scan Mode information and timing <https://www.ospo.noaa.gov/Operations/GOES/16/GOES-16%20Scan%20Mode%206.html>`_
- `Current position of the MESO1 sector <https://www.ospo.noaa.gov/Operations/GOES/east/meso1-img.html>`_
- `Current position of the MESO2 sector <https://www.ospo.noaa.gov/Operations/GOES/east/meso2-img.html>`_
- `Historical Mesoscale regions <https://qcweb.ssec.wisc.edu/web/meso_search/>`_
- `Real time GOES data quality <https://qcweb.ssec.wisc.edu/web/abi_quality_scores/>`_
"""

from __future__ import annotations

import datetime
import enum
import os
import tempfile
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pycontrails.core import cache
from pycontrails.datalib import geo_utils
from pycontrails.datalib.geo_utils import (
    parallax_correct,  # noqa: F401, keep for backwards compatibility
    to_ash,  # keep for backwards compatibility
)
from pycontrails.utils import dependencies

if TYPE_CHECKING:
    import cartopy.crs

try:
    import gcsfs
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="goes module",
        package_name="gcsfs",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )


#: Default bands to use if none are specified. These are the bands
#: required by the SEVIRI (MIT) ash color scheme.
DEFAULT_BANDS = "C11", "C14", "C15"

#: The time at which the GOES scan mode changed from mode 3 to mode 6. This
#: is used to determine the scan time resolution.
#: See `GOES ABI scan information <https://www.goes-r.gov/users/abiScanModeInfo.html>`_.
GOES_SCAN_MODE_CHANGE = datetime.datetime(2019, 4, 2, 16)

#: The date at which GOES-19 data started being available. This is used to
#: determine the source (GOES-16 or GOES-19) of requested. In particular,
#: Mesoscale images are only available for GOES-East from GOES-19 after this date.
#: See the `NOAA press release <https://www.noaa.gov/news-release/noaas-goes-19-satellite-now-operational-providing-critical-new-data-to-forecasters>`_.
GOES_16_19_SWITCH_DATE = datetime.datetime(2025, 4, 4)

#: The GCS bucket for GOES-East data before ``GOES_16_19_SWITCH_DATE``.
GOES_16_BUCKET = "gcp-public-data-goes-16"

#: The GCS bucket for GOES-West data. Note that GOES-17 has degraded data quality
#: and is not recommended for use. This bucket isn't used by the ``GOES`` handler by default.
GOES_18_BUCKET = "gcp-public-data-goes-18"

#: The GCS bucket for GOES-East data after ``GOES_16_19_SWITCH_DATE``.
GOES_19_BUCKET = "gcp-public-data-goes-19"


class GOESRegion(enum.Enum):
    """GOES Region of interest.

    Uses the following conventions.

    - F: Full Disk
    - C: CONUS
    - M1: Mesoscale 1
    - M2: Mesoscale 2
    """

    F = enum.auto()
    C = enum.auto()
    M1 = enum.auto()
    M2 = enum.auto()


def _check_time_resolution(t: datetime.datetime, region: GOESRegion) -> datetime.datetime:
    """Confirm request t is at GOES scan time resolution."""
    if t.second != 0 or t.microsecond != 0:
        raise ValueError(
            "Time must be at GOES scan time resolution. Seconds or microseconds not supported"
        )

    if region == GOESRegion.F:
        # Full Disk: Scan times are available every 10 minutes after
        # 2019-04-02 and every 15 minutes before
        if t >= GOES_SCAN_MODE_CHANGE:
            if t.minute % 10:
                raise ValueError(
                    f"Time must be at GOES scan time resolution for {region}. "
                    f"After {GOES_SCAN_MODE_CHANGE}, time should be a multiple of 10 minutes."
                )
        elif t.minute % 15:
            raise ValueError(
                f"Time must be at GOES scan time resolution for {region}. "
                f"Before {GOES_SCAN_MODE_CHANGE}, time should be a multiple of 15 minutes."
            )
        return t

    if region == GOESRegion.C:
        # CONUS: Scan times are every 5 minutes
        if t.minute % 5:
            raise ValueError(
                f"Time must be at GOES scan time resolution for {region}. "
                "Time should be a multiple of 5 minutes."
            )
        return t

    return t


def _parse_bands(bands: str | Iterable[str] | None) -> set[str]:
    """Check that the bands are valid and return as a set."""
    if bands is None:
        return set(DEFAULT_BANDS)

    if isinstance(bands, str):
        bands = (bands,)

    available = {f"C{i:02d}" for i in range(1, 17)}
    bands = {c.upper() for c in bands}
    if not bands.issubset(available):
        raise ValueError(f"Bands must be in {sorted(available)}")
    return bands


def _check_band_resolution(bands: Iterable[str]) -> None:
    """Confirm request bands have a common horizontal resolution."""
    # https://www.goes-r.gov/spacesegment/abi.html
    res = {
        "C01": 1.0,
        "C02": 1.0,  # XXX: this actually has a resolution of 0.5 km, but we coarsen it to 1 km
        "C03": 1.0,
        "C04": 2.0,
        "C05": 1.0,
        "C06": 2.0,
        "C07": 2.0,
        "C08": 2.0,
        "C09": 2.0,
        "C10": 2.0,
        "C11": 2.0,
        "C12": 2.0,
        "C13": 2.0,
        "C14": 2.0,
        "C15": 2.0,
        "C16": 2.0,
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


def _parse_region(region: GOESRegion | str) -> GOESRegion:
    """Parse region from string."""
    if isinstance(region, GOESRegion):
        return region

    region = region.upper().replace(" ", "").replace("_", "")

    if region in ("F", "FULL", "FULLDISK"):
        return GOESRegion.F
    if region in ("C", "CONUS", "CONTINENTAL"):
        return GOESRegion.C
    if region in ("M1", "MESO1", "MESOSCALE1"):
        return GOESRegion.M1
    if region in ("M2", "MESO2", "MESOSCALE2"):
        return GOESRegion.M2
    raise ValueError(f"Region must be one of {GOESRegion._member_names_} or their abbreviations")


def gcs_goes_path(
    time: datetime.datetime,
    region: GOESRegion,
    bands: str | Iterable[str] | None = None,
    bucket: str | None = None,
    fs: gcsfs.GCSFileSystem | None = None,
) -> list[str]:
    """Return GCS paths to GOES data at the given time for the given region and bands.

    Presently only supported for GOES data whose scan time minute coincides with
    the minute of the time parameter.

    Parameters
    ----------
    time : datetime.datetime
        Time of GOES data. This should be a timezone-naive datetime object or an
        ISO 8601 formatted string.
    region : GOESRegion
        GOES Region of interest.
    bands : str | Iterable[str] | None, optional
        Set of bands or bands for CMIP data. The 16 possible bands are
        represented by the strings "C01" to "C16". For the SEVIRI ash color scheme,
        set ``bands=("C11", "C14", "C15")``. For the true color scheme,
        set ``bands=("C01", "C02", "C03")``. By default, the bands
        required by the SEVIRI ash color scheme are used.
    bucket : str | None
        GCS bucket for GOES data. If None, the bucket is automatically
        set to ``GOES_16_BUCKET`` if ``time`` is before
        ``GOES_16_19_SWITCH_DATE`` and ``GOES_19_BUCKET`` otherwise.
    fs : gcsfs.GCSFileSystem | None
        GCS file system instance. If None, a default anonymous instance is created.

    Returns
    -------
    list[str]
        List of GCS paths to GOES data.

    Examples
    --------
    >>> from pprint import pprint
    >>> t = datetime.datetime(2023, 4, 3, 2, 10)

    >>> paths = gcs_goes_path(t, GOESRegion.F, bands=("C11", "C12", "C13"))
    >>> pprint(paths)
    ['gcp-public-data-goes-16/ABI-L2-CMIPF/2023/093/02/OR_ABI-L2-CMIPF-M6C11_G16_s20230930210203_e20230930219511_c20230930219586.nc',
     'gcp-public-data-goes-16/ABI-L2-CMIPF/2023/093/02/OR_ABI-L2-CMIPF-M6C12_G16_s20230930210203_e20230930219516_c20230930219596.nc',
     'gcp-public-data-goes-16/ABI-L2-CMIPF/2023/093/02/OR_ABI-L2-CMIPF-M6C13_G16_s20230930210203_e20230930219523_c20230930219586.nc']

    >>> paths = gcs_goes_path(t, GOESRegion.C, bands=("C11", "C12", "C13"))
    >>> pprint(paths)
    ['gcp-public-data-goes-16/ABI-L2-CMIPC/2023/093/02/OR_ABI-L2-CMIPC-M6C11_G16_s20230930211170_e20230930213543_c20230930214055.nc',
     'gcp-public-data-goes-16/ABI-L2-CMIPC/2023/093/02/OR_ABI-L2-CMIPC-M6C12_G16_s20230930211170_e20230930213551_c20230930214045.nc',
     'gcp-public-data-goes-16/ABI-L2-CMIPC/2023/093/02/OR_ABI-L2-CMIPC-M6C13_G16_s20230930211170_e20230930213557_c20230930214065.nc']

    >>> t = datetime.datetime(2023, 4, 3, 2, 11)
    >>> paths = gcs_goes_path(t, GOESRegion.M1, bands="C01")
    >>> pprint(paths)
    ['gcp-public-data-goes-16/ABI-L2-CMIPM/2023/093/02/OR_ABI-L2-CMIPM1-M6C01_G16_s20230930211249_e20230930211309_c20230930211386.nc']

    >>> t = datetime.datetime(2025, 5, 4, 3, 2)
    >>> paths = gcs_goes_path(t, GOESRegion.M2, bands="C01")
    >>> pprint(paths)
    ['gcp-public-data-goes-19/ABI-L2-CMIPM/2025/124/03/OR_ABI-L2-CMIPM2-M6C01_G19_s20251240302557_e20251240303014_c20251240303092.nc']

    """
    time = _check_time_resolution(time, region)
    year = time.strftime("%Y")
    yday = time.strftime("%j")
    hour = time.strftime("%H")

    sensor = "ABI"  # Advanced Baseline Imager
    level = "L2"  # Level 2
    product_name = "CMIP"  # Cloud and Moisture Imagery
    product = f"{sensor}-{level}-{product_name}{region.name[0]}"

    if bucket is None:
        bucket = GOES_16_BUCKET if time < GOES_16_19_SWITCH_DATE else GOES_19_BUCKET
    else:
        bucket = bucket.removeprefix("gs://")

    path_prefix = f"gs://{bucket}/{product}/{year}/{yday}/{hour}/"

    # https://www.goes-r.gov/users/abiScanModeInfo.html
    mode = "M6" if time >= GOES_SCAN_MODE_CHANGE else "M3"

    # Example name pattern
    # OR_ABI-L1b-RadF-M3C02_G16_s20171671145342_e20171671156109_c20171671156144.nc
    time_str = time.strftime("%Y%j%H%M")
    if region == GOESRegion.F:
        time_str = time_str[:-1]  # might not work before 2019-04-02?
    elif region == GOESRegion.C:
        # Very crude -- assuming scan time ends with 1 or 6
        if time_str.endswith("0"):
            time_str = f"{time_str[:-1]}1"
        elif time_str.endswith("5"):
            time_str = f"{time_str[:-1]}6"

    name_prefix = f"OR_{product[:-1]}{region.name}-{mode}"

    try:
        satellite_number = int(bucket[-2:])  # 16 or 18 or 19 -- this may fail for custom buckets
    except (ValueError, IndexError) as exc:
        msg = f"Bucket name {bucket} does not end with a valid satellite number."
        raise ValueError(msg) from exc
    name_suffix = f"_G{satellite_number}_s{time_str}*"

    bands = _parse_bands(bands)

    # It's faster to run a single glob with C?? then running a glob for
    # each band. The downside is that we have to filter the results.
    rpath = f"{path_prefix}{name_prefix}C??{name_suffix}"

    fs = fs or gcsfs.GCSFileSystem(token="anon")
    rpaths = fs.glob(rpath)

    out = [r for r in rpaths if _extract_band_from_rpath(r) in bands]
    if not out:
        raise RuntimeError(f"No data found for {time} in {region} for bands {bands}")
    return out


def _extract_band_from_rpath(rpath: str) -> str:
    # Split at the separator between product name and mode
    # This works for both M3 and M6
    sep = "-M"
    suffix = rpath.split(sep, maxsplit=1)[1]
    return suffix[1:4]


class GOES:
    """Support for GOES-16 data access via GCP.

    This interface requires the ``gcsfs`` package.

    Parameters
    ----------
    region : GOESRegion | str, optional
        GOES Region of interest. Uses the following conventions.

        - F: Full Disk
        - C: CONUS
        - M1: Mesoscale 1
        - M2: Mesoscale 2

        By default, Full Disk (F) is used.

    bands : str | Iterable[str] | None
        Set of bands or bands for CMIP data. The 16 possible bands are
        represented by the strings "C01" to "C16". For the SEVIRI ash color scheme,
        set ``bands=("C11", "C14", "C15")``. For the true color scheme,
        set ``bands=("C01", "C02", "C03")``. By default, the bands
        required by the SEVIRI ash color scheme are used. The bands must have
        a common horizontal resolution. The resolutions are:

        - C01: 1.0 km
        - C02: 0.5 km  (treated as 1.0 km)
        - C03: 1.0 km
        - C04: 2.0 km
        - C05: 1.0 km
        - C06 - C16: 2.0 km

    cachestore : cache.CacheStore | None, optional
        Cache store for GOES data. If None, data is downloaded directly into
        memory. By default, a :class:`cache.DiskCacheStore` is used.
    bucket : str | None, optional
        GCP bucket for GOES data. If None, the default option, the bucket is automatically
        set to ``GOES_16_BUCKET`` if the requested time is before
        ``GOES_16_19_SWITCH_DATE`` and ``GOES_19_BUCKET`` otherwise.
        The satellite number used for filename construction is derived from the
        last two characters of this bucket name.

    See Also
    --------
    GOESRegion
    gcs_goes_path

    Examples
    --------
    >>> goes = GOES(region="M1", bands=("C11", "C14"))
    >>> da = goes.get("2021-04-03 02:10:00")
    >>> da.shape
    (2, 500, 500)

    >>> da.dims
    ('band_id', 'y', 'x')

    >>> da.band_id.values
    array([11, 14], dtype=int32)

    >>> # Print out a sample of the data
    >>> da.sel(band_id=11).isel(x=slice(0, 50, 10), y=slice(0, 50, 10)).values
    array([[266.8644 , 265.50812, 271.5592 , 271.45486, 272.75897],
           [250.53697, 273.28064, 273.80225, 270.77673, 274.8977 ],
           [272.8633 , 272.65466, 271.5592 , 274.01093, 273.12415],
           [274.16742, 274.11523, 276.5148 , 273.85443, 270.51593],
           [274.84555, 275.15854, 272.60248, 270.67242, 272.23734]],
          dtype=float32)

    >>> # The data has been cached locally
    >>> assert goes.cachestore.listdir()

    >>> # Download GOES data directly into memory by setting cachestore=None
    >>> goes = GOES(region="M2", bands=("C11", "C12", "C13"), cachestore=None)
    >>> da = goes.get("2021-04-03 02:10:00")

    >>> da.shape
    (3, 500, 500)

    >>> da.dims
    ('band_id', 'y', 'x')

    >>> da.band_id.values
    array([11, 12, 13], dtype=int32)

    >>> da.attrs["long_name"]
    'ABI L2+ Cloud and Moisture Imagery brightness temperature'

    >>> da.sel(band_id=11).values
    array([[251.31944, 249.59802, 249.65018, ..., 270.30725, 270.51593,
            269.83777],
           [250.53697, 249.0242 , 249.12854, ..., 270.15076, 270.30725,
            269.73346],
           [249.1807 , 249.33719, 251.99757, ..., 270.15076, 270.20294,
            268.7945 ],
           ...,
           [277.24512, 277.29727, 277.45377, ..., 274.42822, 274.11523,
            273.7501 ],
           [277.24512, 277.45377, 278.18408, ..., 274.6369 , 274.01093,
            274.06308],
           [276.8278 , 277.14078, 277.7146 , ..., 274.6369 , 273.9066 ,
            274.16742]], shape=(500, 500), dtype=float32)

    """

    __marker = object()

    def __init__(
        self,
        region: GOESRegion | str = GOESRegion.F,
        bands: str | Iterable[str] | None = None,
        *,
        channels: str | Iterable[str] | None = None,  # deprecated alias for bands
        cachestore: cache.CacheStore | None = __marker,  # type: ignore[assignment]
        bucket: str | None = None,
        goes_bucket: str | None = None,  # deprecated alias for bucket
    ) -> None:
        if channels is not None:
            if bands is not None:
                raise ValueError("Only one of channels or bands should be specified")
            warnings.warn(
                "The 'channels' parameter is deprecated and will be removed in a future release. "
                "Use 'bands' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            bands = channels
        if goes_bucket is not None:
            if bucket is not None:
                raise ValueError("Only one of goes_bucket or bucket should be specified")
            warnings.warn(
                "The 'goes_bucket' parameter is deprecated and will be removed in a future release."
                "Use 'bucket' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            bucket = goes_bucket

        self.region = _parse_region(region)
        self.bands = _parse_bands(bands)
        _check_band_resolution(self.bands)

        self.bucket = bucket
        self.fs = gcsfs.GCSFileSystem(token="anon")

        if cachestore is self.__marker:
            cache_root = cache._get_user_cache_dir()
            cache_dir = f"{cache_root}/goes"
            cachestore = cache.DiskCacheStore(cache_dir=cache_dir)
        self.cachestore = cachestore

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"GOES(region='{self.region.name}', bands={sorted(self.bands)}, bucket={self.bucket})"
        )

    def gcs_goes_path(self, time: datetime.datetime, bands: set[str] | None = None) -> list[str]:
        """Return GCS paths to GOES data at given time.

        Presently only supported for GOES data whose scan time minute coincides with
        the minute of the time parameter.

        Parameters
        ----------
        time : datetime.datetime
            Time of GOES data.
        bands : set[str] | None
            Set of bands or bands for CMIP data. If None, the :attr:`bands`
            attribute is used.

        Returns
        -------
        list[str]
            List of GCS paths to GOES data.
        """
        bands = bands or self.bands
        return gcs_goes_path(time, self.region, bands, bucket=self.bucket, fs=self.fs)

    def _lpaths(self, time: datetime.datetime) -> dict[str, str]:
        """Construct names for local netcdf files using the :attr:`cachestore`.

        Returns dictionary of the form ``{band: local_path}``.
        """
        if not self.cachestore:
            raise ValueError("cachestore must be set to use _lpaths")

        t_str = time.strftime("%Y%m%d%H%M")

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
        """Return GOES data at given time.

        Parameters
        ----------
        time : datetime.datetime | str
            Time of GOES data. This should be a timezone-naive datetime object
            or an ISO 8601 formatted string.

        Returns
        -------
        xr.DataArray
            DataArray of GOES data with coordinates:

            - band_id: Channel or band ID
            - x: GOES x-coordinate
            - y: GOES y-coordinate
        """
        t = pd.Timestamp(time).to_pydatetime()

        if self.cachestore is not None:
            return self._get_with_cache(t)
        return self._get_without_cache(t)

    def _get_with_cache(self, time: datetime.datetime) -> xr.DataArray:
        """Download the GOES data to the :attr:`cachestore` at the given time."""
        if self.cachestore is None:
            raise ValueError("cachestore must be set to use _get_with_cache")

        lpaths = self._lpaths(time)
        bands_needed = {c for c, lpath in lpaths.items() if not self.cachestore.exists(lpath)}

        if bands_needed:
            rpaths = self.gcs_goes_path(time, bands_needed)
            for rpath in rpaths:
                band = _extract_band_from_rpath(rpath)
                lpath = lpaths[band]
                self.fs.get(rpath, lpath)

        # Deal with the different spatial resolutions
        kwargs = {
            "concat_dim": "band",
            "combine": "nested",
            "data_vars": ["CMI"],
            "compat": "override",
            "coords": "minimal",
        }
        if len(lpaths) > 1 and "C02" in lpaths:  # xr.open_mfdataset fails after pop if only 1 file
            lpath02 = lpaths.pop("C02")
            ds = xr.open_mfdataset(lpaths.values(), **kwargs).swap_dims(band="band_id")  # type: ignore[arg-type]
            da1 = ds.reset_coords()["CMI"]
            da2 = xr.open_dataset(lpath02).reset_coords()["CMI"].expand_dims(band_id=[2])
            da = (
                geo_utils._coarsen_then_concat(da1, da2)
                .sortby("band_id")
                .assign_coords(t=ds["t"].values)
            )
        else:
            ds = xr.open_mfdataset(lpaths.values(), **kwargs).swap_dims(band="band_id")  # type: ignore[arg-type]
            da = ds["CMI"].sortby("band_id")

        # Attach some useful attrs -- only using goes_imager_projection currently
        da.attrs["goes_imager_projection"] = ds.goes_imager_projection.attrs
        da.attrs["geospatial_lat_lon_extent"] = ds.geospatial_lat_lon_extent.attrs

        return da

    def _get_without_cache(self, time: datetime.datetime) -> xr.DataArray:
        """Download the GOES data into memory at the given time."""
        rpaths = self.gcs_goes_path(time)

        # Load into memory
        data = self.fs.cat(rpaths)

        da_dict = {}
        for rpath, init_bytes in data.items():
            band = _extract_band_from_rpath(rpath)
            ds = _load_via_tempfile(init_bytes)

            da = ds["CMI"]
            da = da.expand_dims(band_id=ds["band_id"].values)
            da_dict[band] = da

        if len(da_dict) > 1 and "C02" in da_dict:  # xr.concat fails after pop if only 1 file
            da2 = da_dict.pop("C02")
            da1 = xr.concat(da_dict.values(), dim="band_id", coords="different", compat="equals")
            da = geo_utils._coarsen_then_concat(da1, da2)
        else:
            da = xr.concat(da_dict.values(), dim="band_id", coords="different", compat="equals")

        da = da.sortby("band_id")

        # Attach some useful attrs -- only using goes_imager_projection currently
        da.attrs["goes_imager_projection"] = ds.goes_imager_projection.attrs
        da.attrs["geospatial_lat_lon_extent"] = ds.geospatial_lat_lon_extent.attrs

        return da


def _load_via_tempfile(data: bytes) -> xr.Dataset:
    """Load xarray dataset via temporary file."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)
    try:
        return xr.load_dataset(tmp.name)
    finally:
        os.remove(tmp.name)


def _cartopy_crs(proj_info: dict[str, Any]) -> cartopy.crs.Geostationary:
    try:
        from cartopy import crs as ccrs
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            name="GOES visualization",
            package_name="cartopy",
            module_not_found_error=exc,
            pycontrails_optional_package="sat",
        )

    globe = ccrs.Globe(
        semimajor_axis=proj_info["semi_major_axis"],
        semiminor_axis=proj_info["semi_minor_axis"],
    )
    return ccrs.Geostationary(
        central_longitude=proj_info["longitude_of_projection_origin"],
        satellite_height=proj_info["perspective_point_height"],
        sweep_axis=proj_info["sweep_angle_axis"],
        globe=globe,
    )


def extract_visualization(
    da: xr.DataArray,
    color_scheme: str = "ash",
    ash_convention: str = "SEVIRI",
    gamma: float = 2.2,
) -> tuple[npt.NDArray[np.float32], cartopy.crs.Geostationary, tuple[float, float, float, float]]:
    """Extract artifacts for visualizing GOES data with the given color scheme.

    Parameters
    ----------
    da : xr.DataArray
        DataArray of GOES data as returned by :meth:`GOES.get`. Must have the bands
        required by :func:`to_ash`.
    color_scheme : str
        Color scheme to use for visualization. Must be one of {"true", "ash"}.
        If "true", the ``da`` must contain bands C01, C02, and C03.
        If "ash", the ``da`` must contain bands C11, C14, and C15 (SEVIRI convention)
        or bands C11, C13, C14, and C15 (standard convention).
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
        The Geostationary projection built from the GOES metadata.
    src_extent : tuple[float, float, float, float]
        Extent of GOES data in the Geostationary projection
    """
    proj_info = da.attrs["goes_imager_projection"]
    h = proj_info["perspective_point_height"]

    src_crs = _cartopy_crs(proj_info)

    if color_scheme == "true":
        rgb = to_true_color(da, gamma)
    elif color_scheme == "ash":
        rgb = to_ash(da, ash_convention)
    else:
        raise ValueError(f"Color scheme must be 'true' or 'ash', not '{color_scheme}'")

    np.nan_to_num(rgb, copy=False)

    x = da["x"].values
    y = da["y"].values

    # Multiply extremes by the satellite height
    src_extent = h * x.min(), h * x.max(), h * y.min(), h * y.max()

    return rgb, src_crs, src_extent


extract_goes_visualization = extract_visualization  # keep for backwards compatibility


def to_true_color(da: xr.DataArray, gamma: float = 2.2) -> npt.NDArray[np.float32]:
    """Compute 3d RGB array for the true color scheme.

    Parameters
    ----------
    da : xr.DataArray
        DataArray of GOES data with bands C01, C02, C03.
    gamma : float, optional
        Gamma correction for the RGB bands.

    Returns
    -------
    npt.NDArray[np.float32]
        3d RGB array with true color scheme.

    References
    ----------
    - `Unidata's true color recipe <https://unidata.github.io/python-gallery/examples/mapping_GOES16_TrueColor.html>`_
    """
    if not np.all(np.isin([1, 2, 3], da["band_id"])):
        msg = "DataArray must contain bands 1, 2, and 3 for true color"
        raise ValueError(msg)

    red = da.sel(band_id=2).values
    veggie = da.sel(band_id=3).values
    blue = da.sel(band_id=1).values

    red = geo_utils._clip_and_scale(red, 0.0, 1.0)
    veggie = geo_utils._clip_and_scale(veggie, 0.0, 1.0)
    blue = geo_utils._clip_and_scale(blue, 0.0, 1.0)

    red = red ** (1 / gamma)
    veggie = veggie ** (1 / gamma)
    blue = blue ** (1 / gamma)

    # Calculate synthetic green band
    green = 0.45 * red + 0.1 * veggie + 0.45 * blue
    green = geo_utils._clip_and_scale(green, 0.0, 1.0)

    return np.dstack([red, green, blue])
