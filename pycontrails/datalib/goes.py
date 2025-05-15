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
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pycontrails.core import cache
from pycontrails.core.met import XArrayType
from pycontrails.utils import dependencies

try:
    import cartopy.crs as ccrs
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="goes module",
        package_name="cartopy",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )

try:
    import gcsfs
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="goes module",
        package_name="gcsfs",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )


#: Default channels to use if none are specified. These are the channels
#: required by the SEVIRI (MIT) ash color scheme.
DEFAULT_CHANNELS = "C11", "C14", "C15"

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


def _parse_channels(channels: str | Iterable[str] | None) -> set[str]:
    """Check that the channels are valid and return as a set."""
    if channels is None:
        return set(DEFAULT_CHANNELS)

    if isinstance(channels, str):
        channels = (channels,)

    available = {f"C{i:02d}" for i in range(1, 17)}
    channels = {c.upper() for c in channels}
    if not channels.issubset(available):
        raise ValueError(f"Channels must be in {sorted(available)}")
    return channels


def _check_channel_resolution(channels: Iterable[str]) -> None:
    """Confirm request channels have a common horizontal resolution."""
    assert channels, "channels must be non-empty"

    # https://www.goes-r.gov/spacesegment/abi.html
    resolutions = {
        "C01": 1.0,
        "C02": 1.0,  # XXX: this actually has a resolution of 0.5 km, but we treat it as 1 km
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

    resolutions = {c: resolutions[c] for c in channels}
    c0, res0 = resolutions.popitem()

    try:
        c1, res1 = next((c, res) for c, res in resolutions.items() if res != res0)
    except StopIteration:
        # All resolutions are the same
        return
    raise ValueError(
        "Channels must have a common horizontal resolution. "
        f"Channel {c0} has resolution {res0} km and channel {c1} has resolution {res1} km."
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
    channels: str | Iterable[str] | None = None,
    bucket: str | None = None,
    fs: gcsfs.GCSFileSystem | None = None,
) -> list[str]:
    """Return GCS paths to GOES data at the given time for the given region and channels.

    Presently only supported for GOES data whose scan time minute coincides with
    the minute of the time parameter.

    Parameters
    ----------
    time : datetime.datetime
        Time of GOES data. This should be a timezone-naive datetime object or an
        ISO 8601 formatted string.
    region : GOESRegion
        GOES Region of interest.
    channels : str | Iterable[str]
        Set of channels or bands for CMIP data. The 16 possible channels are
        represented by the strings "C01" to "C16". For the SEVIRI ash color scheme,
        set ``channels=("C11", "C14", "C15")``. For the true color scheme,
        set ``channels=("C01", "C02", "C03")``. By default, the channels
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

    >>> paths = gcs_goes_path(t, GOESRegion.F, channels=("C11", "C12", "C13"))
    >>> pprint(paths)
    ['gcp-public-data-goes-16/ABI-L2-CMIPF/2023/093/02/OR_ABI-L2-CMIPF-M6C11_G16_s20230930210203_e20230930219511_c20230930219586.nc',
     'gcp-public-data-goes-16/ABI-L2-CMIPF/2023/093/02/OR_ABI-L2-CMIPF-M6C12_G16_s20230930210203_e20230930219516_c20230930219596.nc',
     'gcp-public-data-goes-16/ABI-L2-CMIPF/2023/093/02/OR_ABI-L2-CMIPF-M6C13_G16_s20230930210203_e20230930219523_c20230930219586.nc']

    >>> paths = gcs_goes_path(t, GOESRegion.C, channels=("C11", "C12", "C13"))
    >>> pprint(paths)
    ['gcp-public-data-goes-16/ABI-L2-CMIPC/2023/093/02/OR_ABI-L2-CMIPC-M6C11_G16_s20230930211170_e20230930213543_c20230930214055.nc',
     'gcp-public-data-goes-16/ABI-L2-CMIPC/2023/093/02/OR_ABI-L2-CMIPC-M6C12_G16_s20230930211170_e20230930213551_c20230930214045.nc',
     'gcp-public-data-goes-16/ABI-L2-CMIPC/2023/093/02/OR_ABI-L2-CMIPC-M6C13_G16_s20230930211170_e20230930213557_c20230930214065.nc']

    >>> t = datetime.datetime(2023, 4, 3, 2, 11)
    >>> paths = gcs_goes_path(t, GOESRegion.M1, channels="C01")
    >>> pprint(paths)
    ['gcp-public-data-goes-16/ABI-L2-CMIPM/2023/093/02/OR_ABI-L2-CMIPM1-M6C01_G16_s20230930211249_e20230930211309_c20230930211386.nc']

    >>> t = datetime.datetime(2025, 5, 4, 3, 2)
    >>> paths = gcs_goes_path(t, GOESRegion.M2, channels="C01")
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

    channels = _parse_channels(channels)

    # It's faster to run a single glob with C?? then running a glob for
    # each channel. The downside is that we have to filter the results.
    rpath = f"{path_prefix}{name_prefix}C??{name_suffix}"

    fs = fs or gcsfs.GCSFileSystem(token="anon")
    rpaths: list[str] = fs.glob(rpath)

    out = [r for r in rpaths if _extract_channel_from_rpath(r) in channels]
    if not out:
        raise RuntimeError(f"No data found for {time} in {region} for channels {channels}")
    return out


def _extract_channel_from_rpath(rpath: str) -> str:
    # Split at the separator between product name and mode
    # This works for both M3 and M6
    sep = "-M"
    suffix = rpath.split(sep, maxsplit=1)[1]
    return suffix[1:4]


class GOES:
    """Support for GOES-16 data handling.

    Parameters
    ----------
    region : GOESRegion | str = {"F", "C", "M1", "M2"}
        GOES Region of interest. Uses the following conventions.

        - F: Full Disk
        - C: CONUS
        - M1: Mesoscale 1
        - M2: Mesoscale 2

    channels : str | set[str] | None
        Set of channels or bands for CMIP data. The 16 possible channels are
        represented by the strings "C01" to "C16". For the SEVIRI ash color scheme,
        set ``channels=("C11", "C14", "C15")``. For the true color scheme,
        set ``channels=("C01", "C02", "C03")``. By default, the channels
        required by the SEVIRI ash color scheme are used. The channels must have
        a common horizontal resolution. The resolutions are:

        - C01: 1.0 km
        - C02: 0.5 km  (treated as 1.0 km)
        - C03: 1.0 km
        - C04: 2.0 km
        - C05: 1.0 km
        - C06 - C16: 2.0 km

    cachestore : cache.CacheStore | None
        Cache store for GOES data. If None, data is downloaded directly into
        memory. By default, a :class:`cache.DiskCacheStore` is used.
    goes_bucket : str | None = None
        GCP bucket for GOES data. If None, the bucket is automatically
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
    >>> goes = GOES(region="M1", channels=("C11", "C14"))
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
    >>> goes = GOES(region="M2", channels=("C11", "C12", "C13"), cachestore=None)
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
        channels: str | Iterable[str] | None = None,
        cachestore: cache.CacheStore | None = __marker,  # type: ignore[assignment]
        goes_bucket: str | None = None,
    ) -> None:
        self.region = _parse_region(region)
        self.channels = _parse_channels(channels)
        _check_channel_resolution(self.channels)

        self.goes_bucket = goes_bucket
        self.fs = gcsfs.GCSFileSystem(token="anon")

        if cachestore is self.__marker:
            cache_root = cache._get_user_cache_dir()
            cache_dir = f"{cache_root}/goes"
            cachestore = cache.DiskCacheStore(cache_dir=cache_dir)
        self.cachestore = cachestore

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GOES(region='{self.region}', channels={sorted(self.channels)})"

    def gcs_goes_path(self, time: datetime.datetime, channels: set[str] | None = None) -> list[str]:
        """Return GCS paths to GOES data at given time.

        Presently only supported for GOES data whose scan time minute coincides with
        the minute of the time parameter.

        Parameters
        ----------
        time : datetime.datetime
            Time of GOES data.
        channels : set[str] | None
            Set of channels or bands for CMIP data. If None, the :attr:`channels`
            attribute is used.

        Returns
        -------
        list[str]
            List of GCS paths to GOES data.
        """
        channels = channels or self.channels
        return gcs_goes_path(time, self.region, channels, bucket=self.goes_bucket, fs=self.fs)

    def _lpaths(self, time: datetime.datetime) -> dict[str, str]:
        """Construct names for local netcdf files using the :attr:`cachestore`.

        Returns dictionary of the form ``{channel: local_path}``.
        """
        assert self.cachestore, "cachestore must be set"

        t_str = time.strftime("%Y%m%d%H%M")

        out = {}
        for c in self.channels:
            name = f"{self.region.name}_{t_str}_{c}.nc"
            lpath = self.cachestore.path(name)
            out[c] = lpath

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
        if not isinstance(time, datetime.datetime):
            time = pd.Timestamp(time).to_pydatetime()

        if self.cachestore is not None:
            return self._get_with_cache(time)  # type: ignore[arg-type]
        return self._get_without_cache(time)  # type: ignore[arg-type]

    def _get_with_cache(self, time: datetime.datetime) -> xr.DataArray:
        """Download the GOES data to the :attr:`cachestore` at the given time."""
        assert self.cachestore, "cachestore must be set"

        lpaths = self._lpaths(time)

        channels_needed = set()
        for c, lpath in lpaths.items():
            if not self.cachestore.exists(lpath):
                channels_needed.add(c)

        if channels_needed:
            rpaths = self.gcs_goes_path(time, channels_needed)
            for rpath in rpaths:
                channel = _extract_channel_from_rpath(rpath)
                lpath = lpaths[channel]
                self.fs.get(rpath, lpath)

        # Deal with the different spatial resolutions
        kwargs = {
            "concat_dim": "band",
            "combine": "nested",
            "data_vars": ["CMI"],
            "compat": "override",
            "coords": "minimal",
        }
        if len(lpaths) == 1:
            ds = xr.open_dataset(lpaths.popitem()[1])
            ds["CMI"] = ds["CMI"].expand_dims(band=ds["band_id"].values)
        elif "C02" in lpaths:
            lpath02 = lpaths.pop("C02")
            ds1 = xr.open_mfdataset(lpaths.values(), **kwargs)  # type: ignore[arg-type]
            ds2 = xr.open_dataset(lpath02)
            ds = _concat_c02(ds1, ds2)
        else:
            ds = xr.open_mfdataset(lpaths.values(), **kwargs)  # type: ignore[arg-type]

        da = ds["CMI"]
        da = da.swap_dims({"band": "band_id"}).sortby("band_id")

        # Attach some useful attrs -- only using goes_imager_projection currently
        da.attrs["goes_imager_projection"] = ds.goes_imager_projection.attrs
        da.attrs["geospatial_lat_lon_extent"] = ds.geospatial_lat_lon_extent.attrs

        return da

    def _get_without_cache(self, time: datetime.datetime) -> xr.DataArray:
        """Download the GOES data into memory at the given time."""
        rpaths = self.gcs_goes_path(time)

        # Load into memory
        data = self.fs.cat(rpaths)

        if isinstance(data, dict):
            da_dict = {}
            for rpath, init_bytes in data.items():
                channel = _extract_channel_from_rpath(rpath)
                ds = _load_via_tempfile(init_bytes)

                da = ds["CMI"]
                da = da.expand_dims(band_id=ds["band_id"].values)
                da_dict[channel] = da

            if len(da_dict) == 1:  # This might be redundant with the branch below
                da = da_dict.popitem()[1]
            elif "C02" in da_dict:
                da2 = da_dict.pop("C02")
                da1 = xr.concat(da_dict.values(), dim="band_id")
                da = _concat_c02(da1, da2)
            else:
                da = xr.concat(da_dict.values(), dim="band_id")

        else:
            ds = _load_via_tempfile(data)
            da = ds["CMI"]
            da = da.expand_dims(band_id=ds["band_id"].values)

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


def _concat_c02(ds1: XArrayType, ds2: XArrayType) -> XArrayType:
    """Concatenate two datasets with C01 and C02 data."""
    # Average the C02 data to the C01 resolution
    ds2 = ds2.coarsen(x=2, y=2, boundary="exact").mean()  # type: ignore[attr-defined]

    # Gut check
    np.testing.assert_allclose(ds1["x"], ds2["x"], rtol=0.0005)
    np.testing.assert_allclose(ds1["y"], ds2["y"], rtol=0.0005)

    # Assign the C01 data to the C02 data
    ds2["x"] = ds1["x"]
    ds2["y"] = ds1["y"]

    # Finally, combine the datasets
    dim = "band_id" if "band_id" in ds1.dims else "band"
    return xr.concat([ds1, ds2], dim=dim)


def extract_goes_visualization(
    da: xr.DataArray,
    color_scheme: str = "ash",
    ash_convention: str = "SEVIRI",
    gamma: float = 2.2,
) -> tuple[npt.NDArray[np.float32], ccrs.Geostationary, tuple[float, float, float, float]]:
    """Extract artifacts for visualizing GOES data with the given color scheme.

    Parameters
    ----------
    da : xr.DataArray
        DataArray of GOES data as returned by :meth:`GOES.get`. Must have the channels
        required by :func:`to_ash`.
    color_scheme : str = {"ash", "true"}
        Color scheme to use for visualization.
    ash_convention : str = {"SEVIRI", "standard"}
        Passed into :func:`to_ash`. Only used if ``color_scheme="ash"``.
    gamma : float = 2.2
        Passed into :func:`to_true_color`. Only used if ``color_scheme="true"``.

    Returns
    -------
    rgb : npt.NDArray[np.float32]
        3D RGB array of shape ``(height, width, 3)``. Any nan values are replaced with 0.
    src_crs : ccrs.Geostationary
        The Geostationary projection built from the GOES metadata.
    src_extent : tuple[float, float, float, float]
        Extent of GOES data in the Geostationary projection
    """
    proj_info = da.attrs["goes_imager_projection"]
    h = proj_info["perspective_point_height"]
    lon0 = proj_info["longitude_of_projection_origin"]
    src_crs = ccrs.Geostationary(central_longitude=lon0, satellite_height=h, sweep_axis="x")

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


def to_true_color(da: xr.DataArray, gamma: float = 2.2) -> npt.NDArray[np.float32]:
    """Compute 3d RGB array for the true color scheme.

    Parameters
    ----------
    da : xr.DataArray
        DataArray of GOES data with channels C01, C02, C03.
    gamma : float = 2.2
        Gamma correction for the RGB channels.

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
    green = da.sel(band_id=3).values
    blue = da.sel(band_id=1).values

    red = _clip_and_scale(red, 0.0, 1.0)
    green = _clip_and_scale(green, 0.0, 1.0)
    blue = _clip_and_scale(blue, 0.0, 1.0)

    red = red ** (1 / gamma)
    green = green ** (1 / gamma)
    blue = blue ** (1 / gamma)

    # Calculate "true" green channel
    green = 0.45 * red + 0.1 * green + 0.45 * blue
    green = _clip_and_scale(green, 0.0, 1.0)

    return np.dstack([red, green, blue])


def to_ash(da: xr.DataArray, convention: str = "SEVIRI") -> npt.NDArray[np.float32]:
    """Compute 3d RGB array for the ASH color scheme.

    Parameters
    ----------
    da : xr.DataArray
        DataArray of GOES data with appropriate channels.
    convention : str = {"SEVIRI", "standard"}
        Convention for color space.

        - SEVIRI convention requires channels C11, C14, C15.
          Used in :cite:`kulikSatellitebasedDetectionContrails2019`.
        - Standard convention requires channels C11, C13, C14, C15

    Returns
    -------
    npt.NDArray[np.float32]
        3d RGB array with ASH color scheme according to convention.

    References
    ----------
    - `Ash RGB quick guide (the color space and color interpretations) <https://rammb.cira.colostate.edu/training/visit/quick_guides/GOES_Ash_RGB.pdf>`_
    - :cite:`SEVIRIRGBCal`
    - :cite:`kulikSatellitebasedDetectionContrails2019`

    Examples
    --------
    >>> goes = GOES(region="M2", channels=("C11", "C14", "C15"))
    >>> da = goes.get("2022-10-03 04:34:00")
    >>> rgb = to_ash(da)
    >>> rgb.shape
    (500, 500, 3)

    >>> rgb[0, 0, :]
    array([0.0127004 , 0.22793579, 0.3930847 ], dtype=float32)
    """
    if convention == "standard":
        if not np.all(np.isin([11, 13, 14, 15], da["band_id"])):
            msg = "DataArray must contain bands 11, 13, 14, and 15 for standard ash"
            raise ValueError(msg)
        c11 = da.sel(band_id=11).values  # 8.44
        c13 = da.sel(band_id=13).values  # 10.33
        c14 = da.sel(band_id=14).values  # 11.19
        c15 = da.sel(band_id=15).values  # 12.27

        red = c15 - c13
        green = c14 - c11
        blue = c13

    elif convention in ("SEVIRI", "MIT"):  # retain MIT for backwards compatibility
        if not np.all(np.isin([11, 14, 15], da["band_id"])):
            msg = "DataArray must contain bands 11, 14, and 15 for SEVIRI ash"
            raise ValueError(msg)
        c11 = da.sel(band_id=11).values  # 8.44
        c14 = da.sel(band_id=14).values  # 11.19
        c15 = da.sel(band_id=15).values  # 12.27

        red = c15 - c14
        green = c14 - c11
        blue = c14

    else:
        raise ValueError("Convention must be either 'SEVIRI' or 'standard'")

    # See colostate pdf for slightly wider values
    red = _clip_and_scale(red, -4.0, 2.0)
    green = _clip_and_scale(green, -4.0, 5.0)
    blue = _clip_and_scale(blue, 243.0, 303.0)
    return np.dstack([red, green, blue])


def _clip_and_scale(
    arr: npt.NDArray[np.floating], low: float, high: float
) -> npt.NDArray[np.floating]:
    """Clip array and rescale to the interval [0, 1].

    Array is first clipped to the interval [low, high] and then linearly rescaled
    to the interval [0, 1] so that::

        low -> 0
        high -> 1

    Parameters
    ----------
    arr : npt.NDArray[np.floating]
        Array to clip and scale.
    low : float
        Lower clipping bound.
    high : float
        Upper clipping bound.

    Returns
    -------
    npt.NDArray[np.floating]
        Clipped and scaled array.
    """
    return (arr.clip(low, high) - low) / (high - low)


def parallax_correct(
    longitude: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
    altitude: npt.NDArray[np.floating],
    goes_da: xr.DataArray,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Apply parallax correction to WGS84 geodetic coordinates based on satellite perspective.

    This function considers the ray from the satellite to the points of interest and finds
    the intersection of this ray with the WGS84 ellipsoid. The intersection point is then
    returned as the corrected longitude and latitude coordinates.

    ::

       @ satellite
        \
         \
          \
           \
            \
             * aircraft
              \
               \
                x parallax corrected aircraft
        -------------------------  surface

    If the point of interest is not visible from the satellite (ie, on the opposite side of the
    earth), the function returns nan for the corrected coordinates.

    This function requires the :mod:`pyproj` package to be installed.

    Parameters
    ----------
    longitude : npt.NDArray[np.floating]
        A 1D array of longitudes in degrees.
    latitude : npt.NDArray[np.floating]
        A 1D array of latitudes in degrees.
    altitude : npt.NDArray[np.floating]
        A 1D array of altitudes in meters.
    goes_da : xr.DataArray
        DataArray containing the GOES projection information. Only the ``goes_imager_projection``
        field of the :attr:`xr.DataArray.attrs` is used.

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        A tuple containing the corrected longitude and latitude coordinates.

    """
    goes_imager_projection = goes_da.attrs["goes_imager_projection"]
    sat_lon = goes_imager_projection["longitude_of_projection_origin"]
    sat_lat = goes_imager_projection["latitude_of_projection_origin"]
    sat_alt = goes_imager_projection["perspective_point_height"]

    try:
        import pyproj
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            name="parallax_correct function",
            package_name="pyproj",
            module_not_found_error=exc,
            pycontrails_optional_package="pyproj",
        )

    # Convert from WGS84 to ECEF coordinates
    ecef_crs = pyproj.CRS("EPSG:4978")
    transformer = pyproj.Transformer.from_crs("WGS84", ecef_crs, always_xy=True)

    p0 = np.array(transformer.transform([sat_lon], [sat_lat], [sat_alt]))
    p1 = np.array(transformer.transform(longitude, latitude, altitude))

    # Major and minor axes of the ellipsoid
    a = ecef_crs.ellipsoid.semi_major_metre  # type: ignore[union-attr]
    b = ecef_crs.ellipsoid.semi_minor_metre  # type: ignore[union-attr]
    intersection = _intersection_with_ellipsoid(p0, p1, a, b)

    # Convert back to WGS84 coordinates
    inv_transformer = pyproj.Transformer.from_crs(ecef_crs, "WGS84", always_xy=True)
    return inv_transformer.transform(*intersection)[:2]  # final coord is (close to) 0


def _intersection_with_ellipsoid(
    p0: npt.NDArray[np.floating],
    p1: npt.NDArray[np.floating],
    a: float,
    b: float,
) -> npt.NDArray[np.floating]:
    """Find the intersection of a line with the surface of an ellipsoid."""
    # Calculate the direction vector
    px, py, pz = p0
    v = p1 - p0
    vx, vy, vz = v

    # The line between p0 and p1 in parametric form is p(t) = p0 + t * v
    # We need to find t such that p(t) lies on the ellipsoid
    # x^2 / a^2 + y^2 / a^2 + z^2 / b^2 = 1
    # (px + t * vx)^2 / a^2 + (py + t * vy)^2 / a^2 + (pz + t * vz)^2 / b^2 = 1
    # Rearranging gives a quadratic in t

    # Calculate the coefficients of this quadratic equation
    A = vx**2 / a**2 + vy**2 / a**2 + vz**2 / b**2
    B = 2 * (px * vx / a**2 + py * vy / a**2 + pz * vz / b**2)
    C = px**2 / a**2 + py**2 / a**2 + pz**2 / b**2 - 1.0

    # Calculate the discriminant
    D = B**2 - 4 * A * C
    sqrtD = np.sqrt(D, where=D >= 0, out=np.full_like(D, np.nan))

    # Calculate the two possible solutions for t
    t0 = (-B + sqrtD) / (2.0 * A)
    t1 = (-B - sqrtD) / (2.0 * A)

    # Calculate the intersection points
    intersection0 = p0 + t0 * v
    intersection1 = p0 + t1 * v

    # Pick the intersection point that is closer to the aircraft (p1)
    d0 = np.linalg.norm(intersection0 - p1, axis=0)
    d1 = np.linalg.norm(intersection1 - p1, axis=0)
    out = np.where(d0 < d1, intersection0, intersection1)

    # Fill the points in which the aircraft is not visible by the satellite with nan
    # This occurs when the earth is between the satellite and the aircraft
    # In other words, we can check for t0 < 1 (or t1 < 1)
    opposite_side = t0 < 1.0
    out[:, opposite_side] = np.nan

    return out
