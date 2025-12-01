"""DWD Open Data Server utilities."""

from __future__ import annotations

import asyncio
import contextlib
import warnings
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from html.parser import HTMLParser

from pycontrails.utils import coroutines, dependencies

try:
    import aiohttp
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="dwd.ods module",
        package_name="aiohttp",
        module_not_found_error=exc,
        pycontrails_optional_package="dwd",
    )


def list_forecasts(domain: str) -> list[datetime]:
    """List available forecast cycles.

    Parameters
    ----------
    domain : str
        ICON domain. Must be one of "global", "europe", or "germany".

    Returns
    -------
    list[datetime]
        Start time of available forecast cycles
    """
    return coroutines.run(_list_forecasts_async(domain))


async def _list_forecasts_async(domain: str) -> list[datetime]:
    """Async helper function for _list_forecasts."""
    start_times = []

    async with aiohttp.ClientSession(raise_for_status=True) as session:
        cycles = [c async for c in _ls_async(_root(domain), session=session)]
        tasks = [anext(_ls_async(f"{cycle}/athb_t", session=session)) for cycle in cycles]

        for task in asyncio.as_completed(tasks):
            try:
                sample_grib = await task
            except StopAsyncIteration as e:
                msg = "Could not find OLR GRIB file to read forecast start time."
                raise FileNotFoundError(msg) from e

            try:
                idx = -5 if domain == "germany" else -4
                start_str = sample_grib.split("_")[idx]
                start = datetime.strptime(start_str, "%Y%m%d%H")
            except ValueError as e:
                msg = f"Could not parse date from GRIB file at {sample_grib}"
                raise ValueError(msg) from e

            start_times.append(start)

    return sorted(start_times)


def list_forecast_steps(domain: str, forecast: datetime) -> list[datetime]:
    """List forecast steps available for a given forecast cycle.

    Parameters
    ----------
    domain : str
        ICON domain. Must be one of "global", "europe", or "germany".

    forecast : datetime
        Start time of forecast cycle.

    Returns
    -------
    list[datetime]
        Times of available forecast steps. If no data is available
        for the specified forecast cycle a warning is issued and
        an empty list is returned.

    """
    available = list_forecasts(domain)
    if forecast not in available:
        msg = (
            f"No data available for forecast cycle starting at {forecast}. "
            "Use `list_forecasts` to list available forecast cycles."
        )
        warnings.warn(msg)
        return []

    try:
        gribs = _ls(f"{_root(domain)}/{forecast.hour:02d}/athb_t")
    except Exception as e:
        msg = "Could not find surface pressure GRIB files to read forecast start time."
        raise ValueError(msg) from e

    steps = set()
    for grib in gribs:
        try:
            idx = -4 if domain == "germany" else -3
            step_str = grib.split("_")[idx]
            steps.add(int(step_str))
        except ValueError as e:
            msg = f"Could not parse step from GRIB file at {grib}"
            raise ValueError(msg) from e

    return [forecast + timedelta(hours=step) for step in sorted(steps)]


def global_latitude_rpath(forecast: datetime) -> str:
    """Get path to remote latitude file for global icosahedral grid.

    Parameters
    ----------
    forecast : datetime
        Start time of forecast cycle.

    Returns
    -------
    str
        URL of grib file with cell-center latitudes

    """
    domain = "global"
    return (
        f"{_root(domain)}/{forecast.hour:02d}/clat/"
        f"{_prefix(domain)}_time-invariant_{forecast.strftime('%Y%m%d%H')}"
        "_CLAT.grib2.bz2"
    )


def global_longitude_rpath(forecast: datetime) -> str:
    """Get path to remote longitude file for global icosahedral grid.

    Parameters
    ----------
    forecast : datetime
        Start time of forecast cycle.

    Returns
    -------
    str
        URL of grib file with cell-center longitudes

    """
    domain = "global"
    return (
        f"{_root(domain)}/{forecast.hour:02d}/clon/"
        f"{_prefix(domain)}_time-invariant_{forecast.strftime('%Y%m%d%H')}"
        "_CLON.grib2.bz2"
    )


def rpath(domain: str, forecast: datetime, variable: str, step: int, level: int | None) -> str:
    """Get URL remote data file.

    Parameters
    ----------
    domain : str
        ICON domain. Must be one of "global", "europe", or "germany".

    forecast : datetime
        Start time of forecast cycle.

    variable : str
        Open Data Server variable name.

    step : int
        Forecast step.

    level : int | None
        Model level for 3d variables or ``None`` for 2d variables

    Returns
    -------
    str
        URL of remote data file.

    """
    if domain not in ("global", "europe", "germany"):
        msg = f"Unknown domain {domain}."
        raise ValueError(msg)

    level_type = "model" if level is not None else "single"
    step_level_str = _step_level_str(domain, step, level)

    return (
        f"{_root(domain)}/{forecast.hour:02d}/{variable}/"
        f"{_prefix(domain)}_{level_type}-level_{forecast.strftime('%Y%m%d%H')}"
        f"_{step_level_str}_{variable if domain == 'germany' else variable.upper()}.grib2.bz2"
    )


def get(rpath: str, lpath: str) -> None:
    """Get data file from Open Data Server.

    Parameters
    ----------
    rpath : str
        URL of remote file on Open Data Server

    lpath : str
        Local path where file contents will be saved

    """
    return coroutines.run(_get_async(rpath, lpath, None))


async def _get_async(rpath: str, lpath: str, session: aiohttp.ClientSession | None) -> None:
    """Async helper function for get."""
    try:
        async with (
            _use_or_create_session(session) as _session,
            _session.get(f"https://{rpath}") as response,
        ):
            content = await response.read()
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            msg = (
                f"Remote file {rpath} not found on DWD Open Data Server. "
                "You may be requesting data from a past forecast that is no longer available. "
                "Note that `list_forecasts` and `list_forecast_steps` can be used to check "
                "available forecast cycles and steps."
            )
            raise FileNotFoundError(msg) from e
        msg = f"Error while downloading file at {rpath}"
        raise RuntimeError(msg) from e

    with open(lpath, "wb") as f:
        f.write(content)


class _OpenDataServerParser(HTMLParser):
    """Parser for DWD Open Data Server pages.

    This parser builds a list of links on each page,
    ignoring links to parent directories.
    """

    __slots__ = ("children",)

    def __init__(self) -> None:
        super().__init__()
        self.children: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record link targets, excluding parent directory."""
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value is not None and value != "../":
                self.children.append(value.rstrip("/"))
                return


def _ls(url: str) -> list[str]:
    """List URL of each item in directory."""
    # This uses an async helper function to avoid an additional
    # dependency on a synchronous http request library. The helper
    # only issues a single get request, so there's no significant
    # performance benefit from concurrency.
    return coroutines.materialize(_ls_async(url, None))


async def _ls_async(url: str, session: aiohttp.ClientSession | None) -> AsyncIterator[str]:
    """Async helper function for _ls."""
    try:
        async with (
            _use_or_create_session(session) as _session,
            _session.get(f"https://{url}") as response,
        ):
            text = await response.text()
    except aiohttp.ClientError as e:
        msg = f"Error listing contents of {url}."
        raise RuntimeError(msg) from e

    parser = _OpenDataServerParser()
    parser.feed(text)
    parser.close()

    for child in parser.children:
        yield f"{url}/{child}"


@contextlib.asynccontextmanager
async def _use_or_create_session(
    session: aiohttp.ClientSession | None,
) -> AsyncIterator[aiohttp.ClientSession]:
    """Provide session for async requests, using an existing session if provided."""
    if session is None:
        session = aiohttp.ClientSession(raise_for_status=True)
        local_session = True
    else:
        local_session = False

    try:
        yield session
    finally:
        if local_session:
            await session.close()


def _root(domain: str) -> str:
    """Get Open Data Server root for ICON grib files on specified domain."""
    root = "opendata.dwd.de"

    if domain.lower() == "global":
        return f"{root}/weather/nwp/icon/grib"
    if domain.lower() == "europe":
        return f"{root}/weather/nwp/icon-eu/grib"
    if domain.lower() == "germany":
        return f"{root}/weather/nwp/icon-d2/grib"

    msg = f"Unknown domain {domain}."
    raise ValueError(msg)


def _prefix(domain: str) -> str:
    """Get Open Data Server filename prefix for ICON grid files on specified domain."""
    if domain.lower() == "global":
        return "icon_global_icosahedral"
    if domain.lower() == "europe":
        return "icon-eu_europe_regular-lat-lon"
    if domain.lower() == "germany":
        return "icon-d2_germany_regular-lat-lon"

    msg = f"Unknown domain {domain}."
    raise ValueError(msg)


def _step_level_str(domain: str, step: int, level: int | None) -> str:
    """Return portion of filename identifying step and level."""
    step_str = f"{step:03d}"
    level_str = f"_{level}" if level is not None else ("_2d" if domain == "germany" else "")
    return step_str + level_str
