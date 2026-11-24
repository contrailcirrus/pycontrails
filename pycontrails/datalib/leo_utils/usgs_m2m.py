"""Download Landsat files through the USGS Machine-to-Machine (M2M) API.

The `M2M API <https://m2m.cr.usgs.gov/>`__ is free but requires a
`USGS EROS Registration System <https://ers.cr.usgs.gov/>`__ account with M2M access and an
application token.

Set the environment variables ``USGS_M2M_USERNAME`` and ``USGS_M2M_TOKEN`` to the
ERS username and application token.
"""

import os
import pathlib
from typing import Any

from pycontrails.utils import dependencies

try:
    import requests
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="datalib.leo_utils.usgs_m2m module",
        package_name="requests",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )

#: Root URL of the M2M JSON API
M2M_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"

#: M2M dataset of Landsat 8-9 OLI/TIRS Collection 2 Level-1 products
M2M_DATASET = "landsat_ot_c2_l1"

#: Timeout in seconds for HTTP requests
TIMEOUT = 60


def _post(session: requests.Session, endpoint: str, payload: dict[str, Any] | None) -> Any:
    """Post a request to an M2M endpoint and return the response data."""
    response = session.post(f"{M2M_URL}/{endpoint}", json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    body = response.json()
    if body["errorCode"]:
        msg = f"M2M {endpoint} failed with {body['errorCode']}: {body['errorMessage']}"
        raise RuntimeError(msg)
    return body["data"]


def _scene_id(product_id: str) -> str:
    """Return the scene ID of a Landsat product from its USGS STAC item.

    M2M identifies scenes by the scene ID rather than the product ID.
    """
    url = f"https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items/{product_id}"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()["properties"]["landsat:scene_id"]


def download(product_id: str, sinks: dict[str, str]) -> None:
    """Download files of a Landsat Collection 2 Level-1 product.

    Parameters
    ----------
    product_id : str
        Landsat product ID, for example ``"LC09_L1TP_095022_20250601_20250602_02_T1"``.
    sinks : dict[str, str]
        Map from a file suffix, such as ``"B9.TIF"`` or ``"MTL.txt"``, to the local path
        at which to save the file.
    """
    username = os.environ.get("USGS_M2M_USERNAME")
    token = os.environ.get("USGS_M2M_TOKEN")
    if not username or not token:
        msg = (
            "Set USGS_M2M_USERNAME and USGS_M2M_TOKEN to download Landsat data from USGS. "
            "See the pycontrails.datalib.leo_utils.usgs_m2m module documentation."
        )
        raise RuntimeError(msg)

    scene_id = _scene_id(product_id)

    # The M2M entity ID of <product_id>_B9.TIF is L1_<product_id>_B9_TIF
    entity_ids = {
        f"L1_{product_id}_{suffix.replace('.', '_').upper()}": sink
        for suffix, sink in sinks.items()
    }

    with requests.Session() as session:
        api_key = _post(session, "login-token", {"username": username, "token": token})
        session.headers["X-Auth-Token"] = api_key

        try:
            payload = {"datasetName": M2M_DATASET, "entityIds": [scene_id]}
            options = _post(session, "download-options", payload)

            # All individual files of a scene share one M2M product ID
            file_product_id = options[0]["secondaryDownloads"][0]["id"]
            downloads = [{"productId": file_product_id, "entityId": eid} for eid in entity_ids]
            result = _post(session, "download-request", {"downloads": downloads})
        finally:
            _post(session, "logout", None)

    urls = {d["entityId"]: d["url"] for d in result["availableDownloads"]}
    for entity_id, sink in entity_ids.items():
        _download_url(urls[entity_id], sink)


def _download_url(url: str, sink: str) -> None:
    """Stream a URL to a local file.

    The file is written to ``<sink>.part`` and renamed when complete.
    """
    tmp = pathlib.Path(f"{sink}.part")
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
    tmp.replace(sink)
