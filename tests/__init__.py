"""Pycontrails tests."""

import os
import pathlib
import platform

import requests
from google.cloud import storage

import pycontrails

# Overwrite any CDS env configuration
os.environ["CDSAPI_URL"] = "FAKE"
os.environ["CDSAPI_KEY"] = "FAKE"

# Determine if BADA is available
# Set to BADA_CACHE_DIR environment variable or use default local bada path
default_bada_root = pathlib.Path(*pycontrails.__path__).parents[1] / "bada"
BADA_ROOT = pathlib.Path(os.getenv("BADA_CACHE_DIR", default_bada_root))
BADA3_PATH = BADA_ROOT / "bada3"
BADA4_PATH = BADA_ROOT / "bada4"

BADA_AVAILABLE = BADA3_PATH.exists() and BADA4_PATH.exists()
IS_WINDOWS = platform.system() == "Windows"


try:
    import open3d  # noqa: F401
except ModuleNotFoundError:
    OPEN3D_AVAILABLE = False
else:
    OPEN3D_AVAILABLE = True

try:
    storage.Client()
except Exception:
    GCP_CREDENTIALS = False
else:
    GCP_CREDENTIALS = True

try:
    from google.cloud import bigquery

    bigquery.Client()
except Exception:
    BIGQUERY_ACCESS = False
else:
    BIGQUERY_ACCESS = True

try:
    requests.get("https://github.com", timeout=5)
except Exception:
    OFFLINE = True
else:
    OFFLINE = False
