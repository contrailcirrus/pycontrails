"""Google Contrails Forecast Datalib."""

import datetime
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests
import xarray as xr

from pycontrails import MetDataset, MetVariable
from pycontrails.core.vector import VectorDataset
from pycontrails.physics import units

if TYPE_CHECKING:
    import google.auth.credentials


logger = logging.getLogger(__name__)


Severity = MetVariable(
    short_name="contrails",
    standard_name="contrails",
    long_name="Contrail Severity Index",
    description="The severity (0-4) of forecasted contrail warming.",
)


EnergyForcing = MetVariable(
    short_name="ef_per_m",
    standard_name="expected_effective_energy_forcing",
    long_name="Expected Effective Energy Forcing",
    description="The effective energy forcing of contrail warming.",
)


@dataclass
class GoogleForecastParams:
    """Parameters for :class:`ForecastApi`. to load the Google Contrails Forecast.

    See: https://developers.google.com/contrails.
    """

    #: Credentials to authenticate to Google Cloud. This can be an API key, or any credentials
    #: provided by `google.auth`.
    credentials: str | google.auth.credentials.Credentials

    #: Google Contrails Forecast API URL.
    url: str = "https://contrails.googleapis.com/v2/grids"

    @property
    def request_headers(self) -> dict[str, str]:
        """Request headers for authentication."""
        headers = {}
        if isinstance(self.credentials, str):
            headers["x-goog-api-key"] = self.credentials
        else:
            self.credentials.apply(headers)
        return headers


class ForecastApi:
    """Forecast datalib to download precomputed contrail forecasts from API sources.

    This class provides an interface to the Google Contrails Forecast API.
    It returns a :class:`MetDataset` containing the forecasted energy forcing.

    Parameters
    ----------
    params : GoogleForecastParams
        Parameters for the model.
    """

    def __init__(
        self,
        params: GoogleForecastParams,
        met_variables: tuple[MetVariable, ...] = (Severity,),
    ) -> None:
        self._params = params
        self.met_variables = met_variables
        super().__init__()

    # TODO: Add support for GeoVectorDataset.
    def eval(self, source: VectorDataset) -> MetDataset:
        """Evaluate the model.

        Parameters
        ----------
        source : VectorDataset
            Source data to evaluate the model on.

        Returns
        -------
        MetDataset
            Evaluated model output.
        """
        times = self._infer_times(source)
        grids = [self.get_grid(t) for t in times]

        if not grids:
            raise ValueError("No grids found")

        ds = xr.concat([m.data for m in grids], dim="time")

        # Ensure that the grid fully covers the time range of the source
        # by interpolating to the unique times in the source.
        source_times = np.unique(source["time"])
        ds = ds.interp(time=source_times, method="linear").sel(time=source_times)

        return MetDataset(ds)

    def get_grid(
        self, time: datetime.datetime, met_variables: tuple[MetVariable, ...] | None = None
    ) -> MetDataset:
        """Get grid from Google API."""

        if met_variables is None:
            met_variables = self.met_variables

        params = {"time": time.isoformat(), "data": [v.standard_name for v in met_variables]}

        logger.info("Requesting: %s with params %s", self._params.url, params)
        response = requests.get(
            self._params.url, params=params, headers=self._params.request_headers
        )
        logger.info(
            "Received %s, response with size %s and headers: %s",
            response.status_code,
            len(response.content),
            response.headers,
        )
        response.raise_for_status()

        # Workaround: xarray does not support loading NetCDF4 from a bytes buffer.
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(response.content)
            tmp.flush()
            ds = xr.load_dataset(tmp.name, engine="netcdf4")

        logger.info("Received dataset: %s", ds)

        if "level" not in ds.dims:
            ds["level"] = units.ft_to_pl(ds["flight_level"] * 100)
            ds = ds.swap_dims({"flight_level": "level"})

        return MetDataset(ds)

    def _infer_times(self, source: VectorDataset) -> set[datetime.datetime]:
        # The API only provides forecasts for full hours. In case the user requests odd times,
        # we request the previous and next full hour for an interpolation.
        times = pd.to_datetime(source["time"], utc=True).to_pydatetime()
        required_times = set()

        for t in times:
            full_hour = t.replace(minute=0, second=0, microsecond=0)
            required_times.add(full_hour)
            if t != full_hour:
                required_times.add(full_hour + datetime.timedelta(hours=1))

        return required_times


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = GoogleForecastParams(credentials=os.getenv("GOOGLE_API_KEY"))
    google_forecast = ForecastApi(params)

    source = VectorDataset(dict(time=[datetime.datetime(2026, 1, 28, 12, 30)]))
    mds = google_forecast.eval(source)
    print(mds)
