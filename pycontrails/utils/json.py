"""JSON utilities."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """Custom JSONEncoder for numpy data types.

    Examples
    --------
    >>> import json
    >>> import numpy as np
    >>> from pycontrails.utils.json import NumpyEncoder

    >>> data = np.array([0, 1, 2, 3])
    >>> json.dumps(data, cls=NumpyEncoder)
    '[0, 1, 2, 3]'

    >>> data = np.datetime64(1234567890, "s")
    >>> json.dumps(data, cls=NumpyEncoder)
    '"2009-02-13T23:31:30"'

    Notes
    -----
    Adapted https://github.com/hmallen/numpyencoder/blob/master/numpyencoder/numpyencoder.py
    """

    def default(self, obj: Any) -> Any:
        """Encode numpy data types.

        This method overrides :meth:`default` on the JSONEncoder class.

        Parameters
        ----------
        obj : Any
            Object to encode.

        Returns
        -------
        Any
            Encoded object.
        """
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        # TODO: this is not easily reversible - np.timedelta64(str(np.timedelta64(1, "h"))) raises
        if isinstance(obj, (np.timedelta64)):
            return str(obj)

        if isinstance(obj, (np.datetime64)):
            return str(obj)

        if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        if isinstance(obj, (np.bool_)):
            return bool(obj)

        if isinstance(obj, (np.void)):
            return None

        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.to_numpy().tolist()

        try:
            return json.JSONEncoder.default(self, obj)

        # last ditch attempt by looking for a to_json attribute
        except TypeError as e:
            try:
                return obj.to_json
            except AttributeError:
                raise TypeError from e


def dataframe_to_geojson_points(
    df: pd.DataFrame,
    properties: list[str] | None = None,
    filter_nan: bool | list[str] = False,
) -> dict[str, Any]:
    """Convert a pandas DataFrame to a GeoJSON-like dictionary.

    This function create a Python representation of a GeoJSON FeatureCollection with Point features
    based on a :class:`pandas.DataFrame` with geospatial coordinate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Base dataframe.
        Must contain geospatial coordinate columns ["longitude", "latitude", "altitude", "time"]
    properties : list[str], optional
        Specify columns to include feature properties.
        By default, will use all column data that is not in the coordinate columns.
    filter_nan : bool | list[str], optional
        Filter out points with nan values in any columns, including coordinate columns.
        If `list of str` is input, only `filter_nan` columns will be used for filtering,
        allowing null values in the other columns.

    Returns
    -------
    dict[str, Any]
        Description

    Raises
    ------
    KeyError
        Raises if `properties` or `filter_nan` input contains a column label that does
        not exist in `df`
    """
    # required columns
    coord_cols = ["longitude", "latitude", "altitude", "time"]

    # get all properties if not defined
    if properties is None:
        properties = [c for c in df.columns if c not in coord_cols]
    elif [c for c in properties if c not in df.columns]:
        raise KeyError(
            f"{[c for c in properties if c not in df.columns]} do not exist in dataframe"
        )

    # downselect dataframe
    cols = ["longitude", "latitude", "altitude", "time"] + properties
    df = df[cols]

    # filter out coords with nan values, or filter just on "filter_nan" labels
    if isinstance(filter_nan, bool) and filter_nan:
        df = df[np.all(~np.isnan(df[cols]), axis=1)]
    elif isinstance(filter_nan, list):
        if [c for c in filter_nan if c not in df.columns]:
            raise KeyError(
                f"{[c for c in filter_nan if c not in df.columns]} do not exist in dataframe"
            )
        df = df[np.all(~np.isnan(df[filter_nan]), axis=1)]

    def row_to_feature(row: pd.Series) -> dict[str, str | dict[str, Any]]:
        point = [
            np.round(row.longitude, decimals=4) if not np.isnan(row.longitude) else None,
            np.round(row.latitude, decimals=4) if not np.isnan(row.latitude) else None,
            np.round(row.altitude, decimals=4) if not np.isnan(row.altitude) else None,
        ]
        # converting to int to allow JSON serialization
        properties = {"time": int(row.time.timestamp())}
        used_keys = ["time", "latitude", "longitude", "altitude"]
        unused_keys = [k for k in row.keys() if k not in used_keys]
        properties.update(
            {
                k: (
                    row[k]
                    if not isinstance(row[k], float)
                    or isinstance(row[k], float)
                    and not np.isnan(row[k])
                    else None
                )
                for k in unused_keys
            }
        )

        geometry = {"type": "Point", "coordinates": point}
        return {"type": "Feature", "geometry": geometry, "properties": properties}

    features = []
    df.apply(lambda row: features.append(row_to_feature(row)), axis=1)
    return {"type": "FeatureCollection", "features": features}
