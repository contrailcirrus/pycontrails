"""Download and parse Sentinel metadata."""

import datetime
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Collection

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import xarray as xr
from scipy.interpolate import griddata

from pycontrails.utils import dependencies

BAND_ID_MAPPING = {
    "B01": 0,
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B05": 4,
    "B06": 5,
    "B07": 6,
    "B08": 7,
    "B8A": 8,
    "B09": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}


def _band_id(band: str) -> int:
    """Get band ID used in some metadata files."""
    if band in (f"B{i:2d}" for i in range(1, 9)):
        return int(band[1:]) - 1
    if band == "B8A":
        return 8
    return int(band[1:])


def parse_viewing_incidence_angle_by_detector(
    metadata_path: str, target_detector_id: str, target_band_id: str = "2"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read sensor incidence angles from metadata.

    Parameters
    ----------
    metadata_path : str
        Path to the XML file containing TILE metadata.
    target_detector_id : str
        Target Detector_ID.
    target_band_id : str
        Starts from 0 (e.g. band 2 (blue) = band_id "1")

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Zenith Angles, Azimuth Angles ((23x23) numpy arrays)
    """
    tree = ET.parse(metadata_path)
    root = tree.getroot()

    tile_angles_element = root.find(".//Tile_Angles")
    if tile_angles_element is not None:
        for band in tile_angles_element.findall(".//Viewing_Incidence_Angles_Grids"):
            band_id = band.get("bandId")
            detector_id = band.get("detectorId")

            if band_id == target_band_id and detector_id == target_detector_id:
                zenith_element = band.find(".//Zenith")
                azimuth_element = band.find(".//Azimuth")

                if zenith_element is not None and azimuth_element is not None:
                    zenith_values_list = zenith_element.find(".//Values_List")
                    azimuth_values_list = azimuth_element.find(".//Values_List")

                    zenith_2d_array = []
                    azimuth_2d_array = []

                    if zenith_values_list is not None:
                        for values in zenith_values_list.findall(".//VALUES"):
                            if values.text is not None:
                                zenith_row = list(map(float, values.text.split()))
                                zenith_2d_array.append(zenith_row)

                    if azimuth_values_list is not None:
                        for values in azimuth_values_list.findall(".//VALUES"):
                            if values.text is not None:
                                azimuth_row = list(map(float, values.text.split()))
                                azimuth_2d_array.append(azimuth_row)

                    return np.array(zenith_2d_array), np.array(azimuth_2d_array)

        # If no matching band/detector found, return empty arrays
        return np.array([]), np.array([])

    raise ValueError("Viewing_Incidence_Angles_Grids element not found.")


def parse_viewing_incidence_angles(
    metadata_path: str, target_band_id: str = "2"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read sensor incidence angles from metadata. Returns the total of all detectors.

    Parameters
    ----------
    metadata_path : str
        Path to the XML file containing TILE metadata.
    target_band_id : str
        Starts from 0 (e.g. band 2 (blue) = band_id "1")

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Zenith Angles, Azimuth Angles ((23x23) numpy array)
    """
    total_zenith = np.full((23, 23), np.nan, dtype=np.float64)
    total_azimuth = np.full((23, 23), np.nan, dtype=np.float64)

    # loop over all 12 detector id's
    for detector_id in [str(i) for i in range(1, 13)]:
        zen, azi = parse_viewing_incidence_angle_by_detector(
            metadata_path, detector_id, target_band_id
        )
        if zen is None or azi is None or zen.size == 0 or azi.size == 0:
            continue

        # convert to np array
        zen_array = np.array(zen, dtype=np.float64)
        azi_array = np.array(azi, dtype=np.float64)

        # remove NaN values
        mask = ~np.isnan(zen_array)

        total_zenith[mask] = zen_array[mask]
        total_azimuth[mask] = azi_array[mask]

    return total_zenith, total_azimuth


def parse_high_res_detector_mask(metadata_path: str, scale: int = 10) -> npt.NDArray[np.integer]:
    """
    Load in the detector mask from either JP2 or GML file.

    - JP2: Reads pixel-level mask indicating which detector [1-12] captured each pixel.
    - GML: Converts detector polygons to raster mask, where each pixel corresponds to a detector ID.

    Lower the resolution with 'scale' to speed up processing.
    Scale 1 -> 10m resolution. Scale 10 -> 100m resolution.

    Parameters
    ----------
    metadata_path : str
        Path to metadata file (.jp2 or .gml).
    scale : int
        Factor by which to lower the resolution.

    Returns
    -------
    npt.NDArray[np.integer]
        2D array of detector IDs (1 to 12), shape (height, width).
    """
    try:
        import rasterio
        import rasterio.enums
        import rasterio.features
        import rasterio.transform
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            name="landsat module",
            package_name="rasterio",
            module_not_found_error=exc,
            pycontrails_optional_package="sat",
        )

    scale = int(scale)

    file_ext = os.path.splitext(metadata_path)[1].lower()

    if file_ext == ".jp2":
        # --- Handle JP2 case ---
        with rasterio.open(metadata_path) as src:
            return src.read(
                1,
                out_shape=(int(src.height // scale), int(src.width // scale)),
                resampling=rasterio.enums.Resampling.nearest,
            )

    if file_ext == ".gml":
        # --- Handle GML case ---
        try:
            import geopandas as gpd
        except ModuleNotFoundError as exc:
            dependencies.raise_module_not_found_error(
                name="landsat module",
                package_name="geopandas",
                module_not_found_error=exc,
                pycontrails_optional_package="sat",
            )
        gdf = gpd.read_file(metadata_path)

        # Extract detector_id from gml_id (assuming format contains "-Bxx-<id>-")
        def _extract_detector_id(gml_id: str) -> int:
            match = re.search(r"-B\d+-(\d+)-", gml_id)
            if match:
                return int(match.group(1))
            return 0

        gdf["detector_id"] = gdf["gml_id"].apply(_extract_detector_id)

        # Calculate bounding box and raster size
        minx, miny, maxx, maxy = gdf.total_bounds
        resolution = 10 * scale

        transform = rasterio.transform.from_origin(minx, maxy, resolution, resolution)

        # Create an emmpty instance for the full grid
        full_width = 10980 // scale
        full_height = 10980 // scale
        mask_full = np.full((full_height, full_width), 0, dtype="uint8")

        # Rasterize detector polygons
        local_width = int((maxx - minx) / resolution)
        local_height = int((maxy - miny) / resolution)
        local_mask = rasterio.features.rasterize(
            [(geom, det_id) for geom, det_id in zip(gdf.geometry, gdf.detector_id, strict=False)],
            out_shape=(local_height, local_width),
            transform=transform,
            fill=0,
            dtype="int32",
        )

        # Insert local raster into top-left corner of full grid
        mask_full[:local_height, :local_width] = local_mask.astype("uint8")

        return mask_full

    raise ValueError(f"Unsupported file extension: {file_ext}. Expected .jp2 or .gml.")


def _band_resolution(band: str) -> int:
    """Get band resolution in meters."""
    return (
        60 if band in ("B01", "B09", "B10") else 10 if band in ("B02", "B03", "B04", "B08") else 20
    )


def read_image_coordinates(meta: str, band: str) -> tuple[np.ndarray, np.ndarray]:
    """Read image x and y coordinates."""

    # convenience function that satisfies mypy
    def _text_from_tag(parent: ET.Element, tag: str) -> str:
        elem = parent.find(tag)
        if elem is None or elem.text is None:
            msg = f"Could not find text in {tag} element"
            raise ValueError(msg)
        return elem.text

    resolution = _band_resolution(band)

    # find coordinates of upper left corner and pixel size
    tree = ET.parse(meta)
    elems = tree.findall(".//Geoposition")
    for elem in elems:
        if int(elem.attrib["resolution"]) == resolution:
            ulx = float(_text_from_tag(elem, "ULX"))
            uly = float(_text_from_tag(elem, "ULY"))
            dx = float(_text_from_tag(elem, "XDIM"))
            dy = float(_text_from_tag(elem, "YDIM"))
            break
    else:
        msg = f"Could not find image geoposition for resolution of {resolution} m"
        raise ValueError(msg)

    # find image size
    elems = tree.findall(".//Size")
    for elem in elems:
        if int(elem.attrib["resolution"]) == resolution:
            nx = int(_text_from_tag(elem, "NCOLS"))
            ny = int(_text_from_tag(elem, "NROWS"))
            break
    else:
        msg = f"Could not find image size for resolution of {resolution} m"
        raise ValueError(msg)

    # compute pixel coordinates
    xlim = (ulx, ulx + (nx - 1) * dx)
    ylim = (uly, uly + (ny - 1) * dy)  # dy is < 0
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)

    return x, y


def parse_high_res_viewing_incidence_angles(
    tile_metadata_path: str, detector_band_metadata_path: str, scale: int = 10
) -> xr.Dataset:
    """
    Parse high-resolution viewing incidence angles (zenith and azimuth).

    Parameters
    ----------
    tile_metadata_path : str
        Path to the tile-level metadata file (usually MTD_TL.xml).
    detector_band_metadata_path : str
        Path to the detector-specific metadata file (e.g., MTD_DETFOO_B03.jp2).
    scale : int, optional
        Desired resolution scale (default is 10, e.g., 10 for 10m resolution).

    Returns
    -------
    xr.Dataset
        Dataset with coordinates ('y', 'x') containing:
        - VZA: View Zenith Angle
        - VAA: View Azimuth Angle

    Raises
    ------
    ValueError
        If required data (zenith or azimuth) cannot be parsed.
    """
    try:
        import skimage.transform
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            name="landsat module",
            package_name="scikit-image",
            module_not_found_error=exc,
            pycontrails_optional_package="sat",
        )

    # Load the detector mask
    detector_mask = parse_high_res_detector_mask(detector_band_metadata_path, scale=scale)
    if detector_mask is None:
        raise ValueError("Detector mask could not be parsed.")

    # Load averaged low-resolution zenith angles
    low_res_zenith, _ = parse_viewing_incidence_angles(tile_metadata_path)
    if low_res_zenith is None:
        raise ValueError("Zenith angles could not be parsed.")

    low_res_zenith = extrapolate_array(low_res_zenith)

    # Upsample zenith angles to high resolution
    high_res_zenith = skimage.transform.resize(
        low_res_zenith,
        output_shape=detector_mask.shape,
        order=1,
        mode="edge",
        anti_aliasing=True,
        preserve_range=True,
    )

    # Dictionary to store upsampled azimuth data per detector
    low_res_azimuth_dict = {}

    for detector_id in [str(i) for i in range(1, 13)]:
        zen, azi = parse_viewing_incidence_angle_by_detector(tile_metadata_path, detector_id, "2")
        if zen is None or azi is None or zen.size == 0 or azi.size == 0:
            continue

        azi_array = np.array(azi, dtype=np.float64)
        azi_extrapolated = extrapolate_array(azi_array)

        azi_extrapolated_highres = skimage.transform.resize(
            azi_extrapolated,
            output_shape=detector_mask.shape,
            order=1,
            mode="edge",
            anti_aliasing=True,
            preserve_range=True,
        )

        low_res_azimuth_dict[detector_id] = azi_extrapolated_highres

    if not low_res_azimuth_dict:
        raise ValueError("No azimuth data could be parsed for any detector.")

    # Initialize high-res azimuth array
    high_res_azimuth = np.zeros_like(detector_mask, dtype=np.float32)
    for i in range(detector_mask.shape[0]):
        for j in range(detector_mask.shape[1]):
            pixel_val = detector_mask[i, j]
            high_res_azimuth[i, j] = process_pixel(
                pixel_val, (i, j), detector_mask.shape, low_res_azimuth_dict
            )

    # Get UTM coordinates from the image
    x_img, y_img = read_image_coordinates(tile_metadata_path, "B03")
    x_min, x_max = float(x_img.min()), float(x_img.max())
    y_min, y_max = float(y_img.min()), float(y_img.max())

    # Create evenly spaced coordinate arrays that span the UTM extent
    height, width = high_res_zenith.shape
    x_coords = np.linspace(x_min, x_max, num=width)
    y_coords = np.linspace(y_max, y_min, num=height)  # y decreases in image space

    # Save the extent for metadata
    extent = (x_min, x_max, y_min, y_max)

    # Create xarray.Dataset
    return xr.Dataset(
        data_vars={
            "VZA": (("y", "x"), high_res_zenith.astype(np.float32)),
            "VAA": (("y", "x"), high_res_azimuth),
        },
        coords={
            "x": x_coords,
            "y": y_coords,
        },
        attrs={"title": "Sentinel Viewing Incidence Angles", "scale": scale, "extent": extent},
    )


def parse_ephemeris_sentinel(datatsrip_metadata_path: str) -> pd.DataFrame:
    """Return the ephemeris data from the DATASTRIP xml file.

    Parameters
    ----------
    datatsrip_metadata_path : str
        The location of the DATASTRIP xml file

    Returns
    -------
    pd.DataFrame
        A :class:`pandas.DataFrame` containing the ephemeris track with columns:
        - EPHEMERIS_TIME: Timestamps of the ephemeris data.
        - EPHEMERIS_ECEF_X: ECEF X coordinates.
        - EPHEMERIS_ECEF_Y: ECEF Y coordinates.
        - EPHEMERIS_ECEF_Z: ECEF Z coordinates.
    """
    tree = ET.parse(datatsrip_metadata_path)
    root = tree.getroot()

    ns = root[0].tag.split("}")[0][1:]

    satellite_ancillary_data = root.find(f".//{{{ns}}}Satellite_Ancillary_Data_Info")

    if satellite_ancillary_data is None:
        return pd.DataFrame(
            columns=["EPHEMERIS_TIME", "EPHEMERIS_ECEF_X", "EPHEMERIS_ECEF_Y", "EPHEMERIS_ECEF_Z"]
        )

    records = []

    for elem in satellite_ancillary_data:
        if elem.tag.endswith("Ephemeris"):
            gps_points_list = elem.find("GPS_Points_List")
            if gps_points_list is None:
                continue  # skip if missing

            for point in gps_points_list:
                gps_time_elem = point.find(".//GPS_TIME")
                position_elem = point.find(".//POSITION_VALUES")

                if gps_time_elem is None or gps_time_elem.text is None:
                    continue  # skip if missing

                if position_elem is None or position_elem.text is None:
                    continue  # skip if missing

                gps_time = datetime.datetime.strptime(gps_time_elem.text, "%Y-%m-%dT%H:%M:%S")

                # Convert GPS to UTC time as there is a few seconds between them
                utc_time = gps_to_utc(gps_time).replace(tzinfo=datetime.UTC)

                # Parse positions in ECEF coordinate system
                x, y, z = map(float, position_elem.text.split())

                records.append(
                    {
                        "EPHEMERIS_TIME": pd.Timestamp(utc_time).tz_localize(None),
                        "EPHEMERIS_ECEF_X": x / 1000,
                        "EPHEMERIS_ECEF_Y": y / 1000,
                        "EPHEMERIS_ECEF_Z": z / 1000,
                    }
                )

    return pd.DataFrame(records)


def parse_sentinel_crs(granule_metadata_path: str) -> pyproj.CRS:
    """Parse the CRS in the granule metadata."""
    tree = ET.parse(granule_metadata_path)
    root = tree.getroot()

    # Get the namespace of the XML file
    ns = root[0].tag.split("}")[0][1:]

    # Find the CS code in the XML file
    epsg_elem = root.find(f".//{{{ns}}}Geometric_Info/Tile_Geocoding/HORIZONTAL_CS_CODE")
    if epsg_elem is None or epsg_elem.text is None:
        raise ValueError("HORIZONTAL_CS_CODE element not found or empty in metadata")

    epsg_code = epsg_elem.text.strip()

    return pyproj.CRS.from_string(epsg_code)


def parse_sensing_time(granule_metadata_path: str) -> pd.Timestamp:
    """Parse the sensing_time in the granule metadata."""
    tree = ET.parse(granule_metadata_path)
    root = tree.getroot()

    # Get the namespace of the XML file
    ns = root[0].tag.split("}")[0][1:]

    # Find the SENSING_TIME element
    sensing_elem = root.find(f".//{{{ns}}}General_Info/SENSING_TIME")
    if sensing_elem is None or sensing_elem.text is None:
        raise ValueError("SENSING_TIME element not found or empty in metadata")

    sensing_time = sensing_elem.text.strip()
    return pd.to_datetime(sensing_time)


def get_detector_id(
    detector_band_metadata_path: str,
    tile_metadata_path: str,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    band: str = "B03",
) -> npt.NDArray[np.integer]:
    """
    Return the detector ID that captured a given pixel in a Sentinel-2 image.

    Parameters
    ----------
    detector_band_metadata_path : str
        Path to the MSK_DETFOO_Bxx.jp2 detector band mask file.
    tile_metadata_path : str
        Path to the tile metadata XML file (MTD_TL.xml) containing image geometry.
    x : npt.NDArray[np.floating]
        X coordinate (in UTM coordinate system) of the target pixel.
    y : npt.NDArray[np.floating]
        Y coordinate (in UTM coordinate system) of the target pixel.
    band : str, optional
        Spectral band to use for geometry parsing. Default is "B03".

    Returns
    -------
    npt.NDArray[np.integer]
        The detector ID (in the range 1 to 12) that captured the pixel. Returns 0 if
        the pixel is outside the image bounds or not covered by any detector.
    """
    x, y = np.atleast_1d(x, y)

    detector_mask = parse_high_res_detector_mask(detector_band_metadata_path, scale=10)

    height, width = detector_mask.shape

    x_img, y_img = read_image_coordinates(tile_metadata_path, band)
    x_min, x_max = float(x_img.min()), float(x_img.max())
    y_min, y_max = float(y_img.min()), float(y_img.max())

    # Compute resolution
    pixel_width = (x_max - x_min) / width
    pixel_height = (y_max - y_min) / height

    valid = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    # Convert x, y to column, row
    col = ((x[valid] - x_min) // pixel_width).astype(int)
    row = ((y_max - y[valid]) // pixel_height).astype(int)  # Note: y axis is top-down in images

    out = np.zeros(x.shape, dtype=detector_mask.dtype)
    out[valid] = detector_mask[row, col]

    return out


def get_time_delay_detectors(
    datastrip_metadata_path: str, band: str = "B03"
) -> dict[int, pd.Timedelta]:
    """
    Return the time delay for a given detector.

    Detector id's are positioned in alternating viewing angle.

    Even detectors capture earlier, odd detectors later.
    Check page 41: https://sentiwiki.copernicus.eu/__attachments/1692737/S2-PDGS-CS-DI-PSD%20-%20S2%20Product%20Specification%20Document%202024%20-%2015.0.pdf?inst-v=e48c493c-f3ee-4a19-8673-f60058308b2a.

    This function checks the DATASTRIP xml to find the reference times used
    for intializing the offset. Currently it calculates the average time for a certain
    band_id, and then returns the offset between the detector_id time and the average
    time. (Unsure whether average is actually correct usage)

    Parameters
    ----------
    datastrip_metadata_path : str
        The location of the DATASTRIP xml file
    band : str, optional
        Spectral band to use for geometry parsing. Default is "B03".

    Returns
    -------
    dict[int, pd.Timedelta]
        Time offset for each detector ID (1 to 12) as a dictionary.
    """
    band_id = str(_band_id(band))

    # Parse XML
    tree = ET.parse(datastrip_metadata_path)
    root = tree.getroot()

    ns = root[0].tag.split("}")[0][1:]

    time_information_element = root.find(
        f".//{{{ns}}}Image_Data_Info/Sensor_Configuration/Time_Stamp"
    )
    if time_information_element is None:
        raise ValueError("Time_Stamp element not found in DATASTRIP metadata")

    cband = next((c for c in time_information_element if c.get("bandId") == band_id), None)
    if cband is None:
        raise ValueError(f"Band ID {band_id} not found in Time_Stamp element")

    delays = {}
    for detector in cband:
        detector_id = detector.get("detectorId")
        if detector_id is None:
            continue

        gps_time_elem = detector.find("GPS_TIME")
        if gps_time_elem is None or gps_time_elem.text is None:
            continue

        # Convert detector_id to int and store the GPS time
        delays[int(detector_id)] = gps_time_elem.text

    if not delays:
        raise ValueError(f"No GPS times found for band {band_id}")

    return _calculate_timedeltas(delays)


# -----------------------------------------------------------------------------------
# Time helper functions


def gps_to_utc(gps_time: datetime.datetime) -> datetime.datetime:
    """Convert GPS time (datetime object) to UTC time.

    https://gssc.esa.int/navipedia/index.php/Transformations_between_Time_Systems
    """

    gps_tai_offset = datetime.timedelta(seconds=19)
    utc_tai_offset = datetime.timedelta(seconds=37)

    # Convert GPS time to UTC
    return gps_time + gps_tai_offset - utc_tai_offset


def _calculate_average_time(times: Collection[datetime.datetime]) -> datetime.datetime:
    """Return the average time from a list of times."""
    # Compute the average time
    avg_timestamp = sum(t.timestamp() for t in times) / len(times)
    return datetime.datetime.fromtimestamp(avg_timestamp)


def _calculate_timedeltas(detector_times: dict[int, str]) -> dict[int, pd.Timedelta]:
    """Calculate the time difference between a detector and the average time."""
    detector_times_dt = {
        detector_id: datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f")
        for detector_id, time_str in detector_times.items()
    }

    avg_time = _calculate_average_time(detector_times_dt.values())
    return {
        detector_id: pd.Timedelta(det_time - avg_time)
        for detector_id, det_time in detector_times_dt.items()
    }


# -----------------------------------------------------------------------------------
# Viewing angle correction helper functions


def process_pixel(
    pixel_val: int,
    pixel_location: tuple[int, int],
    image_shape: tuple[int, ...],
    azimuth_dict: dict[str, np.ndarray],
) -> float:
    """Map a pixel value and location to an azimuth value."""
    # Convert dict keys to integers once
    available_detectors = sorted(int(k) for k in azimuth_dict)
    min_det = available_detectors[0]
    max_det = available_detectors[-1]

    # Inside your loop or function:
    pixel_val = int(pixel_val)  # Ensure it's an integer
    pixel_val = max(min(pixel_val, max_det), min_det)  # Clip to valid range
    azi_array = azimuth_dict[str(pixel_val)]

    # remap the pixel location to the 23x23 grid
    i, j = pixel_location
    H, W = image_shape
    low_res_height, low_res_width = azi_array.shape

    # Map (i,j) from high-res to low-res pixel coordinates
    low_res_y = int(i * low_res_height / H)
    low_res_x = int(j * low_res_width / W)

    # Clamp to bounds (just in case)
    low_res_y = min(low_res_y, low_res_height - 1)
    low_res_x = min(low_res_x, low_res_width - 1)

    # Get azimuth value at mapped pixel
    return azi_array[low_res_y, low_res_x]


def extrapolate_array(array: np.ndarray) -> np.ndarray:
    """Extrapolate NaN values in a 2D azimuth array using linear interpolation/extrapolation."""
    # Get the shape
    h, w = array.shape

    # Meshgrid of coordinates
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    # Mask for valid (non-NaN) values
    mask = ~np.isnan(array)

    # Known points and their values
    known_points = np.stack((xx[mask], yy[mask]), axis=-1)
    known_values = array[mask]

    # Points to interpolate (includes all)
    all_points = np.stack((xx.ravel(), yy.ravel()), axis=-1)

    if np.unique(known_points[:, 0]).size < 2 or np.unique(known_points[:, 1]).size < 2:
        # not enough variation in x or y — use nearest neighbor directly
        interpolated = griddata(known_points, known_values, all_points, method="nearest")
    else:
        # Try linear, fallback to nearest
        interpolated = griddata(known_points, known_values, all_points, method="linear")
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            interpolated[nan_mask] = griddata(
                known_points, known_values, all_points[nan_mask], method="nearest"
            )

    return interpolated.reshape((h, w))
