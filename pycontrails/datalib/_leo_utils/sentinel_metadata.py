"""Download and parse Sentinel metadata.
"""

import numpy as np
import xml.etree.ElementTree as ET
import rasterio
from rasterio.enums import Resampling
from skimage.transform import resize
from scipy.interpolate import griddata
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy.typing as npt
import pyproj

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
    "B12": 12
}

def parse_viewing_incidence_angle_by_detector(metadata_path, target_detector_id, target_band_id="2"):
    """
    Read sensor incidence angles from metadata
    
    Parameters:
    - metadata_path (str): Path to the XML file containing TILE metadata.
    - target_detector_id (str): Target Detector_ID. 
    - target_band_id (str): Starts from 0 (e.g. band 2 (blue) = band_id "1")

    Returns:
    - tuple: Zenith Angles, Azimuth Angles ((23x23) numpy array)
    """
    xml_file = metadata_path

    tree = ET.parse(xml_file)
    root = tree.getroot()

    tile_angles_element = root.find('.//Tile_Angles')
    if tile_angles_element is not None:
        for band in tile_angles_element.findall('.//Viewing_Incidence_Angles_Grids'):
            band_id = band.get('bandId')
            detector_id = band.get('detectorId')
            
            if band_id == target_band_id and detector_id == target_detector_id:

                zenith_element = band.find('.//Zenith')
                azimuth_element = band.find('.//Azimuth')
                
                if zenith_element is not None and azimuth_element is not None:
                    # Extract the Values_List from both Zenith and Azimuth
                    zenith_values_list = zenith_element.find('.//Values_List')
                    azimuth_values_list = azimuth_element.find('.//Values_List')
                    
                    # Initialize 2D arrays to hold the values
                    zenith_2d_array = []
                    azimuth_2d_array = []
                    
                    # Extract and store Zenith values in a 2D array
                    if zenith_values_list is not None:
                        zenith_values = zenith_values_list.findall('.//VALUES')
                        for values in zenith_values:
                            # Split the space-separated string into individual values and convert to floats
                            zenith_row = list(map(float, values.text.split()))
                            zenith_2d_array.append(zenith_row)
                    
                    # Extract and store Azimuth values in a 2D array
                    if azimuth_values_list is not None:
                        azimuth_values = azimuth_values_list.findall('.//VALUES')
                        for values in azimuth_values:
                            # Split the space-separated string into individual values and convert to floats
                            azimuth_row = list(map(float, values.text.split()))
                            azimuth_2d_array.append(azimuth_row)
                    
                    return zenith_2d_array, azimuth_2d_array

        else:
            return None, None
    else:
        print("Viewing_Incidence_Angles_Grids element not found.")


def parse_viewing_incidence_angles(metadata_path, target_band_id="2"):
    """
    Read sensor incidence angles from metadata. Returns the total of all detectors
    
    Parameters:
    - xml_file (str): Path to the XML file containing TILE metadata.
    - target_band_id (str): Starts from 0 (e.g. band 2 (blue) = band_id "1")

    Returns:
    - tuple: Zenith Angles, Azimuth Angles ((23x23) numpy array)
    """
    total_zenith = None
    total_azimuth = None

    # loop over all 12 detector id's
    for detector_id in [str(i) for i in range(1, 13)]:
        zen, azi = parse_viewing_incidence_angle_by_detector(metadata_path, detector_id, target_band_id)
        if zen is None or azi is None:
            continue

        # convert to np array
        zen_array = np.array(zen, dtype=np.float64)
        azi_array = np.array(azi, dtype=np.float64)

        # in the first time, initialize to the correct 23x23 shape
        if total_zenith is None:
            total_zenith = np.full(zen_array.shape, np.nan)
            total_azimuth = np.full(azi_array.shape, np.nan)

        # remove NaN values
        mask = ~np.isnan(zen_array)

        total_zenith[mask] = zen_array[mask]
        total_azimuth[mask] = azi_array[mask]

    return total_zenith, total_azimuth


def parse_high_res_detector_mask(metadata_path:str, scale:int=10) -> npt.NDArray[np.integer]:
    """
    Load in the detector mask. Contains a pixel level mask indicating which detector [1-12] 
    captured the pixel. 
    
    Lower the resolution with 'scale' to speed up processing. 
    Scale 1 -> 10m resolution. Scale 10 -> 100m resolution.

    Parameters:
    - metadata_path (str): Path to the file containing metadata. Usually in the 
    {base_url}/GRANULE/{granule_id}/QI_DATA/MSK_DETFOO_B03.jp2 path.
    - scale (int): Indicates by which factor to lower the resolution. 

    Returns:
    - np.ndarray: 2D array of detector IDs (1–12), shape (height, width)
    """
    # Make sure scale is an integer
    scale = int(scale)

    # Load and downsample detector mask
    with rasterio.open(metadata_path) as src:
        # image = src.read(1)
        image = src.read(
            1,
            out_shape=(
                int(src.count),
                int(src.height // scale),
                int(src.width // scale)
            ),
            resampling=Resampling.nearest
        )
        
        # Scale transform for new resolution
        transform = src.transform * src.transform.scale(
            (src.width / image.shape[-1]),
            (src.height / image.shape[-2])
        )

        profile = src.profile
        profile.update({
            'height': image.shape[0],
            'width': image.shape[1],
            'transform': transform
        })

    return image


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


def parse_high_res_viewing_incidence_angles(tile_metadata_path, detector_band_metadata_path, scale=10) -> xr.Dataset:
    """
    Parses and returns high-resolution viewing incidence angles (zenith and azimuth)
    for a given Sentinel-2 tile as an xarray.Dataset.

    Parameters:
    -----------
    tile_metadata_path : str
        Path to the tile-level metadata file (usually MTD_TL.xml).
    detector_band_metadata_path : str
        Path to the detector-specific metadata file (e.g., MTD_DETFOO_B03.jp2).
    scale : int, optional (default=10)
        Desired resolution scale (e.g., 10 for 10m resolution).

    Returns:
    --------
    xarray.Dataset
        Dataset with coordinates ('y', 'x') containing:
        - VZA: View Zenith Angle
        - VAA: View Azimuth Angle

    Raises:
    -------
    ValueError:
        If required data (zenith or azimuth) cannot be parsed.
    """
    try:
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
        high_res_zenith = resize(
            low_res_zenith,
            output_shape=detector_mask.shape,
            order=1,
            mode='edge',
            anti_aliasing=True,
            preserve_range=True
        )

        # Dictionary to store upsampled azimuth data per detector
        low_res_azimuth_dict = {}

        for detector_id in [str(i) for i in range(1, 13)]:
            zen, azi = parse_viewing_incidence_angle_by_detector(tile_metadata_path, detector_id, "2")
            if zen is None or azi is None:
                continue

            azi_array = np.array(azi, dtype=np.float64)
            azi_extrapolated = extrapolate_array(azi_array)

            azi_extrapolated_highres = resize(
                azi_extrapolated,
                output_shape=detector_mask.shape,
                order=1,
                mode='edge',
                anti_aliasing=True,
                preserve_range=True
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
        viewing_angles_ds = xr.Dataset(
            data_vars={
                "VZA": (("y", "x"), high_res_zenith.astype(np.float32)),
                "VAA": (("y", "x"), high_res_azimuth),
            },
            coords={
                "x": x_coords,
                "y": y_coords,
            },
            attrs={
                "title": "High-resolution Viewing Incidence Angles",
                "description": "Zenith and Azimuth angles resampled to high resolution from Sentinel-2 metadata",
                "scale": scale,
                "extent": extent
            }
        )

        return viewing_angles_ds

    except Exception as e:
        raise RuntimeError(f"Failed to parse high-resolution viewing incidence angles: {e}")


def parse_ephemeris_sentinel(datatsrip_metadata_path: str) -> pd.DataFrame:
    """Return the ephemeris data from the DATASTRIP xml file

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

    records = []
    for elem in satellite_ancillary_data:
        if elem.tag.endswith("Ephemeris"):
            gps_points_list = elem.find("GPS_Points_List")
            for point in gps_points_list:
                gps_time_elem = point.find(".//GPS_TIME")
                position_elem = point.find(".//POSITION_VALUES")

                gps_time = datetime.strptime(gps_time_elem.text, "%Y-%m-%dT%H:%M:%S")

                # Convert GPS to UTC time as there is a few seconds between them
                utc_time = gps_to_utc(gps_time).replace(tzinfo=timezone.utc)

                # Parse positions in ECEF coordinate system
                x, y, z = map(float, position_elem.text.split())

                records.append({
                    "EPHEMERIS_TIME": pd.Timestamp(utc_time).tz_localize(None),
                    "EPHEMERIS_ECEF_X": x / 1000,
                    "EPHEMERIS_ECEF_Y": y / 1000,
                    "EPHEMERIS_ECEF_Z": z / 1000
                })

    return pd.DataFrame(records)
    

def parse_sentinel_crs(granule_metadata_path: str) -> pyproj.CRS:
    tree = ET.parse(granule_metadata_path)
    root = tree.getroot()
    
    # Get the namespace of the XML file
    ns = root[0].tag.split("}")[0][1:]

    # Find the CS code in the xml file
    epsg_code = root.find(f".//{{{ns}}}Geometric_Info/Tile_Geocoding/HORIZONTAL_CS_CODE").text

    return pyproj.CRS.from_string(epsg_code)


def parse_sensing_time(granule_metadata_path: str) -> pd.Timestamp:
    tree = ET.parse(granule_metadata_path)
    root = tree.getroot()
    
    # Get the namespace of the XML file
    ns = root[0].tag.split("}")[0][1:]

    # Find the CS code in the xml file
    sensing_time = root.find(f".//{{{ns}}}General_Info/SENSING_TIME").text
    return pd.to_datetime(sensing_time)


def get_detector_id(    
    detector_band_metadata_path: str,
    tile_metadata_path: str,
    x: float,
    y: float,
    band: str = "B03"
) -> int:
    """
    Return the detector ID that captured a given pixel in a Sentinel-2 image.

    Parameters
    ----------
    detector_band_metadata_path : str
        Path to the MSK_DETFOO_Bxx.jp2 detector band mask file.
    tile_metadata_path : str
        Path to the tile metadata XML file (MTD_TL.xml) containing image geometry.
    x : float
        X coordinate (in UTM coordinate system) of the target pixel.
    y : float
        Y coordinate (in UTM coordinate system) of the target pixel.
    band : str, optional
        Spectral band to use for geometry parsing. Default is "B03".

    Returns
    -------
    int : The detector ID (in the range 1–12) that captured the pixel.

    Raises
    ------
    ValueError
        If the (x, y) coordinate is outside the image bounds.
    """
    detector_mask = parse_high_res_detector_mask(detector_band_metadata_path, scale=10)

    height, width = detector_mask.shape

    x_img, y_img = read_image_coordinates(tile_metadata_path, band)
    x_min, x_max = float(x_img.min()), float(x_img.max())
    y_min, y_max = float(y_img.min()), float(y_img.max())

    # Compute resolution
    pixel_width = (x_max - x_min) / width
    pixel_height = (y_max - y_min) / height

    # Convert x, y to column, row
    col = int((x - x_min) / pixel_width)
    row = int((y_max - y) / pixel_height)  # Note: y axis is top-down in images

    if 0 <= row < height and 0 <= col < width:
        return detector_mask[row, col]
    else:
        raise ValueError("Point is outside the image bounds.")


def get_time_delay_detector(datastrip_metadata_path, target_detector_id, band="B03") -> pd.Timedelta:
    """
    Detector id's are positioned in alternating viewing angle. Even detectors capture earlier, odd detectors later.
    Check page 41: https://sentiwiki.copernicus.eu/__attachments/1692737/S2-PDGS-CS-DI-PSD%20-%20S2%20Product%20Specification%20Document%202024%20-%2015.0.pdf?inst-v=e48c493c-f3ee-4a19-8673-f60058308b2a

    This function checks the DATASTRIP xml to find the reference times used for intializing the offset.
    Currently it calculates the average time for a certain band_id, and then returns the offset between the 
    detector_id time and the average time. (Unsure whether average is actually correct usage)

    parameters:
    - target_detector_id (str): Detector ID for which the timedelta needs to be calculated
    - band_id (str): Starts from 0 (e.g. band 2 (blue) = band_id "1")

    returns:
    - timedelta (Datetime Object): time offset for detector. Add to current time.
    """
    if len(target_detector_id) == 1:
        target_detector_id = "0" + target_detector_id

    band_id = str(BAND_ID_MAPPING[band])

    detector_times = []

    # Import and read the XML file
    tree = ET.parse(datastrip_metadata_path)
    root = tree.getroot()

    # Get the namespace of the XML file
    ns = root[0].tag.split("}")[0][1:]

    time_information_element = root.find(f".//{{{ns}}}Image_Data_Info/Sensor_Configuration/Time_Stamp")
    for band in time_information_element:

        bandId = band.get("bandId")
        if bandId == band_id:

            for detector in band:
                detector_id = detector.get('detectorId')
                gps_time = detector.find('GPS_TIME')
                detector_times.append([detector_id, gps_time.text])

    time_difference = calculate_timedelta(detector_times, target_detector_id)
    return pd.to_timedelta(time_difference)

# -----------------------------------------------------------------------------------
# Time helper functions

def gps_to_utc(gps_time: datetime) -> datetime:
    """Convert GPS time (datetime object) to UTC time.
    
    https://gssc.esa.int/navipedia/index.php/Transformations_between_Time_Systems
    """

    gps_tai_offset = timedelta(seconds=19)
    utc_tai_offset = timedelta(seconds=37)

    # Convert GPS time to UTC
    return gps_time + gps_tai_offset - utc_tai_offset


def calculate_average_detector_time(detector_times):
    # Convert string times to datetime objects
    times = [datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f") for _, time_str in detector_times]
    
    # Compute the average time
    avg_timestamp = sum(t.timestamp() for t in times) / len(times)
    avg_time = datetime.fromtimestamp(avg_timestamp)

    return avg_time


def calculate_timedelta(detector_times, target_detector_id) -> timedelta:
    avg_time = calculate_average_detector_time(detector_times)
    
    # Find the time for the target detector ID
    target_time = None
    for detector_id, time_str in detector_times:

        if detector_id == target_detector_id:
            target_time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f")
            break

    if target_time is None:
        raise ValueError(f"Detector ID {target_detector_id} not found")

    # Compute the timedelta
    delta = target_time - avg_time
    return delta

# -----------------------------------------------------------------------------------
# Viewing angle correction helper functions

def process_pixel(pixel_val, pixel_location, image_shape, azimuth_dict):
    # Convert dict keys to integers once
    available_detectors = sorted(int(k) for k in azimuth_dict.keys())
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
    azimuth_val = azi_array[low_res_y, low_res_x]
    
    return azimuth_val


def extrapolate_array(array):
    """
    Extrapolate NaN values in a 2D azimuth array using linear interpolation/extrapolation.
    """
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
        interpolated = griddata(known_points, known_values, all_points, method='nearest')
    else:
        # Try linear, fallback to nearest
        interpolated = griddata(known_points, known_values, all_points, method='linear')
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            interpolated[nan_mask] = griddata(
                known_points, known_values, all_points[nan_mask], method='nearest'
            )

    return interpolated.reshape((h, w))