"""Tools for working with Sentinel-2 and Landsat data."""

from pycontrails.datalib.leo_utils.correction import estimate_scan_time, scan_angle_correction

__all__ = ["estimate_scan_time", "scan_angle_correction"]
