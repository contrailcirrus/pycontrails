"""Support for parsing the Himawari-8/9 header structure.

See the latest user guide for details:
https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/hsd_sample/HS_D_users_guide_en_v13.pdf

If that link breaks, find the correct link here:
https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/sample_hisd.html
"""

import struct
from typing import Any, TypedDict


class _HeaderBlock(TypedDict):
    """An individual Himawari header block."""

    name: str
    fields: list[tuple[str, str, int, int, str | None]]


HEADER_STRUCT_SCHEMA: dict[int, _HeaderBlock] = {
    1: {
        "name": "basic_information",
        "fields": [
            ("header_block_number", "I1", 1, 1, None),
            ("block_length", "I2", 2, 1, None),
            ("total_header_blocks", "I2", 2, 1, None),
            ("byte_order", "I1", 1, 1, None),
            ("satellite_name", "C", 1, 16, None),
            ("processing_center_name", "C", 1, 16, None),
            ("observation_area", "C", 1, 4, None),
            ("other_obs_info", "C", 1, 2, None),
            ("obs_timeline", "I2", 2, 1, None),
            ("obs_start_time", "R8", 8, 1, None),
            ("obs_end_time", "R8", 8, 1, None),
            ("file_creation_time", "R8", 8, 1, None),
            ("total_header_length", "I4", 4, 1, None),
            ("total_data_length", "I4", 4, 1, None),
            ("quality_flag_1", "I1", 1, 1, None),
            ("quality_flag_2", "I1", 1, 1, None),
            ("quality_flag_3", "I1", 1, 1, None),
            ("quality_flag_4", "I1", 1, 1, None),
            ("file_format_version", "C", 1, 32, None),
            ("file_name", "C", 1, 128, None),
            ("spare", "C", 40, 1, None),
        ],
    },
    2: {
        "name": "data_information",
        "fields": [
            ("header_block_number", "I1", 1, 1, None),
            ("block_length", "I2", 2, 1, None),
            ("bits_per_pixel", "I2", 2, 1, None),
            ("num_columns", "I2", 2, 1, None),
            ("num_lines", "I2", 2, 1, None),
            ("compression_flag", "I1", 1, 1, None),
            ("spare", "C", 40, 1, None),
        ],
    },
    3: {
        "name": "projection_information",
        "fields": [
            ("header_block_number", "I1", 1, 1, None),
            ("block_length", "I2", 2, 1, None),
            ("sub_lon", "R8", 8, 1, None),
            ("cfac", "I4", 4, 1, None),
            ("lfac", "I4", 4, 1, None),
            ("coff", "R4", 4, 1, None),
            ("loff", "R4", 4, 1, None),
            ("dist_from_earth_center", "R8", 8, 1, None),
            ("equatorial_radius", "R8", 8, 1, None),
            ("polar_radius", "R8", 8, 1, None),
            ("rec_minus_rpol_div_req_sq", "R8", 8, 1, None),
            ("rpol_sq_div_req_sq", "R8", 8, 1, None),
            ("req_sq_div_rpol_sq", "R8", 8, 1, None),
            ("coeff_for_sd", "R8", 8, 1, None),
            ("resampling_types", "I2", 2, 1, None),
            ("resampling_size", "I2", 2, 1, None),
            ("spare", "C", 40, 1, None),
        ],
    },
    4: {
        "name": "navigation_information",
        "fields": [
            ("header_block_number", "I1", 1, 1, None),
            ("block_length", "I2", 2, 1, None),
            ("nav_info_time", "R8", 8, 1, None),
            ("ssp_longitude", "R8", 8, 1, None),
            ("ssp_latitude", "R8", 8, 1, None),
            ("dist_from_earth_center_to_sat", "R8", 8, 1, None),
            ("nadir_longitude", "R8", 8, 1, None),
            ("nadir_latitude", "R8", 8, 1, None),
            ("sun_position", "R8", 8, 3, None),
            ("moon_position", "R8", 8, 3, None),
            ("spare", "C", 40, 1, None),
        ],
    },
    5: {
        "name": "calibration_information",
        "fields": [
            ("header_block_number", "I1", 1, 1, None),
            ("block_length", "I2", 2, 1, None),
            ("band_number", "I2", 2, 1, None),
            ("central_wavelength", "R8", 8, 1, None),
            ("valid_bits_per_pixel", "I2", 2, 1, None),
            ("count_error_pixels", "I2", 2, 1, None),
            ("count_outside_scan_area", "I2", 2, 1, None),
            ("gain", "R8", 8, 1, None),
            ("constant", "R8", 8, 1, None),
            ("c0", "R8", 8, 1, "IR-BANDS"),
            ("c1", "R8", 8, 1, "IR-BANDS"),
            ("c2", "R8", 8, 1, "IR-BANDS"),
            ("C0", "R8", 8, 1, "IR-BANDS"),
            ("C1", "R8", 8, 1, "IR-BANDS"),
            ("C2", "R8", 8, 1, "IR-BANDS"),
            ("speed_of_light", "R8", 8, 1, "IR-BANDS"),
            ("planck_constant", "R8", 8, 1, "IR-BANDS"),
            ("boltzmann_constant", "R8", 8, 1, "IR-BANDS"),
            ("spare", "C", 40, 1, "IR-BANDS"),
            ("coeff_c_prime", "R8", 8, 1, "NIR-BANDS"),
            ("spare", "C", 104, 1, "NIR-BANDS"),
        ],
    },
    6: {
        "name": "inter_calibration_information",
        "fields": [
            ("header_block_number", "I1", 1, 1, None),
            ("block_length", "I2", 2, 1, None),
            ("gsics_intercept", "R8", 8, 1, None),
            ("gsics_slope", "R8", 8, 1, None),
            ("gsics_quad", "R8", 8, 1, None),
            ("rad_bias_standard", "R8", 8, 1, None),
            ("uncert_rad_bias", "R8", 8, 1, None),
            ("rad_standard_scene", "R8", 8, 1, None),
            ("gsics_validity_start", "R8", 8, 1, None),
            ("gsics_validity_end", "R8", 8, 1, None),
            ("rad_validity_upper", "R4", 4, 1, None),
            ("rad_validity_lower", "R4", 4, 1, None),
            ("gsics_file_name", "C", 1, 128, None),
            ("spare", "C", 56, 1, None),
        ],
    },
    7: {
        "name": "segment_information",
        "fields": [
            ("header_block_number", "I1", 1, 1, None),
            ("block_length", "I2", 2, 1, None),
            ("total_segments", "I1", 1, 1, None),
            ("segment_seq_number", "I1", 1, 1, None),
            ("first_line_number", "I2", 2, 1, None),
            ("spare", "C", 40, 1, None),
        ],
    },
    8: {
        "name": "navigation_correction_information",
        "fields": [
            ("header_block_number", "I1", 1, 1, None),
            ("block_length", "I2", 2, 1, None),
            ("center_col_rot", "R4", 4, 1, None),
            ("center_line_rot", "R4", 4, 1, None),
            ("rot_correction", "R8", 8, 1, None),
            ("num_corr_data", "I2", 2, 1, None),
            # The following fields are variable and depend on 'num_corr_data'
            # These are not currently parsed
            # ("line_after_rot", "I2", 2, 1, None),
            # ("shift_amount_col", "R4", 4, 1, None),
            # ("shift_amount_line", "R4", 4, 1, None),
            # ("spare", "C", 40, 1, None),
        ],
    },
    9: {
        "name": "observation_time_information",
        "fields": [
            ("header_block_number", "I1", 1, 1, None),
            ("block_length", "I2", 2, 1, None),
            ("num_obs_times", "I2", 2, 1, None),
            # The following fields are variable and depend on 'num_obs_times'
            # These are not currently parsed
            # ("line_number", "I2", 2, 1, None),
            # ("obs_time", "R8", 8, 1, None),
            # ("spare", "C", 40, 1, None),
        ],
    },
    10: {
        "name": "error_information",
        "fields": [
            ("header_block_number", "I1", 1, 1, None),
            ("block_length", "I4", 4, 1, None),
            ("num_error_data", "I2", 2, 1, None),
            # The following fields are variable and depend on 'num_error_data'
            # These are not currently parsed
            # ("line_number", "I2", 2, 1, None),
            # ("num_error_pixels", "I2", 2, 1, None),
            # ("spare", "C", 40, 1, None),
        ],
    },
    11: {
        "name": "spare",
        "fields": [
            ("header_block_number", "I1", 1, 1, None),
            ("block_length", "I2", 2, 1, None),
            ("spare", "C", 256, 1, None),
        ],
    },
}


def parse_himawari_header(content: bytes) -> dict[str, dict[str, Any]]:
    """Parse the Himawari header data.

    Skips variable-length fields and spares.
    """
    out = {}
    offset = 0

    # everything is little-endian (see the byte_order field in block #1)
    typ_map = {
        "I1": "B",
        "I2": "H",
        "I4": "I",
        "R4": "f",
        "R8": "d",
        "C": "s",
    }

    for block_num, block_info in HEADER_STRUCT_SCHEMA.items():
        offset_block_start = offset  # blocks 8, 9, 10 are dynamic
        block_data: dict[str, Any] = {}
        block_name = block_info["name"]
        fields = block_info["fields"]
        block_length_value: int | None = None

        for name, typ, size, count, cond in fields:
            if block_num == 5 and cond:  # deal with dynamic block 5
                band_number = block_data["band_number"]
                if cond == "IR-BANDS" and band_number <= 6:
                    continue
                if cond == "NIR-BANDS" and band_number >= 7:
                    continue

            if name == "spare":  # skip spare fields
                offset += size * count
                continue

            fmt = typ_map[typ]
            if typ == "C":
                raw = struct.unpack_from(f"{size * count}s", content, offset)[0]
                value = raw.rstrip(b"\x00").decode("ascii", errors="ignore")
            else:
                value = struct.unpack_from(f"{count}{fmt}", content, offset)
                if count == 1:
                    value = value[0]

            block_data[name] = value
            offset += size * count

            if name == "block_length":
                block_length_value = value

        if block_length_value is None:
            raise ValueError(f"Missing block_length in {block_name}")
        offset = offset_block_start + block_length_value  # only needed for blocks 8, 9, 10

        out[block_name] = block_data

    return out
