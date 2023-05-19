"""Test flightplan module."""

from __future__ import annotations

from pycontrails.core import flightplan


def test_flightplan_one() -> None:
    atc_plan_str = (
        "(FPL-GEC8145-IN -B77L/H-SDE2E3FGHIJ3J4J5M1RWXYZ/SB1D1"
        "-EGGL1040 -N0474F360 IMVUR1Z IMVUR N63 SAM N19 ADKIK DCT "
        "MOPAT DCT  LIMRI/M083F360 DCT 51N020W 47N030W/M083F380 40N040W "
        "34N045W  28N050W/M083F400 24N055W 19N060W DCT AMTTO DCT ANU DCT "
        "-PBN/A1B1C1D1L1O1S1S2 NAV/RNVD1E2A1 DAT/SVM DOF/140501 REG/DALFA "
        "EET/OKAC0037 ORBB0052 LTAA0159 UKFV0308 UKOV0333 LUUU0344 UKLV0406"
        "EPWW0427 ESAA0521 EKDK0540 ENOR0557 SEL/DFBH OPR/GEC RVR/200) "
        "-E/0740 P/3 R/E S/ J/ A/WHITE BLUE TAIL"
    )

    fp_dict = flightplan.parse_atc_plan(atc_plan_str)

    assert fp_dict["callsign"] == "GEC8145"
    assert fp_dict["flight_rules"] == "I"
    assert fp_dict["type_of_flight"] == "N"
    assert fp_dict["type_of_aircraft"] == "B77L"
    assert fp_dict["wake_category"] == "H"
    assert fp_dict["equipment"] == "SDE2E3FGHIJ3J4J5M1RWXYZ"
    assert fp_dict["transponder"] == "SB1D1"
    assert fp_dict["departure_icao"] == "EGGL"
    assert fp_dict["time"] == "1040"
    assert fp_dict["speed_type"] == "N"
    assert fp_dict["speed"] == "0474"
    assert fp_dict["level_type"] == "F"
    assert fp_dict["level"] == "360"
    assert (
        fp_dict["route"]
        == "IMVUR1Z IMVUR N63 SAM N19 ADKIK DCT MOPAT DCT  "
        "LIMRI/M083F360 DCT 51N020W 47N030W/M083F380 40N040W 34N045W  28N050W/M083F400 "
        "24N055W 19N060W DCT AMTTO DCT ANU DCT"
    )
    assert fp_dict["other_info"] == "E/0740 P/3 R/E S/ J/ A/WHITE BLUE TAIL"

    assert (
        flightplan.to_atc_plan(fp_dict)
        == "(FPL-GEC8145-IN\n"
        "-B77L/H-SDE2E3FGHIJ3J4J5M1RWXYZ/SB1D1\n"
        "-EGGL1040\n"
        "-N0474F360 IMVUR1Z IMVUR N63 SAM N19 ADKIK DCT MOPAT DCT  "
        "LIMRI/M083F360 DCT 51N020W 47N030W/M083F380 40N040W 34N045W  "
        "28N050W/M083F400 24N055W 19N060W DCT AMTTO DCT ANU DCT\n"
        "\n"
        "-E/0740 P/3 R/E S/ J/ A/WHITE BLUE TAIL)"
    )


def test_flightplan_two() -> None:
    fp_str = "(FPL-N12345-IG-SR22/L-S/S-KSEA1414-N0220F090 DCT-PAEN0600-DOF/170428 RMK/DO NOT POST)"

    fp_dict = flightplan.parse_atc_plan(fp_str)

    assert fp_dict["callsign"] == "N12345"
    assert fp_dict["flight_rules"] == "I"
    assert fp_dict["type_of_flight"] == "G"
    assert fp_dict["type_of_aircraft"] == "SR22"
    assert fp_dict["wake_category"] == "L"
    assert fp_dict["equipment"] == "S"
    assert fp_dict["transponder"] == "S"
    assert fp_dict["departure_icao"] == "KSEA"
    assert fp_dict["time"] == "1414"
    assert fp_dict["speed_type"] == "N"
    assert fp_dict["speed"] == "0220"
    assert fp_dict["level_type"] == "F"
    assert fp_dict["level"] == "090"
    assert fp_dict["route"] == "DCT"
    assert fp_dict["destination_icao"] == "PAEN"
    assert fp_dict["duration"] == "0600"
    assert fp_dict["other_info"] == "DOF/170428 RMK/DO NOT POST"

    assert (
        flightplan.to_atc_plan(fp_dict)
        == "(FPL-N12345-IG\n"
        "-SR22/L-S/S\n"
        "-KSEA1414\n"
        "-N0220F090 DCT\n"
        "-PAEN0600\n"
        "-DOF/170428 RMK/DO NOT POST)"
    )
