"""ATC Flight Plan Parser."""

import datetime
import io
import re
import xml.etree.ElementTree as ET
from typing import IO, Any, AnyStr

import pandas as pd

from pycontrails.core import flight
from pycontrails.physics import units


def to_atc_plan(plan: dict[str, Any]) -> str:
    """Write dictionary from :func:`parse_atc_plan` as ATC flight plan string.

    Parameters
    ----------
    plan: dict[str, Any]
        Dictionary representation of ATC flight plan returned from :func:`parse_atc_plan`.

    Returns
    -------
    str
        ATC flight plan string conforming to ICAO Doc 4444-ATM/501

    See Also
    --------
    :func:`parse_atc_plan`
    """
    ret = f"(FPL-{plan['callsign']}-{plan['flight_rules']}"
    ret += f"{plan['type_of_flight']}\n"
    ret += "-"
    if "number_aircraft" in plan and plan["number_aircraft"] <= 10:
        ret += plan["number_aircraft"]
    ret += f"{plan['type_of_aircraft']}/{plan['wake_category']}-"
    ret += f"{plan['equipment']}/{plan['transponder']}\n"
    ret += f"-{plan['departure_icao']}{plan['time']}\n"
    ret += f"-{plan['speed_type']}{plan['speed']}{plan['level_type']}"
    ret += f"{plan['level']} {plan['route']}\n"
    if "destination_icao" in plan and "duration" in plan:
        ret += f"-{plan['destination_icao']}{plan['duration']}"
    if "alt_icao" in plan:
        ret += f" {plan['alt_icao']}"
    if "second_alt_icao" in plan:
        ret += f" {plan['second_alt_icao']}"
    ret += "\n"
    ret += f"-{plan['other_info']}"
    if "supplementary_info" in plan:
        ret += "\n-"
        ret += " ".join([f"{i[0]}/{i[1]}" for i in plan["supplementary_info"].items()])

    ret += ")"

    if ret[-1] == "\n":
        ret = ret[:-1]

    return ret


def parse_atc_plan(atc_plan: str) -> dict[str, str]:
    """Parse an ATC flight plan string into a dictionary.

    The route string is not converted to lat/lon in this process.

    Parameters
    ----------
    atc_plan : str
        An ATC flight plan string conforming to ICAO Doc 4444-ATM/501 (Appendix 2)

    Returns
    -------
    dict[str, str]
        A dictionary consisting of parsed components of the ATC flight plan.
        A full ATC plan will contain the keys:

        - ``callsign``: ICAO flight callsign
        - ``flight_rules``: Flight rules ("I", "V", "Y", "Z")
        - ``type_of_flight``: Type of flight ("S", "N", "G", "M", "X")
        - ``number_aircraft``: The number of aircraft, if more than one
        - ``type_of_aircraft``: ICAO aircraft type
        - ``wake_category``: Wake turbulence category
        - ``equipment``: Radiocommunication, navigation and approach aid equipment and capabilities
        - ``transponder``: Surveillance equipment and capabilities
        - ``departure_icao``: ICAO departure airport
        - ``time``: Estimated off-block (departure) time (UTC)
        - ``speed_type``: Speed units ("K": km / hr, "N": knots)
        - ``speed``: Cruise true airspeed in ``speed_type`` units
        - ``level_type``: Level units ("F", "S", "A", "M")
        - ``level``: Cruise level
        - ``route``: Route string
        - ``destination_icao``: ICAO destination airport
        - ``duration``: The total estimated elapsed time for the flight plan
        - ``alt_icao``: ICAO alternate destination airport
        - ``second_alt_icao``: ICAO second alternate destination airport
        - ``other_info``: Other information
        - ``supplementary_info``: Supplementary information

    References
    ----------
    - https://applications.icao.int/tools/ATMiKIT/story_content/external_files/story_content/external_files/DOC%204444_PANS%20ATM_en.pdf

    See Also
    --------
    :func:`to_atc_plan`
    """
    atc_plan = atc_plan.replace("\r", " ")
    atc_plan = atc_plan.replace("\n", " ")
    atc_plan = atc_plan.upper()
    atc_plan = atc_plan.strip()

    if len(atc_plan) == 0:
        raise ValueError("Empty or invalid flight plan")

    atc_plan = atc_plan.replace("(FPL", "")
    atc_plan = atc_plan.replace(")", "")
    atc_plan = atc_plan.replace("--", "-")

    basic = atc_plan.split("-")

    flightplan: dict[str, Any] = {}

    # Callsign
    if len(basic) > 1:
        flightplan["callsign"] = basic[1]

    # Flight Rules
    if len(basic) > 2:
        flightplan["flight_rules"] = basic[2][0]
        flightplan["type_of_flight"] = basic[2][1]

    # Aircraft
    if len(basic) > 3:
        aircraft = basic[3].split("/")
        matches = re.match(r"(\d{1})(\S{3,4})", aircraft[0])
        groups = matches.groups() if matches else ()

        if matches and len(groups) > 2:
            flightplan["number_aircraft"] = groups[1]
            flightplan["type_of_aircraft"] = groups[2]
        else:
            flightplan["type_of_aircraft"] = aircraft[0]

        if len(aircraft) > 1:
            flightplan["wake_category"] = aircraft[1]

    # Equipment
    if len(basic) > 4:
        equip = basic[4].split("/")
        flightplan["equipment"] = equip[0]
        if len(equip) > 1:
            flightplan["transponder"] = equip[1]

    # Dep. airport info
    if len(basic) > 5:
        matches = re.match(r"(\D*)(\d*)", basic[5])
        groups = matches.groups() if matches else ()

        if groups:
            flightplan["departure_icao"] = groups[0]
        if len(groups) > 1:
            flightplan["time"] = groups[1]

    # Speed and route info
    if len(basic) > 6:
        matches = re.match(r"(\D*)(\d*)(\D*)(\d*)", basic[6])
        groups = matches.groups() if matches else ()

        # match speed and level
        if groups:
            flightplan["speed_type"] = groups[0]
            if len(groups) > 1:
                flightplan["speed"] = groups[1]
            if len(groups) > 2:
                flightplan["level_type"] = groups[2]
            if len(groups) > 3:
                flightplan["level"] = groups[3]

            flightplan["route"] = basic[6][len("".join(groups)) :].strip()
        else:
            flightplan["route"] = basic[6].strip()

    # Dest. airport info
    if len(basic) > 7:
        matches = re.match(r"(\D{4})(\d{4})", basic[7])
        groups = matches.groups() if matches else ()

        if groups:
            flightplan["destination_icao"] = groups[0]
        if len(groups) > 1:
            flightplan["duration"] = groups[1]

        matches = re.match(r"(\D{4})(\d{4})(\s{1})(\D{4})", basic[7])
        groups = matches.groups() if matches else ()

        if len(groups) > 3:
            flightplan["alt_icao"] = groups[3]

        matches = re.match(r"(\D{4})(\d{4})(\s{1})(\D{4})(\s{1})(\D{4})", basic[7])
        groups = matches.groups() if matches else ()

        if len(groups) > 5:
            flightplan["second_alt_icao"] = groups[5]

    # Other info
    if len(basic) > 8:
        info = basic[8]
        idx = info.find("DOF")
        if idx != -1:
            flightplan["departure_date"] = info[idx + 4 : idx + 10]

        flightplan["other_info"] = info.strip()

    # Supl. Info
    if len(basic) > 9:
        sup_match = re.findall(r"(\D{1}[\/]{1})", basic[9])
        if sup_match:
            suplInfo = {}
            for i in range(len(sup_match) - 1):
                this_key = sup_match[i]
                this_idx = basic[9].find(this_key)

                next_key = sup_match[i + 1]
                next_idx = basic[9].find(next_key)

                val = basic[9][this_idx + 2 : next_idx - 1]
                suplInfo[this_key[0]] = val

            last_key = sup_match[-1]
            last_idx = basic[9].find(last_key)
            suplInfo[last_key[0]] = basic[9][last_idx + 2 :]

            flightplan["supplementary_info"] = suplInfo

    return flightplan


def parse_ofp_xml(raw_xml: AnyStr | IO[AnyStr]) -> flight.Flight:
    """Parse an ARINC 633 Operational Flight Plan (OFP) XML into a Flight object.

    Extracts waypoint-level information such as latitude, longitude, altitude, and
    time for the main flight plan (departure, waypoints, arrival) to construct a
    :class:`Flight` instance.

    Parameters
    ----------
    raw_xml : str | bytes | IO[str] | IO[bytes]
        String, bytes, or file-like object containing the ARINC 633 XML data.

    Returns
    -------
    Flight
        A :class:`Flight` instance containing the parsed waypoints.
    """
    if isinstance(raw_xml, bytes):
        raw_xml = io.BytesIO(raw_xml)
    elif isinstance(raw_xml, str):
        raw_xml = io.StringIO(raw_xml)
    tree = ET.parse(raw_xml)
    root = tree.getroot()

    flight_el = _find_or_error(root, ".//{*}Flight")
    dep_dt = pd.to_datetime(flight_el.get("scheduledTimeOfDeparture"), utc=True)

    waypoints = [_parse_waypoint(wp, dep_dt) for wp in root.findall("./{*}Waypoints/{*}Waypoint")]
    df = pd.DataFrame(waypoints).drop_duplicates(subset="time", keep="last")
    if df.empty:
        raise ValueError("No waypoints found in ARINC 633 XML.")

    attrs: dict[str, str] = {}

    if val := root.findtext(".//{*}FlightKeyIdentifier"):
        attrs["flight_id"] = val
    if val := root.findtext(".//{*}CommercialFlightNumber"):
        attrs["flight_number"] = val
    if val := root.findtext(".//{*}AircraftICAOType"):
        attrs["aircraft_type"] = val
    if val := root.findtext(".//{*}DepartureAirport/{*}AirportICAOCode"):
        attrs["departure_airport"] = val
    if val := root.findtext(".//{*}ArrivalAirport/{*}AirportICAOCode"):
        attrs["arrival_airport"] = val

    aircraft_el = root.find(".//{*}Aircraft")
    if aircraft_el is not None and (val := aircraft_el.get("aircraftRegistration")):
        attrs["tail_number"] = val

    return flight.Flight(data=df, attrs=attrs)


def _parse_waypoint(wp: ET.Element, departure: pd.Timestamp) -> dict[str, Any]:
    """Parse an ARINC 633 <Waypoint> element."""
    coord = _find_or_error(wp, "{*}Coordinates")

    return {
        "waypoint_name": wp.get("waypointId") or wp.get("waypointName"),
        "country_code": wp.get("countryICAOCode"),
        "latitude": float(coord.attrib["latitude"]) / 3600.0,
        "longitude": float(coord.attrib["longitude"]) / 3600.0,
        "altitude": _parse_altitude_m(wp),
        "time": _parse_time(wp, departure),
    }


def _find_or_error(el: ET.Element, path: str) -> ET.Element:
    result = el.find(path)
    if result is None:
        raise ValueError(f"Could not find {path} in {ET.tostring(el, encoding='unicode')[:200]}")
    return result


def _parse_time(wp: ET.Element, departure: datetime.datetime) -> datetime.datetime:
    """Parse the time from an ARINC 633 <Waypoint> element."""
    cft = wp.find(".//{*}CumulatedFlightTime/{*}EstimatedTime/{*}Value")
    if cft is not None and cft.text:
        return departure + pd.to_timedelta(cft.text)

    tow = wp.find(".//{*}TimeOverWaypoint/{*}EstimatedTime/{*}Value")
    if tow is not None and tow.text:
        return pd.to_datetime(tow.text, utc=True)

    if wp.get("sequenceId") == "1":
        # For the first waypoint (usually the departure airport) we can fall
        # back to the flights departure time.
        return departure

    raise ValueError(f"No time found for waypoint {ET.tostring(wp, encoding='unicode')}")


def _parse_altitude_m(wp: ET.Element) -> float:
    """Parse the altitude in meters from an ARINC 633 <Waypoint> element."""
    alt = wp.find(".//{*}EstimatedAltitude/{*}Value")
    if alt is None or alt.text is None:
        if wp.get("sequenceId") == "1":
            # For the first waypoint (usually the departure airport) we can
            # fall back to 0m altitude.
            return 0.0

        raise ValueError(f"No altitude found for waypoint {ET.tostring(wp, encoding='unicode')}")

    unit = alt.get("unit")
    val = float(alt.text)
    mult = 1.0

    if not unit:
        raise ValueError(f"No unit in altitude in {ET.tostring(wp, encoding='unicode')[:200]}")

    if "/" in unit:  # Convert units like "ft/100".
        unit, mult_str = unit.split("/")
        mult = float(mult_str)

    if unit == "ft":
        return units.ft_to_m(val) * mult

    if unit == "m":
        return val * mult

    raise ValueError(f"Unknown unit: {unit} in {ET.tostring(wp, encoding='unicode')[:200]}")
