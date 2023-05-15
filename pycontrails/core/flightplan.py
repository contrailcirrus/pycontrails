"""ATC Flight Plan Parser."""

import re
from typing import Any


def flightplan_dict_to_str(flightplan: dict[str, Any]) -> str:
    r"""Convert a dictionary of an ATC flight plan to a string.

    The string will conform to the ICAO Doc 4444 standard.


    Parameters
    ----------
    flightplan: dict[str,Any]
        A dictionary consisting of parsed components of the ATC flight plan.

    Returns
    -------
    flightplan_str : str
        An ATC flight plan string conforming to ICAO Doc 4444-ATM/501
    """
    ret = f'(FPL-{flightplan["callsign"]}-{flightplan["flight_rules"]}'
    ret += f'{flightplan["type_of_flight"]}\n'
    ret += "-"
    if "number" in flightplan and flightplan["number"] <= 10:
        ret += flightplan["number"]
    ret += f'{flightplan["type_of_aircraft"]}/{flightplan["wake_cat"]}-'
    ret += f'{flightplan["equipment"]}/{flightplan["transponder"]}\n'
    ret += f'-{flightplan["dep_icao"]}{flightplan["time"]}\n'
    ret += f'-{flightplan["speed_type"]}{flightplan["speed"]}{flightplan["level_type"]}'
    ret += f'{flightplan["level"]} {flightplan["route"]}\n'
    if "dest_icao" in flightplan and "ttl_eeet" in flightplan:
        ret += f'-{flightplan["dest_icao"]}{flightplan["ttl_eeet"]}'
    if "altn_icao" in flightplan:
        ret += f' {flightplan["altn_icao"]}'
    if "second_altn_icao" in flightplan:
        ret += f' {flightplan["second_altn_icao"]}'
    ret += "\n"
    ret += f'-{flightplan["other_info"]})\n'
    if "supl_info" in flightplan:
        ret += " ".join([f"{i[0]}/{i[1]}" for i in flightplan["supl_info"].items()])

    if ret[-1] == "\n":
        ret = ret[:-1]

    return ret


def flightplan_str_to_dict(flightplan_str: str) -> dict[str, Any]:
    r"""Parse an ATC flight plan string into a dictionary.

    The route string is not converted to lat/lon in this process.

    Parameters
    ----------
    flightplan_str : str
        An ATC flight plan string conforming to ICAO Doc 4444-ATM/501

    Returns
    -------
    dict[str,Any]
        A dictionary consisting of parsed components of the ATC flight plan.
    """
    flightplan_str = flightplan_str.replace("\r", "")
    flightplan_str = flightplan_str.replace("\n", "")
    flightplan_str = flightplan_str.upper()
    flightplan_str = flightplan_str.strip()

    if len(flightplan_str) == 0:
        raise ValueError("Empty or invalid flight plan")

    flightplan_str = flightplan_str.replace("(FPL", "")
    flightplan_str = flightplan_str.replace(")", "")
    flightplan_str = flightplan_str.replace("--", "-")

    basic = flightplan_str.split("-")

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
        matches = re.match("(\d{1})(\S{3,4})", aircraft[0])
        if matches:
            groups = matches.groups()
        else:
            groups = ()

        if matches and len(groups) > 2:
            flightplan["number"] = groups[1]
            flightplan["type_of_aircraft"] = groups[2]
        else:
            flightplan["type_of_aircraft"] = aircraft[0]

        if len(aircraft) > 1:
            flightplan["wake_cat"] = aircraft[1]

    # Equipment
    if len(basic) > 4:
        equip = basic[4].split("/")
        flightplan["equipment"] = equip[0]
        if len(equip) > 1:
            flightplan["transponder"] = equip[1]

    # Dep. airport info
    if len(basic) > 5:
        matches = re.match("(\D*)(\d*)", basic[5])
        if matches:
            groups = matches.groups()
        else:
            groups = ()

        if len(groups) > 0:
            flightplan["dep_icao"] = groups[0]
        if len(groups) > 1:
            flightplan["time"] = groups[1]

    # Speed and route info
    if len(basic) > 6:
        matches = re.match("(\D*)(\d*)(\D*)(\d*)", basic[6])
        if matches:
            groups = matches.groups()
        else:
            groups = ()

        # match speed and level
        if len(groups) > 0:
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
        matches = re.match("(\D{4})(\d{4})", basic[7])
        if matches:
            groups = matches.groups()
        else:
            groups = ()

        if len(groups) > 0:
            flightplan["dest_icao"] = groups[0]
        if len(groups) > 1:
            flightplan["ttl_eeet"] = groups[1]

        matches = re.match("(\D{4})(\d{4})(\s{1})(\D{4})", basic[7])
        if matches:
            groups = matches.groups()
        else:
            groups = ()

        if len(groups) > 3:
            flightplan["altn_icao"] = groups[3]

        matches = re.match("(\D{4})(\d{4})(\s{1})(\D{4})(\s{1})(\D{4})", basic[7])
        if matches:
            groups = matches.groups()
        else:
            groups = ()

        if len(groups) > 5:
            flightplan["second_altn_icao"] = groups[5]

    # Other info
    if len(basic) > 8:
        flightplan["other_info"] = basic[8]

    # Supl. Info
    if len(basic) > 9:
        sup_match = re.findall("(\D{1}[\/]{1})", basic[9])
        if len(sup_match) > 0:
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

            flightplan["supl_info"] = suplInfo
    return flightplan
