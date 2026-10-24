"""Estimated drone-flight inspection cost and duration.

The published panel currently has no structure-length column. ``flight_cost``
therefore recognizes common meter-length column names when Part A adds one and
otherwise falls back to material-level planning medians. Those fallbacks are
explicitly marked ``length estimated``; all returned costs and minutes are
planning estimates, never vendor quotes.
"""

from __future__ import annotations

import math
import re

import pandas as pd

# Meter-valued panel column names, in priority order.
LENGTH_COLUMNS = (
    "length_m",
    "structure_length_m",
    "structure_len_mt_049",
    "structure_length",
    "total_length",
    "length",
)

# Planning medians (meters) used only when the panel/row has no usable length.
# Keys are NBI item 043A material codes. They are cost-model assumptions and
# must remain labeled as estimates in every reason.
MATERIAL_MEDIAN_LENGTH_M = {
    "0": 30.0,  # Other
    "1": 25.0,  # Concrete
    "2": 40.0,  # Concrete continuous
    "3": 45.0,  # Steel
    "4": 65.0,  # Steel continuous
    "5": 30.0,  # Prestressed concrete
    "6": 50.0,  # Prestressed continuous
    "7": 18.0,  # Wood/Timber
    "8": 22.0,  # Masonry
    "9": 30.0,  # Aluminum/iron
}
MATERIAL_NAME_MEDIAN_LENGTH_M = {
    "other": 30.0,
    "concrete": 25.0,
    "concrete continuous": 40.0,
    "steel": 45.0,
    "steel continuous": 65.0,
    "prestressed concrete": 30.0,
    "prestressed continuous": 50.0,
    "wood/timber": 18.0,
    "timber": 18.0,
    "masonry": 22.0,
    "aluminum/iron": 30.0,
}
DEFAULT_LENGTH_M = 30.0

DEPOTS = {
    "Boston": (42.36, -71.06),
    "Worcester": (42.26, -71.80),
    "Springfield": (42.10, -72.59),
}
DEFAULT_TRAVEL_KM = 60.0
EARTH_RADIUS_KM = 6371.0088

BASE_COST_USD = 1500.0
LENGTH_COST_PER_M = 8.0
SPECIAL_CROSSING_MULTIPLIER = 1.4
TRAVEL_COST_PER_KM = 2.0

BASE_FLIGHT_MINUTES = 20.0
FLIGHT_MINUTES_PER_M = 0.5
TRAVEL_MINUTES_PER_KM = 1.2

RAIL_PATTERN = re.compile(r"(?:RR\b|\bRAILROAD\b|\bRAILWAY\b|\bMBTA\b)")
WATER_PATTERN = re.compile(
    r"\b(?:RIV|RIVER|CREEK|BROOK|STREAM|WATER|BAY|POND|LAKE|CANAL|HARBOR)\b"
)


def _value(row, key, default=None):
    if hasattr(row, "get"):
        return row.get(key, default)
    return getattr(row, key, default)


def _number(value):
    try:
        if value is None or pd.isna(value):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _material_code(value) -> str:
    parsed = _number(value)
    if parsed is not None and parsed.is_integer():
        return str(int(parsed))
    return str(value).strip() if value is not None else ""


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _length_m(row):
    for column in LENGTH_COLUMNS:
        length = _number(_value(row, column))
        if length is not None and length > 0:
            return length, False

    # score() injects the observed panel median for this row's material when a
    # partially populated length column exists.
    median = _number(_value(row, "_material_median_length_m"))
    if median is not None and median > 0:
        return median, True

    material = _value(row, "material")
    code = _material_code(material)
    by_code = MATERIAL_MEDIAN_LENGTH_M.get(code)
    if by_code is not None:
        return by_code, True
    by_name = MATERIAL_NAME_MEDIAN_LENGTH_M.get(str(material).strip().lower())
    return (by_name if by_name is not None else DEFAULT_LENGTH_M), True


def _nearest_depot(row):
    lat = _number(_value(row, "lat"))
    lon = _number(_value(row, "lon"))
    if (
        lat is None
        or lon is None
        or not -90 <= lat <= 90
        or not -180 <= lon <= 180
    ):
        return None, DEFAULT_TRAVEL_KM, True

    distances = {
        name: _haversine_km(lat, lon, depot_lat, depot_lon)
        for name, (depot_lat, depot_lon) in DEPOTS.items()
    }
    depot = min(distances, key=distances.get)
    return depot, distances[depot], False


def _special_crossing(row) -> bool:
    crosses = _value(row, "crosses")
    if crosses is None or pd.isna(crosses):
        return False
    text = str(crosses).upper()
    return bool(RAIL_PATTERN.search(text) or WATER_PATTERN.search(text))


def flight_cost(row):
    """Return ``(insp_cost_usd, flight_minutes, reason_fragment)``.

    Cost estimate:

    ``(1500 + 8 * length_m) * crossing_factor + 2 * one_way_travel_km``

    where ``crossing_factor`` is 1.4 for water/rail and 1.0 otherwise.

    Duration estimate:

    ``20 + 0.5 * length_m + 2 * one_way_travel_km * 1.2``

    The factor of two makes travel round-trip. The reason fragment always calls
    the values estimates and identifies estimated length/location inputs.
    """
    length_m, length_estimated = _length_m(row)
    depot, travel_km, travel_estimated = _nearest_depot(row)
    crossing_factor = SPECIAL_CROSSING_MULTIPLIER if _special_crossing(row) else 1.0

    operation_cost = (BASE_COST_USD + LENGTH_COST_PER_M * length_m) * crossing_factor
    cost_usd = float(operation_cost + TRAVEL_COST_PER_KM * travel_km)
    flight_minutes = float(
        BASE_FLIGHT_MINUTES
        + FLIGHT_MINUTES_PER_M * length_m
        + 2.0 * travel_km * TRAVEL_MINUTES_PER_KM
    )

    if travel_estimated:
        location_fragment = f"{travel_km:.0f} km default travel"
    else:
        location_fragment = f"{travel_km:.0f} km from {depot}"

    fragments = [
        f"drone est. ${cost_usd:,.0f}",
        f"{flight_minutes:.0f} min",
        location_fragment,
    ]
    if length_estimated:
        fragments.append("length estimated")
    if crossing_factor > 1.0:
        fragments.append("water/rail access factor")
    return cost_usd, flight_minutes, ", ".join(fragments)
