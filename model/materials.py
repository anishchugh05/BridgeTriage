"""NBI item 043A structure-material code -> human-readable name.

Used to make score() reasons readable ("Steel, 48y, cond 5") instead of showing
the raw code ("3, 48y, cond 5"). Part A's panel stores the raw 043A code.
"""

from __future__ import annotations

MATERIAL_NAMES = {
    "0": "Other",
    "1": "Concrete",
    "2": "Concrete continuous",
    "3": "Steel",
    "4": "Steel continuous",
    "5": "Prestressed concrete",
    "6": "Prestressed continuous",
    "7": "Wood/Timber",
    "8": "Masonry",
    "9": "Aluminum/iron",
}


def material_name(code) -> str:
    """Return the material name for an 043A code, or a readable fallback.

    Accepts codes as strings or numbers (e.g. 3, "3", 3.0). Unknown or missing
    codes return "unknown material".
    """
    if code is None:
        return "unknown material"
    try:
        import pandas as pd

        if pd.isna(code):
            return "unknown material"
    except (TypeError, ValueError):
        pass
    key = str(code).strip()
    # Normalize numeric-looking codes like "3.0" -> "3".
    if key.endswith(".0"):
        key = key[:-2]
    return MATERIAL_NAMES.get(key, f"material {key}")
