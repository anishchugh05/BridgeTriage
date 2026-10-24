"""The shared prediction target for BridgeTriage.

A *safety drop* is the event the whole project is built to predict and catch.
Part B trains on it and Part C backtests against it, so the definition lives in
exactly one place and both import it from here.

Definition (condition = worst_rating, i.e. min of deck/super/sub):

    A pair (start, end) is a safety drop when either
      - it crosses into poor territory: end <= 4 AND start >= 5, or
      - it was already poor and got worse: start <= 4 AND end <= start - 1.

    Slips that stay above 4 (e.g. 9 -> 7, 8 -> 5) are NOT safety drops: the
    bridge is still in fair-or-better condition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POOR_THRESHOLD = 4  # <= 4 is "poor"; >= 5 is "fair or better"


def is_safety_drop(start, end) -> bool:
    """Return True if the (start -> end) condition pair is a safety drop.

    ``start`` / ``end`` are condition grades (0-9) or NaN. A pair with a missing
    grade on either side is not a drop (and should normally be excluded from
    training/backtest upstream).
    """
    if start is None or end is None:
        return False
    try:
        if pd.isna(start) or pd.isna(end):
            return False
    except (TypeError, ValueError):
        return False
    s = float(start)
    e = float(end)
    crossed_into_poor = (e <= POOR_THRESHOLD) and (s >= POOR_THRESHOLD + 1)
    already_poor_worsened = (s <= POOR_THRESHOLD) and (e <= s - 1)
    return bool(crossed_into_poor or already_poor_worsened)


def is_safety_drop_series(start: pd.Series, end: pd.Series) -> pd.Series:
    """Vectorized ``is_safety_drop`` over aligned start/end condition series."""
    s = pd.to_numeric(start, errors="coerce")
    e = pd.to_numeric(end, errors="coerce")
    crossed = (e <= POOR_THRESHOLD) & (s >= POOR_THRESHOLD + 1)
    worsened = (s <= POOR_THRESHOLD) & (e <= s - 1)
    out = (crossed | worsened) & s.notna() & e.notna()
    return out.fillna(False).astype(bool)
