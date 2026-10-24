"""Value-of-information (VOI) layer for scored bridges.

A drone flight only *buys information*: it confirms or refutes whether a bridge
has already crossed into a safety drop. That information is most valuable when
we are genuinely uncertain (``p_worse`` near 0.5) and when the bridge matters
(``consequence_norm`` large). Bridges that are already very likely in a drop do
not need a flight to tell us what to do — they should go straight to the
track-and-fix queue.

VOI and the decision label are a per-cohort post-pass over a scored frame,
because the "fly" tier depends on a cohort-relative top-20% VOI cutoff.

Decision rules (per scored cohort):
- ``fix_track`` when ``p_worse >= FIX_TRACK_THRESHOLD`` (0.25).
- ``fly``       when ``p_worse < FIX_TRACK_THRESHOLD`` AND ``voi`` is in the
                top ``FLY_TOP_FRACTION`` (20%) of the cohort's VOI.
- ``monitor``   otherwise.

These three labels are exhaustive and mutually exclusive for every row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Decision thresholds.
FIX_TRACK_THRESHOLD = 0.25   # p_worse at/above this -> fix_track
FLY_TOP_FRACTION = 0.20      # top 20% of VOI among sub-threshold bridges -> fly

DECISION_LABELS = ("fix_track", "fly", "monitor")

# Columns VOI adds to a scored frame.
VOI_COLUMNS = ["voi", "decision"]


def compute_voi(p_worse, consequence_norm):
    """VOI = p_worse * (1 - p_worse) * consequence_norm.

    Works element-wise for scalars or aligned array-likes. The ``p*(1-p)`` term
    peaks at p=0.5 (maximum uncertainty) and vanishes at p=0 or p=1.
    """
    p = np.asarray(p_worse, dtype=float)
    cn = np.asarray(consequence_norm, dtype=float)
    return p * (1.0 - p) * cn


def _consequence_norm(scored: pd.DataFrame) -> pd.Series:
    """Cohort consequence normalized by its median (matches score())."""
    median = float(scored["consequence"].median())
    if not np.isfinite(median) or median <= 0:
        median = 1.0
    return scored["consequence"] / median


def decision_explanation(decision: str, p_worse: float, voi: float) -> str:
    """One-line, human-readable rationale for a decision label."""
    if decision == "fix_track":
        return (
            f"decision fix_track: p_worse {p_worse:.2f} >= "
            f"{FIX_TRACK_THRESHOLD:.2f}, likely already dropped — queue a fix"
        )
    if decision == "fly":
        return (
            f"decision fly: uncertain (p_worse {p_worse:.2f}) and high "
            f"information value (voi {voi:.3f}) — a flight resolves it"
        )
    return (
        f"decision monitor: low information value (voi {voi:.3f}) — "
        f"watch, don't spend a flight yet"
    )


def decision_explanation_human(decision: str) -> str:
    """Plain-English version of the decision rationale."""
    if decision == "fix_track":
        return (
            "Recommended action: put this bridge on the fix-and-track list — "
            "it's likely already in poor condition, so a flight wouldn't tell "
            "us anything new."
        )
    if decision == "fly":
        return (
            "Recommended action: send a drone. We're genuinely unsure about "
            "this bridge and a flight would give us the most useful new "
            "information for the money."
        )
    return (
        "Recommended action: keep monitoring. A flight here wouldn't change "
        "the plan much right now."
    )


def add_voi(scored: pd.DataFrame) -> pd.DataFrame:
    """Return ``scored`` with ``voi`` and ``decision`` columns appended.

    The input frame must have ``p_worse`` and ``consequence`` columns (the
    output of :func:`model.score.score`). Operates on a copy; also appends a
    one-line decision explanation to each row's ``reason`` when present.
    """
    result = scored.copy()
    if result.empty:
        result["voi"] = pd.Series(dtype=float)
        result["decision"] = pd.Series(dtype=object)
        return result

    has_human = "reason_human" in result.columns

    consequence_norm = _consequence_norm(result)
    result["voi"] = compute_voi(result["p_worse"], consequence_norm)

    p_worse = result["p_worse"].to_numpy(dtype=float)
    voi = result["voi"].to_numpy(dtype=float)

    fix_track = p_worse >= FIX_TRACK_THRESHOLD

    # "fly" is the top FLY_TOP_FRACTION of VOI *among bridges below the
    # fix_track threshold*. Rank only those candidates so fix_track bridges
    # never consume fly slots, and pick the highest-VOI candidates.
    decision = np.where(fix_track, "fix_track", "monitor").astype(object)
    candidate_mask = ~fix_track
    n_candidates = int(candidate_mask.sum())
    if n_candidates > 0:
        n_fly = int(np.ceil(n_candidates * FLY_TOP_FRACTION))
        if n_fly > 0:
            candidate_voi = voi[candidate_mask]
            # Threshold = the n_fly-th largest candidate VOI; ties fly together.
            cutoff = np.sort(candidate_voi)[::-1][n_fly - 1]
            fly_mask = candidate_mask & (voi >= cutoff)
            decision[fly_mask] = "fly"
    result["decision"] = decision

    if "reason" in result.columns:
        explanations = [
            decision_explanation(d, p, v)
            for d, p, v in zip(result["decision"], p_worse, voi)
        ]
        result["reason"] = result["reason"].astype(str) + "; " + pd.Series(
            explanations, index=result.index
        )

    if has_human:
        human = [decision_explanation_human(d) for d in result["decision"]]
        result["reason_human"] = result["reason_human"].astype(str) + " " + (
            pd.Series(human, index=result.index)
        )

    return result


def simulate_flight(scored_row, outcome: bool):
    """Return the bridge's updated ``(p_worse, insp_date)`` after a flight.

    A flight is a perfect, current observation of whether the bridge has already
    crossed into a safety drop:

    - ``outcome=True``  (drop confirmed): ``p_worse`` resolves to 1.0.
    - ``outcome=False`` (drop refuted):   ``p_worse`` resolves to 0.0.

    Either way the inspection date becomes "today" (the flight day), because the
    bridge now has a fresh, authoritative inspection. Returned so the app can
    re-rank and show the plan changing after a flight day. ``scored_row`` may be
    a mapping or a pandas row; it is not mutated.
    """
    getter = scored_row.get if hasattr(scored_row, "get") else (
        lambda k, d=None: getattr(scored_row, k, d)
    )
    _ = getter("p_worse")  # present for interface clarity / validation
    updated_p_worse = 1.0 if bool(outcome) else 0.0
    updated_insp_date = pd.Timestamp.now().normalize()
    return updated_p_worse, updated_insp_date
