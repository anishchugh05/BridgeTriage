"""Tests for the value-of-information layer (model/voi.py)."""

import numpy as np
import pandas as pd
import pytest

from model.voi import (
    DECISION_LABELS,
    FIX_TRACK_THRESHOLD,
    add_voi,
    compute_voi,
    simulate_flight,
)


def _synthetic_scored(n=200, seed=0):
    """A scored-shaped frame with a spread of p_worse and consequence."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 0.9, size=n)
    consequence = rng.uniform(0.2, 12.0, size=n)
    return pd.DataFrame(
        {
            "id": [f"B{i:04d}" for i in range(n)],
            "p_worse": p,
            "consequence": consequence,
            "insp_cost": rng.uniform(1600, 3000, size=n),
            "reason": ["base reason"] * n,
        }
    )


def test_compute_voi_matches_formula():
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cn = np.array([1.0, 2.0, 0.5, 4.0, 3.0])
    expected = p * (1 - p) * cn
    assert np.allclose(compute_voi(p, cn), expected)
    # Uncertainty term is zero at the extremes, positive in the middle.
    assert compute_voi(0.0, 5.0) == 0.0
    assert compute_voi(1.0, 5.0) == 0.0
    assert compute_voi(0.5, 1.0) > 0.0


def test_labels_are_exhaustive_and_valid():
    scored = add_voi(_synthetic_scored())
    assert set(scored["decision"].unique()) <= set(DECISION_LABELS)
    # Every row is labeled (no NaN/empty), i.e. exhaustive coverage.
    assert scored["decision"].notna().all()
    assert scored["decision"].isin(DECISION_LABELS).all()


def test_fix_track_has_highest_p_worse():
    scored = add_voi(_synthetic_scored())
    fix = scored[scored["decision"] == "fix_track"]
    others = scored[scored["decision"] != "fix_track"]
    assert not fix.empty
    # fix_track is exactly the p_worse >= threshold set, so its minimum beats
    # every other bridge's p_worse.
    assert (fix["p_worse"] >= FIX_TRACK_THRESHOLD).all()
    assert (others["p_worse"] < FIX_TRACK_THRESHOLD).all()
    assert fix["p_worse"].min() >= others["p_worse"].max()


def test_fly_has_higher_mean_voi_than_monitor():
    scored = add_voi(_synthetic_scored())
    fly = scored[scored["decision"] == "fly"]
    monitor = scored[scored["decision"] == "monitor"]
    assert not fly.empty and not monitor.empty
    assert fly["voi"].mean() > monitor["voi"].mean()
    # fly and monitor are both strictly below the fix_track threshold.
    assert (fly["p_worse"] < FIX_TRACK_THRESHOLD).all()
    assert (monitor["p_worse"] < FIX_TRACK_THRESHOLD).all()


def test_decision_explanation_appended_to_reason():
    scored = add_voi(_synthetic_scored())
    assert scored["reason"].str.contains("decision ", regex=False).all()
    for label in scored["decision"].unique():
        row = scored[scored["decision"] == label].iloc[0]
        assert f"decision {label}" in row["reason"]


def test_empty_frame_gets_voi_columns():
    empty = pd.DataFrame(
        columns=["id", "p_worse", "consequence", "insp_cost", "reason"]
    )
    out = add_voi(empty)
    assert "voi" in out.columns and "decision" in out.columns
    assert out.empty


def test_simulate_flight_confirm_and_refute():
    row = {"id": "B1", "p_worse": 0.4, "insp_date": pd.Timestamp("2015-01-01")}

    confirmed_p, confirmed_date = simulate_flight(row, outcome=True)
    refuted_p, refuted_date = simulate_flight(row, outcome=False)

    assert confirmed_p == 1.0
    assert refuted_p == 0.0
    # Inspection date is refreshed to the flight day (today), newer than before.
    today = pd.Timestamp.now().normalize()
    assert confirmed_date == today
    assert refuted_date == today
    assert confirmed_date > pd.Timestamp("2015-01-01")
    # The original row is not mutated.
    assert row["p_worse"] == 0.4
    assert row["insp_date"] == pd.Timestamp("2015-01-01")


def test_simulate_flight_accepts_pandas_row():
    scored = add_voi(_synthetic_scored(n=10))
    scored["insp_date"] = pd.Timestamp("2016-06-01")
    row = scored.iloc[0]
    p, date = simulate_flight(row, outcome=True)
    assert p == 1.0
    assert date == pd.Timestamp.now().normalize()
