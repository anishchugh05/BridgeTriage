"""Tests for Part B — the decline model and scoring.

Covered invariants (from the task spec):
1. is_safety_drop matches the agreed definition.
2. Rebuilt bridges are excluded from the fit.
3. fit_decline ignores any year after `end` (no lookahead).
4. score output has no NaNs and no empty reasons; reasons use material names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.fake_data import make_panel
from model.score import fit_decline, fit_logistic, score
from model.target import is_safety_drop


@pytest.fixture(scope="module")
def panel():
    return make_panel(n_bridges=800, seed=11)


@pytest.fixture(scope="module")
def rates(panel):
    return fit_decline(panel, 2012, 2016)


# --------------------------------------------------------------------------- #
# 1. Monotonicity: lower rating -> not-lower p_worse, holding group fixed.
# --------------------------------------------------------------------------- #


def test_is_safety_drop_definition():
    """A safety drop is: end <= 4 and start >= 5, OR start <= 4 and it fell by
    at least one grade. Slips that stay above 4 are not drops."""
    # Crossing into poor territory (end <= 4, start >= 5).
    assert is_safety_drop(5, 4) is True
    assert is_safety_drop(9, 4) is True
    assert is_safety_drop(6, 3) is True

    # Already poor and got worse by >= 1 grade.
    assert is_safety_drop(4, 3) is True
    assert is_safety_drop(3, 1) is True

    # NOT drops: high-grade slips that stay above 4.
    assert is_safety_drop(9, 7) is False
    assert is_safety_drop(8, 5) is False
    assert is_safety_drop(6, 5) is False

    # NOT drops: no change, or already poor but flat.
    assert is_safety_drop(5, 5) is False
    assert is_safety_drop(4, 4) is False
    assert is_safety_drop(3, 3) is False

    # NOT drops: improvement.
    assert is_safety_drop(4, 6) is False

    # Missing data is not a drop.
    assert is_safety_drop(np.nan, 3) is False
    assert is_safety_drop(5, np.nan) is False


# --------------------------------------------------------------------------- #
# 2. Rebuilt bridges are excluded from the fit.
# --------------------------------------------------------------------------- #


def test_rebuilt_bridges_excluded():
    """A bridge whose year_recon changes between the two years, or whose
    condition improved, must not count as an observed decline sample."""
    # Two-year panel: 2012 and 2016 only, three bridges.
    def row(bid, year, cond, recon):
        return {
            "id": bid, "year": year, "county": "Suffolk",
            "year_built": 1970, "year_recon": recon, "material": "Steel",
            "deck": float(cond), "super": float(cond), "sub": float(cond),
            "culvert": np.nan, "adt": 5000,
            "insp_date": pd.Timestamp(year=year, month=6, day=1),
            "insp_freq_mo": 24, "fracture_critical": False, "underwater": False,
            "lat": 42.0, "lon": -71.0, "crosses": "HWY", "insp_cost": 5000.0,
        }

    rows = [
        # genuine safety drop (6 -> 4 crosses into poor): counted, positive.
        row("DROP", 2012, 6, np.nan), row("DROP", 2016, 4, np.nan),
        # rebuilt: recon year changed -> excluded.
        row("RECON", 2012, 6, 2005), row("RECON", 2016, 4, 2014),
        # improved condition -> excluded.
        row("UP", 2012, 5, np.nan), row("UP", 2016, 7, np.nan),
    ]
    panel = pd.DataFrame(rows)

    rates = fit_decline(panel, 2012, 2016)
    # Only the genuine DROP survives pairing. With exactly one kept pair (a
    # positive at cond 6), the shrunk cond-6 rate sits above the global prior.
    # Had the rebuilt/improved rows leaked in, there would be 3 pairs and the
    # rate would be pulled down toward 1/3.
    assert rates.group_prob, "no groups learned"
    kept_cond6 = rates.group_prob.get(("c", 6))
    assert kept_cond6 is not None
    assert kept_cond6 > rates.global_prob - 1e-9
    # The single kept pair is a positive, so the global rate reflects 1/1.
    assert rates.global_prob > 0.5


# --------------------------------------------------------------------------- #
# 3. No lookahead: fit_decline ignores years after `end`.
# --------------------------------------------------------------------------- #


def test_fit_decline_ignores_years_after_end(panel):
    """Appending wildly different post-`end` rows must not change the fit."""
    rates_clean = fit_decline(panel, 2012, 2016)

    # Poison every year strictly after 2016 with implausible all-zero
    # conditions. If fit_decline peeked past `end`, the rates would shift.
    poisoned = panel.copy()
    future = poisoned["year"] > 2016
    for c in ["deck", "super", "sub"]:
        poisoned.loc[future, c] = 0.0
    poisoned.loc[future, "culvert"] = np.nan
    rates_poisoned = fit_decline(poisoned, 2012, 2016)

    assert rates_clean.group_prob.keys() == rates_poisoned.group_prob.keys()
    for k in rates_clean.group_prob:
        assert rates_clean.group_prob[k] == pytest.approx(
            rates_poisoned.group_prob[k]
        )
    assert rates_clean.global_prob == pytest.approx(rates_poisoned.global_prob)


def test_fit_decline_rejects_bad_interval(panel):
    with pytest.raises(ValueError):
        fit_decline(panel, 2016, 2012)


# --------------------------------------------------------------------------- #
# 4. score output: no NaNs, no empty reasons, exact schema.
# --------------------------------------------------------------------------- #


def test_score_output_clean(panel, rates):
    table = panel[panel["year"] == 2018].copy()
    scored = score(table, rates, asof="2018-06-01")

    assert list(scored.columns) == [
        "id", "p_worse", "consequence", "insp_cost", "score", "reason"
    ]
    assert len(scored) == len(table)

    # No NaNs anywhere.
    assert not scored.isna().any().any(), "score output contains NaN"

    # p_worse is a probability; cost and score are positive.
    assert (scored["p_worse"] >= 0).all() and (scored["p_worse"] <= 1).all()
    assert (scored["insp_cost"] > 0).all()
    assert (scored["score"] >= 0).all()

    # Every reason is a non-empty string.
    assert scored["reason"].map(lambda s: isinstance(s, str) and len(s.strip()) > 0).all()


def test_score_discloses_modeled_estimate_when_cost_missing(rates):
    """A missing insp_cost yields a positive cost and discloses its estimate."""
    row = {
        "id": "NOCOST", "year_built": 1965, "material": "Timber",
        "deck": 4.0, "super": 4.0, "sub": 5.0, "culvert": np.nan,
        "adt": 8000, "insp_date": pd.Timestamp("2016-01-01"),
        "fracture_critical": True, "underwater": True,
        "crosses": "WATER RIVER", "insp_cost": np.nan,
    }
    scored = score(pd.DataFrame([row]), rates, asof="2018-06-01")
    assert scored.loc[0, "insp_cost"] > 0
    assert "modeled inspection-cost estimate" in scored.loc[0, "reason"].lower()


def test_score_handles_na_condition_culvert_fallback(rates):
    """deck/super/sub all NaN must fall back to culvert without crashing."""
    row = {
        "id": "CULV", "year_built": 1980, "material": "Concrete",
        "deck": np.nan, "super": np.nan, "sub": np.nan, "culvert": 6.0,
        "adt": 3000, "insp_date": pd.Timestamp("2017-01-01"),
        "fracture_critical": False, "underwater": False,
        "crosses": "CRANBERRY BOG", "insp_cost": 4000.0,
    }
    scored = score(pd.DataFrame([row]), rates, asof="2018-06-01")
    assert not scored.isna().any().any()
    assert scored.loc[0, "p_worse"] >= 0


# --------------------------------------------------------------------------- #
# 5. Missing insp_date must not push a bridge to the top of the ranking.
# --------------------------------------------------------------------------- #


def test_missing_insp_date_does_not_inflate_rank(rates):
    """Three otherwise-identical bridges, differing only by last inspection:
    one inspected long ago (stale), one with a missing date, one inspected
    recently. The missing-date bridge must be treated as a moderate, recent-ish
    inspection (one required cycle ago) — never as the old 10-year assumption.
    So it must NOT outrank the genuinely stale bridge, and its reason must flag
    the unknown date."""
    base = {
        "year_built": 1975,
        "material": "Steel",
        "deck": 5.0,
        "super": 5.0,
        "sub": 5.0,
        "culvert": np.nan,
        "adt": 12000,
        "insp_freq_mo": 24,
        "fracture_critical": False,
        "underwater": False,
        "crosses": "WATER CHARLES RIVER",
        "insp_cost": 5000.0,
    }
    stale = dict(base, id="STALE", insp_date=pd.Timestamp("2008-06-01"))  # ~10y
    missing = dict(base, id="MISSING", insp_date=pd.NaT)                  # 24 mo
    recent = dict(base, id="RECENT", insp_date=pd.Timestamp("2018-01-01"))

    scored = score(pd.DataFrame([stale, missing, recent]), rates, asof="2018-06-01")
    scored = scored.set_index("id")

    # The missing-date bridge is assumed inspected one cycle ago (24 mo), not a
    # decade ago, so it cannot reach the genuinely-stale bridge's risk.
    assert scored.loc["MISSING", "p_worse"] <= scored.loc["STALE", "p_worse"] + 1e-9
    assert scored.loc["MISSING", "score"] <= scored.loc["STALE", "score"] + 1e-9

    # It must never be the single highest-scoring bridge here.
    assert scored["score"].idxmax() != "MISSING"

    assert "inspection date unknown" in scored.loc["MISSING", "reason"]
    assert "inspection date unknown" not in scored.loc["RECENT", "reason"]


def test_missing_date_uses_default_when_no_freq(rates):
    """With no insp_date and no insp_freq_mo, the fallback is the default
    window (not a long-ago date), and the reason flags the unknown date."""
    row = {
        "id": "NOFREQ", "year_built": 1975, "material": "Steel",
        "deck": 5.0, "super": 5.0, "sub": 5.0, "culvert": np.nan,
        "adt": 12000, "insp_date": pd.NaT, "insp_freq_mo": pd.NA,
        "fracture_critical": False, "underwater": False,
        "crosses": "HWY ROUTE 9", "insp_cost": 5000.0,
    }
    scored = score(pd.DataFrame([row]), rates, asof="2018-06-01")
    assert not scored.isna().any().any()
    assert "inspection date unknown" in scored.loc[0, "reason"]


# --------------------------------------------------------------------------- #
# 6. Rail/water crossing detection uses word boundaries.
# --------------------------------------------------------------------------- #


def test_crossing_word_boundaries():
    """"RR" must not match inside CHERRY/TERRACE; whole-word rail tokens must."""
    from model.score import _crossing_flags

    # Not rail: "RR" is a substring of CHERRY and TERRACE, never a whole word.
    assert _crossing_flags("CHERRY ST") == (False, False)
    assert _crossing_flags("TERRACE AVE") == (False, False)

    # Rail: standalone RR, MBTA, and railroad abbreviations like BMRR.
    assert _crossing_flags("RR MBTA")[0] is True
    assert _crossing_flags("BMRR")[0] is True
    assert _crossing_flags("RAILROAD CSX")[0] is True
    assert _crossing_flags("RAILWAY BR")[0] is True

    # Water: whole-word matches only.
    assert _crossing_flags("CHARLES RIVER")[1] is True
    assert _crossing_flags("WATER STREAM")[1] is True
    # "LAKED" / "WATERFORD" shouldn't false-positive as water (word boundary).
    assert _crossing_flags("WATERFORD RD") == (False, False)


# --------------------------------------------------------------------------- #
# 7. Material-code names appear in reasons.
# --------------------------------------------------------------------------- #


def test_reason_uses_material_name(rates):
    """A score reason must render the 043A material code as a name."""
    from model.materials import material_name

    assert material_name("3") == "Steel"
    assert material_name(5) == "Prestressed concrete"
    assert material_name("7") == "Wood/Timber"
    assert material_name(None) == "unknown material"

    row = {
        "id": "MAT", "year": 2018, "year_built": 1980, "material": "3",
        "deck": 5.0, "super": 5.0, "sub": 5.0, "culvert": np.nan,
        "worst_rating": 5.0, "adt": 9000,
        "insp_date": pd.NaT, "insp_freq_mo": 24,
        "fracture_critical": False, "underwater": False, "crosses": "HWY",
    }
    scored = score(pd.DataFrame([row]), rates, asof="2018-06-01")
    assert "Steel" in scored.loc[0, "reason"]


# --------------------------------------------------------------------------- #
# 8. Logistic model fits and scores like the grouped model.
# --------------------------------------------------------------------------- #


def test_logistic_model_scores(panel):
    """fit_logistic produces a model score() can use, with the same clean
    output as the grouped model."""
    logit = fit_logistic(panel, 2012, 2016)
    table = panel[panel["year"] == 2018].copy()
    scored = score(table, logit, asof="2018-06-01")

    assert list(scored.columns) == [
        "id", "p_worse", "consequence", "insp_cost", "score", "reason"
    ]
    assert not scored.isna().any().any()
    assert (scored["p_worse"] >= 0).all() and (scored["p_worse"] <= 1).all()


# --------------------------------------------------------------------------- #
# 9. months_since_insp: capped, reason text, missing-date flag.
# --------------------------------------------------------------------------- #


def test_months_since_insp_in_reason_and_capped(rates):
    """A known insp_date yields 'last inspected N mo ago' (capped at 60); a
    missing date falls back and says so."""
    # Inspected ~20 months before asof 2018-04-01 -> "last inspected 20 mo ago".
    known = {
        "id": "KNOWN", "year": 2018, "year_built": 1980, "material": "3",
        "deck": 5.0, "super": 5.0, "sub": 5.0, "culvert": np.nan,
        "worst_rating": 5.0, "adt": 9000,
        "insp_date": pd.Timestamp("2016-08-01"), "insp_freq_mo": 24,
        "fracture_critical": False, "underwater": False, "crosses": "HWY",
    }
    # Very old date -> months capped at 60.
    old = dict(known, id="OLD", insp_date=pd.Timestamp("2000-01-01"))
    # Missing date -> fallback + flag.
    missing = dict(known, id="MISS", insp_date=pd.NaT)

    scored = score(pd.DataFrame([known, old, missing]), rates,
                   asof="2018-04-01").set_index("id")

    assert "last inspected 20 mo ago" in scored.loc["KNOWN", "reason"]
    assert "last inspected 60 mo ago" in scored.loc["OLD", "reason"]
    assert "inspection date unknown" in scored.loc["MISS", "reason"]


def test_logistic_coefficients_named(panel):
    """A fitted logistic exposes named coefficients sorted by magnitude, and
    months_since_insp appears only when the feature is enabled."""
    with_msi = fit_logistic(panel, 2012, 2016, use_months_since_insp=True)
    no_msi = fit_logistic(panel, 2012, 2016, use_months_since_insp=False)

    c_with = with_msi.coefficients()
    assert "months_since_insp" in c_with.index
    assert "months_since_insp" not in no_msi.coefficients().index
    # Sorted by descending magnitude.
    mags = c_with.abs().to_numpy()
    assert (mags[:-1] + 1e-12 >= mags[1:]).all()
