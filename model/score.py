"""Part B — decline model and scoring for BridgeTriage.

Implements the two contract functions:

- ``fit_decline(panel, start, end)`` learns, per (material, age band, condition)
  group, the probability of the shared safety-drop target over the
  ``start`` -> ``end`` interval.
- ``score(table, rates, asof)`` turns a fitted grouped/logistic probability into
  ``p_worse``, combines it with consequence and the drone-flight inspection
  estimate, and returns the scored table defined in
  ``.kiro/steering/contract.md``.

Rules honored here (see product.md / contract.md):
- No lookahead: ``fit_decline`` reads only rows in years ``start`` and ``end``.
- NBI "N" is not-applicable: it arrives as NaN and is treated as missing, never
  as a rating of 0.
- Every scored row carries a one-line ``reason`` with its estimated drone
  flight cost, duration, travel, and any estimated inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .flight_cost import LENGTH_COLUMNS, flight_cost
from .materials import material_name
from .target import is_safety_drop_series
from .voi import add_voi

# --------------------------------------------------------------------------- #
# Constants / knobs
# --------------------------------------------------------------------------- #

CONDITION_COLS = ["deck", "super", "sub"]  # primary condition inputs
CULVERT_COL = "culvert"  # fallback when deck/super/sub are all N/A

# Age bands (years). Edges are left-inclusive; last band is open-ended.
AGE_BAND_EDGES = [0, 25, 50, 75, np.inf]
AGE_BAND_LABELS = ["0-24", "25-49", "50-74", "75+"]

# Shrinkage: an effective sample size controlling how fast a small group's
# estimate is pulled toward its coarser parent group.
SHRINKAGE_K = 20.0

# Consequence shaping.
CONSEQUENCE_RAIL_MULT = 1.5
CONSEQUENCE_WATER_MULT = 1.3
CONSEQUENCE_POOR_MULT = 1.4  # extra weight when condition <= 5
POOR_CONDITION_THRESHOLD = 5.0

# Forward-looking window added to "months since last inspection" so that a
# freshly inspected bridge still carries some forward risk.
LOOKAHEAD_MONTHS = 12.0

# Fallback used when insp_date is missing: assume the bridge was inspected one
# required cycle ago (insp_freq_mo), or this many months if that is also
# missing. We do NOT assume a very old date — that would wrongly push an
# unknown-date bridge toward the top of the ranking.
DEFAULT_MONTHS_NO_DATE = 24.0

# months_since_insp is capped so a very stale date can't dominate; NBI
# inspection cycles are 24-48 months, so 60 is a generous ceiling.
MONTHS_SINCE_INSP_CAP = 60.0

# The NBI yearly file is a snapshot taken around the start of the data year; we
# use April 1 of the row's year as the reference "as of" date for a start-year
# row when computing months since its last inspection.
NBI_SNAPSHOT_MONTH = 4
NBI_SNAPSHOT_DAY = 1

# Word-boundary crossing keywords. NBI "crosses" is free text, so match whole
# tokens only — otherwise "RR" hits CHERRY/TERRACE and "WATER" partials leak in.
# For the "RR" railroad token we anchor only the right side (RR\b): this matches
# a standalone "RR" and railroad suffixes like "BMRR"/"NYRR", but NOT CHERRY or
# TERRACE, where RR is followed by more letters (no right-hand boundary).
RAIL_PATTERN = re.compile(r"(?:RR\b|\bRAILROAD\b|\bRAILWAY\b|\bMBTA\b)")
WATER_PATTERN = re.compile(
    r"\b(?:RIV|RIVER|CREEK|BROOK|STREAM|WATER|BAY|POND|LAKE|CANAL|HARBOR)\b"
)

OUTPUT_COLUMNS = [
    "id",
    "p_worse",
    "consequence",
    "insp_cost",
    "flight_minutes",
    "expected_drops_per_10k",
    "score",
    "voi",
    "decision",
    "reason",
    "reason_human",
]


# --------------------------------------------------------------------------- #
# Rates container
# --------------------------------------------------------------------------- #


@dataclass
class DeclineRates:
    """Grouped-model decline rates returned by :func:`fit_decline`.

    ``group_prob`` maps a group key to the probability of a *safety drop* (see
    :mod:`model.target`) over the fit horizon (``horizon_months``). Keys exist
    at three granularities so lookups can fall back from fine to coarse:

    - fine:   ``("mat", material, age_band, condition)``
    - mid:    ``("mc", material, condition)``
    - coarse: ``("c", condition)``

    ``global_prob`` is the final fallback when nothing else matches.
    """

    group_prob: Dict[Tuple, float]
    global_prob: float
    horizon_months: float
    name: str = "grouped"

    def prob_for(self, material: str, age_band: str, condition: float) -> float:
        """Best available horizon drop probability for a bridge."""
        if condition is None or _isnan(condition):
            return self.global_prob
        cond = int(round(condition))
        for key in (
            ("mat", material, age_band, cond),
            ("mc", material, cond),
            ("c", cond),
        ):
            if key in self.group_prob:
                return self.group_prob[key]
        return self.global_prob

    def interval_prob(self, rec: dict, asof=None) -> float:
        """Horizon drop probability for a scored row ``rec``."""
        condition = _condition(rec)
        material = rec.get("material")
        yb = rec.get("year_built")
        if not _isnan(yb) and asof is not None:
            age = pd.Timestamp(asof).year - float(yb)
        elif not _isnan(rec.get("age")):
            age = float(rec.get("age"))
        elif not _isnan(yb) and not _isnan(rec.get("year")):
            age = float(rec.get("year")) - float(yb)
        else:
            age = np.nan
        return self.prob_for(material, _age_band(age), condition)


@dataclass
class ConditionBaseline:
    """Per-condition empirical safety-drop model used by CV and selection."""

    per_condition: Dict[int, float]
    global_prob: float
    horizon_months: float
    name: str = "baseline"

    def interval_prob(self, rec: dict, asof=None) -> float:
        condition = _condition(rec)
        if _isnan(condition):
            return self.global_prob
        return self.per_condition.get(int(round(condition)), self.global_prob)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _condition(row_like) -> float:
    """Condition for one bridge-row.

    Prefers Part A's precomputed ``worst_rating`` when present (that column is
    the min of deck/super/sub, null only when all three are null). Otherwise
    falls back to min(deck, super, sub), then culvert. NaN means
    not-applicable and is skipped; returns NaN only when everything is missing.
    """
    if "worst_rating" in row_like:
        worst = row_like.get("worst_rating")
        # In Part A's published panel, null worst_rating explicitly means the
        # condition is not applicable. Do not reconstruct it from components.
        return float(worst) if not _isnan(worst) else float("nan")
    vals = [row_like.get(c) for c in CONDITION_COLS]
    vals = [v for v in vals if v is not None and not _isnan(v)]
    if vals:
        return float(min(vals))
    culvert = row_like.get(CULVERT_COL)
    if culvert is not None and not _isnan(culvert):
        return float(culvert)
    return float("nan")


def _condition_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized condition.

    Uses ``worst_rating`` where available, falling back to min(deck/super/sub),
    then culvert. A null ``worst_rating`` is not-applicable and is skipped (the
    bridge gets NaN unless the fallbacks supply a value).
    """
    if "worst_rating" in df.columns:
        # Null is an explicit not-applicable value in Part A's panel; callers
        # skip it rather than filling it from component ratings.
        return pd.to_numeric(df["worst_rating"], errors="coerce")
    primary_cols = [c for c in CONDITION_COLS if c in df.columns]
    if primary_cols:
        primary = df[primary_cols].min(axis=1, skipna=True)
    else:
        primary = pd.Series(np.nan, index=df.index)
    if CULVERT_COL in df.columns:
        primary = primary.where(primary.notna(), df[CULVERT_COL])
    return primary


def _isnan(v) -> bool:
    """True for None, float NaN, and pandas NA / NaT (nullable dtypes)."""
    if v is None:
        return True
    try:
        # pd.isna handles np.nan, pd.NA, pd.NaT, and scalars uniformly.
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _age_band(age: float) -> str:
    if age is None or _isnan(age):
        return "unknown"
    idx = int(np.digitize([age], AGE_BAND_EDGES, right=False)[0]) - 1
    idx = max(0, min(idx, len(AGE_BAND_LABELS) - 1))
    return AGE_BAND_LABELS[idx]


def _age_band_series(age: pd.Series) -> pd.Series:
    bands = pd.cut(
        age, bins=AGE_BAND_EDGES, labels=AGE_BAND_LABELS, right=False
    ).astype("object")
    return bands.where(age.notna(), "unknown")


def _prob_to_monthly_hazard(prob: float, interval_months: float) -> float:
    """Convert P(drop within interval) to a constant per-month hazard.

    Uses the exponential survival relation: p = 1 - exp(-h * T), so
    h = -ln(1 - p) / T.
    """
    prob = float(np.clip(prob, 1e-9, 1 - 1e-9))
    return -np.log1p(-prob) / interval_months


def _shrink(k_success: float, n: float, parent_rate: float) -> float:
    """Shrink an observed rate toward ``parent_rate`` by effective weight K."""
    return (k_success + SHRINKAGE_K * parent_rate) / (n + SHRINKAGE_K)


def _is_rebuilt_end(ra, rb) -> bool:
    """Whether a start->end pair is a rebuild/repair to exclude from training.

    Uses Part A's end-year ``rebuilt`` flag when present; otherwise falls back
    to a year_recon change (0 means never). An improvement in condition is also
    treated as a repair.
    """
    if "rebuilt" in rb.index:
        if bool(rb.get("rebuilt")):
            return True
    else:
        recon_a, recon_b = ra.get("year_recon"), rb.get("year_recon")
        ra_val = None if (_isnan(recon_a) or float(recon_a) == 0) else float(recon_a)
        rb_val = None if (_isnan(recon_b) or float(recon_b) == 0) else float(recon_b)
        if (ra_val is not None and rb_val is not None and ra_val != rb_val) or (
            rb_val is not None and ra_val is None
        ):
            return True
    return False


def _build_pairs(panel: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Pair bridges across ``start``->``end`` and label the safety-drop target.

    Shared by the grouped and logistic fits (and mirrored by validate / Part C).
    Reads only the two interval years (no lookahead). Skips null-condition rows
    and rebuilt/improved pairs. Carries the feature columns the logistic model
    needs. The label uses :func:`model.target.is_safety_drop`.
    """
    sub = panel[panel["year"].isin([start, end])].copy()
    a = sub[sub["year"] == start].copy()
    b = sub[sub["year"] == end].copy()
    a["cond"] = _condition_series(a)
    b["cond"] = _condition_series(b)
    a = a.set_index("id")
    b = b.set_index("id")
    common = a.index.intersection(b.index)

    records = []
    for bid in common:
        ra, rb = a.loc[bid], b.loc[bid]
        cond_a, cond_b = ra["cond"], rb["cond"]
        if _isnan(cond_a) or _isnan(cond_b):
            continue
        if _is_rebuilt_end(ra, rb):
            continue
        if cond_b > cond_a:  # condition improved -> repair/rebuild, exclude
            continue

        yb = ra.get("year_built")
        if not _isnan(yb):
            age = start - float(yb)
        elif not _isnan(ra.get("age")):
            age = float(ra.get("age"))
        else:
            age = np.nan
        adt = ra.get("adt")
        # months since last inspection, measured to the start-year NBI snapshot
        # (April 1 of the start year), capped at 60. Falls back to insp_freq_mo
        # only when insp_date is missing.
        ref = _snapshot_date(start)
        msi, msi_known = _months_since(
            ra.get("insp_date"), ref, ra.get("insp_freq_mo")
        )
        records.append(
            {
                "id": bid,
                "material": ra.get("material"),
                "age": age,
                "age_band": _age_band(age),
                "cond": int(round(cond_a)),
                "adt": float(adt) if not _isnan(adt) else np.nan,
                "insp_freq_mo": ra.get("insp_freq_mo"),
                "months_since_insp": msi,
                "months_since_insp_known": msi_known,
                "fracture_critical": bool(ra.get("fracture_critical", False)),
                "underwater": bool(ra.get("underwater", False)),
                "dropped": int(is_safety_drop_series(
                    pd.Series([cond_a]), pd.Series([cond_b])
                ).iloc[0]),
            }
        )
    return pd.DataFrame.from_records(records)


def fit_condition_baseline(panel: pd.DataFrame, start: int, end: int) -> ConditionBaseline:
    """Fit the condition-only comparison model on a four-year interval."""
    _validate_fit_interval(start, end)
    pairs = _build_pairs(panel, start, end)
    if pairs.empty:
        return ConditionBaseline({}, 0.1, (end - start) * 12.0)
    per = pairs.groupby("cond")["dropped"].mean().to_dict()
    return ConditionBaseline(
        per_condition={int(k): float(v) for k, v in per.items()},
        global_prob=float(pairs["dropped"].mean()),
        horizon_months=(end - start) * 12.0,
    )


# --------------------------------------------------------------------------- #
# fit_decline
# --------------------------------------------------------------------------- #


def _validate_fit_interval(start: int, end: int) -> None:
    """Require the four-year horizon shared by training and backtesting."""
    if end - start != 4:
        raise ValueError(
            f"fit interval must be exactly 4 years; got {start}->{end}. "
            "The 2016->2018 two-year gap must never be fit."
        )


def fit_decline(panel: pd.DataFrame, start: int, end: int) -> DeclineRates:
    """Learn decline rates from the ``start`` -> ``end`` interval only.

    Pairs each bridge's ``start`` row with its ``end`` row by ``id``, drops
    rebuilt bridges, and estimates P(condition drops >= 1 grade) per
    (material, age band, start-condition) group with shrinkage toward coarser
    groups. Returns monthly hazards.

    No row from any year other than ``start`` or ``end`` is read.
    """
    _validate_fit_interval(start, end)

    interval_months = (end - start) * 12.0
    paired = _build_pairs(panel, start, end)
    if paired.empty:
        return DeclineRates(group_prob={}, global_prob=0.1,
                            horizon_months=interval_months)

    global_prob = float(paired["dropped"].mean())

    # Grouped estimate at three granularities, each shrunk toward its coarser
    # parent group. No monotonicity clamp: the safety-drop target excludes
    # high-grade slips, so the old "worse grade drops more" / floor-effect
    # ordering no longer applies — we let the grouped rates speak.
    group_prob: Dict[Tuple, float] = {}

    cond_group = paired.groupby("cond")["dropped"].agg(["sum", "count"])
    cond_prob: Dict[int, float] = {}
    for cond, r in cond_group.iterrows():
        p = _shrink(r["sum"], r["count"], global_prob)
        cond_prob[int(cond)] = p
        group_prob[("c", int(cond))] = p

    mc_group = paired.groupby(["material", "cond"])["dropped"].agg(["sum", "count"])
    mc_prob: Dict[Tuple, float] = {}
    for (mat, cond), r in mc_group.iterrows():
        parent = cond_prob.get(int(cond), global_prob)
        p = _shrink(r["sum"], r["count"], parent)
        mc_prob[(mat, int(cond))] = p
        group_prob[("mc", mat, int(cond))] = p

    fine_group = paired.groupby(["material", "age_band", "cond"])["dropped"].agg(
        ["sum", "count"]
    )
    for (mat, band, cond), r in fine_group.iterrows():
        parent = mc_prob.get((mat, int(cond)), cond_prob.get(int(cond), global_prob))
        p = _shrink(r["sum"], r["count"], parent)
        group_prob[("mat", mat, band, int(cond))] = p

    return DeclineRates(
        group_prob=group_prob,
        global_prob=global_prob,
        horizon_months=interval_months,
    )


# --------------------------------------------------------------------------- #
# Logistic model
# --------------------------------------------------------------------------- #

# Logistic feature set, in the order the design matrix is built.
LOGIT_COND_GRADES = list(range(0, 10))  # one-hot columns for grades 0..9


@dataclass
class LogisticModel:
    """Logistic-regression drop model with the same interface as DeclineRates.

    ``use_months_since_insp`` toggles the months-since-last-inspection feature
    so validate.py can compare the model with and without it.
    """

    clf: object  # fitted sklearn LogisticRegression
    materials: list  # training material vocabulary (for one-hot alignment)
    horizon_months: float
    age_median: float
    use_months_since_insp: bool = True
    feature_names: list = field(default_factory=list)
    constant_prob: Optional[float] = None
    name: str = "logistic"

    def _row_matrix(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        cond = pd.to_numeric(df["cond"], errors="coerce").to_numpy()
        cond_oh = np.zeros((n, len(LOGIT_COND_GRADES)), dtype=float)
        for j, g in enumerate(LOGIT_COND_GRADES):
            cond_oh[:, j] = (np.round(cond) == g).astype(float)

        mat = df["material"].astype("string").fillna("<NA>").to_numpy()
        mat_oh = np.zeros((n, len(self.materials)), dtype=float)
        index = {m: j for j, m in enumerate(self.materials)}
        for i, m in enumerate(mat):
            j = index.get(m)
            if j is not None:
                mat_oh[i, j] = 1.0

        age = pd.to_numeric(df["age"], errors="coerce").fillna(
            self.age_median
        ).to_numpy()
        adt = pd.to_numeric(df["adt"], errors="coerce").fillna(0.0).to_numpy()
        log_adt = np.log1p(np.clip(adt, 0, None))
        freq = pd.to_numeric(df["insp_freq_mo"], errors="coerce").fillna(
            24.0
        ).to_numpy()
        fc = df["fracture_critical"].astype(bool).astype(float).to_numpy()
        uw = df["underwater"].astype(bool).astype(float).to_numpy()

        blocks = [cond_oh, mat_oh,
                  np.column_stack([age, log_adt, freq, fc, uw])]
        if self.use_months_since_insp:
            msi = pd.to_numeric(
                df.get("months_since_insp"), errors="coerce"
            ).fillna(DEFAULT_MONTHS_NO_DATE).clip(
                upper=MONTHS_SINCE_INSP_CAP
            ).to_numpy()
            blocks.append(msi.reshape(-1, 1))
        return np.hstack(blocks)

    def _build_feature_names(self) -> list:
        names = [f"cond={g}" for g in LOGIT_COND_GRADES]
        names += [f"material={material_name(m)}" for m in self.materials]
        names += ["age", "log_adt", "insp_freq_mo", "fracture_critical",
                  "underwater"]
        if self.use_months_since_insp:
            names.append("months_since_insp")
        return names

    def _row_from_rec(self, rec: dict, asof: pd.Timestamp) -> pd.DataFrame:
        cond = _condition(rec)
        yb = rec.get("year_built")
        if not _isnan(yb):
            age = pd.Timestamp(asof).year - float(yb)
        elif not _isnan(rec.get("age")):
            age = float(rec.get("age"))
        else:
            age = self.age_median
        msi, _ = _months_since(rec.get("insp_date"), asof, rec.get("insp_freq_mo"))
        return pd.DataFrame(
            [
                {
                    "cond": -1 if _isnan(cond) else int(round(cond)),
                    "material": rec.get("material"),
                    "age": age,
                    "adt": rec.get("adt"),
                    "insp_freq_mo": rec.get("insp_freq_mo"),
                    "months_since_insp": msi,
                    "fracture_critical": bool(rec.get("fracture_critical", False)),
                    "underwater": bool(rec.get("underwater", False)),
                }
            ]
        )

    def interval_prob(self, rec: dict, asof=None) -> float:
        """Horizon drop probability for a single scored row ``rec``.

        ``asof`` sets the reference date for months_since_insp; defaults to the
        row's NBI snapshot (April 1 of its year) when available.
        """
        if asof is None:
            asof = _snapshot_date(rec.get("year")) or pd.Timestamp.now()
        if self.constant_prob is not None:
            return float(self.constant_prob)
        X = self._row_matrix(self._row_from_rec(rec, pd.Timestamp(asof)))
        return float(self.clf.predict_proba(X)[0, 1])

    def interval_prob_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized horizon probabilities for a pairs frame (for validate)."""
        if self.constant_prob is not None:
            return np.full(len(df), self.constant_prob, dtype=float)
        X = self._row_matrix(df)
        return self.clf.predict_proba(X)[:, 1]

    def coefficients(self) -> pd.Series:
        """Fitted coefficients indexed by feature name, sorted by |value|."""
        if self.constant_prob is not None:
            return pd.Series(0.0, index=self.feature_names)
        coefs = pd.Series(self.clf.coef_[0], index=self.feature_names)
        return coefs.reindex(coefs.abs().sort_values(ascending=False).index)


def fit_logistic(panel: pd.DataFrame, start: int, end: int,
                 train_ids=None, use_months_since_insp: bool = True
                 ) -> LogisticModel:
    """Fit a logistic regression on the safety-drop target.

    Features: condition one-hot, material one-hot, age, log ADT, insp_freq_mo,
    fracture_critical, underwater, and (optionally) months_since_insp. Trains
    only on ``train_ids`` when given. Reads only the ``start``/``end`` years.
    """
    from sklearn.linear_model import LogisticRegression

    _validate_fit_interval(start, end)
    if train_ids is not None:
        panel = panel[panel["id"].isin(train_ids)]
    pairs = _build_pairs(panel, start, end)
    if pairs.empty:
        raise ValueError("no training pairs for logistic model")

    materials = sorted(pairs["material"].astype("string").fillna("<NA>").unique())
    age_median = float(pd.to_numeric(pairs["age"], errors="coerce").median())

    model = LogisticModel(
        clf=None, materials=materials, horizon_months=(end - start) * 12.0,
        age_median=age_median, use_months_since_insp=use_months_since_insp,
    )
    model.feature_names = model._build_feature_names()
    X = model._row_matrix(pairs)
    y = pairs["dropped"].to_numpy()
    if np.unique(y).size < 2:
        # Small CV folds/subsets can contain one class. A constant empirical
        # probability keeps validation and scoring defined instead of letting
        # sklearn abort the entire comparison.
        model.constant_prob = float(np.clip(y.mean(), 1e-9, 1 - 1e-9))
        return model
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    model.clf = clf
    return model


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #


def _snapshot_date(year) -> Optional[pd.Timestamp]:
    """April 1 of ``year`` — the NBI snapshot reference date for that row."""
    if _isnan(year):
        return None
    return pd.Timestamp(year=int(year), month=NBI_SNAPSHOT_MONTH,
                        day=NBI_SNAPSHOT_DAY)


def _months_since(insp_date, asof: pd.Timestamp, insp_freq_mo=None):
    """Months since the last inspection, capped at ``MONTHS_SINCE_INSP_CAP``.

    Returns ``(months, date_known)``. When ``insp_date`` is missing we do NOT
    assume a long time ago (that would inflate p_worse and wrongly rank the
    bridge near the top). Instead we fall back to ``insp_freq_mo`` if present,
    else ``DEFAULT_MONTHS_NO_DATE``, and flag ``date_known=False``.
    """
    if insp_date is None or pd.isna(insp_date):
        if insp_freq_mo is not None and not _isnan(insp_freq_mo):
            months = float(insp_freq_mo)
        else:
            months = DEFAULT_MONTHS_NO_DATE
        return min(months, MONTHS_SINCE_INSP_CAP), False
    delta_days = (asof - pd.Timestamp(insp_date)).days
    months = max(0.0, delta_days / 30.4375)
    return min(months, MONTHS_SINCE_INSP_CAP), True


def months_since_insp(insp_date, ref: pd.Timestamp, insp_freq_mo=None):
    """Public helper: months from ``insp_date`` to ``ref``, capped at 60.

    Thin wrapper over :func:`_months_since` with the same fallback/flag
    semantics; shared by the logistic feature builder, Part C, and score().
    """
    return _months_since(insp_date, ref, insp_freq_mo)


def _human_reason(*, material, age, condition, rail, water, months_since,
                  date_known, p_worse, insp_cost, flight_minutes,
                  has_real_cost) -> str:
    """Plain-English explanation of a score, for a non-technical reader.

    Mirrors the technical ``reason`` but in full sentences a planner or the
    public could read. Kept alongside (not replacing) the technical string.
    """
    mat = material_name(material)
    grade = int(round(condition))
    chance = int(round(p_worse * 100))
    if _isnan(age):
        lead = f"This {mat.lower()} bridge is currently rated {grade} out of 9."
    else:
        lead = (
            f"This {mat.lower()} bridge, built about {int(age)} years ago, is "
            f"currently rated {grade} out of 9."
        )

    crossing = ""
    if rail and water:
        crossing = " It spans both water and a rail line, which complicate access."
    elif water:
        crossing = " It crosses water, which complicates access."
    elif rail:
        crossing = " It crosses a rail line, which complicates access."

    if date_known:
        recency = f"It was last inspected about {months_since:.0f} months ago."
    else:
        recency = (
            "We don't have its last inspection date, so we assumed a typical "
            f"{months_since:.0f}-month gap."
        )

    cost_phrase = (
        f"A real inspection cost of ${insp_cost:,.0f} was on file."
        if has_real_cost
        else (
            f"We estimate a drone flight would cost about ${insp_cost:,.0f} "
            f"and take roughly {flight_minutes:.0f} minutes."
        )
    )

    return (
        f"{lead} We estimate about a {chance}% chance it has quietly slipped "
        f"into poor condition. {recency}{crossing} {cost_phrase} "
        "All figures are planning estimates, not official measurements."
    )


def _crossing_flags(crosses) -> Tuple[bool, bool]:
    """Return (crosses_rail, crosses_water) from the free-text crosses field.

    Uses word-boundary matching so rail tokens like "RR" don't fire inside
    CHERRY or TERRACE, and water tokens match whole words only.
    """
    if crosses is None or (isinstance(crosses, float) and pd.isna(crosses)):
        return False, False
    text = str(crosses).upper()
    rail = bool(RAIL_PATTERN.search(text))
    water = bool(WATER_PATTERN.search(text))
    return rail, water


def score(table: pd.DataFrame, rates, asof) -> pd.DataFrame:
    """Score each bridge as of ``asof``. See contract.md for the output schema.

    ``rates`` is a fitted grouped :class:`DeclineRates` or
    :class:`LogisticModel`. Logistic probabilities are scaled directly from
    the fit horizon to 12 months; only the grouped model uses the recency-based
    hazard path. Returns exactly: id, p_worse, consequence, insp_cost,
    flight_minutes, expected_drops_per_10k, score, voi, decision, reason,
    reason_human.
    """
    asof = pd.Timestamp(asof)
    has_cost_col = "insp_cost" in table.columns
    horizon = float(getattr(rates, "horizon_months", 48.0)) or 48.0

    # If Part A adds a partially populated length column, use each material's
    # observed median for missing rows. flight_cost() supplies explicit
    # material-level planning medians when no usable panel length exists.
    length_column = next((c for c in LENGTH_COLUMNS if c in table.columns), None)
    material_length_medians = {}
    if length_column is not None and "material" in table.columns:
        lengths = pd.to_numeric(table[length_column], errors="coerce")
        valid = lengths.notna() & np.isfinite(lengths) & (lengths > 0)
        if valid.any():
            material_length_medians = (
                pd.DataFrame(
                    {
                        "material": table.loc[valid, "material"],
                        "length_m": lengths.loc[valid],
                    }
                )
                .groupby("material")["length_m"]
                .median()
                .to_dict()
            )

    out = []
    for _, row in table.iterrows():
        rec = row.to_dict()
        condition = _condition(rec)
        # Part A defines null worst_rating as not applicable. Such rows cannot
        # be trained or ranked against numeric-condition bridges.
        if _isnan(condition):
            continue
        material_median = material_length_medians.get(rec.get("material"))
        if material_median is not None:
            rec["_material_median_length_m"] = float(material_median)

        # --- p_worse --------------------------------------------------------
        yb = rec.get("year_built")
        age = (asof.year - float(yb)) if not _isnan(yb) else (
            float(rec.get("age")) if not _isnan(rec.get("age")) else np.nan
        )
        material = rec.get("material")

        # months since last inspection, measured to asof, capped at 60. Falls
        # back to insp_freq_mo only if insp_date is missing (date_known=False).
        months_since, date_known = _months_since(
            rec.get("insp_date"), asof, rec.get("insp_freq_mo")
        )
        # Make the feature available to any model that reads it from the record.
        rec["months_since_insp"] = months_since

        # Both models give P(safety drop) over their fit horizon.
        p_horizon = float(rates.interval_prob(rec, asof=asof))
        p_horizon = float(np.clip(p_horizon, 1e-9, 1 - 1e-9))

        if isinstance(rates, (LogisticModel, ConditionBaseline)):
            # Logistic models already include months_since_insp in their design
            # when selected with that feature. Baseline also has no recency
            # hazard. Scale the fitted horizon probability directly to 12 mo.
            risk_window = 12.0
            p_worse = 1.0 - (1.0 - p_horizon) ** (risk_window / horizon)
        elif isinstance(rates, DeclineRates):
            # The grouped model has no months-since-inspection feature, so keep
            # the original constant-hazard recency path.
            hazard = _prob_to_monthly_hazard(p_horizon, horizon)
            risk_window = months_since + LOOKAHEAD_MONTHS
            p_worse = 1.0 - np.exp(-hazard * risk_window)
        else:
            raise TypeError(
                "rates must be a fitted DeclineRates, LogisticModel, or "
                "ConditionBaseline"
            )
        p_worse = float(np.clip(p_worse, 0.0, 1.0))

        # --- consequence ----------------------------------------------------
        adt = rec.get("adt")
        adt_val = 0.0 if (adt is None or _isnan(adt)) else max(0.0, float(adt))
        consequence = np.log1p(adt_val)
        if consequence <= 0:
            consequence = 0.1  # a bridge still matters even with no ADT on file

        rail, water = _crossing_flags(rec.get("crosses"))
        if rail:
            consequence *= CONSEQUENCE_RAIL_MULT
        if water:
            consequence *= CONSEQUENCE_WATER_MULT
        poor = (not _isnan(condition)) and condition <= POOR_CONDITION_THRESHOLD
        if poor:
            consequence *= CONSEQUENCE_POOR_MULT
        consequence = float(consequence)

        # --- inspection cost / flight duration -----------------------------
        estimated_cost, flight_minutes, flight_reason = flight_cost(rec)
        raw_cost = rec.get("insp_cost") if has_cost_col else None
        has_real_cost = (
            raw_cost is not None
            and not _isnan(raw_cost)
            and np.isfinite(float(raw_cost))
            and float(raw_cost) > 0
        )
        insp_cost = float(raw_cost) if has_real_cost else estimated_cost

        # --- reason ---------------------------------------------------------
        cond_txt = f"cond {int(round(condition))}"
        age_txt = "age ?" if _isnan(age) else f"{int(age)}y"
        mat_txt = material_name(material)
        window_txt = (
            f"{risk_window:.0f} mo risk window"
            if isinstance(rates, DeclineRates)
            else "12 mo forecast"
        )
        bits = [f"{mat_txt}, {age_txt}, {cond_txt}", window_txt]
        crossing_bits = []
        if rail:
            crossing_bits.append("rail")
        if water:
            crossing_bits.append("water")
        if crossing_bits:
            bits.append("crosses " + "+".join(crossing_bits))
        if poor:
            bits.append("poor condition weight")
        if date_known:
            bits.append(f"last inspected {months_since:.0f} mo ago")
        else:
            bits.append(
                f"inspection date unknown (assumed {months_since:.0f} mo)"
            )
        bits.append(f"p_worse {p_worse:.2f}")
        bits.append(flight_reason)
        if has_real_cost:
            bits.append(f"provided inspection cost ${insp_cost:,.0f} used")
        reason = "; ".join(bits)

        # --- plain-language reason (for non-technical readers) --------------
        reason_human = _human_reason(
            material=material,
            age=age,
            condition=condition,
            rail=rail,
            water=water,
            months_since=months_since,
            date_known=date_known,
            p_worse=p_worse,
            insp_cost=insp_cost,
            flight_minutes=flight_minutes,
            has_real_cost=has_real_cost,
        )

        out.append(
            {
                "id": rec.get("id"),
                "p_worse": p_worse,
                "consequence": consequence,
                "insp_cost": insp_cost,
                "flight_minutes": flight_minutes,
                "expected_drops_per_10k": np.nan,
                "score": np.nan,
                "reason": reason,
                "reason_human": reason_human,
            }
        )

    result = pd.DataFrame(out)
    if result.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Normalize consequence within this scored cohort so the metric has a
    # stable interpretation: expected consequence-weighted drops per $10k.
    consequence_median = float(result["consequence"].median())
    if not np.isfinite(consequence_median) or consequence_median <= 0:
        consequence_median = 1.0
    consequence_norm = result["consequence"] / consequence_median
    expected = (
        result["p_worse"] * consequence_norm / result["insp_cost"] * 10_000.0
    )
    result["expected_drops_per_10k"] = expected.astype(float)
    result["score"] = result["expected_drops_per_10k"]

    # Value-of-information layer: voi + decision columns, and a one-line
    # decision rationale appended to each reason. Cohort-relative (top-20%
    # VOI defines the "fly" tier), so it runs as a post-pass over the frame.
    result = add_voi(result)
    return result[OUTPUT_COLUMNS]
