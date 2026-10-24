"""Backtest harness for bridge inspection prioritization strategies.

Evaluates selection methods (baselines, oracle, model scores) that pick
bridges for inspection within a budget, using 2018 condition data as
inputs and 2022 condition outcomes as ground truth. See
.kiro/specs/backtest-harness/{requirements,design,tasks}.md.

Conventions (per requirements.md):
- Higher score = picked first, for ALL scoring functions.
- Ratings are integers 0-9; NaN means "not applicable" ("N" in source).
- Budgets are fractions of the candidate pool's total inspection cost.
- Labels: worse = is_safety_drop(min_2018, min_2022), defined locally
  below (never imported from model).
"""

import logging
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved to files, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

BAND_K = 3                # band/CI width multiplier (plots and sanity test)
TARGET_RATE = 0.15        # target decline rate for noise=0 generation
# Calibrated by simulation (task 2): at signal=0, noise=1 the latent reduces
# to BASE_RATE_LOGIT + N(0, 1), so the expected decline rate is
# E[logistic(BASE_RATE_LOGIT + Z)], independent of bridge features. Exclusions
# at signal=0 are also independent of the drawn worse labels, so the realized
# POOL rate (R14.8, measurable only after task 3's apply_exclusion_and_label)
# matches the pre-exclusion generated rate this value was calibrated against.
# -2.0 yields a generated decline rate of ~0.156 (E[logistic(B + Z)], Z~N(0,1),
# confirmed by simulation over 20 seeds at n=5000), inside the required
# [0.12, 0.22] band. Under the safety-drop label the pipeline-computed label
# equals the drawn label by construction (R14.9), so the pool rate equals the
# generated rate up to label-independent exclusions (see make_fake docstring).
BASE_RATE_LOGIT = -2.0
REL_TOL = 1e-9            # relative tolerance for budget comparisons
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 0
RANDOM_SEEDS = range(200)       # random_avg production seeds
RANDOM_SEEDS_TEST = range(20)   # random_avg seeds used in property tests
FAKE_SEEDS = range(50)          # oracle property seeds
SANITY_SEEDS = range(20)        # sanity Leg B and base-rate test seeds

RATING_COLS = ("deck", "super", "sub")
COLS_2018 = (
    "structure_id", "county", "year_built", "adt",
    "deck", "super", "sub", "inspect_cost",
)
COLS_2022 = ("structure_id", "deck", "super", "sub", "rebuilt")


# ---------------------------------------------------------------------------
# Algorithm 1: load_and_validate (R1)
# ---------------------------------------------------------------------------

def _clean_ratings(df, columns, table_name):
    """Convert rating columns to float 0-9 with "N"/empty -> NaN.

    NaN ratings are preserved; no silent conversion to 0 or 9 (R1.3).
    Raises ValueError on any non-NaN value that is not an integer 0-9.
    """
    out = df.copy()
    for col in columns:
        s = out[col]
        if s.dtype == object:
            s = s.where(~s.astype(str).str.strip().isin(("", "N")), np.nan)
        s = pd.to_numeric(s, errors="coerce")
        non_na = s.dropna()
        bad = non_na[(non_na % 1 != 0) | (non_na < 0) | (non_na > 9)]
        if not bad.empty:
            raise ValueError(
                f"{table_name}: column {col!r} has ratings outside integer "
                f"0-9 range: {sorted(bad.unique())[:5]}"
            )
        out[col] = s.astype(float)
    return out


def _check_columns(df, required, table_name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name}: missing required columns: {missing}")


def load_and_validate(df_2018, df_2022):
    """Validate and normalize the 2018 inputs and 2022 outcomes (Algorithm 1).

    Returns (df_2018, df_2022, index) where index maps structure_id -> row
    label in the 2022 table, for fast outcome lookup.

    Raises ValueError on: missing required columns; a `score` column in the
    2018 table (scores are produced by scoring functions, never read from
    input, R1.1); non-integer or out-of-range ratings; any inspect_cost
    NaN or <= 0; any adt < 0 (NaN adt passes load and is excluded at pool
    time, R1.4). Input tables are never modified.
    """
    _check_columns(df_2018, COLS_2018, "df_2018")
    _check_columns(df_2022, COLS_2022, "df_2022")
    if "score" in df_2018.columns:
        raise ValueError(
            "df_2018 must not contain a 'score' column; scores are produced "
            "by scoring functions, never read from input (R1.1)"
        )

    out18 = _clean_ratings(df_2018, RATING_COLS, "df_2018")
    out22 = _clean_ratings(df_2022, RATING_COLS, "df_2022")

    out18["structure_id"] = out18["structure_id"].astype(str)
    out22["structure_id"] = out22["structure_id"].astype(str)

    inspect_cost = pd.to_numeric(out18["inspect_cost"], errors="coerce")
    if inspect_cost.isna().any() or (inspect_cost <= 0).any():
        raise ValueError(
            "df_2018: inspect_cost must be present and > 0 for every bridge "
            f"(R1.4); found {(inspect_cost.isna() | (inspect_cost <= 0)).sum()} "
            "violating rows"
        )
    out18["inspect_cost"] = inspect_cost.astype(float)

    adt = pd.to_numeric(out18["adt"], errors="coerce")
    if (adt < 0).any():
        raise ValueError(
            "df_2018: adt must be >= 0 for every bridge (R1.4); NaN adt is "
            "allowed at load and excluded at pool time"
        )
    out18["adt"] = adt.astype(float)

    index = dict(zip(out22["structure_id"], out22.index))
    return out18, out22, index

# ---------------------------------------------------------------------------
# Algorithm 2: apply_exclusion_and_label (R1.4, R2, R3, R4.4)
# ---------------------------------------------------------------------------

EXCLUSION_REASONS = (
    "missing_2022", "rebuilt", "nan_min_2018", "nan_min_2022",
    "missing_yearbuilt_adt", "duplicate_id",
)


def apply_exclusion_and_label(df_2018, df_2022, index, county_fee):
    """Build the candidate pool, outcomes, and pool cost info (Algorithm 2).

    Exclusion reasons, checked in order per 2018 row (R2): the id is
    duplicated within either year's table (ALL rows of a duplicated id are
    excluded, counted under 'duplicate_id'); the id is absent from 2022
    ('missing_2022' — bridges replaced under a new id fall out here, R2.4);
    rebuilt in 2022; min_2018 NaN; min_2022 NaN; year_built NaN or <= 0 or
    adt NaN ('missing_yearbuilt_adt'). min = min(deck, super, sub) ignoring
    NaN (R3.1); an all-NaN triple yields NaN.

    worse = is_safety_drop(min_2018, min_2022) = (min_2022 < min_2018) and
    (min_2022 <= 4) (R3.2); a decline staying at 5 or above (e.g. 7 -> 5)
    is NOT worse. The filter uses 2022 information and is identical for all
    methods (R2.2). Input tables are never modified (R2.3).

    Returns (candidates, outcomes, pool_cost_info):
      candidates: DataFrame indexed by structure_id (str) with columns
        county (str), inspect_cost (float), adt (float), year_built (int),
        min_2018 (float).
      outcomes: dict structure_id -> {min_2018, min_2022, worse, adt}.
      pool_cost_info: {'total_cost_pool': sum(inspect_cost over POOL) +
        n_unique_pool_counties * county_fee (POOL ONLY, R4.4),
        'unique_counties': set[str], 'exclusion_counts': per-reason ints}.
    """
    # Step 0: duplicated structure_ids within either year — exclude all rows
    # of any duplicated id, counted under 'duplicate_id' (R2.1).
    dup18 = df_2018["structure_id"].duplicated(keep=False)
    dup22 = df_2022["structure_id"].duplicated(keep=False)
    dup_ids = (
        set(df_2018.loc[dup18, "structure_id"])
        | set(df_2022.loc[dup22, "structure_id"])
    )

    # 2022 lookups, keyed by structure_id (values for duplicated ids are
    # never used: those 2018 rows are excluded above).
    min22_by_id = dict(
        zip(df_2022["structure_id"], df_2022[list(RATING_COLS)].min(axis=1))
    )
    rebuilt_flags = pd.to_numeric(df_2022["rebuilt"], errors="coerce")
    rebuilt_by_id = dict(
        zip(df_2022["structure_id"], rebuilt_flags.fillna(0).astype(bool))
    )

    # 2018 columns as arrays; min ignores NaN, all-NaN triple -> NaN.
    ids = df_2018["structure_id"].to_numpy()
    counties = df_2018["county"].astype(str).to_numpy()
    year_built = pd.to_numeric(df_2018["year_built"], errors="coerce").to_numpy(dtype=float)
    adt = df_2018["adt"].to_numpy(dtype=float)
    inspect_cost = df_2018["inspect_cost"].to_numpy(dtype=float)
    min18 = df_2018[list(RATING_COLS)].min(axis=1).to_numpy()

    exclusion_counts = {reason: 0 for reason in EXCLUSION_REASONS}
    pool_rows = {}
    outcomes = {}
    for sid, county, yb, a, cost, m18 in zip(
        ids, counties, year_built, adt, inspect_cost, min18
    ):
        if sid in dup_ids:
            exclusion_counts["duplicate_id"] += 1
            continue
        if sid not in index:
            exclusion_counts["missing_2022"] += 1
            continue
        if rebuilt_by_id[sid]:
            exclusion_counts["rebuilt"] += 1
            continue
        if np.isnan(m18):
            exclusion_counts["nan_min_2018"] += 1
            continue
        m22 = min22_by_id[sid]
        if np.isnan(m22):
            exclusion_counts["nan_min_2022"] += 1
            continue
        if np.isnan(yb) or yb <= 0 or np.isnan(a):
            exclusion_counts["missing_yearbuilt_adt"] += 1
            continue
        pool_rows[sid] = (county, float(cost), float(a), int(yb), float(m18))
        outcomes[sid] = {
            "min_2018": float(m18),
            "min_2022": float(m22),
            "worse": bool(is_safety_drop(m18, m22)),
            "adt": float(a),
        }

    candidates = pd.DataFrame.from_dict(
        pool_rows,
        orient="index",
        columns=["county", "inspect_cost", "adt", "year_built", "min_2018"],
    )
    candidates.index.name = "structure_id"
    candidates = candidates.astype({
        "county": str, "inspect_cost": float, "adt": float,
        "year_built": int, "min_2018": float,
    })

    unique_counties = set(candidates["county"])
    pool_cost_info = {
        "total_cost_pool": float(candidates["inspect_cost"].sum())
        + len(unique_counties) * float(county_fee),
        "unique_counties": unique_counties,
        "exclusion_counts": exclusion_counts,
    }
    return candidates, outcomes, pool_cost_info

# ---------------------------------------------------------------------------
# Algorithm 3: Cost — one rule, two functions (R4)
# ---------------------------------------------------------------------------

def marginal_cost(county, inspect_cost, counties_seen, county_fee):
    """Marginal cost of selecting one bridge (R4.1).

    inspect_cost + county_fee when this is the FIRST selected bridge of its
    county (county not in counties_seen), else inspect_cost: a county's fee
    is paid at most once, and only when at least one of its bridges is
    selected (R4.1, R4.3). This is the ONLY cost rule in the codebase —
    pick() accumulates it incrementally and cost_accounting() iterates it
    over a finished selection (R4.2). inspect_cost, county_fee, and budget
    arithmetic share one unit (R4.6).
    """
    return inspect_cost + (county_fee if county not in counties_seen else 0.0)


def cost_accounting(selected_ids, candidates, county_fee):
    """Total cost of a selection, accumulated in selection order (R4.2).

    Iterates selected_ids in order, accumulating marginal_cost while
    tracking counties seen, so total_cost = sum(inspect_cost) +
    n_unique_selected_counties * county_fee (R4.3) — NEVER
    sum(inspect_cost + county_fee): a county's fee is paid at most once
    (R4.6). Returns (total_cost, n_unique_selected_counties).
    """
    seen = set()
    total = 0.0
    for sid in selected_ids:
        county = candidates.loc[sid, "county"]
        total += marginal_cost(
            county, candidates.loc[sid, "inspect_cost"], seen, county_fee
        )
        seen.add(county)
    return total, len(seen)

# ---------------------------------------------------------------------------
# Algorithm 4: pick (R5)
# ---------------------------------------------------------------------------

def pick(scores, budget, candidates, pool_cost_info, county_fee,
         rank_by_value_per_cost=False, tiebreak="adt", rng=None):
    """Greedy budgeted selection by score (Algorithm 4, R5).

    scores: dict structure_id -> float; higher = picked first, all methods,
    no exceptions (R5.2). Returns selected structure_ids in selection order.

    Validation (R5.3): ValueError unless ALL scores are finite (both modes);
    when rank_by_value_per_cost=True, additionally ValueError if any score
    is negative; ValueError if tiebreak="random" and rng is None.

    Ranking (R5.4): key = score, or score / inspect_cost when
    rank_by_value_per_cost=True (model scores and oracle). The value-per-
    cost division happens HERE, never inside scoring functions (R5).

    Tie-break (R5.5): tiebreak="adt" sorts by (key DESC, adt DESC), stable;
    tiebreak="random" shuffles the candidate order with rng FIRST, then
    stable-sorts by key DESC only (equal keys keep shuffled order).

    Greedy (R5.6, R4.2): iterate in rank order, tentatively add each bridge
    via marginal_cost (the ONLY cost rule), keep it while
    spent + marginal <= available * (1 + REL_TOL), with
    available = budget * pool_cost_info['total_cost_pool'] (R4.4, R4.5).
    O(n log n) overall. Candidates missing from scores are skipped with a
    logged warning; scores for non-candidates are ignored (R5.7).
    """
    values = np.asarray(list(scores.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "pick: all scores must be finite (no NaN/inf) (R5.3)"
        )
    if rank_by_value_per_cost and (values < 0).any():
        raise ValueError(
            "pick: rank_by_value_per_cost=True requires non-negative "
            "scores (R5.3)"
        )
    if tiebreak == "random" and rng is None:
        raise ValueError(
            'pick: tiebreak="random" requires an rng (R5.3)'
        )

    available = budget * pool_cost_info["total_cost_pool"]

    # Candidates missing from scores are skipped (R5.7); scores keyed by
    # non-candidates never enter ids and are ignored.
    ids = np.array([sid for sid in candidates.index if sid in scores])
    n_skipped = len(candidates) - len(ids)
    if n_skipped:
        logger.warning(
            "pick: %d candidate(s) missing from scores; skipped (R5.7)",
            n_skipped,
        )

    keys = np.array([scores[sid] for sid in ids], dtype=float)
    inspect_cost = candidates.loc[ids, "inspect_cost"].to_numpy(dtype=float)
    if rank_by_value_per_cost:
        keys = keys / inspect_cost
    adt = candidates.loc[ids, "adt"].to_numpy(dtype=float)

    if tiebreak == "random":
        # Shuffle FIRST, then stable sort by key only (R5.5).
        perm = rng.permutation(len(ids))
        ids, keys, inspect_cost, adt = (
            ids[perm], keys[perm], inspect_cost[perm], adt[perm]
        )
        order = np.argsort(-keys, kind="stable")
    else:
        # lexsort: indirect stable sort, LAST key is primary:
        # (key DESC, adt DESC) (R5.5).
        order = np.lexsort((-adt, -keys))

    counties = candidates.loc[ids, "county"].to_numpy()
    selected = []
    seen = set()
    spent = 0.0
    for i in order:
        m = marginal_cost(counties[i], inspect_cost[i], seen, county_fee)
        if spent + m <= available * (1 + REL_TOL):
            selected.append(ids[i])
            spent += m
            seen.add(counties[i])
    return selected

# ---------------------------------------------------------------------------
# Algorithm 5: Scoring (R6-R9) — all return dict structure_id -> float,
# higher = picked first; tie-break behavior comes from pick's tiebreak
# argument (R5.5), never from the scoring functions themselves.
# ---------------------------------------------------------------------------

def score_worst_first(candidates):
    """worst_first baseline (R6): score = -min_2018.

    Rank by score only (rank_by_value_per_cost=False): the bridge with the
    lowest 2018 condition min gets the highest score and is picked first.
    Returns a dict structure_id -> float keyed by the candidate index.
    """
    return {sid: -float(m) for sid, m in candidates["min_2018"].items()}


def score_oldest_first(candidates):
    """oldest_first baseline (R7): score = -year_built.

    Rank by score only: the oldest bridge gets the highest score and is
    picked first. Returns a dict structure_id -> float keyed by the
    candidate index.
    """
    return {sid: -float(y) for sid, y in candidates["year_built"].items()}


def score_random(candidates, seed):
    """random_avg baseline (R8): score = uniform random draw per candidate.

    One independent draw from np.random.default_rng(seed) per candidate,
    ranked by score only. Scores are almost surely distinct, so pick's
    tiebreak has no effect. Returns a dict structure_id -> float keyed by
    the candidate index.
    """
    rng = np.random.default_rng(seed)
    return {sid: float(s) for sid, s in zip(candidates.index, rng.random(len(candidates)))}


def score_oracle(outcomes):
    """Oracle (R9): score = log1p(adt) if worse, else 0.

    Uses the realized outcomes' worse labels — the best achievable signal.
    Ranked by value per cost (rank_by_value_per_cost=True in pick, which
    does the /inspect_cost division): best achievable weighted recall per
    dollar under the budget rule. Scores are non-negative, satisfying
    pick's value-per-cost validation (R5.3). Returns a dict
    structure_id -> float.
    """
    return {
        sid: float(np.log1p(o["adt"])) if o["worse"] else 0.0
        for sid, o in outcomes.items()
    }


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _z(x):
    """Standardize over the dataset; constant columns map to zeros."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


# ---------------------------------------------------------------------------
# Algorithm 6: evaluate_metrics (R10)
# ---------------------------------------------------------------------------

@dataclass
class EvaluationMetrics:
    """Per (method, budget) evaluation record (R10, design Data Model).

    Unweighted and weighted metrics are always separate (R10.3); score is
    never a weight. cost_total is left NaN here: it needs candidates and
    county_fee, so run_backtest fills it in via cost_accounting (R4.2).
    """
    tp: int = 0
    fp: int = 0
    weighted_tp: float = 0.0
    weighted_fp: float = 0.0
    recall: float = 0.0
    weighted_recall: float = 0.0
    count_selected: int = 0
    count_pool: int = 0
    count_worse_in_pool: int = 0
    cost_total: float = float("nan")


def evaluate_metrics(selected_ids, outcomes, adt_dict):
    """Evaluate one selection against realized outcomes (Algorithm 6, R10).

    selected_ids: ids in selection order (from pick; duplicates would be a
      bug — the set here mirrors Algorithm 6's S). outcomes: pool outcomes
      dict from apply_exclusion_and_label. adt_dict: structure_id -> adt
      (evaluation weights come from HERE, never from score, R10.3).

    Headline (R10.1):
      recall          = tp / count of worse bridges in pool  (0.0 if none)
      weighted_recall = sum log1p(adt) over selected worse /
                        sum log1p(adt) over all pool worse   (0.0 if 0)
    Supporting (R10.2): tp, fp, weighted_tp, weighted_fp (weights are
    log1p(adt) per bridge), count_selected, count_pool, count_worse_in_pool.
    All metrics are within the candidate pool; cost_total is NaN (filled by
    run_backtest via cost_accounting).
    """
    worse_ids = {sid for sid, o in outcomes.items() if o["worse"]}
    selected = set(selected_ids)

    tp = fp = 0
    weighted_tp = weighted_fp = 0.0
    for sid in selected:
        w = float(np.log1p(adt_dict[sid]))
        if sid in worse_ids:
            tp += 1
            weighted_tp += w
        else:
            fp += 1
            weighted_fp += w

    total_worse_weight = float(
        sum(float(np.log1p(adt_dict[sid])) for sid in worse_ids)
    )
    return EvaluationMetrics(
        tp=tp,
        fp=fp,
        weighted_tp=weighted_tp,
        weighted_fp=weighted_fp,
        recall=tp / len(worse_ids) if worse_ids else 0.0,
        weighted_recall=(
            weighted_tp / total_worse_weight if total_worse_weight > 0 else 0.0
        ),
        count_selected=len(selected),
        count_pool=len(outcomes),
        count_worse_in_pool=len(worse_ids),
    )

# ---------------------------------------------------------------------------
# Label (R3): safety drop — defined locally, never imported from model
# ---------------------------------------------------------------------------

def is_safety_drop(start, end):
    """Safety-drop label (R3): (end < start) and (end <= 4) on integer mins.

    NaN on either side yields False (such bridges are excluded from the
    pool, R2). Defined locally so the harness stands alone; do NOT import
    this from model. Accepts scalars or numpy arrays; returns bool(s).
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return (end < start) & (end <= 4)


def make_fake(n, seed, signal, noise):
    """Generate a synthetic (df_2018, df_2022, truth) triple (Algorithm 9, R13).

    truth is indexed by structure_id with columns latent, p_decline (R13.1),
    plus a `worse` column carrying the generator's drawn label (generator
    internals, used by the generator-consistency property R14.9).

    Every bridge has min_2018 >= 1 and at least one non-NaN 2018 rating
    (R13.2). Feature term (R13.3):
        f      = 2.0*z(9 - min_2018) + 1.0*z(age) + 0.5*z(log1p(adt))
        latent = BASE_RATE_LOGIT + signal*f + noise*N(0, 1)
    with age = 2018 - year_built and z standardized over the dataset.

    noise == 0: requires signal > 0 (else ValueError); worse = the top
    round(TARGET_RATE * n) bridges by latent — deterministic, ~15% (R13.4).
    noise > 0: worse ~ bernoulli(p_decline) (R13.5).

    Rating evolution (R13.6, consistent with the safety-drop label by
    construction): worse bridges with min_2018 >= 5 get min_2022 uniform
    in 0..4; worse bridges with min_2018 <= 4 fall by drop in
    1..min(3, min_2018). Non-worse bridges are NEVER safety drops:
    min_2018 <= 4 -> min_2022 >= min_2018; min_2018 >= 5 -> min_2022 >= 5
    (a bridge at 5 can never be labeled non-worse by landing on 4).
    2022 rating columns are regenerated consistent with the new min
    (one column anchored exactly at min_2022).

    Plausible distributions (R13.8): integer ratings 0-9 skewed toward 5-8
    with ~15% NaN per column, lognormal adt and inspect_cost (positive),
    ~14 counties, ~2% rebuilt and ~3% missing-from-2022 bridges so the
    exclusion rules are exercised.

    Calibration note (R13.7): at signal=0, noise=1 the latent is
    BASE_RATE_LOGIT + N(0, 1), so features do not matter and exclusions
    (rebuilt / missing / NaN) are independent of the drawn worse labels.
    BASE_RATE_LOGIT was therefore calibrated against the pre-exclusion
    generated decline rate (a proxy for the pool rate, measurable only
    after apply_exclusion_and_label exists): a seed sweep at n=5000 with
    signal=0, noise=1 gives a mean generated rate of ~0.156, inside the
    required [0.12, 0.22] band.
    """
    if noise == 0 and signal <= 0:
        raise ValueError("make_fake: noise=0 requires signal > 0 (R13.4)")

    rng = np.random.default_rng(seed)

    structure_ids = np.array([f"S{i:07d}" for i in range(n)])
    counties = np.array([f"C{c:03d}" for c in rng.integers(0, 14, n)])
    year_built = rng.integers(1850, 2022, n)
    adt = rng.lognormal(mean=8.0, sigma=1.0, size=n)          # positive, skewed
    inspect_cost = 0.5 + rng.lognormal(mean=0.0, sigma=0.5, size=n)  # > 0

    # 2018 ratings: integers 0-9 skewed toward 5-8, ~15% NaN per column.
    ratings18 = np.clip(np.rint(rng.normal(6.5, 1.5, size=(n, 3))), 0, 9)
    ratings18[rng.random((n, 3)) < 0.15] = np.nan
    # At least one non-NaN 2018 rating per bridge (R13.2).
    all_nan = np.isnan(ratings18).all(axis=1)
    if all_nan.any():
        rows = np.flatnonzero(all_nan)
        ratings18[rows, rng.integers(0, 3, rows.size)] = rng.integers(1, 10, rows.size)
    # Enforce min_2018 >= 1 for every bridge (R13.2).
    below_one = ratings18 < 1
    if below_one.any():
        ratings18[below_one] = rng.integers(1, 10, int(below_one.sum()))
    min18 = np.nanmin(ratings18, axis=1)

    # Feature term and latent (R13.3).
    age = 2018.0 - year_built.astype(float)
    f = 2.0 * _z(9.0 - min18) + 1.0 * _z(age) + 0.5 * _z(np.log1p(adt))
    latent = BASE_RATE_LOGIT + signal * f + noise * rng.standard_normal(n)
    p_decline = _logistic(latent)

    # Draw worse labels (R13.4 / R13.5).
    if noise == 0:
        k = int(round(TARGET_RATE * n))
        worse = np.zeros(n, dtype=bool)
        worse[np.argsort(-latent, kind="stable")[:k]] = True
    else:
        worse = rng.random(n) < p_decline

    # Rating evolution (R13.6) under the safety-drop label: the drawn
    # labels are made exactly consistent with is_safety_drop, so the
    # pipeline label equals the generator's drawn label (R14.9).
    start = min18.astype(int)                                  # >= 1 (R13.2)
    hi = start >= 5
    drop_cap = np.minimum(3, start)
    drop = 1 + np.floor(rng.random(n) * drop_cap)              # uniform 1..cap
    end_worse = np.where(hi, rng.integers(0, 5, n), start - drop)
    end_ok = np.where(
        hi,
        rng.integers(5, 10, n),                                # 5..9: may fall, never <= 4
        start + np.floor(rng.random(n) * (10 - start)),        # start..9: never falls
    )
    min22 = np.where(worse, end_worse, end_ok).astype(float)

    # Regenerate 2022 ratings consistent with the new min: every column is
    # >= min_2022 and one anchored column equals min_2022 exactly.
    up_cap = np.minimum((9 - min22).astype(int), 3)
    ratings22 = min22[:, None] + np.floor(
        rng.random((n, 3)) * (up_cap[:, None] + 1)
    )
    anchor = rng.integers(0, 3, n)
    ratings22[np.arange(n), anchor] = min22
    nan22 = rng.random((n, 3)) < 0.15
    nan22[np.arange(n), anchor] = False  # anchor stays non-NaN; min_2022 defined
    ratings22[nan22] = np.nan

    df_2018 = pd.DataFrame({
        "structure_id": structure_ids,
        "county": counties,
        "year_built": year_built.astype(int),
        "adt": adt,
        "deck": ratings18[:, 0],
        "super": ratings18[:, 1],
        "sub": ratings18[:, 2],
        "inspect_cost": inspect_cost,
    })

    # Exercise exclusions: ~2% rebuilt flags and ~3% rows absent from 2022.
    rebuilt = rng.random(n) < 0.02
    keep = rng.random(n) >= 0.03
    df_2022 = pd.DataFrame({
        "structure_id": structure_ids,
        "deck": ratings22[:, 0],
        "super": ratings22[:, 1],
        "sub": ratings22[:, 2],
        "rebuilt": rebuilt,
    })[keep].reset_index(drop=True)

    truth = pd.DataFrame(
        {"latent": latent, "p_decline": p_decline, "worse": worse},
        index=pd.Index(structure_ids, name="structure_id"),
    )
    return df_2018, df_2022, truth


# ---------------------------------------------------------------------------
# Algorithm 10: adapt_panel (real-data adapter)
# ---------------------------------------------------------------------------

PANEL_COLS_2018 = (
    "structure_id", "county", "year_built", "adt",
    "deck", "super", "sub", "inspect_cost",
)
PANEL_COLS_2022 = ("structure_id", "deck", "super", "sub", "rebuilt")


def adapt_panel(path="bridges/panel.parquet", flat_inspect_cost=1.0):
    """Adapt bridges/panel.parquet to the R1 (df_2018, df_2022) formats.

    The parquet is never mutated; columns are selected by name only; year
    2012 and 2016 rows are ignored.

    df_2018 = rows with year == 2018: structure_id (stripped str), county,
    year_built = 2018 - age, adt, deck, super, sub, inspect_cost =
    flat_inspect_cost. The flat cost is a "placeholder units" ASSUMPTION
    (the panel's improvement_cost column is an improvement cost, not an
    inspection cost, and is NOT used).

    df_2022 = rows with year == 2022: structure_id (stripped), deck, super,
    sub, rebuilt — taken DIRECTLY from the panel's rebuilt column (A's
    definition: the bridge improved or was reconstructed); it is NOT
    recomputed here.

    Prints row counts per year, the count of rebuilt=True in 2022, and both
    unmatched-ID counts (2018-absent-from-2022 and 2022-absent-from-2018).
    """
    panel = pd.read_parquet(path)

    counts = panel["year"].value_counts().sort_index()
    for yr, n in counts.items():
        note = "" if yr in (2018, 2022) else "  (ignored)"
        print(f"year {yr}: {n} rows{note}")

    rows18 = panel.loc[panel["year"] == 2018]
    rows22 = panel.loc[panel["year"] == 2022]

    df_2018 = pd.DataFrame({
        "structure_id": rows18["structure_id"].astype(str).str.strip(),
        "county": rows18["county"],
        "year_built": 2018 - rows18["age"].astype(int),
        "adt": rows18["adt"].astype(float),
        "deck": rows18["deck"],
        "super": rows18["super"],
        "sub": rows18["sub"],
        "inspect_cost": float(flat_inspect_cost),  # ASSUMPTION: placeholder units
    })

    df_2022 = pd.DataFrame({
        "structure_id": rows22["structure_id"].astype(str).str.strip(),
        "deck": rows22["deck"],
        "super": rows22["super"],
        "sub": rows22["sub"],
        # A's definition (improved or reconstructed); NOT recomputed here.
        "rebuilt": rows22["rebuilt"].astype(bool),
    })

    print(f"rebuilt=True in 2022: {int(df_2022['rebuilt'].sum())}")
    ids18, ids22 = set(df_2018["structure_id"]), set(df_2022["structure_id"])
    print(f"2018 ids absent from 2022: {len(ids18 - ids22)}")
    print(f"2022 ids absent from 2018: {len(ids22 - ids18)}")

    return df_2018, df_2022

# ---------------------------------------------------------------------------
# Algorithm 7: run_backtest (R11)
# ---------------------------------------------------------------------------

RESULT_COLUMNS = (
    "method", "budget", "seed", "tiebreak", "tp", "fp", "weighted_tp",
    "weighted_fp", "recall", "weighted_recall", "count_selected",
    "cost_total", "is_aggregate", "stat",
)


def run_backtest(df_2018, df_2022, county_fee,
                 budgets=[0.01, 0.02, 0.05, 0.10, 0.20],
                 model_scores=None, tiebreaks=("adt",)):
    """Run every selection method over every budget on ONE shared pool (R11).

    Canonical signature (R11.1). The candidate pool is built ONCE via
    load_and_validate + apply_exclusion_and_label and shared by all methods
    (R11.2), so no method gains an advantage from the filter (R2.2).

    Methods per budget (R11.2):
      worst_first / oldest_first — the deterministic baselines. For each
        entry in `tiebreaks`: "adt" runs each baseline ONCE with
        tiebreak="adt"; "random" instead runs each over RANDOM_SEEDS_TEST
        (20 seeds) with tiebreak="random" and reports the MEAN of recall and
        weighted_recall as ONE aggregate row with tiebreak="random",
        is_aggregate=True, stat="mean" (R11.3). Other metric columns are NaN
        on that row: like the random_avg aggregates (R8.3), only recall and
        weighted_recall are aggregated. The default call tiebreaks=("adt",)
        therefore yields exactly ONE worst_first row per budget (and one per
        other deterministic method) (R11.3). Unknown tiebreak entries raise
        ValueError.
      random_avg — one run per seed in RANDOM_SEEDS (200 production seeds,
        R8.2); every per-seed row is kept (is_aggregate=False) and each
        budget also gets four aggregate rows (is_aggregate=True,
        stat in {"mean", "std", "min", "max"}) aggregating ONLY recall and
        weighted_recall (R8.3, R11.4); std is the sample std (ddof=1).
      oracle — one run per budget, ranked by value per cost (R9).
      model — pick(model_scores, ..., rank_by_value_per_cost=True), one run
        per budget, only when model_scores is given (R11.2); pick's
        validation applies (finite, non-negative scores, R5.3).

    Returns a DataFrame with columns RESULT_COLUMNS (design Data Model):
    one row per (method, budget [, seed, tiebreak]) plus random_avg's
    aggregate rows (R11.4). seed/tiebreak are NaN where not applicable
    (deterministic single runs have seed NaN; random_avg, oracle, and model
    rows have tiebreak NaN). cost_total is filled via cost_accounting, the
    ONLY cost rule (R4.2); tp/fp/weighted_tp/weighted_fp/recall/
    weighted_recall come from evaluate_metrics (R10) with weights from the
    outcomes' adt (never from score, R10.3).
    """
    unknown = set(tiebreaks) - {"adt", "random"}
    if unknown:
        raise ValueError(
            f"run_backtest: unknown tiebreak entries {sorted(unknown)}; "
            'valid entries are "adt" and "random"'
        )

    df_2018, df_2022, index = load_and_validate(df_2018, df_2022)
    candidates, outcomes, pool_cost_info = apply_exclusion_and_label(
        df_2018, df_2022, index, county_fee
    )
    logger.info(
        "run_backtest: pool=%d bridges (%d worse); exclusions=%s",
        len(candidates),
        sum(o["worse"] for o in outcomes.values()),
        pool_cost_info["exclusion_counts"],
    )
    adt_dict = {sid: o["adt"] for sid, o in outcomes.items()}

    # Budget-independent score dicts, computed once and shared by all budgets.
    baseline_scores = {
        "worst_first": score_worst_first(candidates),
        "oldest_first": score_oldest_first(candidates),
    }
    oracle_scores = score_oracle(outcomes)
    random_scores = {s: score_random(candidates, s) for s in RANDOM_SEEDS}

    def _row(method, budget, selected, seed, tiebreak):
        m = evaluate_metrics(selected, outcomes, adt_dict)
        cost_total, _ = cost_accounting(selected, candidates, county_fee)
        return {
            "method": method, "budget": budget, "seed": seed,
            "tiebreak": tiebreak, "tp": m.tp, "fp": m.fp,
            "weighted_tp": m.weighted_tp, "weighted_fp": m.weighted_fp,
            "recall": m.recall, "weighted_recall": m.weighted_recall,
            "count_selected": m.count_selected, "cost_total": cost_total,
            "is_aggregate": False, "stat": None,
        }

    def _aggregate_row(method, budget, tiebreak, stat, recalls, weighted_recalls):
        # Only recall and weighted_recall are aggregated (R8.3); all other
        # metric columns are NaN on aggregate rows.
        rec = np.asarray(recalls, dtype=float)
        wrec = np.asarray(weighted_recalls, dtype=float)
        reduce = {
            "mean": lambda x: float(np.mean(x)),
            "std": lambda x: float(np.std(x, ddof=1)),
            "min": lambda x: float(np.min(x)),
            "max": lambda x: float(np.max(x)),
        }[stat]
        return {
            "method": method, "budget": budget, "seed": None,
            "tiebreak": tiebreak, "tp": None, "fp": None,
            "weighted_tp": None, "weighted_fp": None,
            "recall": reduce(rec), "weighted_recall": reduce(wrec),
            "count_selected": None, "cost_total": None,
            "is_aggregate": True, "stat": stat,
        }

    rows = []
    for budget in budgets:
        # Deterministic baselines, once per tiebreak entry (R11.3).
        for tb in tiebreaks:
            for method, scores in baseline_scores.items():
                if tb == "adt":
                    rows.append(_row(
                        method, budget,
                        pick(scores, budget, candidates, pool_cost_info,
                             county_fee, tiebreak="adt"),
                        None, "adt",
                    ))
                else:  # "random": 20 seeded runs, MEAN reported as ONE row.
                    recalls, weighted_recalls = [], []
                    for s in RANDOM_SEEDS_TEST:
                        m = evaluate_metrics(
                            pick(scores, budget, candidates, pool_cost_info,
                                 county_fee, tiebreak="random",
                                 rng=np.random.default_rng(s)),
                            outcomes, adt_dict,
                        )
                        recalls.append(m.recall)
                        weighted_recalls.append(m.weighted_recall)
                    rows.append(_aggregate_row(
                        method, budget, "random", "mean",
                        recalls, weighted_recalls,
                    ))

        # random_avg: all per-seed rows kept, plus mean/std/min/max (R8.3).
        per_seed_recalls, per_seed_weighted = [], []
        for s in RANDOM_SEEDS:
            row = _row(
                "random_avg", budget,
                pick(random_scores[s], budget, candidates, pool_cost_info,
                     county_fee),
                s, None,
            )
            rows.append(row)
            per_seed_recalls.append(row["recall"])
            per_seed_weighted.append(row["weighted_recall"])
        for stat in ("mean", "std", "min", "max"):
            rows.append(_aggregate_row(
                "random_avg", budget, None, stat,
                per_seed_recalls, per_seed_weighted,
            ))

        # Oracle: best achievable weighted recall per dollar (R9).
        rows.append(_row(
            "oracle", budget,
            pick(oracle_scores, budget, candidates, pool_cost_info,
                 county_fee, rank_by_value_per_cost=True, tiebreak="adt"),
            None, None,
        ))

        # Model scores, ranked by value per cost, only when given (R11.2).
        if model_scores is not None:
            rows.append(_row(
                "model", budget,
                pick(model_scores, budget, candidates, pool_cost_info,
                     county_fee, rank_by_value_per_cost=True, tiebreak="adt"),
                None, None,
            ))

    result = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
    result["is_aggregate"] = result["is_aggregate"].astype(bool)
    return result

# ---------------------------------------------------------------------------
# Algorithm 8: plot (R12) — headline figures, with bootstrap_gap_ci (R12.2)
# providing per-method gap-CI error bars, drawn by _plot_metric.
# ---------------------------------------------------------------------------

def bootstrap_gap_ci(selections, outcomes, adt, metric):
    """Bootstrap 95% CI on each method's metric gap vs worst_first (R12.2).

    selections: method -> {budget: selected_ids}, the selected_ids from the
      actual run — FIXED throughout, never re-picked. Must include
      "worst_first" (every gap is computed against it at the SAME budget).
    outcomes: pool outcomes dict from apply_exclusion_and_label (the fixed
      shared pool, R11.2). adt: structure_id -> adt; evaluation weights are
      log1p(adt), never score (R10.3). metric: "recall" (Fig 1) or
      "weighted_recall" (Fig 2) (R10.1).

    Per replicate (BOOTSTRAP_REPS = 1000, rng = default_rng(BOOTSTRAP_SEED)):
        w = rng.multinomial(n_pool, uniform)  # weights over pool row indices
        numerator AND denominator recomputed as w-weighted sums:
            recall          = sum_i w_i * 1[i selected] * worse_i
                              / sum_i w_i * worse_i
            weighted_recall = sum_i w_i * 1[i selected] * worse_i * log1p(adt_i)
                              / sum_i w_i * worse_i * log1p(adt_i)
        (a replicate with denominator 0 contributes 0.0, matching
        evaluate_metrics), then
            gap = method_metric - worst_first_metric
        under the SAME w (paired replicates). The CI is the 2.5/97.5
        percentile band of the per-replicate gaps; worst_first's gap vs
        itself is exactly (0, 0).

    Returns {method: {budget: (lo, hi)}} for every method in selections.

    Because selections stay fixed, these CIs capture outcome-sampling
    variance only, not selection variance (R12.3) — any report of them must
    state that.
    """
    if metric not in ("recall", "weighted_recall"):
        raise ValueError(
            'bootstrap_gap_ci: metric must be "recall" or '
            f'"weighted_recall", got {metric!r}'
        )
    if "worst_first" not in selections:
        raise ValueError(
            "bootstrap_gap_ci: selections must include 'worst_first' "
            "(gaps are vs worst_first, R12.2)"
        )

    pool_ids = np.array(list(outcomes))          # fixed pool row order (R11.2)
    n_pool = len(pool_ids)
    worse = np.array(
        [outcomes[sid]["worse"] for sid in pool_ids], dtype=float
    )
    log_adt = np.array(
        [float(np.log1p(adt[sid])) for sid in pool_ids], dtype=float
    )
    # Per-row denominator contributions before the replicate weights apply;
    # numerators are the same rows masked down to each selection.
    denom_rows = worse if metric == "recall" else worse * log_adt

    # One selected-row mask per (method, budget), in insertion order.
    pairs = [
        (method, budget)
        for method, per_budget in selections.items()
        for budget in per_budget
    ]
    sel_matrix = np.array(
        [np.isin(pool_ids, list(selections[method][budget]))
         for method, budget in pairs],
        dtype=float,
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    uniform = np.full(n_pool, 1.0 / n_pool)
    gaps = np.empty((len(pairs), BOOTSTRAP_REPS))
    for rep in range(BOOTSTRAP_REPS):
        w = rng.multinomial(n_pool, uniform)
        contrib = w * denom_rows
        denom = contrib.sum()
        values = (
            (sel_matrix @ contrib) / denom
            if denom > 0 else np.zeros(len(pairs))
        )
        wf_values = {
            budget: value
            for (method, budget), value in zip(pairs, values)
            if method == "worst_first"
        }
        gaps[:, rep] = [
            value - wf_values[budget]
            for (method, budget), value in zip(pairs, values)
        ]

    result = {}
    for (method, budget), pair_gaps in zip(pairs, gaps):
        lo, hi = np.percentile(pair_gaps, [2.5, 97.5])
        result.setdefault(method, {})[budget] = (float(lo), float(hi))
    return result

# Canonical legend/line order; methods outside this list keep their order of
# appearance after these.
_METHOD_ORDER = ("worst_first", "oldest_first", "random_avg", "oracle", "model")


def _ordered_methods(results_df):
    """Methods present in results_df, in canonical plot order."""
    present = list(dict.fromkeys(results_df["method"]))
    ordered = [m for m in _METHOD_ORDER if m in present]
    return ordered + [m for m in present if m not in _METHOD_ORDER]


def _reference_series(results_df, method, metric):
    """budget -> metric value for `method`'s canonical line (R12.2 wiring).

    Uses the rows _plot_metric draws for a deterministic method (single runs
    plus tiebreak="random" MEAN rows, R11.3), preferring the tiebreak="adt"
    line when several exist — the canonical deterministic run whose
    selections bootstrap_gap_ci receives.
    """
    sub = results_df[
        (results_df["method"] == method)
        & (~results_df["is_aggregate"] | (results_df["stat"] == "mean"))
    ]
    adt_rows = sub[sub["tiebreak"] == "adt"]
    if not adt_rows.empty:
        sub = adt_rows
    sub = sub.sort_values("budget")
    return dict(zip(sub["budget"], sub[metric]))


def _plot_metric(results_df, metric, ylabel, title, gap_ci=None):
    """Draw ONE headline figure (budget vs `metric`) and return its Figure.

    One line per method; oracle dashed (R12.1). random_avg is drawn from its
    aggregate rows only: the stat="mean" line plus a shaded band of
    mean +/- BAND_K*std over the per-seed runs (BAND_K = 3, the same constant
    as sanity property 7, R12.1a). Deterministic methods use their single-run
    rows (is_aggregate=False); a baseline run under several tiebreaks gets
    one line per tiebreak value, the tiebreak="random" line coming from the
    single MEAN row run_backtest reports for it (R11.3).

    Bootstrap gap CIs vs worst_first (R12.2, bootstrap_gap_ci) are drawn
    when gap_ci is given: each method with a CI gets error bars on its
    canonical line (the tiebreak="adt" line when a baseline has several),
    the bar at budget b spanning [worst_first(b) + lo, worst_first(b) + hi]
    on the method's own metric scale — so the bar covers the worst_first
    line exactly when the 95% CI contains a zero gap. worst_first itself
    gets no bar (its gap vs itself is exactly (0, 0)); random_avg keeps its
    band instead. metric="recall" feeds Fig 1, "weighted_recall" Fig 2.
    Because selections stay fixed, these CIs capture outcome-sampling
    variance only, not selection variance (R12.3).
    """
    wf_ref = None
    if gap_ci is not None:
        wf_ref = _reference_series(results_df, "worst_first", metric)
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in _ordered_methods(results_df):
        sub = results_df[results_df["method"] == method]
        if method == "random_avg":
            agg = sub[sub["is_aggregate"]].set_index("budget").sort_index()
            mean = agg.loc[agg["stat"] == "mean", metric]
            std = agg.loc[agg["stat"] == "std", metric].reindex(mean.index)
            budgets = mean.index.to_numpy(dtype=float)
            center = mean.to_numpy(dtype=float)
            half = BAND_K * std.to_numpy(dtype=float)
            (line,) = ax.plot(budgets, center, marker="o", label="random_avg")
            ax.fill_between(
                budgets, center - half, center + half,
                color=line.get_color(), alpha=0.2,
                label=f"random_avg ± {BAND_K}·std",
            )
            continue
        # Single runs, plus the tiebreak="random" MEAN row when present
        # (R11.3); deterministic baselines never produce other aggregates.
        sub = sub[~sub["is_aggregate"] | (sub["stat"] == "mean")]
        multi = sub["tiebreak"].nunique(dropna=False) > 1
        for tb, grp in sub.groupby("tiebreak", dropna=False):
            grp = grp.sort_values("budget")
            label = f"{method} (tiebreak={tb})" if multi else method
            style = {"linestyle": "--"} if method == "oracle" else {}
            budgets = grp["budget"].to_numpy(dtype=float)
            values = grp[metric].to_numpy(dtype=float)
            (line,) = ax.plot(budgets, values, marker="o", label=label, **style)
            # R12.2 gap-CI error bars, shifted onto this method's metric
            # scale by worst_first's line value (see docstring).
            if (
                gap_ci is not None
                and method in gap_ci
                and method != "worst_first"
                and (not multi or tb == "adt")
                and wf_ref
            ):
                xs, ys, below, above = [], [], [], []
                for b, v in zip(budgets, values):
                    if b in gap_ci[method] and b in wf_ref:
                        lo, hi = gap_ci[method][b]
                        xs.append(b)
                        ys.append(v)
                        below.append(max(v - (wf_ref[b] + lo), 0.0))
                        above.append(max((wf_ref[b] + hi) - v, 0.0))
                if xs:
                    ax.errorbar(
                        xs, ys, yerr=[below, above], fmt="none",
                        ecolor=line.get_color(), elinewidth=1.2,
                        capsize=3, alpha=0.7,
                    )
    ax.set_xlabel("Budget (fraction of candidate-pool inspection cost)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot(results_df, output_path=None, selections=None, outcomes=None,
         adt=None):
    """Headline figures for a run_backtest result (Algorithm 8, R12).

    Two figures only — no precision-recall curves: the budget fixes the
    selection size, so PR curves are degenerate (R12.1):
      1. budget vs recall (unweighted headline, R10.1);
      2. budget vs weighted recall (log1p(adt) weights, R10.1).
    Both figures: one line per method, oracle dashed, random_avg band =
    mean +/- BAND_K*std over its per-seed runs (BAND_K = 3), legend, grid,
    axis labels, title (R12.1, R12.4).

    output_path is a PREFIX (R12.4): when given, writes
    "{output_path}_recall.png" and "{output_path}_weighted.png"; when None,
    nothing is written. Returns (fig_recall, fig_weighted) either way; the
    caller owns the open figures.

    When selections (method -> {budget: selected_ids}, as returned by pick),
    outcomes, and adt are given, both figures also carry the R12.2
    bootstrap gap-CI error bars vs worst_first, computed via
    bootstrap_gap_ci with metric="recall" for Fig 1 and
    metric="weighted_recall" for Fig 2 (selections without outcomes and adt
    raises ValueError). Because selections stay fixed, those CIs capture
    outcome-sampling variance only, not selection variance (R12.3) — any
    report of these figures must state that.
    """
    if selections is not None and (outcomes is None or adt is None):
        raise ValueError(
            "plot: selections require outcomes and adt "
            "(bootstrap_gap_ci inputs, R12.2)"
        )
    ci_recall = ci_weighted = None
    if selections is not None:
        ci_recall = bootstrap_gap_ci(selections, outcomes, adt, "recall")
        ci_weighted = bootstrap_gap_ci(
            selections, outcomes, adt, "weighted_recall"
        )
    fig_recall = _plot_metric(
        results_df, "recall", "Recall",
        "Budget vs recall (share of pool decliners caught)",
        gap_ci=ci_recall,
    )
    fig_weighted = _plot_metric(
        results_df, "weighted_recall", "Weighted recall",
        "Budget vs weighted recall (weights: log1p(adt))",
        gap_ci=ci_weighted,
    )
    if output_path is not None:
        fig_recall.savefig(
            f"{output_path}_recall.png", dpi=150, bbox_inches="tight"
        )
        fig_weighted.savefig(
            f"{output_path}_weighted.png", dpi=150, bbox_inches="tight"
        )
    return fig_recall, fig_weighted
