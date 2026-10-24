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

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved to files, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
# Algorithm 9: make_fake synthetic data generator (R13)
# ---------------------------------------------------------------------------

def _logistic(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _z(x):
    """Standardize over the dataset; constant columns map to zeros."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


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
