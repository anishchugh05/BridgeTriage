"""Backtest harness for bridge inspection prioritization strategies.

Evaluates selection methods (baselines, oracle, model scores) that pick
bridges for inspection within a budget, using 2018 condition data as
inputs and 2022 condition outcomes as ground truth. See
.kiro/specs/backtest-harness/{requirements,design,tasks}.md.

Conventions (per requirements.md):
- Higher score = picked first, for ALL scoring functions.
- Ratings are integers 0-9; NaN means "not applicable" ("N" in source).
- Budgets are fractions of the candidate pool's total inspection cost.
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
BASE_RATE_LOGIT = None    # placeholder; calibrated by simulation in task 2
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
