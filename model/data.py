"""Adapter from Part A's published panel to the names Part B's model uses.

Part A owns ``bridges/panel.parquet`` and its schema is fixed (see
``bridges/README.md``). We never edit anything under ``bridges/`` — instead
``load_panel`` reads that file and returns a frame whose column names and
derived fields match what :mod:`model.score` expects.

Mapping applied here:
- ``structure_id``      -> ``id``
- ``insp_freq_months``  -> ``insp_freq_mo``
- ``year_built``        derived as ``year - age``
- ``worst_rating``      kept as-is; score.py uses it as the condition when
                        present, and a null ``worst_rating`` is not-applicable
                        and skipped.
- ``year_recon`` == 0   -> ``NaN`` (0 means "never rebuilt")
- ``rebuilt``           passed through; fit_decline excludes a start->end pair
                        when the END year's rebuilt flag is True.
- ``improvement_cost``  deliberately NOT mapped to ``insp_cost`` — it is a
                        repair cost, not an inspection cost. score.py uses the
                        estimated drone-flight cost when no real inspection
                        cost exists.
- ``insp_date``         not in the panel yet. If Part A adds it later, it is
                        picked up automatically; otherwise score.py falls back
                        to ``insp_freq_mo``.

The real panel's 4-year gaps are 2012->2016 and 2018->2022. Never fit on the
2016->2018 gap (2-year horizon) — see ``FOUR_YEAR_PAIRS``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PANEL_PATH = Path(__file__).resolve().parents[1] / "bridges" / "panel.parquet"

# The only interval pairs whose horizon matches (4 years). The 2016->2018 gap
# is 2 years and must never be used for fitting or backtest.
FOUR_YEAR_PAIRS = [(2012, 2016), (2018, 2022)]


def load_panel(path=None) -> pd.DataFrame:
    """Load Part A's panel and adapt column names/derived fields for the model.

    Does not mutate any file under ``bridges/``. Returns a new DataFrame.
    """
    path = Path(path) if path is not None else DEFAULT_PANEL_PATH
    raw = pd.read_parquet(path)
    df = raw.copy()

    # --- renames ------------------------------------------------------------
    renames = {}
    if "structure_id" in df.columns:
        renames["structure_id"] = "id"
    if "insp_freq_months" in df.columns:
        renames["insp_freq_months"] = "insp_freq_mo"
    df = df.rename(columns=renames)

    # --- derive year_built = year - age ------------------------------------
    if "year_built" not in df.columns and {"year", "age"}.issubset(df.columns):
        df["year_built"] = df["year"] - df["age"]

    # --- year_recon 0 means "never rebuilt" -> NaN -------------------------
    if "year_recon" in df.columns:
        df["year_recon"] = df["year_recon"].where(df["year_recon"] != 0, other=pd.NA)
        df["year_recon"] = pd.to_numeric(df["year_recon"], errors="coerce")

    # --- insp_date: pick it up automatically if Part A adds it later -------
    if "insp_date" in df.columns:
        df["insp_date"] = pd.to_datetime(df["insp_date"], errors="coerce")
    # If absent, we intentionally do NOT create it; score.py falls back to
    # insp_freq_mo.

    # improvement_cost is left untouched and NOT renamed to insp_cost.

    return df


if __name__ == "__main__":
    p = load_panel()
    print(f"loaded {len(p)} rows, {p['id'].nunique()} bridges, years {sorted(p['year'].unique())}")
    print("has insp_date:", "insp_date" in p.columns)
    print("has insp_cost:", "insp_cost" in p.columns)
    print(p[["id", "year", "year_built", "worst_rating", "insp_freq_mo", "rebuilt"]].head())
