"""Narrative proof: our pick vs. worst-rating-first, checked against 2022.

Using the 2018 score table and the 2022 panel, this builds two lists:

1. The top 5 bridges by our ``score`` that actually had a safety drop by 2022 —
   the bridges our triage would have flagged and that genuinely got worse.
2. The top 5 the "worst-rating-first" baseline would pick (lowest 2018 grade),
   and whether each actually dropped.

It writes ``docs/stories.md`` with both tables (2018 and 2022 grades, carries,
crosses, county, ADT) and InfoBridge lookup instructions so a reviewer can pull
up any structure by its NBI number.

Run:
    .venv/bin/python -m model.stories
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import load_panel
from .target import is_safety_drop

ROOT = Path(__file__).resolve().parents[1]
SCORES_2018 = ROOT / "data" / "processed" / "scores_2018.parquet"
DOCS_PATH = ROOT / "docs" / "stories.md"

PICK_YEAR, OUTCOME_YEAR = 2018, 2022
TOP_N = 5

# Columns pulled from the 2018 panel row for context in the tables.
CONTEXT_COLS = ["carries", "crosses", "county", "adt"]


def _paired_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per bridge present in both years, with grades and drop label."""
    a = panel[panel["year"] == PICK_YEAR].set_index("id")
    b = panel[panel["year"] == OUTCOME_YEAR].set_index("id")
    common = a.index.intersection(b.index)

    records = []
    for bid in common:
        ra, rb = a.loc[bid], b.loc[bid]
        g18, g22 = ra.get("worst_rating"), rb.get("worst_rating")
        if pd.isna(g18) or pd.isna(g22):
            continue
        record = {
            "id": bid,
            "grade_2018": int(round(float(g18))),
            "grade_2022": int(round(float(g22))),
            "is_drop": bool(is_safety_drop(float(g18), float(g22))),
        }
        for col in CONTEXT_COLS:
            record[col] = ra.get(col)
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_stories(scores_path=None, panel_path=None) -> dict:
    """Compute the two top-5 lists. Returns a dict of DataFrames."""
    scores = pd.read_parquet(scores_path or SCORES_2018)
    panel = load_panel(panel_path)
    outcomes = _paired_outcomes(panel)

    merged = scores.merge(outcomes, on="id", how="inner")

    our_hits = (
        merged[merged["is_drop"]]
        .sort_values("score", ascending=False)
        .head(TOP_N)
        .copy()
    )
    # Worst-rating-first: lowest 2018 grade; break ties by our score so the
    # baseline is shown at its strongest.
    worst_first = (
        merged.sort_values(["grade_2018", "score"], ascending=[True, False])
        .head(TOP_N)
        .copy()
    )
    return {
        "our_hits": our_hits,
        "worst_first": worst_first,
        "n_paired": len(merged),
        "n_drops": int(merged["is_drop"].sum()),
    }


def _fmt_adt(value) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{int(value):,}"


def _table(df: pd.DataFrame, include_score: bool) -> str:
    if include_score:
        header = (
            "| NBI # | our score | 2018 | 2022 | carries | crosses | county "
            "| ADT |\n|---|---|---|---|---|---|---|---|"
        )
    else:
        header = (
            "| NBI # | 2018 | 2022 | dropped? | carries | crosses | county "
            "| ADT |\n|---|---|---|---|---|---|---|---|"
        )
    lines = [header]
    for _, r in df.iterrows():
        carries = str(r.get("carries") or "").strip() or "—"
        crosses = str(r.get("crosses") or "").strip() or "—"
        county = str(r.get("county") or "").strip() or "—"
        adt = _fmt_adt(r.get("adt"))
        if include_score:
            lines.append(
                f"| {r['id']} | {r['score']:.3f} | {r['grade_2018']} | "
                f"{r['grade_2022']} | {carries} | {crosses} | {county} | {adt} |"
            )
        else:
            dropped = "yes" if r["is_drop"] else "no"
            lines.append(
                f"| {r['id']} | {r['grade_2018']} | {r['grade_2022']} | "
                f"{dropped} | {carries} | {crosses} | {county} | {adt} |"
            )
    return "\n".join(lines)


def render_markdown(stories: dict) -> str:
    our_hits = stories["our_hits"]
    worst_first = stories["worst_first"]
    wf_hits = int(worst_first["is_drop"].sum())

    return f"""# BridgeTriage: picks vs. what actually happened

We picked with **2018** data and checked against **2022** reality. A "safety
drop" means a bridge crossed into poor condition (NBI grade ≤ 4) or got worse
once already there. All grades are the worst of deck/superstructure/
substructure (NBI item 058/059/060). Scores and costs are planning estimates.

Of {stories['n_paired']:,} bridges present in both years, {stories['n_drops']}
had a safety drop by 2022.

## Top 5 BridgeTriage picks that actually dropped

These are the highest-scoring bridges in our 2018 ranking that genuinely slipped
into poor condition by 2022 — the ones a drone program would have caught early.

{_table(our_hits, include_score=True)}

## Top 5 that "worst-rating-first" would pick

The simplest baseline inspects the lowest-rated bridges first. Of its top 5,
**{wf_hits} of 5** actually dropped. Low-rated bridges are often already near the
floor (little room to fall) or get rebuilt, so chasing the lowest grade spends
inspections where they reveal the least.

{_table(worst_first, include_score=False)}

## Look up any bridge in InfoBridge

FHWA's InfoBridge lets you pull the full NBI record for any structure number
above:

1. Open InfoBridge: https://infobridge.fhwa.dot.gov/
2. Go to **Data** → **Bridge Explorer** (or the search box).
3. Choose state **Massachusetts (25)**.
4. Search by the **NBI structure number** in the first column (NBI item 008).
5. Open the record for inspection history, condition ratings by year, traffic,
   and structure details. Cross-check the 2018 and 2022 grades shown above.

Note: InfoBridge structure numbers may carry formatting (dashes/spaces) that
differs from the raw ID; match on the digits if an exact string search fails.
"""


def main() -> None:
    stories = build_stories()
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(stories)
    DOCS_PATH.write_text(markdown)
    print(markdown)
    print(f"\nwrote {DOCS_PATH}")


if __name__ == "__main__":
    main()
