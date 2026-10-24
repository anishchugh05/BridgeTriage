"""Run the backtest baselines on the real panel data (task 22).

Pipeline: adapt_panel() -> load_and_validate -> apply_exclusion_and_label
(county_fee 5.0) -> run_backtest(..., county_fee=5.0, tiebreaks=("adt",
"random")) on the adapted bridges/panel.parquet data (R10, R11, R12).

Prints, in order: the adapter's year/unmatched-ID counts, the pool size,
exclusion counts by reason, the number of safety-drop decliners in the
pool, and recall / weighted_recall per method per budget — random_avg
shown as its aggregate MEAN row only (no per-seed rows), blank tiebreak
filled with "adt". Headline figures are written to /tmp/real_recall.png
and /tmp/real_weighted.png via plot()'s output-path prefix.

Runnable as either `python -m planner.run_real` or
`python planner/run_real.py`, from any cwd.
"""

import sys
from pathlib import Path

# Repo root on sys.path so `planner` imports when run as a script
# (`python planner/run_real.py` puts planner/ on sys.path, not the root);
# harmless under `python -m planner.run_real`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planner.harness import (  # noqa: E402
    adapt_panel,
    apply_exclusion_and_label,
    load_and_validate,
    plot,
    run_backtest,
)

COUNTY_FEE = 5.0
TIEBREAKS = ("adt", "random")
PANEL_PATH = REPO_ROOT / "bridges" / "panel.parquet"
PLOT_PREFIX = "/tmp/real"


def main():
    print("=== Adapt panel (bridges/panel.parquet) ===")
    df_2018, df_2022 = adapt_panel(str(PANEL_PATH))

    print("\n=== Load and validate ===")
    df_2018, df_2022, index = load_and_validate(df_2018, df_2022)
    print(f"2018 rows: {len(df_2018)}; 2022 rows: {len(df_2022)}")

    print(f"\n=== Exclusion and labeling (county_fee={COUNTY_FEE}) ===")
    candidates, outcomes, pool_cost_info = apply_exclusion_and_label(
        df_2018, df_2022, index, COUNTY_FEE
    )
    print(f"pool size (candidate bridges): {len(candidates)}")
    print("exclusion counts by reason:")
    for reason, count in pool_cost_info["exclusion_counts"].items():
        print(f"  {reason}: {count}")
    n_worse = sum(o["worse"] for o in outcomes.values())
    print(f"safety-drop decliners in pool (worse): {n_worse}")

    print(f"\n=== Run backtest (tiebreaks={TIEBREAKS}) ===")
    results = run_backtest(df_2018, df_2022, COUNTY_FEE, tiebreaks=TIEBREAKS)

    print("\n=== Recall and weighted recall per method per budget ===")
    # random_avg: aggregate MEAN row only — no per-seed rows and no
    # std/min/max aggregate rows (per user constraint, taking precedence
    # over the tasks.md demo filter, which keeps all four aggregate
    # stats). Baselines keep both their tiebreak="adt" single runs and
    # their tiebreak="random" MEAN rows (R11.3); blank tiebreak is shown
    # as "adt".
    random_avg = results["method"] == "random_avg"
    drop = (random_avg & ~results["is_aggregate"]) | (
        random_avg & results["is_aggregate"] & (results["stat"] != "mean")
    )
    out = results.loc[~drop].copy()
    out["tiebreak"] = out["tiebreak"].fillna("adt")
    out = out.round({"recall": 4, "weighted_recall": 4})
    print(
        out[["method", "budget", "tiebreak", "recall", "weighted_recall"]]
        .to_string(index=False)
    )

    print(
        "\nNOTE: inspect_cost is flat (placeholder units, the adapt_panel "
        "default), so value-per-cost ranking (score / inspect_cost) is "
        "identical to raw-score ranking; the oracle's value-per-cost "
        "advantage is not exercised on this data."
    )

    print("\n=== Plots ===")
    plot(results, PLOT_PREFIX)
    print(f"wrote {PLOT_PREFIX}_recall.png and {PLOT_PREFIX}_weighted.png")


if __name__ == "__main__":
    main()
