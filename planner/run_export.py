"""Export the real model backtest and fixed-selection bootstrap summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from planner.harness import (  # noqa: E402
    adapt_panel,
    apply_exclusion_and_label,
    bootstrap_gap_ci,
    load_and_validate,
    pick,
    run_backtest,
    score_oldest_first,
    score_oracle,
    score_worst_first,
)

PANEL_PATH = REPOSITORY_ROOT / "bridges" / "panel.parquet"
SCORES_PATH = REPOSITORY_ROOT / "data" / "processed" / "scores_2018.parquet"
RESULTS_DIR = REPOSITORY_ROOT / "results"
DEFAULT_BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.20)
COUNTY_FEE = 5.0
FIXED_METHODS = ("worst_first", "oldest_first", "oracle", "model")
EXPORT_COLUMNS = (
    "method",
    "budget",
    "tiebreak",
    "recall",
    "weighted_recall",
    "gap_lo",
    "gap_hi",
)
BOOTSTRAP_CAVEAT = (
    "Bootstrap caveat: fixed selections capture outcome-sampling variance "
    "only, not selection variance."
)


def load_model_scores(
    path: str | Path,
    candidate_ids: Iterable[object],
) -> dict[str, float]:
    """Load B's 2018 scores with strict identity, value, and coverage checks."""
    frame = pd.read_parquet(Path(path))
    missing_columns = [column for column in ("id", "score") if column not in frame]
    if missing_columns:
        raise ValueError(
            f"score file is missing required columns: {missing_columns}"
        )
    if "structure_id" in frame.columns:
        raise ValueError("score file must use id before boundary renaming")

    ids = frame["id"]
    if ids.isna().any():
        raise ValueError("score ids must be non-null and nonempty")
    normalized_ids = ids.map(lambda value: str(value).strip())
    if normalized_ids.eq("").any():
        raise ValueError("score ids must be non-null and nonempty")
    if normalized_ids.duplicated(keep=False).any():
        raise ValueError("score ids must be unique after normalization")

    numeric_scores = pd.to_numeric(frame["score"], errors="coerce")
    if numeric_scores.isna().any() or not np.isfinite(
        numeric_scores.to_numpy(dtype=float)
    ).all():
        raise ValueError("scores must be finite numeric values")

    normalized_candidates = [str(candidate_id).strip() for candidate_id in candidate_ids]
    if any(not candidate_id for candidate_id in normalized_candidates):
        raise ValueError("candidate ids must be nonempty")
    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise ValueError("candidate ids must be unique")

    available_ids = set(normalized_ids)
    missing_candidates = sorted(set(normalized_candidates) - available_ids)
    if missing_candidates:
        raise ValueError(
            "score file does not provide complete candidate coverage: "
            f"{len(missing_candidates)} candidate(s) missing"
        )

    renamed = pd.DataFrame(
        {
            "structure_id": normalized_ids,
            "score": numeric_scores.astype(float),
        }
    )
    return dict(zip(renamed["structure_id"], renamed["score"]))


def _fixed_selections(
    candidates: pd.DataFrame,
    outcomes: dict,
    pool_cost_info: dict,
    model_scores: dict[str, float],
    budgets: Sequence[float],
    county_fee: float,
) -> dict[str, dict[float, list[str]]]:
    score_sets = {
        "worst_first": (score_worst_first(candidates), False),
        "oldest_first": (score_oldest_first(candidates), False),
        "oracle": (score_oracle(outcomes), True),
        "model": (model_scores, True),
    }
    return {
        method: {
            float(budget): pick(
                scores,
                float(budget),
                candidates,
                pool_cost_info,
                county_fee,
                rank_by_value_per_cost=value_per_cost,
                tiebreak="adt",
            )
            for budget in budgets
        }
        for method, (scores, value_per_cost) in score_sets.items()
    }


def export_backtest(
    panel_path: str | Path = PANEL_PATH,
    scores_path: str | Path = SCORES_PATH,
    output_dir: str | Path = RESULTS_DIR,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    county_fee: float = COUNTY_FEE,
) -> tuple[Path, Path]:
    """Run the real model backtest and write the requested CSV/JSON artifacts.

    The seven-column CSV has one numeric CI pair, so ``gap_lo``/``gap_hi``
    report the unweighted-recall gap versus ``worst_first``. Both required
    recall and weighted-recall bootstrap intervals are computed and printed.
    """
    budget_values = tuple(float(budget) for budget in budgets)
    df_2018, df_2022 = adapt_panel(str(Path(panel_path)))
    validated_2018, validated_2022, index = load_and_validate(df_2018, df_2022)
    candidates, outcomes, pool_cost_info = apply_exclusion_and_label(
        validated_2018,
        validated_2022,
        index,
        county_fee,
    )
    model_scores = load_model_scores(scores_path, candidates.index)

    results = run_backtest(
        validated_2018,
        validated_2022,
        county_fee,
        budgets=list(budget_values),
        model_scores=model_scores,
    )
    selections = _fixed_selections(
        candidates,
        outcomes,
        pool_cost_info,
        model_scores,
        budget_values,
        county_fee,
    )
    adt = {structure_id: values["adt"] for structure_id, values in outcomes.items()}
    recall_ci = bootstrap_gap_ci(selections, outcomes, adt, "recall")
    weighted_ci = bootstrap_gap_ci(
        selections,
        outcomes,
        adt,
        "weighted_recall",
    )

    exported = results[
        (~results["is_aggregate"])
        & results["method"].isin(FIXED_METHODS)
    ].copy()
    exported["tiebreak"] = exported["tiebreak"].fillna("adt")
    exported["gap_lo"] = [
        recall_ci[row.method][float(row.budget)][0]
        for row in exported.itertuples()
    ]
    exported["gap_hi"] = [
        recall_ci[row.method][float(row.budget)][1]
        for row in exported.itertuples()
    ]
    exported = exported.loc[:, list(EXPORT_COLUMNS)]
    method_order = {method: position for position, method in enumerate(FIXED_METHODS)}
    exported["_method_order"] = exported["method"].map(method_order)
    exported = exported.sort_values(["budget", "_method_order"]).drop(
        columns="_method_order"
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "backtest.csv"
    summary_path = destination / "summary.json"
    exported.to_csv(csv_path, index=False)

    summary = {
        "pool_size": int(len(candidates)),
        "decliners": int(sum(values["worse"] for values in outcomes.values())),
        "exclusion_counts": {
            reason: int(count)
            for reason, count in pool_cost_info["exclusion_counts"].items()
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(BOOTSTRAP_CAVEAT)
    print("backtest.csv gap_lo/gap_hi are recall gaps versus worst_first.")
    for metric, intervals in (
        ("recall", recall_ci),
        ("weighted_recall", weighted_ci),
    ):
        print(f"{metric} gap 95% bootstrap CIs versus worst_first:")
        for method in FIXED_METHODS:
            values = ", ".join(
                f"{budget:g}=({intervals[method][budget][0]:+.4f},"
                f" {intervals[method][budget][1]:+.4f})"
                for budget in budget_values
            )
            print(f"  {method}: {values}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    return csv_path, summary_path


def main() -> None:
    export_backtest()


if __name__ == "__main__":
    main()
