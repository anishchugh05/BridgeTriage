"""Focused unit checks for task 5.2 backtest outcome evaluation."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from model.data import MODELED_COST_DISCLOSURE, SCORED_COLUMNS
from planner import (
    BACKTEST_METHODS,
    BacktestPlan,
    BacktestResultValidationError,
    BacktestSelection,
    OutcomeEvaluationError,
    evaluate_outcomes,
    run_backtest,
    validate_backtest_result,
)


def _frozen_plan() -> BacktestPlan:
    budget = Decimal("10")
    candidate_ids = ("A", "B", "C", "D", "E")
    return BacktestPlan(
        candidate_ids=candidate_ids,
        modeled_costs=tuple(
            (identifier, Decimal("1")) for identifier in candidate_ids
        ),
        budgets=(budget,),
        selections=(
            BacktestSelection(budget, "ours", ("C", "D", "E")),
            BacktestSelection(budget, "worst_first", ("A", "B")),
            BacktestSelection(budget, "oldest_first", ("C", "A")),
        ),
        worst_ratings_2018=(
            ("A", 5.0),
            ("B", 3.0),
            ("C", 6.0),
            ("D", 4.0),
            ("E", 3.0),
        ),
    )


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": "A", "year": 2022, "worst_rating": 4, "rebuilt": False},
            {"id": "B", "year": 2022, "worst_rating": 2, "rebuilt": False},
            {"id": "C", "year": 2022, "worst_rating": 6, "rebuilt": False},
            {"id": "D", "year": 2022, "worst_rating": None, "rebuilt": False},
            {"id": "E", "year": 2022, "worst_rating": 1, "rebuilt": True},
        ]
    )


def test_evaluate_outcomes_preserves_measured_baseline_victory_and_audit():
    calls: list[tuple[float, float]] = []

    def target_spy(start: object, end: object) -> bool:
        pair = (float(start), float(end))
        calls.append(pair)
        return pair in {(5.0, 4.0), (3.0, 2.0)}

    run = evaluate_outcomes(
        _frozen_plan(), _outcomes(), target_fn=target_spy
    )

    assert tuple(run.result.columns) == ("budget", "method", "caught")
    assert run.result.to_dict("records") == [
        {"budget": Decimal("10"), "method": "ours", "caught": 0},
        {"budget": Decimal("10"), "method": "worst_first", "caught": 2},
        {"budget": Decimal("10"), "method": "oldest_first", "caught": 1},
    ]
    assert dict(run.eligible_selected_count) == {
        (Decimal("10"), "ours"): 1,
        (Decimal("10"), "worst_first"): 2,
        (Decimal("10"), "oldest_first"): 2,
    }
    assert len(calls) == 3
    assert set(calls) == {(6.0, 6.0), (5.0, 4.0), (3.0, 2.0)}


def test_duplicate_selected_outcome_is_rejected_before_target_access():
    duplicate = pd.concat(
        [_outcomes(), _outcomes().iloc[[0]]], ignore_index=True
    )
    calls: list[tuple[object, object]] = []

    with pytest.raises(OutcomeEvaluationError) as caught:
        evaluate_outcomes(
            _frozen_plan(),
            duplicate,
            target_fn=lambda start, end: bool(calls.append((start, end))),
        )

    assert caught.value.issues == ("backtest.outcomes.selected_id.unique",)
    assert calls == []


def test_result_validator_rejects_noncanonical_order_and_eligible_overcount():
    run = evaluate_outcomes(_frozen_plan(), _outcomes())
    reordered = run.result.iloc[::-1].reset_index(drop=True)

    with pytest.raises(BacktestResultValidationError) as caught:
        validate_backtest_result(
            reordered,
            run.selection_audit,
            run.eligible_selected_count,
        )
    assert "backtest.result.canonical_order" in caught.value.issues

    overstated = run.result.copy(deep=True)
    overstated.loc[0, "caught"] = 2
    with pytest.raises(BacktestResultValidationError) as caught:
        validate_backtest_result(
            overstated,
            run.selection_audit,
            run.eligible_selected_count,
        )
    assert "backtest.result.caught.eligible_bound" in caught.value.issues


def test_run_backtest_plans_before_evaluating_2022_and_deduplicates_budgets():
    rows: list[dict[str, object]] = []
    ratings = {
        2012: {"A": 7, "B": 5},
        2016: {"A": 6, "B": 4},
        2018: {"A": 5, "B": 3},
        2022: {"A": 4, "B": 3},
    }
    for year in (2012, 2016, 2018, 2022):
        for identifier in ("A", "B"):
            rows.append(
                {
                    "id": identifier,
                    "year": year,
                    "worst_rating": ratings[year][identifier],
                    "age": 20 if identifier == "A" else 40,
                    "rebuilt": False,
                }
            )
    panel = pd.DataFrame(rows)
    calls: list[tuple[str, object]] = []
    rates = object()

    def fit_spy(table: pd.DataFrame, start: int, end: int) -> object:
        calls.append(("fit", tuple(sorted(table["year"].unique()))))
        assert (start, end) == (2012, 2016)
        return rates

    def score_spy(table: pd.DataFrame, state: object, as_of: object) -> pd.DataFrame:
        calls.append(("score", tuple(sorted(table["year"].unique()))))
        assert state is rates
        assert pd.Timestamp(as_of) == pd.Timestamp("2018-04-01")
        return pd.DataFrame(
            [
                {
                    "id": "A",
                    "p_worse": 0.5,
                    "consequence": 2.0,
                    "insp_cost": 1.0,
                    "score": 1.0,
                    "reason": MODELED_COST_DISCLOSURE,
                },
                {
                    "id": "B",
                    "p_worse": 0.5,
                    "consequence": 1.0,
                    "insp_cost": 1.0,
                    "score": 0.5,
                    "reason": MODELED_COST_DISCLOSURE,
                },
            ],
            columns=SCORED_COLUMNS,
        )

    result = run_backtest(
        panel,
        [10, 10],
        fit_fn=fit_spy,
        score_fn=score_spy,
    )

    assert calls == [("fit", (2012, 2016)), ("score", (2018,))]
    assert tuple(result.columns) == ("budget", "method", "caught")
    assert result["method"].tolist() == list(BACKTEST_METHODS)
    assert result["caught"].tolist() == [1, 1, 1]
    assert result["budget"].tolist() == [Decimal("10")] * 3
