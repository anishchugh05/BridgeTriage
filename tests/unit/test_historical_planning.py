"""Focused checks for the leakage-free historical planning phase."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pandas as pd
import pytest

from model.data import MODELED_COST_DISCLOSURE, SCORED_COLUMNS
from planner import BACKTEST_METHODS, BacktestPlan, build_historical_plan
from planner.backtest import HistoricalPlanningError


def _historical_panel() -> pd.DataFrame:
    attributes = {
        "A": {"worst_rating": None, "age": 20},
        "B": {"worst_rating": 3, "age": 80},
        "C": {"worst_rating": 3, "age": 80},
        "D": {"worst_rating": 5, "age": 10},
    }
    rows: list[dict[str, object]] = []
    for year in (2012, 2016):
        for identifier in ("A", "B", "C", "D"):
            rows.append(
                {
                    "id": identifier,
                    "year": year,
                    "worst_rating": attributes[identifier]["worst_rating"],
                    "age": attributes[identifier]["age"],
                    "rebuilt": False,
                }
            )
    # Deliberately noncanonical input order: every method must set its own
    # deterministic order rather than inherit DataFrame row order.
    for identifier in ("D", "A", "C", "B"):
        rows.append(
            {
                "id": identifier,
                "year": 2018,
                "worst_rating": attributes[identifier]["worst_rating"],
                "age": attributes[identifier]["age"],
                "rebuilt": False,
            }
        )
    return pd.DataFrame(rows)


def _valid_scores() -> pd.DataFrame:
    # p_worse * consequence / insp_cost equals score for every row.
    specifications = {
        "A": (0.5, 22.0, 110.0, 0.10),
        "B": (0.5, 10.8, 60.0, 0.09),
        "C": (0.5, 6.4, 40.0, 0.08),
        "D": (0.5, 4.2, 30.0, 0.07),
    }
    return pd.DataFrame(
        [
            {
                "id": identifier,
                "p_worse": probability,
                "consequence": consequence,
                "insp_cost": cost,
                "score": score,
                "reason": f"{identifier}; {MODELED_COST_DISCLOSURE}",
            }
            for identifier, (
                probability,
                consequence,
                cost,
                score,
            ) in specifications.items()
        ],
        columns=SCORED_COLUMNS,
    )


def test_historical_plan_bounds_model_calls_and_freezes_common_decisions():
    calls: dict[str, object] = {}
    rates = object()

    def fit_spy(table: pd.DataFrame, start: int, end: int) -> object:
        calls["fit_years"] = tuple(sorted(table["year"].unique().tolist()))
        calls["fit_interval"] = (start, end)
        calls["fit_ids"] = tuple(sorted(table["id"].unique().tolist()))
        return rates

    def score_spy(table: pd.DataFrame, state: object, as_of: object) -> pd.DataFrame:
        calls["score_years"] = tuple(sorted(table["year"].unique().tolist()))
        calls["score_ids"] = tuple(sorted(table["id"].tolist()))
        calls["score_as_of"] = pd.Timestamp(as_of)
        calls["score_state"] = state
        return _valid_scores()

    plan = build_historical_plan(
        _historical_panel(),
        [1000, 100, 100],
        fit_fn=fit_spy,
        score_fn=score_spy,
    )

    assert isinstance(plan, BacktestPlan)
    assert calls == {
        "fit_years": (2012, 2016),
        "fit_interval": (2012, 2016),
        "fit_ids": ("A", "B", "C", "D"),
        "score_years": (2018,),
        "score_ids": ("A", "B", "C", "D"),
        "score_as_of": pd.Timestamp("2018-04-01"),
        "score_state": rates,
    }

    assert plan.fit_years == (2012, 2016)
    assert plan.candidate_year == 2018
    assert plan.as_of == pd.Timestamp("2018-04-01")
    assert plan.candidate_ids == ("A", "B", "C", "D")
    assert plan.budgets == (Decimal("100"), Decimal("1000"))
    assert dict(plan.modeled_costs) == {
        "A": Decimal("110.0"),
        "B": Decimal("60.0"),
        "C": Decimal("40.0"),
        "D": Decimal("30.0"),
    }
    assert dict(plan.worst_ratings_2018) == {
        "A": None,
        "B": 3.0,
        "C": 3.0,
        "D": 5.0,
    }

    # All methods continue beyond unaffordable candidates under the same
    # budget traversal, but use their specified deterministic orders.
    assert plan.selection_for(100, "ours").ids == ("B", "C")
    assert plan.selection_for(100, "worst_first").ids == ("B", "C")
    assert plan.selection_for(100, "oldest_first").ids == ("B", "C")
    assert plan.selection_for(1000, "ours").ids == ("A", "B", "C", "D")
    assert plan.selection_for(1000, "worst_first").ids == ("B", "C", "D", "A")
    assert plan.selection_for(1000, "oldest_first").ids == ("B", "C", "A", "D")
    assert tuple(selection.method for selection in plan.selections[:3]) == (
        BACKTEST_METHODS
    )

    with pytest.raises(FrozenInstanceError):
        plan.candidate_ids = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan.cost_by_id["A"] = Decimal("1")  # type: ignore[index]


def test_future_year_is_rejected_before_fit_or_score_can_run():
    future_row = {
        "id": "A",
        "year": 2022,
        "worst_rating": 0,
        "age": 100,
        "rebuilt": True,
    }
    panel = pd.concat(
        [_historical_panel(), pd.DataFrame([future_row])], ignore_index=True
    )
    calls: list[str] = []

    def forbidden_fit(*_args: object) -> object:
        calls.append("fit")
        return object()

    def forbidden_score(*_args: object) -> pd.DataFrame:
        calls.append("score")
        return _valid_scores()

    with pytest.raises(HistoricalPlanningError) as caught:
        build_historical_plan(
            panel,
            [100],
            fit_fn=forbidden_fit,
            score_fn=forbidden_score,
        )

    assert caught.value.issues == ("backtest.panel.future_year",)
    assert calls == []


def test_duplicate_2018_candidate_fails_before_model_access():
    panel = pd.concat(
        [
            _historical_panel(),
            pd.DataFrame(
                [
                    {
                        "id": "A",
                        "year": 2018,
                        "worst_rating": 4,
                        "age": 20,
                        "rebuilt": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    calls: list[str] = []

    def forbidden(*_args: object) -> object:
        calls.append("called")
        return object()

    with pytest.raises(HistoricalPlanningError) as caught:
        build_historical_plan(
            panel,
            [100],
            fit_fn=forbidden,
            score_fn=forbidden,  # type: ignore[arg-type]
        )

    assert "backtest.panel.id_year.unique" in caught.value.issues
    assert calls == []
