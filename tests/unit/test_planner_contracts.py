"""Focused checks for planner input and immutable result contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pandas as pd
import pytest

from model.data import MODELED_COST_DISCLOSURE, SCORED_COLUMNS
from planner.contracts import (
    ScoredInputValidationError,
    SelectionMetricsPostconditionError,
    SelectionPostconditionError,
    SelectionRequestValidationError,
    build_selection,
    build_selection_metrics,
    validate_scored_input,
    validate_selection_request,
)


def _valid_scored() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "A",
                "p_worse": 0.6,
                "consequence": 10.0,
                "insp_cost": 100.0,
                "score": 0.06,
                "reason": f"A; {MODELED_COST_DISCLOSURE}",
            },
            {
                "id": "B",
                "p_worse": 0.2,
                "consequence": 5.0,
                "insp_cost": 50.0,
                "score": 0.02,
                "reason": f"B; {MODELED_COST_DISCLOSURE}",
            },
        ],
        columns=SCORED_COLUMNS,
    )


def test_scored_input_uses_exact_contract_and_rejects_duplicate_ids():
    scored = _valid_scored()
    validate_scored_input(scored)

    duplicate = scored.copy(deep=True)
    duplicate.loc[1, "id"] = "A"
    with pytest.raises(ScoredInputValidationError) as caught:
        validate_scored_input(duplicate)

    assert "id.unique" in caught.value.issues


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_scored_input_rejects_boolean_and_nonfinite_numeric_fields(value: object):
    scored = _valid_scored()
    scored["p_worse"] = scored["p_worse"].astype(object)
    scored.loc[0, "p_worse"] = value

    with pytest.raises(ScoredInputValidationError) as caught:
        validate_scored_input(scored)

    assert "p_worse.finite_numeric" in caught.value.issues


def test_selection_request_requires_one_valid_mode_and_canonicalizes_values():
    scored = _valid_scored()

    budget_request = validate_selection_request(scored, budget=100.25)
    count_request = validate_selection_request(scored, count=1)

    assert budget_request.mode == "budget"
    assert budget_request.constraint == Decimal("100.25")
    assert count_request.mode == "count"
    assert count_request.constraint == 1

    for arguments in ({}, {"budget": 100, "count": 1}):
        with pytest.raises(SelectionRequestValidationError) as caught:
            validate_selection_request(scored, **arguments)
        assert "selection_mode.exactly_one" in caught.value.issues


@pytest.mark.parametrize(
    "arguments",
    [
        {"budget": True},
        {"budget": -1},
        {"budget": float("nan")},
        {"budget": float("inf")},
        {"budget": "100"},
        {"count": True},
        {"count": -1},
        {"count": 1.0},
        {"count": float("nan")},
        {"count": "1"},
    ],
)
def test_selection_request_rejects_invalid_budget_and_count_values(arguments):
    with pytest.raises(SelectionRequestValidationError):
        validate_selection_request(_valid_scored(), **arguments)


def test_selection_factory_enforces_membership_order_uniqueness_and_immutability():
    scored = _valid_scored()
    request = validate_selection_request(scored, budget=150)
    selection = build_selection(
        scored,
        request,
        ["A", "B"],
        ordered_candidate_ids=["A", "B"],
    )

    assert selection.ids == ("A", "B")
    with pytest.raises(FrozenInstanceError):
        selection.mode = "count"  # type: ignore[misc]

    for selected_ids in (["A", "A"], ["foreign"], ["B", "A"]):
        with pytest.raises(SelectionPostconditionError):
            build_selection(
                scored,
                request,
                selected_ids,
                ordered_candidate_ids=["A", "B"],
            )


def test_selection_metrics_are_atomic_bounded_and_immutable():
    scored = _valid_scored()
    request = validate_selection_request(scored, count=1)
    selection = build_selection(
        scored,
        request,
        ["A"],
        ordered_candidate_ids=["A", "B"],
    )
    metrics = build_selection_metrics(
        selection,
        selected_total_cost=Decimal("100"),
        expected_safety_drop_coverage=0.6,
    )

    assert metrics.selected_count == 1
    assert metrics.selected_total_cost == Decimal("100")
    with pytest.raises(FrozenInstanceError):
        metrics.selected_count = 2  # type: ignore[misc]

    with pytest.raises(SelectionMetricsPostconditionError):
        build_selection_metrics(
            selection,
            selected_total_cost=True,
            expected_safety_drop_coverage=0.6,
        )
