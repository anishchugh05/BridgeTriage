"""Focused examples for the immutable shared application state."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pandas as pd
import pytest

from app.config import Provenance, RuntimeMode
from app.state import (
    ACTIVE_BUDGET_KEY,
    ACTIVE_BUDGET_VALUES,
    INITIAL_ACTIVE_BUDGET,
    LAST_VALID_BUDGET_KEY,
    SelectionContext,
    ViewContext,
    build_selection_identity,
    get_active_budget,
    is_valid_active_budget,
)
from planner import Selection, SelectionMetrics


def _selection_context(
    *,
    budget: int = INITIAL_ACTIVE_BUDGET,
    ids: tuple[str, ...] = ("B-2", "A-1"),
    score_identity: str = "score-population-v1",
) -> SelectionContext:
    selection = Selection(
        mode="budget",
        constraint=Decimal(budget),
        ids=ids,
    )
    metrics = SelectionMetrics(
        selected_count=len(ids),
        selected_total_cost=Decimal(100_000 * len(ids)),
        expected_safety_drop_coverage=0.4 * len(ids),
    )
    records = pd.DataFrame(
        {
            "structure_id": list(ids),
            "county": ["Hampshire"] * len(ids),
        }
    )
    return SelectionContext(
        active_budget=budget,
        selection=selection,
        metrics=metrics,
        selected_records=records,
        score_identity=score_identity,
    )


def _provenance(mode: RuntimeMode = RuntimeMode.PRODUCTION) -> Provenance:
    return Provenance(
        mode=mode,
        data_source="Real_Dataset",
        model_source="Real_Model_Service",
        planner_source="Real_Planner",
        backtest_source="Backtest_Engine",
    )


def test_active_budget_initializes_and_retains_last_valid_grid_member() -> None:
    state: dict[str, object] = {}

    assert get_active_budget(state, None) == INITIAL_ACTIVE_BUDGET
    assert state == {
        ACTIVE_BUDGET_KEY: INITIAL_ACTIVE_BUDGET,
        LAST_VALID_BUDGET_KEY: INITIAL_ACTIVE_BUDGET,
    }

    assert get_active_budget(state, 550_000) == 550_000
    for invalid in (99_999, 3_000_001, 525_000, 550_000.0, True, "550000", None):
        assert get_active_budget(state, invalid) == 550_000
        assert state[ACTIVE_BUDGET_KEY] == 550_000
        assert state[LAST_VALID_BUDGET_KEY] == 550_000


def test_active_budget_validates_injected_state_and_exact_boundaries() -> None:
    assert ACTIVE_BUDGET_VALUES == range(100_000, 3_000_001, 50_000)
    assert is_valid_active_budget(100_000)
    assert is_valid_active_budget(3_000_000)
    assert not is_valid_active_budget(3_000_000 - 1)
    assert not is_valid_active_budget(False)

    state: dict[str, object] = {
        ACTIVE_BUDGET_KEY: "invalid",
        LAST_VALID_BUDGET_KEY: 650_000,
    }
    assert get_active_budget(state) == 650_000

    state[ACTIVE_BUDGET_KEY] = object()
    state[LAST_VALID_BUDGET_KEY] = object()
    assert get_active_budget(state) == INITIAL_ACTIVE_BUDGET


def test_selection_context_is_one_affordable_budget_selection_and_snapshot() -> None:
    source_records = pd.DataFrame(
        {
            "structure_id": ["B-2", "A-1"],
            "county": ["Hampshire", "Berkshire"],
        }
    )
    selection = Selection(
        mode="budget",
        constraint=Decimal("500000.00"),
        ids=("B-2", "A-1"),
    )
    metrics = SelectionMetrics(
        selected_count=2,
        selected_total_cost=Decimal("325000"),
        expected_safety_drop_coverage=1.25,
    )
    context = SelectionContext(
        active_budget=500_000,
        selection=selection,
        metrics=metrics,
        selected_records=source_records,
        score_identity="scored-table-sha256",
    )

    source_records.loc[0, "county"] = "changed outside"
    first_projection = context.selected_records
    first_projection.loc[0, "county"] = "changed by renderer"

    assert context.selected_records["county"].tolist() == [
        "Hampshire",
        "Berkshire",
    ]
    assert context.selected_ids is selection.ids
    assert context.identity == build_selection_identity(
        "scored-table-sha256", selection
    )
    assert len(context.identity) == 64
    with pytest.raises(FrozenInstanceError):
        context.active_budget = 600_000  # type: ignore[misc]


def test_selection_identity_is_canonical_order_sensitive_and_filter_independent() -> None:
    equivalent_a = Selection(
        mode="budget",
        constraint=Decimal("500000.00"),
        ids=("A", "B"),
    )
    equivalent_b = Selection(
        mode="budget",
        constraint=Decimal("5E+5"),
        ids=("A", "B"),
    )
    reordered = Selection(
        mode="budget",
        constraint=Decimal("500000"),
        ids=("B", "A"),
    )

    identity = build_selection_identity("score-v1", equivalent_a)
    assert identity == build_selection_identity("score-v1", equivalent_b)
    assert identity != build_selection_identity("score-v1", reordered)
    assert identity != build_selection_identity("score-v2", equivalent_a)


def test_selection_context_rejects_disconnected_selection_projections() -> None:
    records = pd.DataFrame({"structure_id": ["A"]})
    count_selection = Selection(mode="count", constraint=1, ids=("A",))
    metrics = SelectionMetrics(
        selected_count=1,
        selected_total_cost=Decimal("100000"),
        expected_safety_drop_coverage=0.5,
    )
    with pytest.raises(ValueError, match="budget mode"):
        SelectionContext(500_000, count_selection, metrics, records, "score-v1")

    budget_selection = Selection(
        mode="budget",
        constraint=Decimal("550000"),
        ids=("A",),
    )
    with pytest.raises(ValueError, match="equal active_budget"):
        SelectionContext(500_000, budget_selection, metrics, records, "score-v1")

    matching_selection = Selection(
        mode="budget",
        constraint=Decimal("500000"),
        ids=("A",),
    )
    wrong_records = pd.DataFrame({"structure_id": ["B"]})
    with pytest.raises(ValueError, match="match the statewide selection"):
        SelectionContext(
            500_000,
            matching_selection,
            metrics,
            wrong_records,
            "score-v1",
        )


def test_view_context_delegates_every_selection_value_to_one_shared_context() -> None:
    selection_context = _selection_context()
    source = object()
    ranking = object()
    backtest = object()
    context = ViewContext(
        mode=RuntimeMode.PRODUCTION,
        provenance=_provenance(),
        source=source,
        current_ranking=ranking,
        selection_context=selection_context,
        backtest=backtest,
    )

    assert context.source is source
    assert context.current_ranking is ranking
    assert context.backtest is backtest
    assert context.selection_context is selection_context
    assert context.active_budget == selection_context.active_budget
    assert context.selection is selection_context.selection
    assert context.metrics is selection_context.metrics
    assert context.selection_identity == selection_context.identity
    with pytest.raises(FrozenInstanceError):
        context.selection_context = _selection_context()  # type: ignore[misc]


def test_view_context_rejects_mixed_provenance_and_operation_callbacks() -> None:
    selection_context = _selection_context()
    with pytest.raises(ValueError, match="Provenance mode"):
        ViewContext(
            mode=RuntimeMode.PRODUCTION,
            provenance=_provenance(RuntimeMode.DEMO),
            source=object(),
            current_ranking=object(),
            selection_context=selection_context,
        )

    with pytest.raises(TypeError, match="not an operation callback"):
        ViewContext(
            mode=RuntimeMode.PRODUCTION,
            provenance=_provenance(),
            source=lambda: object(),
            current_ranking=object(),
            selection_context=selection_context,
        )
