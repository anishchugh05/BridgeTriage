"""Deterministic budget and fixed-count bridge selection.

The public functions in this module operate only on a validated scored table.
Candidate ordering is shared by both modes, while all returned values are
constructed through :mod:`planner.contracts` so no partial or mutable result
can escape a failed validation.
"""

from __future__ import annotations

import math
from decimal import Decimal
from numbers import Integral

import pandas as pd

from .contracts import (
    BUDGET_MODE,
    COUNT_MODE,
    BudgetValue,
    Selection,
    SelectionMetrics,
    SelectionPostconditionError,
    SelectionRequest,
    build_selection,
    build_selection_metrics,
    validate_scored_input,
    validate_selection_postconditions,
    validate_selection_request,
)


def _ordered_candidate_ids(scored: pd.DataFrame) -> tuple[str, ...]:
    """Return IDs ordered by descending score and ascending string ID.

    Scored-table validation guarantees finite numeric scores and unique,
    nonempty string identifiers before this helper is called. Two stable sorts
    make the tie-break explicit without changing the caller's frame.
    """
    candidates = list(
        scored.loc[:, ["id", "score"]].itertuples(index=False, name=None)
    )
    candidates.sort(key=lambda candidate: str(candidate[0]))
    candidates.sort(key=lambda candidate: candidate[1], reverse=True)
    return tuple(str(identifier) for identifier, _score in candidates)


def _as_decimal(value: object) -> Decimal:
    """Convert one validated model numeric value to its decimal spelling."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _costs_by_id(scored: pd.DataFrame) -> dict[str, Decimal]:
    """Return validated modeled inspection costs as exact decimals."""
    return {
        str(identifier): _as_decimal(cost)
        for identifier, cost in scored.loc[:, ["id", "insp_cost"]].itertuples(
            index=False, name=None
        )
    }


def _select_ids(
    ordered_ids: tuple[str, ...],
    costs: dict[str, Decimal],
    request: SelectionRequest,
) -> tuple[str, ...]:
    """Apply one already-validated request to deterministic candidates."""
    if request.mode == COUNT_MODE:
        count = request.constraint
        if not isinstance(count, int):  # Defensive; SelectionRequest guarantees it.
            raise SelectionPostconditionError(("selection.count_constraint",))
        return ordered_ids[: min(count, len(ordered_ids))]

    if request.mode != BUDGET_MODE or not isinstance(
        request.constraint, Decimal
    ):
        raise SelectionPostconditionError(("selection.budget_constraint",))

    remaining = request.constraint
    selected: list[str] = []
    for identifier in ordered_ids:
        cost = costs[identifier]
        if cost <= remaining:
            selected.append(identifier)
            remaining -= cost
    return tuple(selected)


def select(
    scored: pd.DataFrame,
    *,
    budget: BudgetValue | None = None,
    count: Integral | None = None,
) -> Selection:
    """Select bridges under exactly one budget or fixed-count constraint.

    Candidates are visited by descending ``score`` and then ascending string
    ``id``. Budget mode uses ``Decimal`` values for every affordability check
    and keeps traversing after an oversized candidate. Count mode returns the
    deterministic prefix, capped at the scored population size.
    """
    request = validate_selection_request(scored, budget=budget, count=count)
    ordered_ids = _ordered_candidate_ids(scored)
    costs = _costs_by_id(scored)
    selected_ids = _select_ids(ordered_ids, costs, request)

    selection = build_selection(
        scored,
        request,
        selected_ids,
        ordered_candidate_ids=ordered_ids,
    )

    if request.mode == BUDGET_MODE:
        selected_cost = sum(
            (costs[identifier] for identifier in selection.ids), Decimal(0)
        )
        if selected_cost > request.constraint:
            raise SelectionPostconditionError(("selection.budget_affordability",))

    return selection


def summarize_selection(
    scored: pd.DataFrame, selection: Selection
) -> SelectionMetrics:
    """Project exact aggregate metrics from one validated ``Selection``.

    The selected count comes from the immutable ID tuple, modeled inspection
    cost is summed after ``Decimal(str(value))`` conversion, and expected
    Safety_Drop coverage uses ``math.fsum`` over the selected probabilities.
    No ordering, filtering, or independent selection is performed here.
    """
    validate_scored_input(scored)
    if not isinstance(selection, Selection):
        raise SelectionPostconditionError(("selection.type",))

    ordered_ids = _ordered_candidate_ids(scored)
    validate_selection_postconditions(
        scored,
        selection,
        ordered_candidate_ids=ordered_ids,
    )

    row_by_id = {
        str(row.id): row
        for row in scored.loc[:, ["id", "insp_cost", "p_worse"]].itertuples(
            index=False
        )
    }
    selected_total_cost = sum(
        (_as_decimal(row_by_id[identifier].insp_cost) for identifier in selection.ids),
        Decimal(0),
    )
    expected_coverage = math.fsum(
        float(row_by_id[identifier].p_worse) for identifier in selection.ids
    )

    return build_selection_metrics(
        selection,
        selected_total_cost=selected_total_cost,
        expected_safety_drop_coverage=expected_coverage,
    )


__all__ = ["select", "summarize_selection"]
