"""Validation and immutable value contracts for deterministic planning.

This module contains no selection algorithm.  It validates planner inputs
before candidates can be ordered and provides atomic constructors for the
immutable values returned by :mod:`planner.selection`.

The authoritative scored-table contract remains in :mod:`model.data`; the
planner deliberately delegates to that boundary instead of maintaining a
second schema definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Integral, Rational, Real
from typing import Final, Iterable, Literal, TypeAlias

import pandas as pd

from model.data import (
    SCORED_COLUMNS,
    ScoredTableValidationError,
    validate_scored_table,
)

SelectionMode: TypeAlias = Literal["budget", "count"]
BudgetValue: TypeAlias = Real | Decimal
SelectionConstraint: TypeAlias = Decimal | int

BUDGET_MODE: Final[SelectionMode] = "budget"
COUNT_MODE: Final[SelectionMode] = "count"
SELECTION_MODES: Final[tuple[SelectionMode, ...]] = (
    BUDGET_MODE,
    COUNT_MODE,
)


class PlannerContractError(ValueError):
    """Base class for planner validation failures with stable issue codes."""

    def __init__(self, issues: Iterable[str]):
        unique_issues = tuple(dict.fromkeys(issues))
        if not unique_issues:
            unique_issues = ("planner.unknown",)
        self.issues = unique_issues
        super().__init__("Planner contract failed: " + ", ".join(unique_issues))


class ScoredInputValidationError(PlannerContractError):
    """Raised when planner input violates the authoritative score contract."""


class SelectionRequestValidationError(PlannerContractError):
    """Raised when budget/count mode arguments do not form one valid request."""


class SelectionPostconditionError(PlannerContractError):
    """Raised when a proposed selection violates immutable postconditions."""


class SelectionMetricsPostconditionError(PlannerContractError):
    """Raised when proposed selection metrics violate their postconditions."""


def _decimal_budget(value: object) -> Decimal | None:
    """Return the canonical finite nonnegative budget, or ``None``.

    Booleans are rejected explicitly even though ``bool`` subclasses ``int``.
    Monetary comparisons use ``Decimal`` values so later selection code does
    not perform affordability checks with binary floating-point accumulation.
    """
    if isinstance(value, bool):
        return None

    try:
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, Integral):
            result = Decimal(int(value))
        elif isinstance(value, Rational):
            result = Decimal(value.numerator) / Decimal(value.denominator)
        elif isinstance(value, Real):
            result = Decimal(str(value))
        else:
            return None
    except (
        InvalidOperation,
        TypeError,
        ValueError,
        OverflowError,
        ZeroDivisionError,
    ):
        return None

    if not result.is_finite() or result < 0:
        return None
    return result


def _integral_count(value: object) -> int | None:
    """Return a canonical nonnegative integral count, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    result = int(value)
    return result if result >= 0 else None


def _finite_nonnegative_real(value: object) -> float | None:
    """Return a finite nonnegative real value, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result) or result < 0.0:
        return None
    return result


def _materialize_ids(
    values: object, *, issue_prefix: str
) -> tuple[tuple[object, ...], list[str]]:
    """Materialize an ID iterable once without exposing values in errors."""
    if isinstance(values, (str, bytes, bytearray)):
        return (), [f"{issue_prefix}.iterable"]
    try:
        materialized = tuple(values)  # type: ignore[arg-type]
    except TypeError:
        return (), [f"{issue_prefix}.iterable"]

    issues: list[str] = []
    if not all(isinstance(value, str) and bool(value.strip()) for value in materialized):
        issues.append(f"{issue_prefix}.nonempty_string")
    elif len(set(materialized)) != len(materialized):
        issues.append(f"{issue_prefix}.unique")
    return materialized, issues


def validate_scored_input(scored: pd.DataFrame) -> None:
    """Validate one complete planner input before candidate ordering.

    Exact columns/order, nonempty unique string identifiers, complete values,
    finite/bounded numeric fields, score arithmetic, and reason disclosure are
    delegated to the sole scored-table boundary in :mod:`model.data`.  Since a
    planner receives a self-contained scored population, its IDs are supplied
    as the expected ranking population after the exact schema is established.
    The input frame is never mutated.
    """
    if not isinstance(scored, pd.DataFrame):
        raise ScoredInputValidationError(("scored.frame_type",))
    if tuple(scored.columns) != SCORED_COLUMNS:
        raise ScoredInputValidationError(("scored.columns.exact_order",))

    ranking_ids = tuple(scored["id"].tolist())
    try:
        validate_scored_table(scored, ranking_ids)
    except ScoredTableValidationError as exc:
        raise ScoredInputValidationError(exc.issues) from exc


@dataclass(frozen=True, slots=True)
class SelectionRequest:
    """A validated, exclusive planner mode and canonical constraint."""

    mode: SelectionMode
    constraint: SelectionConstraint

    def __post_init__(self) -> None:
        if self.mode == BUDGET_MODE:
            normalized = _decimal_budget(self.constraint)
            if normalized is None:
                raise SelectionRequestValidationError(
                    ("budget.finite_nonnegative_numeric",)
                )
        elif self.mode == COUNT_MODE:
            normalized = _integral_count(self.constraint)
            if normalized is None:
                raise SelectionRequestValidationError(
                    ("count.nonnegative_integral",)
                )
        else:
            raise SelectionRequestValidationError(("selection_mode.valid",))

        object.__setattr__(self, "constraint", normalized)


def validate_selection_request(
    scored: pd.DataFrame,
    *,
    budget: object | None = None,
    count: object | None = None,
) -> SelectionRequest:
    """Validate scored input and resolve exactly one selection mode.

    ``None`` denotes an omitted mode.  Supplying both modes or neither mode is
    rejected before any candidate ordering.  The function returns only a
    complete immutable request; no partial selection is constructed on any
    failure path.
    """
    validate_scored_input(scored)

    budget_supplied = budget is not None
    count_supplied = count is not None
    if budget_supplied == count_supplied:
        raise SelectionRequestValidationError(
            ("selection_mode.exactly_one",)
        )

    if budget_supplied:
        normalized_budget = _decimal_budget(budget)
        if normalized_budget is None:
            raise SelectionRequestValidationError(
                ("budget.finite_nonnegative_numeric",)
            )
        return SelectionRequest(BUDGET_MODE, normalized_budget)

    normalized_count = _integral_count(count)
    if normalized_count is None:
        raise SelectionRequestValidationError(
            ("count.nonnegative_integral",)
        )
    return SelectionRequest(COUNT_MODE, normalized_count)


@dataclass(frozen=True, slots=True)
class Selection:
    """An immutable ordered selection returned by the real planner.

    Structural invariants (valid mode/constraint, immutable tuple IDs,
    nonempty strings, uniqueness, and count bound) are enforced directly.
    Population membership and preservation of planner order require scored
    input context and are enforced by :func:`build_selection` before return.
    """

    mode: SelectionMode
    constraint: SelectionConstraint
    ids: tuple[str, ...]

    def __post_init__(self) -> None:
        issues: list[str] = []
        normalized_constraint: SelectionConstraint | None = None

        if self.mode == BUDGET_MODE:
            normalized_constraint = _decimal_budget(self.constraint)
            if normalized_constraint is None:
                issues.append("selection.budget_constraint")
        elif self.mode == COUNT_MODE:
            normalized_constraint = _integral_count(self.constraint)
            if normalized_constraint is None:
                issues.append("selection.count_constraint")
        else:
            issues.append("selection.mode")

        if not isinstance(self.ids, tuple):
            issues.append("selection.ids.immutable_tuple")
        else:
            if not all(
                isinstance(identifier, str) and bool(identifier.strip())
                for identifier in self.ids
            ):
                issues.append("selection.ids.nonempty_string")
            elif len(set(self.ids)) != len(self.ids):
                issues.append("selection.ids.unique")

        if (
            self.mode == COUNT_MODE
            and isinstance(normalized_constraint, int)
            and isinstance(self.ids, tuple)
            and len(self.ids) > normalized_constraint
        ):
            issues.append("selection.count_bound")

        if issues:
            raise SelectionPostconditionError(issues)

        object.__setattr__(self, "constraint", normalized_constraint)


def validate_selection_postconditions(
    scored: pd.DataFrame,
    selection: Selection,
    *,
    ordered_candidate_ids: Iterable[str],
) -> None:
    """Validate selection membership and candidate-order preservation.

    ``ordered_candidate_ids`` is produced by the selection algorithm.  It must
    be a unique permutation of the complete scored population.  Selection IDs
    must then form a unique subsequence of that order, which permits budget
    mode to continue past unaffordable candidates while preserving order.
    """
    validate_scored_input(scored)
    if not isinstance(selection, Selection):
        raise SelectionPostconditionError(("selection.type",))

    candidates, issues = _materialize_ids(
        ordered_candidate_ids, issue_prefix="candidate_order.ids"
    )
    scored_ids = tuple(scored["id"].tolist())

    candidates_are_valid = not issues
    if candidates_are_valid and (
        len(candidates) != len(scored_ids)
        or set(candidates) != set(scored_ids)
    ):
        issues.append("candidate_order.complete_population")
        candidates_are_valid = False

    scored_id_set = set(scored_ids)
    if any(identifier not in scored_id_set for identifier in selection.ids):
        issues.append("selection.ids.membership")

    if candidates_are_valid and all(
        identifier in set(candidates) for identifier in selection.ids
    ):
        positions = {
            identifier: position
            for position, identifier in enumerate(candidates)
        }
        selected_positions = [positions[identifier] for identifier in selection.ids]
        if any(
            later <= earlier
            for earlier, later in zip(
                selected_positions, selected_positions[1:]
            )
        ):
            issues.append("selection.ids.planner_order")

    if issues:
        raise SelectionPostconditionError(issues)


def build_selection(
    scored: pd.DataFrame,
    request: SelectionRequest,
    selected_ids: Iterable[str],
    *,
    ordered_candidate_ids: Iterable[str],
) -> Selection:
    """Atomically validate and construct an immutable ``Selection``.

    No ``Selection`` is returned unless the scored input, request, selected ID
    structure, population membership, and planner-order postconditions all
    pass.  This is the construction path intended for ``planner.selection``.
    """
    validate_scored_input(scored)
    if not isinstance(request, SelectionRequest):
        raise SelectionPostconditionError(("selection.request_type",))

    materialized_ids, issues = _materialize_ids(
        selected_ids, issue_prefix="selection.ids"
    )
    if issues:
        raise SelectionPostconditionError(issues)

    selection = Selection(
        mode=request.mode,
        constraint=request.constraint,
        ids=materialized_ids,  # type: ignore[arg-type]
    )
    validate_selection_postconditions(
        scored,
        selection,
        ordered_candidate_ids=ordered_candidate_ids,
    )
    return selection


@dataclass(frozen=True, slots=True)
class SelectionMetrics:
    """Immutable aggregate postconditions for one validated selection."""

    selected_count: int
    selected_total_cost: Decimal
    expected_safety_drop_coverage: float

    def __post_init__(self) -> None:
        issues: list[str] = []
        count = _integral_count(self.selected_count)
        cost = _decimal_budget(self.selected_total_cost)
        coverage = _finite_nonnegative_real(
            self.expected_safety_drop_coverage
        )

        if count is None:
            issues.append("metrics.selected_count.nonnegative_integral")
        if cost is None:
            issues.append("metrics.selected_total_cost.finite_nonnegative")
        if coverage is None:
            issues.append("metrics.expected_coverage.finite_nonnegative")

        if count is not None and cost is not None:
            if count == 0 and cost != 0:
                issues.append("metrics.empty_selection.zero_cost")
            elif count > 0 and cost <= 0:
                issues.append("metrics.nonempty_selection.positive_cost")

        if count is not None and coverage is not None:
            if count == 0 and coverage != 0.0:
                issues.append("metrics.empty_selection.zero_coverage")
            elif coverage > count:
                issues.append("metrics.expected_coverage.count_bound")

        if issues:
            raise SelectionMetricsPostconditionError(issues)

        object.__setattr__(self, "selected_count", count)
        object.__setattr__(self, "selected_total_cost", cost)
        object.__setattr__(
            self, "expected_safety_drop_coverage", coverage
        )


def validate_selection_metrics_postconditions(
    selection: Selection, metrics: SelectionMetrics
) -> None:
    """Validate that immutable metrics identify the same selection size."""
    issues: list[str] = []
    if not isinstance(selection, Selection):
        issues.append("selection.type")
    if not isinstance(metrics, SelectionMetrics):
        issues.append("metrics.type")
    if (
        isinstance(selection, Selection)
        and isinstance(metrics, SelectionMetrics)
        and metrics.selected_count != len(selection.ids)
    ):
        issues.append("metrics.selected_count.selection_size")
    if issues:
        raise SelectionMetricsPostconditionError(issues)


def build_selection_metrics(
    selection: Selection,
    *,
    selected_total_cost: BudgetValue,
    expected_safety_drop_coverage: Real | Decimal,
) -> SelectionMetrics:
    """Atomically construct metrics whose count is derived from selection."""
    if not isinstance(selection, Selection):
        raise SelectionMetricsPostconditionError(("selection.type",))

    metrics = SelectionMetrics(
        selected_count=len(selection.ids),
        selected_total_cost=selected_total_cost,  # type: ignore[arg-type]
        expected_safety_drop_coverage=expected_safety_drop_coverage,  # type: ignore[arg-type]
    )
    validate_selection_metrics_postconditions(selection, metrics)
    return metrics


__all__ = [
    "BUDGET_MODE",
    "COUNT_MODE",
    "SELECTION_MODES",
    "BudgetValue",
    "PlannerContractError",
    "ScoredInputValidationError",
    "Selection",
    "SelectionConstraint",
    "SelectionMetrics",
    "SelectionMetricsPostconditionError",
    "SelectionMode",
    "SelectionPostconditionError",
    "SelectionRequest",
    "SelectionRequestValidationError",
    "build_selection",
    "build_selection_metrics",
    "validate_scored_input",
    "validate_selection_metrics_postconditions",
    "validate_selection_postconditions",
    "validate_selection_request",
]
