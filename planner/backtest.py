"""Two-phase leakage-free planning and outcome evaluation.

Historical planning receives no observations after 2018: it fits the existing
model on an explicitly bounded 2012/2016 slice, scores the complete 2018
population, and freezes every method's budget selection.  Only after that plan
exists does outcome evaluation obtain 2022 records, apply the shared
``model.is_safety_drop`` target to eligible selected endpoints, and validate
the exact public result table.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from numbers import Integral, Real
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import pandas as pd

from model import fit_decline, is_safety_drop, score
from model.data import project_scored_table, validate_scored_table

from .contracts import (
    BUDGET_MODE,
    PlannerContractError,
    SelectionRequest,
)
from .selection import _costs_by_id, _ordered_candidate_ids, _select_ids

BacktestMethod: TypeAlias = Literal["ours", "worst_first", "oldest_first"]
FitFunction: TypeAlias = Callable[[pd.DataFrame, int, int], object]
ScoreFunction: TypeAlias = Callable[[pd.DataFrame, object, object], pd.DataFrame]

BACKTEST_METHODS: Final[tuple[BacktestMethod, ...]] = (
    "ours",
    "worst_first",
    "oldest_first",
)
BACKTEST_FIT_START_YEAR: Final[int] = 2012
BACKTEST_FIT_END_YEAR: Final[int] = 2016
BACKTEST_CANDIDATE_YEAR: Final[int] = 2018
BACKTEST_AS_OF: Final[pd.Timestamp] = pd.Timestamp("2018-04-01")
_REQUIRED_HISTORICAL_YEARS: Final[frozenset[int]] = frozenset(
    {
        BACKTEST_FIT_START_YEAR,
        BACKTEST_FIT_END_YEAR,
        BACKTEST_CANDIDATE_YEAR,
    }
)


class HistoricalPlanningError(PlannerContractError):
    """Raised when a historical panel cannot produce one atomic plan."""


@dataclass(frozen=True, slots=True)
class BacktestSelection:
    """One immutable method selection frozen before outcomes are available."""

    budget: Decimal
    method: BacktestMethod
    ids: tuple[str, ...]

    def __post_init__(self) -> None:
        issues: list[str] = []

        try:
            request = SelectionRequest(BUDGET_MODE, self.budget)
        except PlannerContractError as exc:
            issues.extend(exc.issues)
            normalized_budget: Decimal | None = None
        else:
            normalized_budget = request.constraint
            if not isinstance(normalized_budget, Decimal):  # pragma: no cover
                issues.append("backtest.selection.budget")

        if self.method not in BACKTEST_METHODS:
            issues.append("backtest.selection.method")

        if not isinstance(self.ids, tuple):
            issues.append("backtest.selection.ids.immutable_tuple")
        elif not all(
            isinstance(identifier, str) and bool(identifier.strip())
            for identifier in self.ids
        ):
            issues.append("backtest.selection.ids.nonempty_string")
        elif len(set(self.ids)) != len(self.ids):
            issues.append("backtest.selection.ids.unique")

        if issues:
            raise HistoricalPlanningError(issues)

        object.__setattr__(self, "budget", normalized_budget)


@dataclass(frozen=True, slots=True)
class BacktestPlan:
    """Immutable audit record for all decisions made with data through 2018.

    ``candidate_ids`` is a canonical ascending-ID representation of the common
    candidate set. ``modeled_costs`` and ``worst_ratings_2018`` are immutable
    key/value tuples aligned to that set. ``budgets`` contains sorted distinct
    canonical decimal budgets, and ``selections`` contains one record for each
    budget/method pair in canonical method order.
    """

    candidate_ids: tuple[str, ...]
    modeled_costs: tuple[tuple[str, Decimal], ...]
    budgets: tuple[Decimal, ...]
    selections: tuple[BacktestSelection, ...]
    worst_ratings_2018: tuple[tuple[str, float | None], ...]
    fit_start_year: int = field(default=BACKTEST_FIT_START_YEAR, init=False)
    fit_end_year: int = field(default=BACKTEST_FIT_END_YEAR, init=False)
    candidate_year: int = field(default=BACKTEST_CANDIDATE_YEAR, init=False)
    as_of: pd.Timestamp = field(default=BACKTEST_AS_OF, init=False)

    def __post_init__(self) -> None:
        issues: list[str] = []

        if (
            not isinstance(self.candidate_ids, tuple)
            or not self.candidate_ids
            or not all(
                isinstance(identifier, str) and bool(identifier.strip())
                for identifier in self.candidate_ids
            )
        ):
            issues.append("backtest.candidate_ids.nonempty_string_tuple")
        elif len(set(self.candidate_ids)) != len(self.candidate_ids):
            issues.append("backtest.candidate_ids.unique")
        elif self.candidate_ids != tuple(sorted(self.candidate_ids)):
            issues.append("backtest.candidate_ids.canonical_order")

        cost_ids: tuple[str, ...] = ()
        cost_by_id: dict[str, Decimal] = {}
        if not isinstance(self.modeled_costs, tuple):
            issues.append("backtest.modeled_costs.immutable_tuple")
        else:
            cost_entries_are_structural = all(
                isinstance(entry, tuple) and len(entry) == 2
                for entry in self.modeled_costs
            )
            if not cost_entries_are_structural:
                issues.append("backtest.modeled_costs.entries")
            else:
                cost_ids = tuple(entry[0] for entry in self.modeled_costs)
                for identifier, cost in self.modeled_costs:
                    if (
                        not isinstance(identifier, str)
                        or not isinstance(cost, Decimal)
                        or not cost.is_finite()
                        or cost <= 0
                    ):
                        issues.append("backtest.modeled_costs.positive_decimal")
                        break
                    cost_by_id[identifier] = cost
                if cost_ids != self.candidate_ids:
                    issues.append("backtest.modeled_costs.complete_population")

        rating_ids: tuple[str, ...] = ()
        if not isinstance(self.worst_ratings_2018, tuple):
            issues.append("backtest.worst_ratings.immutable_tuple")
        else:
            rating_entries_are_structural = all(
                isinstance(entry, tuple) and len(entry) == 2
                for entry in self.worst_ratings_2018
            )
            if not rating_entries_are_structural:
                issues.append("backtest.worst_ratings.entries")
            else:
                rating_ids = tuple(entry[0] for entry in self.worst_ratings_2018)
                for identifier, rating in self.worst_ratings_2018:
                    if not isinstance(identifier, str):
                        issues.append("backtest.worst_ratings.identifier")
                        break
                    if rating is not None and (
                        isinstance(rating, bool)
                        or not isinstance(rating, Real)
                        or not math.isfinite(float(rating))
                        or not 0.0 <= float(rating) <= 9.0
                    ):
                        issues.append("backtest.worst_ratings.valid")
                        break
                if rating_ids != self.candidate_ids:
                    issues.append("backtest.worst_ratings.complete_population")

        if not isinstance(self.budgets, tuple) or not self.budgets:
            issues.append("backtest.budgets.nonempty_tuple")
        elif any(
            not isinstance(budget, Decimal)
            or not budget.is_finite()
            or budget < 0
            for budget in self.budgets
        ):
            issues.append("backtest.budgets.finite_nonnegative_decimal")
        elif self.budgets != tuple(sorted(set(self.budgets))):
            issues.append("backtest.budgets.sorted_distinct")

        expected_keys = tuple(
            (budget, method)
            for budget in self.budgets
            for method in BACKTEST_METHODS
        )
        if not isinstance(self.selections, tuple) or not all(
            isinstance(selection, BacktestSelection)
            for selection in self.selections
        ):
            issues.append("backtest.selections.immutable_records")
        else:
            actual_keys = tuple(
                (selection.budget, selection.method)
                for selection in self.selections
            )
            if actual_keys != expected_keys:
                issues.append("backtest.selections.canonical_complete")

            candidate_set = set(self.candidate_ids)
            for selection in self.selections:
                if any(identifier not in candidate_set for identifier in selection.ids):
                    issues.append("backtest.selections.membership")
                    break
                if cost_by_id and sum(
                    (cost_by_id[identifier] for identifier in selection.ids),
                    Decimal(0),
                ) > selection.budget:
                    issues.append("backtest.selections.affordable")
                    break

        if issues:
            raise HistoricalPlanningError(issues)

    @property
    def fit_years(self) -> tuple[int, int]:
        """Return the exact historical model interval."""
        return self.fit_start_year, self.fit_end_year

    @property
    def cost_by_id(self) -> Mapping[str, Decimal]:
        """Expose modeled 2018 costs through a read-only mapping."""
        return MappingProxyType(dict(self.modeled_costs))

    @property
    def starting_rating_by_id(self) -> Mapping[str, float | None]:
        """Expose frozen 2018 starting ratings for later outcome evaluation."""
        return MappingProxyType(dict(self.worst_ratings_2018))

    def selection_for(
        self, budget: Real | Decimal, method: BacktestMethod
    ) -> BacktestSelection:
        """Return the frozen record for one canonical budget/method pair."""
        try:
            request = SelectionRequest(BUDGET_MODE, budget)
        except PlannerContractError as exc:
            raise KeyError((budget, method)) from exc
        normalized_budget = request.constraint
        for selection in self.selections:
            if selection.budget == normalized_budget and selection.method == method:
                return selection
        raise KeyError((normalized_budget, method))


def _integer_year(value: object) -> int | None:
    """Return an exact integer year, rejecting booleans and float coercion."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def _is_missing(value: object) -> bool:
    """Return whether a candidate scalar is missing."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _finite_number(value: object) -> float | None:
    """Return one finite real/decimal value, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _historical_slices(
    panel_through_2018: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the time boundary and return isolated fit/candidate copies.

    Only the ``year`` column is inspected before the future-year rejection.
    Consequently a frame containing a 2019-or-later record fails before any
    identifier, rating, age, model, or scoring access can influence planning.
    """
    if not isinstance(panel_through_2018, pd.DataFrame):
        raise HistoricalPlanningError(("backtest.panel.frame_type",))
    if "year" not in panel_through_2018.columns:
        raise HistoricalPlanningError(("backtest.panel.year.required",))

    years = tuple(
        _integer_year(value) for value in panel_through_2018["year"].tolist()
    )
    if any(year is None for year in years):
        raise HistoricalPlanningError(("backtest.panel.year.integral",))
    normalized_years = tuple(year for year in years if year is not None)

    if any(year > BACKTEST_CANDIDATE_YEAR for year in normalized_years):
        raise HistoricalPlanningError(("backtest.panel.future_year",))

    missing_years = _REQUIRED_HISTORICAL_YEARS - set(normalized_years)
    if missing_years:
        raise HistoricalPlanningError(("backtest.panel.required_years",))

    required_columns = {"id", "worst_rating", "age"}
    if not required_columns.issubset(panel_through_2018.columns):
        raise HistoricalPlanningError(("backtest.panel.candidate_columns",))

    year_series = pd.Series(normalized_years, index=panel_through_2018.index)
    relevant = panel_through_2018.loc[
        year_series.isin(_REQUIRED_HISTORICAL_YEARS)
    ].copy(deep=True)

    identifiers = relevant["id"].tolist()
    if not all(
        isinstance(identifier, str) and bool(identifier.strip())
        for identifier in identifiers
    ):
        raise HistoricalPlanningError(("backtest.panel.id.nonempty_string",))

    identity_keys = pd.DataFrame(
        {"id": identifiers, "year": year_series.loc[relevant.index].tolist()}
    )
    if identity_keys.duplicated(subset=["id", "year"]).any():
        raise HistoricalPlanningError(("backtest.panel.id_year.unique",))

    candidates = relevant.loc[
        year_series.loc[relevant.index].eq(BACKTEST_CANDIDATE_YEAR)
    ].copy(deep=True)
    candidate_ids = tuple(candidates["id"].tolist())
    if len(set(candidate_ids)) != len(candidate_ids):
        raise HistoricalPlanningError(("backtest.candidates.id.unique",))

    for rating in candidates["worst_rating"].tolist():
        if _is_missing(rating):
            continue
        numeric_rating = _finite_number(rating)
        if (
            numeric_rating is None
            or not numeric_rating.is_integer()
            or not 0.0 <= numeric_rating <= 9.0
        ):
            raise HistoricalPlanningError(
                ("backtest.candidates.worst_rating.valid",)
            )

    for age in candidates["age"].tolist():
        numeric_age = _finite_number(age)
        if numeric_age is None or numeric_age < 0.0:
            raise HistoricalPlanningError(("backtest.candidates.age.valid",))

    fit_rows = relevant.loc[
        year_series.loc[relevant.index].isin(
            {BACKTEST_FIT_START_YEAR, BACKTEST_FIT_END_YEAR}
        )
    ].copy(deep=True)
    return fit_rows, candidates


def _canonical_budgets(budgets: Iterable[Real | Decimal]) -> tuple[Decimal, ...]:
    """Materialize, validate, deduplicate, and sort requested budgets."""
    if isinstance(budgets, (str, bytes, bytearray)):
        raise HistoricalPlanningError(("backtest.budgets.iterable",))
    try:
        requested = tuple(budgets)
    except TypeError as exc:
        raise HistoricalPlanningError(("backtest.budgets.iterable",)) from exc
    if not requested:
        raise HistoricalPlanningError(("backtest.budgets.nonempty",))

    normalized: set[Decimal] = set()
    for value in requested:
        try:
            request = SelectionRequest(BUDGET_MODE, value)
        except PlannerContractError as exc:
            raise HistoricalPlanningError(exc.issues) from exc
        budget = request.constraint
        if not isinstance(budget, Decimal):  # pragma: no cover - contract guard
            raise HistoricalPlanningError(("backtest.budget.decimal",))
        normalized.add(budget)
    return tuple(sorted(normalized))


def _worst_first_order(candidates: pd.DataFrame) -> tuple[str, ...]:
    """Order 2018 candidates by rating ascending, null last, then ID."""
    records = tuple(
        candidates.loc[:, ["id", "worst_rating"]].itertuples(
            index=False, name=None
        )
    )

    def key(record: tuple[object, object]) -> tuple[bool, float, str]:
        identifier, rating = record
        missing = _is_missing(rating)
        numeric = 0.0 if missing else float(rating)
        return missing, numeric, str(identifier)

    return tuple(str(identifier) for identifier, _rating in sorted(records, key=key))


def _oldest_first_order(candidates: pd.DataFrame) -> tuple[str, ...]:
    """Order 2018 candidates by age descending and ascending string ID."""
    records = tuple(
        candidates.loc[:, ["id", "age"]].itertuples(index=False, name=None)
    )
    return tuple(
        str(identifier)
        for identifier, _age in sorted(
            records,
            key=lambda record: (-float(record[1]), str(record[0])),
        )
    )


def build_historical_plan(
    panel_through_2018: pd.DataFrame,
    budgets: Iterable[Real | Decimal],
    *,
    fit_fn: FitFunction = fit_decline,
    score_fn: ScoreFunction = score,
) -> BacktestPlan:
    """Build and freeze all backtest selections using data through 2018 only.

    The model receives exactly 2012/2016 rows and interval arguments. The
    scorer receives every unique 2018 candidate and the exact April 1, 2018
    as-of timestamp. The authoritative scored-table validator establishes an
    identifier bijection and valid modeled costs before any ordering occurs.
    All methods then use one shared affordability implementation from
    :mod:`planner.selection`; only their deterministic candidate orders differ.
    """
    fit_rows, candidates = _historical_slices(panel_through_2018)
    canonical_budgets = _canonical_budgets(budgets)

    requested_candidate_ids = tuple(sorted(candidates["id"].tolist()))
    rates = fit_fn(
        fit_rows.copy(deep=True),
        BACKTEST_FIT_START_YEAR,
        BACKTEST_FIT_END_YEAR,
    )
    full_scored = score_fn(
        candidates.copy(deep=True),
        rates,
        BACKTEST_AS_OF,
    )
    scored = project_scored_table(full_scored)

    if "id" not in scored.columns:
        raise HistoricalPlanningError(("backtest.scored.id.required",))
    scored_ids = tuple(scored["id"].tolist())
    requested_set = set(requested_candidate_ids)
    scored_set = set(scored_ids)
    if not scored_ids or not scored_set.issubset(requested_set):
        raise HistoricalPlanningError(("backtest.scored.population",))

    # The model's established contract omits records whose published
    # worst_rating is not applicable. Only those rows may be absent from the
    # common planning population; all methods then use the identical scored
    # IDs and modeled costs.
    missing_ids = requested_set - scored_set
    if missing_ids:
        missing_rows = candidates.loc[candidates["id"].isin(missing_ids)]
        if missing_rows["worst_rating"].notna().any():
            raise HistoricalPlanningError(
                ("backtest.scored.complete_model_eligible_population",)
            )
        candidates = candidates.loc[candidates["id"].isin(scored_set)].copy(
            deep=True
        )

    candidate_ids = tuple(sorted(scored_ids))
    validate_scored_table(scored, candidate_ids)

    costs = _costs_by_id(scored)
    method_orders: dict[BacktestMethod, tuple[str, ...]] = {
        "ours": _ordered_candidate_ids(scored),
        "worst_first": _worst_first_order(candidates),
        "oldest_first": _oldest_first_order(candidates),
    }
    candidate_set = set(candidate_ids)
    if any(
        len(order) != len(candidate_ids)
        or len(set(order)) != len(order)
        or set(order) != candidate_set
        for order in method_orders.values()
    ):
        raise HistoricalPlanningError(("backtest.method_order.population",))

    selections: list[BacktestSelection] = []
    for budget in canonical_budgets:
        request = SelectionRequest(BUDGET_MODE, budget)
        for method in BACKTEST_METHODS:
            selected_ids = _select_ids(method_orders[method], costs, request)
            selections.append(
                BacktestSelection(
                    budget=budget,
                    method=method,
                    ids=selected_ids,
                )
            )

    candidate_by_id = candidates.set_index("id", verify_integrity=True)
    modeled_costs = tuple(
        (identifier, costs[identifier]) for identifier in candidate_ids
    )
    worst_ratings = tuple(
        (
            identifier,
            None
            if _is_missing(candidate_by_id.at[identifier, "worst_rating"])
            else float(candidate_by_id.at[identifier, "worst_rating"]),
        )
        for identifier in candidate_ids
    )

    return BacktestPlan(
        candidate_ids=candidate_ids,
        modeled_costs=modeled_costs,
        budgets=canonical_budgets,
        selections=tuple(selections),
        worst_ratings_2018=worst_ratings,
    )


BACKTEST_OUTCOME_YEAR: Final[int] = 2022
BACKTEST_RESULT_COLUMNS: Final[tuple[str, str, str]] = (
    "budget",
    "method",
    "caught",
)

BacktestKey: TypeAlias = tuple[Decimal, BacktestMethod]
BacktestResult: TypeAlias = pd.DataFrame
TargetFunction: TypeAlias = Callable[[object, object], bool]


class OutcomeEvaluationError(PlannerContractError):
    """Raised when frozen selections cannot be evaluated atomically."""


class BacktestResultValidationError(PlannerContractError):
    """Raised when a public backtest result violates its exact contract."""


def _normalize_backtest_budget(value: object) -> Decimal | None:
    """Return a canonical result budget using the planner's money contract."""
    try:
        request = SelectionRequest(BUDGET_MODE, value)  # type: ignore[arg-type]
    except PlannerContractError:
        return None
    budget = request.constraint
    return budget if isinstance(budget, Decimal) else None


def _nonnegative_integer(value: object) -> int | None:
    """Return one nonnegative integer count, rejecting booleans and floats."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    normalized = int(value)
    return normalized if normalized >= 0 else None


def _expected_backtest_keys(plan: BacktestPlan) -> tuple[BacktestKey, ...]:
    """Return canonical budget-major, method-minor result keys."""
    return tuple(
        (budget, method)
        for budget in plan.budgets
        for method in BACKTEST_METHODS
    )


def _validated_eligible_counts(
    plan: BacktestPlan,
    eligible_selected_count: Mapping[BacktestKey, int],
) -> dict[BacktestKey, int]:
    """Validate and normalize the private eligible-count audit mapping."""
    if not isinstance(plan, BacktestPlan):
        raise BacktestResultValidationError(("backtest.plan.type",))
    if not isinstance(eligible_selected_count, Mapping):
        raise BacktestResultValidationError(
            ("backtest.eligible_counts.mapping",)
        )

    issues: list[str] = []
    normalized: dict[BacktestKey, int] = {}
    for raw_key, raw_count in eligible_selected_count.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            issues.append("backtest.eligible_counts.key")
            continue

        raw_budget, raw_method = raw_key
        budget = _normalize_backtest_budget(raw_budget)
        if budget is None:
            issues.append("backtest.eligible_counts.budget")
        if raw_method not in BACKTEST_METHODS:
            issues.append("backtest.eligible_counts.method")
        count = _nonnegative_integer(raw_count)
        if count is None:
            issues.append("backtest.eligible_counts.nonnegative_integral")

        if budget is None or raw_method not in BACKTEST_METHODS or count is None:
            continue
        key: BacktestKey = (budget, raw_method)
        if key in normalized:
            issues.append("backtest.eligible_counts.unique_keys")
            continue
        normalized[key] = count

    expected_keys = _expected_backtest_keys(plan)
    if set(normalized) != set(expected_keys):
        issues.append("backtest.eligible_counts.complete_keys")

    selections = {
        (selection.budget, selection.method): selection
        for selection in plan.selections
    }
    for key, count in normalized.items():
        selection = selections.get(key)
        if selection is not None and count > len(selection.ids):
            issues.append("backtest.eligible_counts.selection_bound")
            break

    if issues:
        raise BacktestResultValidationError(issues)
    return normalized


def validate_backtest_result(
    result: pd.DataFrame,
    plan: BacktestPlan,
    eligible_selected_count: Mapping[BacktestKey, int],
) -> None:
    """Validate one public result against its frozen plan and outcome audit.

    The validator accepts no repaired or partial table.  It verifies exact
    columns, cardinality, budget/method sets, canonical row order, integral
    counts, and each count's corresponding eligible-selection bound.
    """
    eligible_counts = _validated_eligible_counts(
        plan, eligible_selected_count
    )

    if not isinstance(result, pd.DataFrame):
        raise BacktestResultValidationError(("backtest.result.frame_type",))
    if tuple(result.columns) != BACKTEST_RESULT_COLUMNS:
        raise BacktestResultValidationError(
            ("backtest.result.columns.exact_order",)
        )

    issues: list[str] = []
    expected_keys = _expected_backtest_keys(plan)
    records = tuple(result.itertuples(index=False, name=None))
    if len(records) != len(expected_keys):
        issues.append("backtest.result.row_cardinality")

    actual_keys: list[BacktestKey] = []
    budgets_are_valid = True
    methods_are_valid = True
    counts_by_key: dict[BacktestKey, int] = {}

    for raw_budget, raw_method, raw_caught in records:
        budget = _normalize_backtest_budget(raw_budget)
        if budget is None:
            issues.append("backtest.result.budget.numeric")
            budgets_are_valid = False

        if raw_method not in BACKTEST_METHODS:
            issues.append("backtest.result.method.valid")
            methods_are_valid = False

        caught = _nonnegative_integer(raw_caught)
        if caught is None:
            issues.append("backtest.result.caught.nonnegative_integral")

        if budget is None or raw_method not in BACKTEST_METHODS:
            continue
        key: BacktestKey = (budget, raw_method)
        actual_keys.append(key)
        if key in counts_by_key:
            issues.append("backtest.result.key.unique")
        elif caught is not None:
            counts_by_key[key] = caught

    if budgets_are_valid and set(budget for budget, _ in actual_keys) != set(
        plan.budgets
    ):
        issues.append("backtest.result.budget_set")
    if methods_are_valid and set(method for _, method in actual_keys) != set(
        BACKTEST_METHODS
    ):
        issues.append("backtest.result.method_set")

    if budgets_are_valid and methods_are_valid:
        if set(actual_keys) != set(expected_keys):
            issues.append("backtest.result.key_set")
        if tuple(actual_keys) != expected_keys:
            issues.append("backtest.result.canonical_order")

    for key, caught in counts_by_key.items():
        eligible = eligible_counts.get(key)
        if eligible is not None and caught > eligible:
            issues.append("backtest.result.caught.eligible_bound")
            break

    if issues:
        raise BacktestResultValidationError(issues)


@dataclass(frozen=True, slots=True)
class BacktestRun:
    """A validated result plus private eligibility and selection audit data."""

    result: BacktestResult
    eligible_selected_count: Mapping[BacktestKey, int]
    selection_audit: BacktestPlan

    def __post_init__(self) -> None:
        result_copy: object = self.result
        if isinstance(self.result, pd.DataFrame):
            result_copy = self.result.copy(deep=True)

        eligible_copy: object = self.eligible_selected_count
        if isinstance(self.eligible_selected_count, Mapping):
            eligible_copy = dict(self.eligible_selected_count)

        validate_backtest_result(
            result_copy,  # type: ignore[arg-type]
            self.selection_audit,
            eligible_copy,  # type: ignore[arg-type]
        )
        object.__setattr__(self, "result", result_copy)
        object.__setattr__(
            self,
            "eligible_selected_count",
            MappingProxyType(eligible_copy),  # type: ignore[arg-type]
        )


def _selected_identifier_order(plan: BacktestPlan) -> tuple[str, ...]:
    """Return selected IDs once, preserving first frozen selection order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for selection in plan.selections:
        for identifier in selection.ids:
            if identifier not in seen:
                seen.add(identifier)
                ordered.append(identifier)
    return tuple(ordered)


def _selected_outcome_rows(
    plan: BacktestPlan, outcomes_2022: pd.DataFrame
) -> dict[str, tuple[object, object]]:
    """Match selected identifiers to at most one raw 2022 endpoint row."""
    if not isinstance(outcomes_2022, pd.DataFrame):
        raise OutcomeEvaluationError(("backtest.outcomes.frame_type",))

    required_columns = {"id", "worst_rating", "rebuilt"}
    if not required_columns.issubset(outcomes_2022.columns):
        raise OutcomeEvaluationError(("backtest.outcomes.columns.required",))

    if "year" in outcomes_2022.columns:
        years = tuple(
            _integer_year(value) for value in outcomes_2022["year"].tolist()
        )
        if any(year != BACKTEST_OUTCOME_YEAR for year in years):
            raise OutcomeEvaluationError(("backtest.outcomes.year.exact",))

    selected = set(_selected_identifier_order(plan))
    rows_by_id: dict[str, tuple[object, object]] = {}
    duplicate_selected_id = False
    for identifier, rating, rebuilt in outcomes_2022.loc[
        :, ["id", "worst_rating", "rebuilt"]
    ].itertuples(index=False, name=None):
        if not isinstance(identifier, str) or identifier not in selected:
            continue
        if identifier in rows_by_id:
            duplicate_selected_id = True
            continue
        rows_by_id[identifier] = (rating, rebuilt)

    if duplicate_selected_id:
        raise OutcomeEvaluationError(
            ("backtest.outcomes.selected_id.unique",)
        )
    return rows_by_id


def evaluate_outcomes(
    plan: BacktestPlan,
    outcomes_2022: pd.DataFrame,
    *,
    target_fn: TargetFunction = is_safety_drop,
) -> BacktestRun:
    """Evaluate frozen selections against separately supplied 2022 outcomes.

    The plan is the sole 2018 endpoint source.  Missing endpoint rows, null
    ratings, and rebuilt 2022 endpoints are excluded.  The shared target is
    called exactly once for each distinct selected identifier that remains
    eligible; measured classifications are then projected onto every frozen
    budget/method selection without cross-method adjustments.
    """
    if not isinstance(plan, BacktestPlan):
        raise OutcomeEvaluationError(("backtest.plan.type",))
    if not callable(target_fn):
        raise OutcomeEvaluationError(("backtest.target.callable",))

    outcome_rows = _selected_outcome_rows(plan, outcomes_2022)
    starting_ratings = plan.starting_rating_by_id

    eligible_pairs: dict[str, tuple[float, float]] = {}
    issues: list[str] = []
    for identifier in _selected_identifier_order(plan):
        starting_rating = starting_ratings.get(identifier)
        if starting_rating is None:
            continue

        endpoint = outcome_rows.get(identifier)
        if endpoint is None:
            continue
        ending_rating, rebuilt = endpoint

        if not pd.api.types.is_bool(rebuilt):
            issues.append("backtest.outcomes.rebuilt.boolean")
            continue
        if bool(rebuilt):
            continue
        if _is_missing(ending_rating):
            continue

        numeric_ending = _finite_number(ending_rating)
        if (
            numeric_ending is None
            or not numeric_ending.is_integer()
            or not 0.0 <= numeric_ending <= 9.0
        ):
            issues.append("backtest.outcomes.worst_rating.valid")
            continue
        eligible_pairs[identifier] = (
            float(starting_rating),
            numeric_ending,
        )

    if issues:
        raise OutcomeEvaluationError(issues)

    drops: dict[str, bool] = {}
    for identifier, (starting_rating, ending_rating) in eligible_pairs.items():
        try:
            classified = target_fn(starting_rating, ending_rating)
        except Exception as exc:
            raise OutcomeEvaluationError(
                ("backtest.target.execution",)
            ) from exc
        if not pd.api.types.is_bool(classified):
            raise OutcomeEvaluationError(("backtest.target.boolean_result",))
        drops[identifier] = bool(classified)

    rows: list[dict[str, object]] = []
    eligible_counts: dict[BacktestKey, int] = {}
    for selection in plan.selections:
        key: BacktestKey = (selection.budget, selection.method)
        eligible_count = sum(
            identifier in eligible_pairs for identifier in selection.ids
        )
        caught = sum(
            drops.get(identifier, False) for identifier in selection.ids
        )
        eligible_counts[key] = eligible_count
        rows.append(
            {
                "budget": selection.budget,
                "method": selection.method,
                "caught": int(caught),
            }
        )

    result = pd.DataFrame(rows, columns=BACKTEST_RESULT_COLUMNS)
    return BacktestRun(
        result=result,
        eligible_selected_count=eligible_counts,
        selection_audit=plan,
    )


def _validated_orchestration_years(
    adapted_panel: pd.DataFrame,
) -> pd.Series:
    """Validate only the year axis needed to establish the phase boundary."""
    if not isinstance(adapted_panel, pd.DataFrame):
        raise OutcomeEvaluationError(("backtest.panel.frame_type",))
    if "year" not in adapted_panel.columns:
        raise OutcomeEvaluationError(("backtest.panel.year.required",))

    normalized = tuple(
        _integer_year(value) for value in adapted_panel["year"].tolist()
    )
    if any(year is None for year in normalized):
        raise OutcomeEvaluationError(("backtest.panel.year.integral",))

    allowed_years = _REQUIRED_HISTORICAL_YEARS | {BACKTEST_OUTCOME_YEAR}
    if any(year not in allowed_years for year in normalized):
        raise OutcomeEvaluationError(("backtest.panel.year.allowed",))
    return pd.Series(normalized, index=adapted_panel.index, dtype="int64")


def run_backtest(
    adapted_panel: pd.DataFrame,
    budgets: Iterable[Real | Decimal],
    *,
    fit_fn: FitFunction = fit_decline,
    score_fn: ScoreFunction = score,
    target_fn: TargetFunction = is_safety_drop,
) -> BacktestResult:
    """Run historical planning first, then evaluate separately read outcomes.

    Before planning completes, orchestration reads only the full panel's year
    axis and copies records dated 2018 or earlier.  The 2022 frame is obtained
    only after :func:`build_historical_plan` has frozen every selection.  The
    returned frame is a defensive copy of the validated public result.
    """
    years = _validated_orchestration_years(adapted_panel)
    historical = adapted_panel.loc[
        years.le(BACKTEST_CANDIDATE_YEAR)
    ].copy(deep=True)

    plan = build_historical_plan(
        historical,
        budgets,
        fit_fn=fit_fn,
        score_fn=score_fn,
    )

    outcomes_2022 = adapted_panel.loc[
        years.eq(BACKTEST_OUTCOME_YEAR)
    ].copy(deep=True)
    run = evaluate_outcomes(
        plan,
        outcomes_2022,
        target_fn=target_fn,
    )
    return run.result.copy(deep=True)


__all__ = [
    "BACKTEST_AS_OF",
    "BACKTEST_CANDIDATE_YEAR",
    "BACKTEST_FIT_END_YEAR",
    "BACKTEST_FIT_START_YEAR",
    "BACKTEST_METHODS",
    "BACKTEST_OUTCOME_YEAR",
    "BACKTEST_RESULT_COLUMNS",
    "BacktestMethod",
    "BacktestPlan",
    "BacktestResult",
    "BacktestResultValidationError",
    "BacktestRun",
    "BacktestSelection",
    "HistoricalPlanningError",
    "OutcomeEvaluationError",
    "build_historical_plan",
    "evaluate_outcomes",
    "run_backtest",
    "validate_backtest_result",
]
