"""Immutable application state shared by every presentation renderer.

This module is deliberately a data-only boundary.  It imports the public
planner value types, but never calls the model, planner, cache, provider, or
demo implementations.  The composition root validates the active budget,
creates one statewide selection, and then gives every renderer the same
:class:`ViewContext`.

County filtering is intentionally absent from selection state.  It is a map
projection argument and therefore cannot change the statewide selection or its
identity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Integral
from typing import Final, Protocol

import pandas as pd

from app.config import Provenance, RuntimeMode
from planner import Selection, SelectionMetrics

MIN_ACTIVE_BUDGET: Final[int] = 100_000
MAX_ACTIVE_BUDGET: Final[int] = 3_000_000
ACTIVE_BUDGET_STEP: Final[int] = 50_000
INITIAL_ACTIVE_BUDGET: Final[int] = 500_000
ACTIVE_BUDGET_KEY: Final[str] = "active_budget"
LAST_VALID_BUDGET_KEY: Final[str] = "last_valid_budget"
ACTIVE_BUDGET_VALUES: Final[range] = range(
    MIN_ACTIVE_BUDGET,
    MAX_ACTIVE_BUDGET + 1,
    ACTIVE_BUDGET_STEP,
)

_SELECTION_IDENTITY_VERSION: Final[bytes] = b"selection-context-v1"
_MISSING: Final[object] = object()


class SessionStateStore(Protocol):
    """Minimal mutable state interface needed by :func:`get_active_budget`."""

    def get(self, key: str, default: object = None, /) -> object: ...

    def __setitem__(self, key: str, value: object, /) -> None: ...


class ViewRenderer(Protocol):
    """Interface for a renderer that consumes one shared view snapshot."""

    def __call__(self, context: "ViewContext", /) -> object: ...


class MapViewRenderer(Protocol):
    """Map renderer interface keeping county projection outside selection."""

    def __call__(
        self,
        context: "ViewContext",
        /,
        *,
        county: str | None = None,
    ) -> object: ...


def _normalize_active_budget(value: object) -> int | None:
    """Return an exact allowed integer budget, rejecting coercions and bools."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    budget = int(value)
    if budget not in ACTIVE_BUDGET_VALUES:
        return None
    return budget


def is_valid_active_budget(value: object) -> bool:
    """Return whether ``value`` is an exact member of the allowed budget grid."""

    return _normalize_active_budget(value) is not None


def get_active_budget(
    session_state: SessionStateStore,
    requested_value: object = _MISSING,
) -> int:
    """Validate and persist the one active statewide budget.

    Both state keys start at ``500000``.  An exact integer member of
    ``range(100000, 3000001, 50000)`` becomes both the active and last-valid
    budget.  An invalid request or injected active value restores the most
    recent valid budget; if that remembered value is itself invalid or absent,
    the initial ``500000`` value is restored.

    Omitting ``requested_value`` validates the current ``active_budget`` state.
    This supports reruns where the widget has already populated session state.
    """

    if not callable(getattr(session_state, "get", None)) or not hasattr(
        session_state, "__setitem__"
    ):
        raise TypeError("session_state must provide get() and item assignment.")

    remembered = _normalize_active_budget(
        session_state.get(LAST_VALID_BUDGET_KEY, INITIAL_ACTIVE_BUDGET)
    )
    if remembered is None:
        remembered = INITIAL_ACTIVE_BUDGET

    if requested_value is _MISSING:
        candidate = _normalize_active_budget(
            session_state.get(ACTIVE_BUDGET_KEY, INITIAL_ACTIVE_BUDGET)
        )
    else:
        candidate = _normalize_active_budget(requested_value)

    active_budget = remembered if candidate is None else candidate
    session_state[ACTIVE_BUDGET_KEY] = active_budget
    session_state[LAST_VALID_BUDGET_KEY] = active_budget
    return active_budget


def _identity_part(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(8, byteorder="big", signed=False) + encoded


def _constraint_token(selection: Selection) -> str:
    if selection.mode == "budget":
        constraint = selection.constraint
        if not isinstance(constraint, Decimal):  # defensive; Selection validates it
            raise TypeError("A budget selection must have a Decimal constraint.")
        if constraint == 0:
            return "0"
        return format(constraint.normalize(), "f")

    if selection.mode == "count":
        constraint = selection.constraint
        if isinstance(constraint, bool) or not isinstance(constraint, int):
            raise TypeError("A count selection must have an integer constraint.")
        return str(constraint)

    raise ValueError("Selection mode is invalid.")


def build_selection_identity(
    score_identity: str,
    selection: Selection,
) -> str:
    """Build a deterministic identity from score and selection semantics.

    The digest includes a version marker, the complete score identity, planner
    mode, canonical constraint, and ordered selected IDs.  Presentation-only
    inputs such as county filters are intentionally excluded.
    """

    if not isinstance(score_identity, str) or not score_identity:
        raise ValueError("score_identity must be a nonempty string.")
    if not isinstance(selection, Selection):
        raise TypeError("selection must be a Selection.")

    digest = hashlib.sha256()
    digest.update(_identity_part(_SELECTION_IDENTITY_VERSION.decode("ascii")))
    digest.update(_identity_part(score_identity))
    digest.update(_identity_part(selection.mode))
    digest.update(_identity_part(_constraint_token(selection)))
    digest.update(len(selection.ids).to_bytes(8, byteorder="big", signed=False))
    for identifier in selection.ids:
        digest.update(_identity_part(identifier))
    return digest.hexdigest()


# Domain-friendly concise alias for callers that do not use the builder prefix.
selection_identity = build_selection_identity


def _snapshot_selected_records(
    selected_records: pd.DataFrame,
    selected_ids: tuple[str, ...],
) -> pd.DataFrame:
    """Validate exact selected membership and retain a defensive snapshot."""

    if not isinstance(selected_records, pd.DataFrame):
        raise TypeError("selected_records must be a pandas DataFrame.")
    if "structure_id" not in selected_records.columns:
        raise ValueError("selected_records must contain structure_id.")

    snapshot = selected_records.copy(deep=True)
    record_ids = tuple(snapshot["structure_id"].tolist())
    if not all(
        isinstance(identifier, str) and bool(identifier.strip())
        for identifier in record_ids
    ):
        raise ValueError("selected_records identifiers must be nonempty strings.")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("selected_records identifiers must be unique.")
    if len(record_ids) != len(selected_ids) or set(record_ids) != set(selected_ids):
        raise ValueError("selected_records must match the statewide selection.")
    return snapshot


@dataclass(frozen=True, slots=True, init=False)
class SelectionContext:
    """One immutable statewide budget selection and its exact projections.

    ``selected_records`` is copied on input and on access so a renderer cannot
    mutate the shared snapshot.  ``Selection`` and ``SelectionMetrics`` are
    themselves frozen planner values.  The deterministic ``identity`` excludes
    map-only county state.
    """

    active_budget: int
    selection: Selection
    metrics: SelectionMetrics
    score_identity: str = field(repr=False)
    identity: str
    _selected_records: pd.DataFrame = field(
        repr=False,
        compare=False,
        hash=False,
    )

    def __init__(
        self,
        active_budget: object,
        selection: Selection,
        metrics: SelectionMetrics,
        selected_records: pd.DataFrame,
        score_identity: str,
    ) -> None:
        normalized_budget = _normalize_active_budget(active_budget)
        if normalized_budget is None:
            raise ValueError("active_budget is not an allowed budget value.")
        if not isinstance(selection, Selection):
            raise TypeError("selection must be a Selection.")
        if selection.mode != "budget":
            raise ValueError("Application selection state must use budget mode.")
        if selection.constraint != Decimal(normalized_budget):
            raise ValueError("Selection constraint must equal active_budget.")
        if not isinstance(metrics, SelectionMetrics):
            raise TypeError("metrics must be SelectionMetrics.")
        if metrics.selected_count != len(selection.ids):
            raise ValueError("Selection metrics must describe the same selection.")
        if metrics.selected_total_cost > Decimal(normalized_budget):
            raise ValueError("Selected total cost must not exceed active_budget.")

        snapshot = _snapshot_selected_records(selected_records, selection.ids)
        identity = build_selection_identity(score_identity, selection)

        object.__setattr__(self, "active_budget", normalized_budget)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "score_identity", score_identity)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "_selected_records", snapshot)

    @property
    def selected_records(self) -> pd.DataFrame:
        """Return an isolated copy of the selected presentation records."""

        return self._selected_records.copy(deep=True)

    @property
    def selected_ids(self) -> tuple[str, ...]:
        """Return the planner's immutable ordered selected identifiers."""

        return self.selection.ids


def build_selection_context(
    *,
    active_budget: object,
    selection: Selection,
    metrics: SelectionMetrics,
    selected_records: pd.DataFrame,
    score_identity: str,
) -> SelectionContext:
    """Construct the sole validated selection context for one rerun."""

    return SelectionContext(
        active_budget=active_budget,
        selection=selection,
        metrics=metrics,
        selected_records=selected_records,
        score_identity=score_identity,
    )


@dataclass(frozen=True, slots=True)
class ViewContext:
    """Complete data-only snapshot supplied to presentation renderers.

    ``source`` and ``current_ranking`` are validated bundle values assembled by
    the composition root; ``backtest`` may be its independent success/failure
    outcome.  No operation callbacks or provider interfaces are stored here.
    Selection-derived properties delegate to the one shared
    :class:`SelectionContext`, preventing any renderer from requesting or
    representing a second selection.
    """

    mode: RuntimeMode
    provenance: Provenance
    source: object
    current_ranking: object
    selection_context: SelectionContext
    backtest: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RuntimeMode):
            raise TypeError("mode must be a RuntimeMode.")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be Provenance.")
        if self.provenance.mode is not self.mode:
            raise ValueError("Provenance mode must match the view mode.")
        if self.source is None:
            raise ValueError("source must be a validated data bundle.")
        if self.current_ranking is None:
            raise ValueError("current_ranking must be a validated ranking.")
        if not isinstance(self.selection_context, SelectionContext):
            raise TypeError("selection_context must be SelectionContext.")
        for name, value in (
            ("source", self.source),
            ("current_ranking", self.current_ranking),
            ("backtest", self.backtest),
        ):
            if value is not None and callable(value):
                raise TypeError(f"{name} must be data, not an operation callback.")

    @property
    def active_budget(self) -> int:
        return self.selection_context.active_budget

    @property
    def selection(self) -> Selection:
        return self.selection_context.selection

    @property
    def metrics(self) -> SelectionMetrics:
        return self.selection_context.metrics

    @property
    def selected_records(self) -> pd.DataFrame:
        return self.selection_context.selected_records

    @property
    def selection_identity(self) -> str:
        return self.selection_context.identity


__all__ = [
    "ACTIVE_BUDGET_KEY",
    "ACTIVE_BUDGET_STEP",
    "ACTIVE_BUDGET_VALUES",
    "INITIAL_ACTIVE_BUDGET",
    "LAST_VALID_BUDGET_KEY",
    "MAX_ACTIVE_BUDGET",
    "MIN_ACTIVE_BUDGET",
    "MapViewRenderer",
    "SelectionContext",
    "SessionStateStore",
    "ViewContext",
    "ViewRenderer",
    "build_selection_context",
    "build_selection_identity",
    "get_active_budget",
    "is_valid_active_budget",
    "selection_identity",
]
