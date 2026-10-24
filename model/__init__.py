"""Authoritative public API for bridge data, modeling, scoring, and targets.

The strict source/scored-data boundary is implemented only in
:mod:`model.data` and re-exported here for application and planner consumers.
``load_panel`` remains the compatibility entry point over that same reader,
validator, and adapter; it does not define a second schema mapping.
Part B's decline model, flight-cost, and value-of-information layers are
exported alongside it.
"""

from .data import (
    DEFAULT_PANEL_PATH,
    FOUR_YEAR_PAIRS,
    MODELED_COST_DISCLOSURE,
    REQUIRED_YEARS,
    SCORED_COLUMNS,
    SOURCE_COLUMNS,
    IdentifierReconciliationError,
    ScoredDataError,
    ScoredTableValidationError,
    SourceAdaptationError,
    SourceDataError,
    SourceReadError,
    SourceSchemaError,
    adapt_source_panel,
    load_panel,
    project_scored_table,
    read_source_panel,
    reconcile_scored_table,
    validate_scored_table,
    validate_source_panel,
)
from .flight_cost import flight_cost
from .materials import MATERIAL_NAMES, material_name
from .score import (
    ConditionBaseline,
    DeclineRates,
    LogisticModel,
    fit_condition_baseline,
    fit_decline,
    fit_logistic,
    months_since_insp,
    score,
)
from .target import is_safety_drop, is_safety_drop_series
from .voi import (
    DECISION_LABELS,
    FIX_TRACK_THRESHOLD,
    FLY_TOP_FRACTION,
    add_voi,
    compute_voi,
    simulate_flight,
)

__all__ = [
    # Strict source and score boundary (Part A).
    "DEFAULT_PANEL_PATH",
    "SOURCE_COLUMNS",
    "REQUIRED_YEARS",
    "SCORED_COLUMNS",
    "MODELED_COST_DISCLOSURE",
    "FOUR_YEAR_PAIRS",
    "SourceDataError",
    "SourceReadError",
    "SourceSchemaError",
    "SourceAdaptationError",
    "ScoredDataError",
    "ScoredTableValidationError",
    "IdentifierReconciliationError",
    "read_source_panel",
    "validate_source_panel",
    "adapt_source_panel",
    "project_scored_table",
    "validate_scored_table",
    "reconcile_scored_table",
    "load_panel",
    # Decline model, flight cost, VOI (Part B).
    "fit_decline",
    "fit_logistic",
    "score",
    "DeclineRates",
    "LogisticModel",
    "ConditionBaseline",
    "fit_condition_baseline",
    "months_since_insp",
    "flight_cost",
    "material_name",
    "MATERIAL_NAMES",
    "add_voi",
    "compute_voi",
    "simulate_flight",
    "DECISION_LABELS",
    "FIX_TRACK_THRESHOLD",
    "FLY_TOP_FRACTION",
    # Shared Safety_Drop target.
    "is_safety_drop",
    "is_safety_drop_series",
]
