"""Authoritative public API for bridge data, modeling, and targets.

The strict source/scored-data boundary is implemented only in
:mod:`model.data` and re-exported here for application and planner consumers.
``load_panel`` remains the compatibility entry point over that same reader,
validator, and adapter; it does not define a second schema mapping.
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
    read_source_panel,
    reconcile_scored_table,
    validate_scored_table,
    validate_source_panel,
)
from .materials import MATERIAL_NAMES, material_name
from .score import (
    DeclineRates,
    LogisticModel,
    fit_decline,
    fit_logistic,
    months_since_insp,
    score,
)
from .target import is_safety_drop, is_safety_drop_series

__all__ = [
    # Strict source and score boundary.
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
    "validate_scored_table",
    "reconcile_scored_table",
    "load_panel",
    # Existing model, material, and shared Safety_Drop API.
    "fit_decline",
    "fit_logistic",
    "score",
    "DeclineRates",
    "LogisticModel",
    "is_safety_drop",
    "is_safety_drop_series",
    "months_since_insp",
    "material_name",
    "MATERIAL_NAMES",
]
