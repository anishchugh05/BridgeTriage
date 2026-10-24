"""Part B — decline model and scoring.

Public API (see .kiro/steering/contract.md):
    load_panel(path=None)              -> DataFrame (adapts Part A's panel)
    is_safety_drop(start, end)         -> bool  (shared target, also for Part C)
    fit_decline(panel, start, end)     -> DeclineRates (grouped model)
    fit_logistic(panel, start, end)    -> LogisticModel
    flight_cost(row)                    -> (cost_usd, minutes, reason_fragment)
    score(table, model, asof)          -> DataFrame[id, p_worse, consequence,
                                                     insp_cost, flight_minutes,
                                                     expected_drops_per_10k,
                                                     score, reason]
"""

from .data import FOUR_YEAR_PAIRS, load_panel
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

__all__ = [
    "fit_decline",
    "fit_logistic",
    "score",
    "DeclineRates",
    "LogisticModel",
    "ConditionBaseline",
    "fit_condition_baseline",
    "load_panel",
    "FOUR_YEAR_PAIRS",
    "is_safety_drop",
    "is_safety_drop_series",
    "months_since_insp",
    "flight_cost",
    "material_name",
    "MATERIAL_NAMES",
]
