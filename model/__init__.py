"""Part B — decline model and scoring.

Public API (see .kiro/steering/contract.md):
    load_panel(path=None)              -> DataFrame (adapts Part A's panel)
    is_safety_drop(start, end)         -> bool  (shared target, also for Part C)
    fit_decline(panel, start, end)     -> DeclineRates (grouped model)
    fit_logistic(panel, start, end)    -> LogisticModel
    score(table, model, asof)          -> DataFrame[id, p_worse, consequence,
                                                     insp_cost, score, reason]
"""

from .data import FOUR_YEAR_PAIRS, load_panel
from .materials import MATERIAL_NAMES, material_name
from .score import DeclineRates, LogisticModel, fit_decline, fit_logistic, score
from .target import is_safety_drop, is_safety_drop_series

__all__ = [
    "fit_decline",
    "fit_logistic",
    "score",
    "DeclineRates",
    "LogisticModel",
    "load_panel",
    "FOUR_YEAR_PAIRS",
    "is_safety_drop",
    "is_safety_drop_series",
    "material_name",
    "MATERIAL_NAMES",
]
