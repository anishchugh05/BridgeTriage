"""Focused examples for the scored-output and reconciliation boundary."""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from model.data import (
    MODELED_COST_DISCLOSURE,
    SCORED_COLUMNS,
    SOURCE_COLUMNS,
    IdentifierReconciliationError,
    ScoredTableValidationError,
    reconcile_scored_table,
    validate_scored_table,
)


def _source_2022() -> pd.DataFrame:
    rows = [
        {
            "structure_id": "A-1",
            "year": 2022,
            "county": "Middlesex",
            "material": "Concrete",
            "age": 42,
            "adt": 12000,
            "deck": 6.0,
            "super": 5.0,
            "sub": 7.0,
            "worst_rating": 5.0,
            "insp_freq_months": 24,
            "fracture_critical": False,
            "underwater": False,
            "year_recon": 0,
            "rebuilt": False,
            "lat": 42.3,
            "lon": -71.2,
            "carries": "MAIN ST",
            "crosses": "RIVER",
            "improvement_cost": 250000.0,
            "insp_date": pd.Timestamp("2022-03-01"),
        },
        {
            "structure_id": "B-2",
            "year": 2022,
            "county": "Essex",
            "material": "Steel",
            "age": 67,
            "adt": 5000,
            "deck": 4.0,
            "super": 6.0,
            "sub": 5.0,
            "worst_rating": 4.0,
            "insp_freq_months": 12,
            "fracture_critical": True,
            "underwater": False,
            "year_recon": 1985,
            "rebuilt": False,
            "lat": 42.7,
            "lon": -70.9,
            "carries": "STATE ROUTE",
            "crosses": "LOCAL ROAD",
            "improvement_cost": None,
            "insp_date": pd.Timestamp("2022-01-01"),
        },
    ]
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def _valid_scored() -> pd.DataFrame:
    rows = [
        {
            "id": "A-1",
            "p_worse": 0.25,
            "consequence": 10.0,
            "insp_cost": 5000.0,
            "score": 0.25 * 10.0 / 5000.0,
            "reason": f"risk evidence; {MODELED_COST_DISCLOSURE} $5,000",
        },
        {
            "id": "B-2",
            "p_worse": 0.8,
            "consequence": 7.5,
            "insp_cost": 6000.0,
            "score": 0.8 * 7.5 / 6000.0,
            "reason": f"risk evidence; {MODELED_COST_DISCLOSURE} $6,000",
        },
    ]
    return pd.DataFrame(rows, columns=SCORED_COLUMNS)


def test_valid_scored_table_reconciles_source_facing_rows_without_mutation():
    source = _source_2022()
    scored = _valid_scored().iloc[::-1].reset_index(drop=True)
    source_before = source.copy(deep=True)
    scored_before = scored.copy(deep=True)

    validate_scored_table(scored, source["structure_id"])
    reconciled = reconcile_scored_table(source, scored)

    assert tuple(reconciled.columns) == SOURCE_COLUMNS + SCORED_COLUMNS[1:]
    assert reconciled["structure_id"].tolist() == ["A-1", "B-2"]
    assert reconciled["score"].tolist() == [
        _valid_scored().loc[0, "score"],
        _valid_scored().loc[1, "score"],
    ]
    assert "id" not in reconciled.columns
    assert_frame_equal(source, source_before)
    assert_frame_equal(scored, scored_before)


@pytest.mark.parametrize(
    ("case", "expected_issue"),
    [
        ("columns", "scored.columns.exact_order"),
        ("null", "scored.complete_non_null"),
        ("probability", "p_worse.bounds"),
        ("consequence", "consequence.positive"),
        ("cost", "insp_cost.positive"),
        ("score", "score.nonnegative"),
        ("arithmetic", "score.arithmetic"),
        ("blank_reason", "reason.nonblank_string"),
        ("disclosure", "reason.modeled_cost_disclosure"),
        ("numeric_text", "p_worse.finite_numeric"),
    ],
)
def test_scored_contract_rejects_each_representative_invalid_output(
    case: str, expected_issue: str
):
    scored = _valid_scored()
    if case == "columns":
        scored = scored.loc[:, list(reversed(SCORED_COLUMNS))]
    elif case == "null":
        scored.loc[0, "reason"] = None
    elif case == "probability":
        scored.loc[0, "p_worse"] = 1.1
    elif case == "consequence":
        scored.loc[0, "consequence"] = 0.0
    elif case == "cost":
        scored.loc[0, "insp_cost"] = 0.0
    elif case == "score":
        scored.loc[0, "score"] = -1.0
    elif case == "arithmetic":
        scored.loc[0, "score"] += 0.01
    elif case == "blank_reason":
        scored.loc[0, "reason"] = "   "
    elif case == "disclosure":
        scored.loc[0, "reason"] = "risk evidence only"
    elif case == "numeric_text":
        scored["p_worse"] = scored["p_worse"].astype(object)
        scored.loc[0, "p_worse"] = "0.25"

    with pytest.raises(ScoredTableValidationError) as caught:
        validate_scored_table(scored, ("A-1", "B-2"))

    assert expected_issue in caught.value.issues


@pytest.mark.parametrize(
    "scored_ids",
    [
        ("A-1",),
        ("A-1", "A-1"),
        ("A-1", "C-3"),
    ],
)
def test_reconciliation_fails_atomically_for_missing_duplicate_or_unexpected_ids(
    scored_ids: tuple[str, ...]
):
    source = _source_2022()
    scored = _valid_scored().iloc[: len(scored_ids)].copy()
    scored["id"] = list(scored_ids)
    source_before = source.copy(deep=True)
    scored_before = scored.copy(deep=True)

    with pytest.raises(IdentifierReconciliationError):
        reconcile_scored_table(source, scored)

    assert_frame_equal(source, source_before)
    assert_frame_equal(scored, scored_before)


def test_reconciliation_rejects_duplicate_source_identifier():
    source = _source_2022()
    source.loc[1, "structure_id"] = "A-1"

    with pytest.raises(IdentifierReconciliationError) as caught:
        reconcile_scored_table(source, _valid_scored())

    assert "source_structure_id.unique" in caught.value.issues
