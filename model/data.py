"""Adapter from Part A's published panel to the names Part B's model uses.

Part A owns ``bridges/panel.parquet`` and its schema is fixed (see
``bridges/README.md``). We never edit anything under ``bridges/`` — instead
``load_panel`` reads that file and returns a frame whose column names and
derived fields match what :mod:`model.score` expects.

Mapping applied here:
- ``structure_id``      -> ``id``
- ``insp_freq_months``  -> ``insp_freq_mo``
- ``year_built``        derived as ``year - age``
- ``worst_rating``      kept as-is; score.py uses it as the condition when
                        present, and a null ``worst_rating`` is not-applicable
                        and skipped.
- ``year_recon`` == 0   -> ``NaN`` (0 means "never rebuilt")
- ``rebuilt``           passed through; fit_decline excludes a start->end pair
                        when the END year's rebuilt flag is True.
- ``improvement_cost``  deliberately NOT mapped to ``insp_cost`` — it is a
                        repair cost, not an inspection cost. score.py uses the
                        estimated drone-flight cost when no real inspection
                        cost exists.
- ``insp_date``         not in the panel yet. If Part A adds it later, it is
                        picked up automatically; otherwise score.py falls back
                        to ``insp_freq_mo``.

The real panel's 4-year gaps are 2012->2016 and 2018->2022. Never fit on the
2016->2018 gap (2-year horizon) — see ``FOUR_YEAR_PAIRS``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final, Iterable

import numpy as np
import pandas as pd

DEFAULT_PANEL_PATH = Path(__file__).resolve().parents[1] / "bridges" / "panel.parquet"

# The only interval pairs whose horizon matches (4 years). The 2016->2018 gap
# is 2 years and must never be used for fitting or backtest.
FOUR_YEAR_PAIRS = [(2012, 2016), (2018, 2022)]


def load_panel(path=None) -> pd.DataFrame:
    """Load Part A's panel and adapt column names/derived fields for the model.

    Does not mutate any file under ``bridges/``. Returns a new DataFrame.
    """
    path = Path(path) if path is not None else DEFAULT_PANEL_PATH
    raw = pd.read_parquet(path)
    df = raw.copy()

    # --- renames ------------------------------------------------------------
    renames = {}
    if "structure_id" in df.columns:
        renames["structure_id"] = "id"
    if "insp_freq_months" in df.columns:
        renames["insp_freq_months"] = "insp_freq_mo"
    df = df.rename(columns=renames)

    # --- derive year_built = year - age ------------------------------------
    if "year_built" not in df.columns and {"year", "age"}.issubset(df.columns):
        df["year_built"] = df["year"] - df["age"]

    # --- year_recon 0 means "never rebuilt" -> NaN -------------------------
    if "year_recon" in df.columns:
        df["year_recon"] = df["year_recon"].where(df["year_recon"] != 0, other=pd.NA)
        df["year_recon"] = pd.to_numeric(df["year_recon"], errors="coerce")

    # --- insp_date: pick it up automatically if Part A adds it later -------
    if "insp_date" in df.columns:
        df["insp_date"] = pd.to_datetime(df["insp_date"], errors="coerce")
    # If absent, we intentionally do NOT create it; score.py falls back to
    # insp_freq_mo.

    # improvement_cost is left untouched and NOT renamed to insp_cost.

    return df


if __name__ == "__main__":
    p = load_panel()
    print(f"loaded {len(p)} rows, {p['id'].nunique()} bridges, years {sorted(p['year'].unique())}")
    print("has insp_date:", "insp_date" in p.columns)
    print("has insp_cost:", "insp_cost" in p.columns)
    print(p[["id", "year", "year_built", "worst_rating", "insp_freq_mo", "rebuilt"]].head())


# ---------------------------------------------------------------------------
# Scored-output / reconciliation boundary (restored from Part A's app so the
# names ``model.data`` is expected to export remain available). Definitions
# below are copied verbatim from Part A and are independent of the adapter
# above.
# ---------------------------------------------------------------------------

SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "structure_id",
    "year",
    "county",
    "material",
    "age",
    "adt",
    "deck",
    "super",
    "sub",
    "worst_rating",
    "insp_freq_months",
    "fracture_critical",
    "underwater",
    "year_recon",
    "rebuilt",
    "lat",
    "lon",
    "carries",
    "crosses",
    "improvement_cost",
    "insp_date",
)


class SourceDataError(Exception):
    """Base class for failures at the source-data boundary."""


class SourceSchemaError(SourceDataError):
    """Raised when a source frame violates the published schema.

    ``issues`` contains stable predicate identifiers only. It deliberately
    excludes row values so callers can classify failures without copying raw
    source records into diagnostics.
    """

    def __init__(self, issues: Iterable[str]):
        unique_issues = tuple(dict.fromkeys(issues))
        if not unique_issues:
            unique_issues = ("schema.unknown",)
        self.issues = unique_issues
        super().__init__(
            "Source panel failed validation: " + ", ".join(unique_issues)
        )


def _is_missing(value: object) -> bool:
    """Return whether ``value`` is one scalar pandas missing value."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _as_finite_number(value: object) -> float | None:
    """Return a finite numeric value, rejecting booleans and text."""
    if _is_missing(value) or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (str, bytes, bytearray, complex)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _as_integer(value: object) -> int | None:
    """Return a mathematically integral finite value, rejecting booleans."""
    number = _as_finite_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


SCORED_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "p_worse",
    "consequence",
    "insp_cost",
    "score",
    "reason",
)
MODELED_COST_DISCLOSURE: Final[str] = "modeled inspection-cost estimate"

# Score is calculated directly from the other three numeric fields. These
# tolerances permit only ordinary floating-point roundoff, not materially
# different model output.
SCORE_RELATIVE_TOLERANCE: Final[float] = 1e-12
SCORE_ABSOLUTE_TOLERANCE: Final[float] = 1e-15


class ScoredDataError(SourceDataError):
    """Base class for invalid model output or score reconciliation."""


class ScoredTableValidationError(ScoredDataError):
    """Raised when model output violates the complete scored-table contract.

    ``issues`` contains stable predicate identifiers and never row values.
    """

    def __init__(self, issues: Iterable[str]):
        unique_issues = tuple(dict.fromkeys(issues))
        if not unique_issues:
            unique_issues = ("scored.unknown",)
        self.issues = unique_issues
        super().__init__(
            "Scored table failed validation: " + ", ".join(unique_issues)
        )


class IdentifierReconciliationError(ScoredDataError):
    """Raised when ranking and model identifiers are not a bijection.

    The exception exposes stable issue identifiers only, so missing or
    unexpected bridge identifiers are not copied into public diagnostics.
    """

    def __init__(self, issues: Iterable[str]):
        unique_issues = tuple(dict.fromkeys(issues))
        if not unique_issues:
            unique_issues = ("identifier.unknown",)
        self.issues = unique_issues
        super().__init__(
            "Ranking identifiers failed reconciliation: "
            + ", ".join(unique_issues)
        )


def _materialize_ranking_ids(
    ranking_ids: Iterable[str],
) -> tuple[tuple[object, ...], list[str]]:
    """Materialize ranking IDs once and return stable contract issues."""
    if isinstance(ranking_ids, (str, bytes, bytearray)):
        return (), ["ranking_ids.iterable"]
    try:
        values = tuple(ranking_ids)
    except TypeError:
        return (), ["ranking_ids.iterable"]

    issues: list[str] = []
    if not all(_is_nonempty_string(value) for value in values):
        issues.append("ranking_ids.nonempty_string")
    elif len(set(values)) != len(values):
        issues.append("ranking_ids.unique")
    return values, issues


def _scored_identifier_issues(
    scored_ids: tuple[object, ...], ranking_ids: tuple[object, ...]
) -> list[str]:
    """Return identifier-bijection issues without exposing identifier values."""
    issues: list[str] = []
    scored_ids_are_valid = all(
        _is_nonempty_string(value) for value in scored_ids
    )
    ranking_ids_are_valid = all(
        _is_nonempty_string(value) for value in ranking_ids
    )

    if not scored_ids_are_valid:
        issues.append("id.nonempty_string")
    elif len(set(scored_ids)) != len(scored_ids):
        issues.append("id.unique")

    if len(scored_ids) != len(ranking_ids):
        issues.append("id.cardinality")

    if scored_ids_are_valid and ranking_ids_are_valid:
        scored_set = set(scored_ids)
        ranking_set = set(ranking_ids)
        if ranking_set - scored_set:
            issues.append("id.missing_ranking_identifier")
        if scored_set - ranking_set:
            issues.append("id.unexpected_identifier")

    return issues


def validate_scored_table(
    scored: pd.DataFrame, ranking_ids: Iterable[str]
) -> None:
    """Validate complete model output for exactly one ranking population.

    Validation is non-mutating and all predicates are checked before success:
    exact ordered columns, an identifier bijection, complete non-null values,
    finite and bounded numeric fields, score arithmetic within
    ``SCORE_RELATIVE_TOLERANCE``/``SCORE_ABSOLUTE_TOLERANCE``, and nonblank
    reasons containing the exact modeled-cost disclosure.

    Production source records do not contain a per-bridge inspection-cost
    field, so every score produced from the adapted source must disclose that
    ``insp_cost`` is a ``modeled inspection-cost estimate``.
    """
    if not isinstance(scored, pd.DataFrame):
        raise ScoredTableValidationError(("scored.frame_type",))

    ranking_id_values, issues = _materialize_ranking_ids(ranking_ids)
    if tuple(scored.columns) != SCORED_COLUMNS:
        issues.append("scored.columns.exact_order")
        raise ScoredTableValidationError(issues)

    scored_ids = tuple(scored["id"].tolist())
    issues.extend(_scored_identifier_issues(scored_ids, ranking_id_values))

    if any(
        _is_missing(value)
        for column in SCORED_COLUMNS
        for value in scored[column].tolist()
    ):
        issues.append("scored.complete_non_null")

    numeric_columns = ("p_worse", "consequence", "insp_cost", "score")
    numeric_values: dict[str, list[float | None]] = {}
    for column in numeric_columns:
        values = [_as_finite_number(value) for value in scored[column].tolist()]
        numeric_values[column] = values
        if any(value is None for value in values):
            issues.append(f"{column}.finite_numeric")

    probabilities = numeric_values["p_worse"]
    if any(
        value is not None and not 0.0 <= value <= 1.0
        for value in probabilities
    ):
        issues.append("p_worse.bounds")

    consequences = numeric_values["consequence"]
    if any(value is not None and value <= 0.0 for value in consequences):
        issues.append("consequence.positive")

    costs = numeric_values["insp_cost"]
    if any(value is not None and value <= 0.0 for value in costs):
        issues.append("insp_cost.positive")

    scores = numeric_values["score"]
    if any(value is not None and value < 0.0 for value in scores):
        issues.append("score.nonnegative")

    arithmetic_is_valid = True
    for probability, consequence, cost, observed_score in zip(
        probabilities, consequences, costs, scores
    ):
        if (
            probability is None
            or consequence is None
            or cost is None
            or cost <= 0.0
            or observed_score is None
        ):
            continue
        expected_score = probability * consequence / cost
        if not math.isclose(
            observed_score,
            expected_score,
            rel_tol=SCORE_RELATIVE_TOLERANCE,
            abs_tol=SCORE_ABSOLUTE_TOLERANCE,
        ):
            arithmetic_is_valid = False
            break
    if not arithmetic_is_valid:
        issues.append("score.arithmetic")

    reasons = scored["reason"].tolist()
    if not all(_is_nonempty_string(reason) for reason in reasons):
        issues.append("reason.nonblank_string")
    if any(
        not isinstance(reason, str)
        or MODELED_COST_DISCLOSURE not in reason
        for reason in reasons
    ):
        issues.append("reason.modeled_cost_disclosure")

    if issues:
        raise ScoredTableValidationError(issues)


def _validate_ranking_source_for_reconciliation(
    source_2022: pd.DataFrame,
) -> tuple[str, ...]:
    """Validate the source-facing identity boundary for a 2022 slice."""
    if not isinstance(source_2022, pd.DataFrame):
        raise SourceSchemaError(("ranking_source.frame_type",))
    if tuple(source_2022.columns) != SOURCE_COLUMNS:
        raise SourceSchemaError(("ranking_source.columns.exact_order",))
    if source_2022.empty:
        raise SourceSchemaError(("ranking_source.nonempty",))

    years = tuple(_as_integer(value) for value in source_2022["year"].tolist())
    if any(year != 2022 for year in years):
        raise SourceSchemaError(("ranking_source.year_2022",))

    source_ids = tuple(source_2022["structure_id"].tolist())
    issues: list[str] = []
    if not all(_is_nonempty_string(value) for value in source_ids):
        issues.append("source_structure_id.nonempty_string")
    elif len(set(source_ids)) != len(source_ids):
        issues.append("source_structure_id.unique")
    if issues:
        raise IdentifierReconciliationError(issues)
    return source_ids  # type: ignore[return-value]


def reconcile_scored_table(
    source_2022: pd.DataFrame, scored: pd.DataFrame
) -> pd.DataFrame:
    """Return a source-facing one-to-one merge of 2022 records and scores.

    The source slice must contain the exact published source columns and only
    2022 rows. Missing, duplicate, invalid, or unexpected identifiers fail
    before a result can escape. Remaining score predicates are then validated,
    and pandas' ``one_to_one`` merge validation provides a final cardinality
    guard. Neither input frame is mutated.
    """
    source_ids = _validate_ranking_source_for_reconciliation(source_2022)

    # Identifier failures at this boundary are reconciliation failures rather
    # than generic score-contract failures. Schema failures still retain their
    # scored-table category.
    if not isinstance(scored, pd.DataFrame):
        validate_scored_table(scored, source_ids)
    if tuple(scored.columns) != SCORED_COLUMNS:
        validate_scored_table(scored, source_ids)

    scored_ids = tuple(scored["id"].tolist())
    identifier_issues = _scored_identifier_issues(scored_ids, source_ids)
    if identifier_issues:
        raise IdentifierReconciliationError(identifier_issues)

    validate_scored_table(scored, source_ids)

    score_values = scored.copy(deep=True).rename(
        columns={"id": "structure_id"}
    )
    try:
        reconciled = source_2022.copy(deep=True).merge(
            score_values,
            on="structure_id",
            how="left",
            sort=False,
            validate="one_to_one",
            indicator="_score_merge",
        )
    except (pd.errors.MergeError, KeyError, ValueError) as exc:
        raise IdentifierReconciliationError(
            ("identifier.merge_one_to_one",)
        ) from exc

    expected_columns = SOURCE_COLUMNS + SCORED_COLUMNS[1:]
    if (
        len(reconciled) != len(source_2022)
        or not reconciled["_score_merge"].eq("both").all()
        or tuple(
            column for column in reconciled.columns if column != "_score_merge"
        )
        != expected_columns
    ):
        raise IdentifierReconciliationError(
            ("identifier.merge_complete",)
        )

    return reconciled.drop(columns="_score_merge")
