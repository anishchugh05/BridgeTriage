"""Strict source-data boundary for the bridge decline model.

The published bridge panel has one source schema. This module reads that
repository asset without fallback, validates it without coercing or mutating
it, and creates the model-facing copy through the only supported mapping:

* ``structure_id`` -> ``id``
* ``insp_freq_months`` -> ``insp_freq_mo``
* ``year_built`` is appended as ``year - age``

Every other source column and value is preserved. In particular,
``year_recon == 0`` continues to mean "never reconstructed", and
``improvement_cost`` remains a repair cost; it is never mapped to an
inspection cost.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final, Iterable

import numpy as np
import pandas as pd

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_PANEL_PATH: Final[Path] = REPOSITORY_ROOT / "bridges" / "panel.parquet"

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

REQUIRED_YEARS: Final[frozenset[int]] = frozenset({2012, 2016, 2018, 2022})
RATING_COLUMNS: Final[tuple[str, ...]] = (
    "deck",
    "super",
    "sub",
    "worst_rating",
)
BOOLEAN_COLUMNS: Final[tuple[str, ...]] = (
    "fracture_critical",
    "underwater",
    "rebuilt",
)
REQUIRED_STRING_COLUMNS: Final[tuple[str, ...]] = (
    "county",
    "material",
    "carries",
    "crosses",
)

# The only interval pairs whose horizons are four years. The 2016 -> 2018 gap
# is two years and must not be used for fitting or backtesting.
FOUR_YEAR_PAIRS = [(2012, 2016), (2018, 2022)]


class SourceDataError(Exception):
    """Base class for failures at the source-data boundary."""


class SourceReadError(SourceDataError):
    """Raised when the configured source panel cannot be read."""


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


class SourceAdaptationError(SourceDataError):
    """Raised when the declared lossless mapping cannot be established."""


def _resolve_source_path(path: str | Path | None) -> Path:
    """Resolve a panel path independently of the caller's working directory."""
    if path is None:
        return DEFAULT_PANEL_PATH
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def read_source_panel(path: str | Path | None = None) -> pd.DataFrame:
    """Read the source panel from a repository-relative path.

    Relative arguments are interpreted from the repository root, never from
    the process working directory. The function performs no fallback,
    generation, normalization, or adaptation. Read failures are wrapped in a
    typed boundary exception while retaining the original exception as the
    private cause.
    """
    source_path = _resolve_source_path(path)
    try:
        source = pd.read_parquet(source_path)
    except Exception as exc:
        raise SourceReadError("Production bridge data could not be read.") from exc
    if not isinstance(source, pd.DataFrame):
        raise SourceReadError("Production bridge data was not a table.")
    return source


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


def _is_exact_boolean(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _as_timestamp(value: object) -> pd.Timestamp | None:
    """Parse one datetime-like scalar without changing its source value."""
    if _is_missing(value) or isinstance(
        value, (bool, np.bool_, int, float, complex, np.number)
    ):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return None if pd.isna(timestamp) else timestamp


def validate_source_panel(source: pd.DataFrame) -> None:
    """Validate ``source`` against the exact published 21-column contract.

    Validation is observational: this function does not assign, coerce, trim,
    sort, or otherwise mutate ``source``. All detected categories are reported
    together as stable issue identifiers in :class:`SourceSchemaError`.
    """
    if not isinstance(source, pd.DataFrame):
        raise SourceSchemaError(("frame.type",))

    if tuple(source.columns) != SOURCE_COLUMNS:
        # Other predicates cannot be evaluated safely with a missing,
        # additional, duplicated, or reordered column.
        raise SourceSchemaError(("columns.exact_order",))

    issues: list[str] = []

    year_values = source["year"].tolist()
    normalized_years = [_as_integer(value) for value in year_values]
    years_are_integral = all(value is not None for value in normalized_years)
    if not years_are_integral or any(
        value not in REQUIRED_YEARS
        for value in normalized_years
        if value is not None
    ):
        issues.append("year.allowed_integral")

    represented_years = {
        value for value in normalized_years if value in REQUIRED_YEARS
    }
    if represented_years != REQUIRED_YEARS:
        issues.append("year.required_coverage")

    structure_ids = source["structure_id"].tolist()
    ids_are_valid = all(_is_nonempty_string(value) for value in structure_ids)
    if not ids_are_valid:
        issues.append("structure_id.nonempty_string")

    if ids_are_valid and years_are_integral:
        keys = tuple(zip(structure_ids, normalized_years))
        if len(set(keys)) != len(keys):
            issues.append("key.structure_id_year_unique")

    for column in REQUIRED_STRING_COLUMNS:
        if not all(_is_nonempty_string(value) for value in source[column].tolist()):
            issues.append(f"{column}.nonempty_string")

    for column in ("age", "adt"):
        values = [_as_integer(value) for value in source[column].tolist()]
        if any(value is None or value < 0 for value in values):
            issues.append(f"{column}.nonnegative_integral")

    frequencies = [
        _as_integer(value) for value in source["insp_freq_months"].tolist()
    ]
    if any(value is None or value <= 0 for value in frequencies):
        issues.append("insp_freq_months.positive_integral")

    for column in RATING_COLUMNS:
        invalid_rating = False
        for value in source[column].tolist():
            if _is_missing(value):
                continue
            rating = _as_integer(value)
            if rating is None or not 0 <= rating <= 9:
                invalid_rating = True
                break
        if invalid_rating:
            issues.append(f"{column}.nullable_rating")

    rowwise_minimum_is_valid = True
    for deck, superstructure, substructure, worst in source[
        ["deck", "super", "sub", "worst_rating"]
    ].itertuples(index=False, name=None):
        component_values: list[int] = []
        components_are_valid = True
        for value in (deck, superstructure, substructure):
            if _is_missing(value):
                continue
            rating = _as_integer(value)
            if rating is None or not 0 <= rating <= 9:
                components_are_valid = False
                break
            component_values.append(rating)

        if not components_are_valid:
            continue
        if not component_values:
            if not _is_missing(worst):
                rowwise_minimum_is_valid = False
                break
            continue

        worst_value = _as_integer(worst)
        if worst_value is None or worst_value != min(component_values):
            rowwise_minimum_is_valid = False
            break

    if not rowwise_minimum_is_valid:
        issues.append("worst_rating.rowwise_minimum")

    for column in BOOLEAN_COLUMNS:
        if not all(_is_exact_boolean(value) for value in source[column].tolist()):
            issues.append(f"{column}.exact_boolean")

    reconstruction_is_valid = True
    for reconstruction, year in zip(
        source["year_recon"].tolist(), normalized_years
    ):
        reconstruction_year = _as_integer(reconstruction)
        if reconstruction_year is None or (
            reconstruction_year != 0
            and (
                reconstruction_year < 1700
                or year is None
                or reconstruction_year > year
            )
        ):
            reconstruction_is_valid = False
            break
    if not reconstruction_is_valid:
        issues.append("year_recon.valid_for_record_year")

    for value in source["improvement_cost"].tolist():
        if _is_missing(value):
            continue
        cost = _as_finite_number(value)
        if cost is None or cost < 0:
            issues.append("improvement_cost.nullable_nonnegative_finite")
            break

    for column, lower, upper in (
        ("lat", 41.0, 43.0),
        ("lon", -74.0, -69.5),
    ):
        for value in source[column].tolist():
            if _is_missing(value):
                continue
            coordinate = _as_finite_number(value)
            if coordinate is None or not lower <= coordinate <= upper:
                issues.append(f"{column}.nullable_finite_bounds")
                break

    inspection_dates_are_valid = True
    for value, year in zip(source["insp_date"].tolist(), normalized_years):
        timestamp = _as_timestamp(value)
        if (
            timestamp is None
            or timestamp.day != 1
            or year is None
            or (timestamp.year, timestamp.month, timestamp.day) > (year, 4, 1)
        ):
            inspection_dates_are_valid = False
            break
    if not inspection_dates_are_valid:
        issues.append("insp_date.month_start_by_april_first")

    if issues:
        raise SourceSchemaError(issues)


def _assert_adaptation_postconditions(
    source: pd.DataFrame, adapted: pd.DataFrame
) -> None:
    expected_columns = tuple(
        "id"
        if column == "structure_id"
        else "insp_freq_mo"
        if column == "insp_freq_months"
        else column
        for column in SOURCE_COLUMNS
    ) + ("year_built",)
    if tuple(adapted.columns) != expected_columns:
        raise SourceAdaptationError("Adapted panel has an invalid column mapping.")
    if len(adapted) != len(source) or not adapted.index.equals(source.index):
        raise SourceAdaptationError("Adapted panel did not preserve source rows.")

    for source_column in SOURCE_COLUMNS:
        adapted_column = {
            "structure_id": "id",
            "insp_freq_months": "insp_freq_mo",
        }.get(source_column, source_column)
        if not source[source_column].equals(adapted[adapted_column]):
            raise SourceAdaptationError(
                "Adapted panel did not preserve a source column."
            )

    expected_year_built = source["year"] - source["age"]
    if not expected_year_built.equals(adapted["year_built"]):
        raise SourceAdaptationError("Adapted panel has an invalid year_built value.")
    if "insp_cost" in adapted.columns:
        raise SourceAdaptationError(
            "Adapted panel must not map repair cost to inspection cost."
        )


def adapt_source_panel(source: pd.DataFrame) -> pd.DataFrame:
    """Validate and create a lossless model-facing copy of ``source``.

    No adapted frame is returned when source validation fails. The input frame
    is never modified. Only the two declared renames are applied, and
    ``year_built`` is the sole added column.
    """
    validate_source_panel(source)

    adapted = source.copy(deep=True).rename(
        columns={
            "structure_id": "id",
            "insp_freq_months": "insp_freq_mo",
        }
    )
    adapted["year_built"] = adapted["year"] - adapted["age"]

    _assert_adaptation_postconditions(source, adapted)
    return adapted


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


def load_panel(path: str | Path | None = None) -> pd.DataFrame:
    """Compatibility wrapper over the strict read/validate/adapt boundary."""
    return adapt_source_panel(read_source_panel(path))


if __name__ == "__main__":
    panel = load_panel()
    print(
        f"loaded {len(panel)} rows, {panel['id'].nunique()} bridges, "
        f"years {sorted(panel['year'].unique())}"
    )
    print("has insp_date:", "insp_date" in panel.columns)
    print("has insp_cost:", "insp_cost" in panel.columns)
    print(
        panel[
            [
                "id",
                "year",
                "year_built",
                "worst_rating",
                "insp_freq_mo",
                "rebuilt",
            ]
        ].head()
    )
