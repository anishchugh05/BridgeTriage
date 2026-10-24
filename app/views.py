"""Pure presentation helpers for shared ranking and evidence views.

The module consumes one immutable :class:`app.state.ViewContext` and never
calls model, planner, cache, provider, or demo implementations. Preparation
helpers return defensive projections; Streamlit renderers add presentation
side effects only. Backtest values are displayed without adjustment, while
bridge history remains sparse and explicitly labels unavailable observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Real
from typing import Any, Final, Iterable, Mapping

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from app.config import RuntimeMode
from app.state import ViewContext

DEMO_INDICATOR: Final[str] = "DEMO MODE - FABRICATED DATA"
CURRENT_RANKING_LABEL: Final[str] = "Current ranking"
MAP_SECTION_LABEL: Final[str] = "Map"
BUDGET_SECTION_LABEL: Final[str] = "Budget"
SECTION_LABELS: Final[tuple[str, ...]] = (
    MAP_SECTION_LABEL,
    BUDGET_SECTION_LABEL,
    "Backtest",
    "Bridge detail",
    "Hazards",
    "Export",
    "Methods",
)
PRIMARY_VIEW_ORDER: Final[tuple[str, ...]] = (
    CURRENT_RANKING_LABEL,
    MAP_SECTION_LABEL,
)

MODELED_COST_PHRASE: Final[str] = "modeled inspection-cost estimate"
MODELED_COST_COLUMN_LABEL: Final[str] = "Modeled inspection-cost estimate ($)"
SELECTED_MARKER_COLOR: Final[tuple[int, int, int]] = (229, 72, 77)
UNSELECTED_COLOR_LOW: Final[tuple[int, int, int]] = (219, 234, 254)
UNSELECTED_COLOR_HIGH: Final[tuple[int, int, int]] = (30, 64, 175)
MAP_RADIUS_BASE: Final[float] = 40.0
MAP_RADIUS_SCALE: Final[float] = 2.5
MAP_VIEW_STATE: Final[Mapping[str, float]] = {
    "latitude": 42.2,
    "longitude": -71.8,
    "zoom": 7.2,
}
MAP_TOOLTIP: Final[Mapping[str, str]] = {
    "text": (
        "Bridge: {structure_id}\n"
        "Carries: {carries}\n"
        "Crosses: {crosses}\n"
        "County: {county}\n"
        "Score: {score}"
    )
}

_MAP_COLUMNS: Final[tuple[str, ...]] = (
    "structure_id",
    "lat",
    "lon",
    "county",
    "score",
    "adt",
    "carries",
    "crosses",
)
_BUDGET_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "structure_id",
    "county",
    "worst_rating",
    "score",
    "insp_cost",
    "underwater",
    "fracture_critical",
    "carries",
    "crosses",
    "reason",
)
BUDGET_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "structure_id",
    "county",
    "worst_rating",
    "score",
    "insp_cost",
    "hazard",
    "carries",
    "crosses",
    "reason",
)


@dataclass(frozen=True, slots=True)
class CurrentRankingSummary:
    """Exact values shown by the current-ranking headline."""

    active_budget: int
    selected_count: int
    ranking_population_count: int
    selected_total_cost: Decimal
    expected_safety_drop_coverage: float
    selection_identity: str


def _require_context(context: ViewContext) -> None:
    if not isinstance(context, ViewContext):
        raise TypeError("context must be a ViewContext.")


def _ui_or_streamlit(ui: Any | None) -> Any:
    return st if ui is None else ui


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    frame_name: str,
) -> None:
    missing = tuple(column for column in required if column not in frame.columns)
    if missing:
        raise ValueError(
            f"{frame_name} is missing required presentation columns: "
            + ", ".join(missing)
        )


def _ranking_records(context: ViewContext) -> pd.DataFrame:
    """Return an isolated complete reconciled ranking population.

    The approved ``CurrentRanking`` value exposes a ``records`` frame.  Direct
    DataFrame and mapping forms are also accepted so the presentation boundary
    remains independently testable while that service value is assembled.
    """

    _require_context(context)
    ranking = context.current_ranking
    if isinstance(ranking, pd.DataFrame):
        records = ranking
    elif isinstance(ranking, Mapping):
        records = ranking.get("records")
    else:
        records = getattr(ranking, "records", None)

    if not isinstance(records, pd.DataFrame):
        raise TypeError("current_ranking must expose a records DataFrame.")
    _require_columns(records, ("structure_id",), frame_name="ranking records")

    identifiers = tuple(records["structure_id"].tolist())
    if not all(
        isinstance(identifier, str) and bool(identifier.strip())
        for identifier in identifiers
    ):
        raise ValueError("ranking records must contain nonempty string identifiers.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("ranking records must contain unique identifiers.")
    return records.copy(deep=True)


def _finite_real(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, Decimal)
    ):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _exact_boolean(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _format_currency(value: Decimal | int) -> str:
    try:
        amount = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:  # defensive
        raise ValueError("Currency value must be a finite decimal.") from exc
    if not amount.is_finite():
        raise ValueError("Currency value must be a finite decimal.")

    plain = format(amount, "f")
    whole, separator, fraction = plain.partition(".")
    grouped = f"{int(whole):,}"
    fraction = fraction.rstrip("0")
    return f"${grouped}{separator + fraction if fraction else ''}"


def _render_metrics(ui: Any, metrics: tuple[tuple[str, object], ...]) -> None:
    columns = ui.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)


def render_mode_notice(context: ViewContext, *, ui: Any | None = None) -> bool:
    """Render the exact demo indicator, and only for explicit demo mode.

    Every data-derived renderer calls this function before preparing or
    displaying its values.  The Boolean return is useful to verify guard
    placement without inspecting Streamlit internals.
    """

    _require_context(context)
    if context.mode is RuntimeMode.DEMO:
        _ui_or_streamlit(ui).warning(DEMO_INDICATOR)
        return True
    return False


def prepare_current_ranking_summary(
    context: ViewContext,
) -> CurrentRankingSummary:
    """Project the headline exclusively from shared context values."""

    records = _ranking_records(context)
    metrics = context.metrics
    return CurrentRankingSummary(
        active_budget=context.active_budget,
        selected_count=metrics.selected_count,
        ranking_population_count=len(records),
        selected_total_cost=metrics.selected_total_cost,
        expected_safety_drop_coverage=metrics.expected_safety_drop_coverage,
        selection_identity=context.selection_identity,
    )


def render_current_ranking(
    context: ViewContext,
    *,
    ui: Any | None = None,
) -> CurrentRankingSummary:
    """Render the current-ranking headline and exact statewide summary."""

    ui = _ui_or_streamlit(ui)
    render_mode_notice(context, ui=ui)
    summary = prepare_current_ranking_summary(context)

    ui.header(CURRENT_RANKING_LABEL)
    ui.markdown(
        f'<div class="headline"><span class="accent">'
        f"{summary.selected_count}</span> bridges selected</div>",
        unsafe_allow_html=True,
    )
    _render_metrics(
        ui,
        (
            ("Active budget", _format_currency(summary.active_budget)),
            ("Selected bridges", summary.selected_count),
            ("Ranking population", summary.ranking_population_count),
            (
                "Selected total modeled inspection-cost estimate",
                _format_currency(summary.selected_total_cost),
            ),
            (
                "Expected Safety_Drop coverage",
                summary.expected_safety_drop_coverage,
            ),
        ),
    )
    provenance = context.provenance
    ui.caption(
        f"Data: {provenance.data_source} · "
        f"Scoring: {provenance.model_source} · "
        f"Selection: {provenance.planner_source} · "
        f"Backtest: {provenance.backtest_source}"
    )
    return summary


def is_map_eligible(lat: object, lon: object) -> bool:
    """Return the exact inclusive Massachusetts map-coordinate predicate."""

    latitude = _finite_real(lat)
    longitude = _finite_real(lon)
    return (
        latitude is not None
        and longitude is not None
        and 41.0 <= latitude <= 43.0
        and -74.0 <= longitude <= -69.5
    )


def normalize_score_values(scores: Iterable[object]) -> tuple[float, ...]:
    """Min-max normalize finite scores onto a fixed nondecreasing scale.

    A constant sequence maps to the fixed midpoint, so equal values always
    receive equal encodings without introducing a divide-by-zero special case
    in callers.
    """

    if isinstance(scores, (str, bytes, bytearray)):
        raise TypeError("scores must be an iterable of finite numbers.")
    try:
        raw_values = tuple(scores)
    except TypeError as exc:
        raise TypeError("scores must be an iterable of finite numbers.") from exc

    values: list[float] = []
    for value in raw_values:
        numeric = _finite_real(value)
        if numeric is None:
            raise ValueError("scores must contain only finite numbers.")
        values.append(numeric)
    if not values:
        return ()

    low = min(values)
    high = max(values)
    if high == low:
        return tuple(0.5 for _ in values)
    span = high - low
    return tuple(min(1.0, max(0.0, (value - low) / span)) for value in values)


# Concise public alias used by presentation tests and callers.
normalize_scores = normalize_score_values


def score_to_color(normalized_score: object) -> tuple[int, int, int]:
    """Map a normalized score to one fixed continuous sequential palette."""

    value = _finite_real(normalized_score)
    if value is None or not 0.0 <= value <= 1.0:
        raise ValueError("normalized_score must be finite and between 0 and 1.")
    return tuple(
        int(round(low + (high - low) * value))
        for low, high in zip(UNSELECTED_COLOR_LOW, UNSELECTED_COLOR_HIGH)
    )


def adt_to_radius(adt: object) -> float:
    """Map validated nonnegative ADT to a fixed nondecreasing radius."""

    value = _finite_real(adt)
    if value is None or value < 0.0:
        raise ValueError("adt must be a finite nonnegative number.")
    return MAP_RADIUS_BASE + MAP_RADIUS_SCALE * math.sqrt(value)


def prepare_map_markers(
    context: ViewContext,
    *,
    county: str | None = None,
) -> pd.DataFrame:
    """Return marker rows without changing the statewide population/selection.

    Coordinate eligibility and the optional exact county match affect this
    defensive projection only.  All reconciled tooltip fields retain their
    source/model values verbatim.
    """

    if county is not None and not isinstance(county, str):
        raise TypeError("county must be a string or None.")

    records = _ranking_records(context)
    _require_columns(records, _MAP_COLUMNS, frame_name="ranking records")
    eligible = pd.Series(
        (
            is_map_eligible(lat, lon)
            for lat, lon in records.loc[:, ["lat", "lon"]].itertuples(
                index=False, name=None
            )
        ),
        index=records.index,
        dtype=bool,
    )
    markers = records.loc[eligible].copy(deep=True)
    if county is not None:
        markers = markers.loc[markers["county"].eq(county)].copy(deep=True)
    markers.reset_index(drop=True, inplace=True)

    scale_values = normalize_score_values(markers["score"].tolist())
    selected_ids = set(context.selection.ids)
    selected = markers["structure_id"].isin(selected_ids)

    markers["selected"] = selected.astype(bool)
    markers["score_color_value"] = scale_values
    markers["fill_color"] = [
        SELECTED_MARKER_COLOR if is_selected else score_to_color(scale_value)
        for is_selected, scale_value in zip(selected.tolist(), scale_values)
    ]
    markers["radius"] = [adt_to_radius(value) for value in markers["adt"]]
    return markers


def render_map(
    context: ViewContext,
    *,
    county: str | None = None,
    ui: Any | None = None,
) -> pd.DataFrame:
    """Render the required map from a marker-only projection."""

    ui = _ui_or_streamlit(ui)
    render_mode_notice(context, ui=ui)
    markers = prepare_map_markers(context, county=county)

    ui.subheader(MAP_SECTION_LABEL)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=markers,
        get_position=["lon", "lat"],
        get_fill_color="fill_color",
        get_radius="radius",
        pickable=True,
        opacity=0.8,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(**MAP_VIEW_STATE),
        tooltip=dict(MAP_TOOLTIP),
    )
    ui.pydeck_chart(deck, use_container_width=True)
    ui.caption(
        "Red markers are in the shared statewide selection. "
        "Unselected color represents score; marker size represents ADT."
    )
    return markers


def _hazard_values(records: pd.DataFrame) -> tuple[bool, ...]:
    _require_columns(
        records,
        ("underwater", "fracture_critical"),
        frame_name="selected records",
    )
    hazards: list[bool] = []
    for underwater, fracture_critical in records.loc[
        :, ["underwater", "fracture_critical"]
    ].itertuples(index=False, name=None):
        underwater_value = _exact_boolean(underwater)
        fracture_value = _exact_boolean(fracture_critical)
        if underwater_value is None or fracture_value is None:
            raise ValueError("Hazard fields must contain exact Boolean values.")
        hazards.append(underwater_value or fracture_value)
    return tuple(hazards)


def order_selected_evidence(records: pd.DataFrame) -> pd.DataFrame:
    """Add exact hazard classification and apply canonical evidence order."""

    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a pandas DataFrame.")
    _require_columns(
        records,
        ("structure_id", "score", "underwater", "fracture_critical"),
        frame_name="selected records",
    )
    ordered = records.copy(deep=True)
    ordered["hazard"] = _hazard_values(ordered)
    ordered.sort_values(
        by=["hazard", "score", "structure_id"],
        ascending=[False, False, True],
        kind="mergesort",
        inplace=True,
    )
    ordered.reset_index(drop=True, inplace=True)
    return ordered


def prepare_budget_rows(context: ViewContext) -> pd.DataFrame:
    """Return the shared selection's exact values in Budget display order."""

    _require_context(context)
    selected_records = context.selected_records
    _require_columns(
        selected_records,
        _BUDGET_SOURCE_COLUMNS,
        frame_name="selected records",
    )
    ordered = order_selected_evidence(selected_records)
    return ordered.loc[:, BUDGET_TABLE_COLUMNS].copy(deep=True)


def render_budget(
    context: ViewContext,
    *,
    ui: Any | None = None,
) -> pd.DataFrame:
    """Render Budget metrics and rows from the one shared selection."""

    ui = _ui_or_streamlit(ui)
    render_mode_notice(context, ui=ui)
    rows = prepare_budget_rows(context)
    metrics = context.metrics

    ui.subheader(BUDGET_SECTION_LABEL)
    _render_metrics(
        ui,
        (
            ("Active budget", _format_currency(context.active_budget)),
            ("Selected bridges", metrics.selected_count),
            (
                "Selected total modeled inspection-cost estimate",
                _format_currency(metrics.selected_total_cost),
            ),
            (
                "Expected Safety_Drop coverage",
                metrics.expected_safety_drop_coverage,
            ),
        ),
    )
    ui.caption(
        "Every insp_cost value is a modeled inspection-cost estimate, "
        "not a quote or repair cost."
    )
    ui.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={"insp_cost": MODELED_COST_COLUMN_LABEL},
    )
    return rows


BACKTEST_SECTION_LABEL: Final[str] = "Backtest"
BRIDGE_DETAIL_SECTION_LABEL: Final[str] = "Bridge detail"
BACKTEST_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "budget",
    "method",
    "caught",
)
BACKTEST_METHODS: Final[tuple[str, ...]] = (
    "ours",
    "worst_first",
    "oldest_first",
)
BASELINE_METHODS: Final[tuple[str, ...]] = (
    "worst_first",
    "oldest_first",
)
BRIDGE_DETAIL_YEARS: Final[tuple[int, ...]] = (2012, 2016, 2018, 2022)
BRIDGE_RATING_COLUMNS: Final[tuple[str, ...]] = ("deck", "super", "sub")
UNAVAILABLE: Final[str] = "Unavailable"
INFOBRIDGE_BASE_URL: Final[str] = "https://infobridge.fhwa.dot.gov/Data"


@dataclass(frozen=True, slots=True)
class BridgeDetail:
    """One reconciled bridge descriptor and its sparse source history."""

    structure_id: str
    history: pd.DataFrame
    reason: str
    carries: str
    crosses: str
    county: str
    infobridge_url: str

    @property
    def fhwa_url(self) -> str:
        """Readable alias for callers using the FHWA terminology."""

        return self.infobridge_url


def _backtest_frame(value: object) -> pd.DataFrame:
    """Extract an isolated public Backtest_Result frame.

    Production currently supplies a DataFrame directly.  The documented
    ``BacktestRun.result`` shape and mapping-based fixtures are accepted as
    data-only wrappers; callbacks and computations are never invoked here.
    """

    if isinstance(value, ViewContext):
        _require_context(value)
        value = value.backtest

    if isinstance(value, pd.DataFrame):
        frame = value
    elif isinstance(value, Mapping):
        frame = value.get("result")
    else:
        frame = getattr(value, "result", None)
        if not isinstance(frame, pd.DataFrame):
            frame = getattr(value, "value", None)

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("backtest must expose a result DataFrame.")
    if tuple(frame.columns) != BACKTEST_RESULT_COLUMNS:
        raise ValueError(
            "backtest result columns must be exactly budget, method, caught."
        )
    if frame.empty:
        raise ValueError("backtest result must not be empty.")
    return frame.copy(deep=True)


def _canonical_budget(value: object) -> Decimal:
    numeric = _finite_real(value)
    if numeric is None or numeric < 0.0:
        raise ValueError("Backtest budgets must be finite nonnegative numbers.")
    try:
        budget = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(
            "Backtest budgets must be finite nonnegative numbers."
        ) from exc
    if not budget.is_finite() or budget < 0:
        raise ValueError("Backtest budgets must be finite nonnegative numbers.")
    return budget


def _caught_count(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("Backtest caught values must be nonnegative integers.")
    caught = int(value)
    if caught < 0:
        raise ValueError("Backtest caught values must be nonnegative integers.")
    return caught


def _validated_backtest_groups(
    value: object,
) -> dict[Decimal, dict[str, tuple[object, object]]]:
    """Validate one row per budget/method without altering measured counts."""

    frame = _backtest_frame(value)
    groups: dict[Decimal, dict[str, tuple[object, object]]] = {}
    for budget_value, method, caught_value in frame.itertuples(
        index=False, name=None
    ):
        budget = _canonical_budget(budget_value)
        if not isinstance(method, str) or method not in BACKTEST_METHODS:
            raise ValueError("Backtest methods are invalid.")
        _caught_count(caught_value)

        by_method = groups.setdefault(budget, {})
        if method in by_method:
            raise ValueError(
                "Backtest result must contain one row per budget and method."
            )
        by_method[method] = (budget_value, caught_value)

    required_methods = set(BACKTEST_METHODS)
    if any(set(by_method) != required_methods for by_method in groups.values()):
        raise ValueError(
            "Every backtest budget must contain all three measured methods."
        )
    return groups


def prepare_backtest_chart(value: object) -> pd.DataFrame:
    """Return measured ``caught`` series with no superiority adjustment.

    Rows are placed in ascending budget order and columns in the published
    method order.  Every plotted cell is the corresponding Backtest_Result
    value; no aggregation, rounding, interpolation, or baseline rewriting is
    performed.
    """

    groups = _validated_backtest_groups(value)
    budgets = sorted(groups)
    index_values = [groups[budget]["ours"][0] for budget in budgets]
    chart = pd.DataFrame(
        {
            method: [groups[budget][method][1] for budget in budgets]
            for method in BACKTEST_METHODS
        },
        index=pd.Index(index_values, name="budget"),
    )
    chart.columns.name = None
    return chart


def backtest_counts_at_budget(
    value: object,
    budget: object,
) -> dict[str, int]:
    """Return exact method counts for one and only one requested budget."""

    groups = _validated_backtest_groups(value)
    requested = _canonical_budget(budget)
    if requested not in groups:
        raise KeyError(budget)
    return {
        method: _caught_count(groups[requested][method][1])
        for method in BACKTEST_METHODS
    }


def _format_ratio(ours_caught: int, baseline_caught: int) -> str:
    ratio = Decimal(ours_caught) / Decimal(baseline_caught)
    rendered = format(ratio, ".6f").rstrip("0").rstrip(".")
    return f"{ours_caught}/{baseline_caught} = {rendered}×"


def format_backtest_comparison(
    *,
    budget: object,
    ours_caught: object,
    baseline_caught: object,
    baseline_method: str,
) -> str:
    """Format one truthful same-budget comparison.

    A winning comparison against a positive baseline reports the specified
    absolute difference and ``ours / baseline`` ratio.  A zero baseline emits
    counts and absolute difference only.  Ties and losses emit exact counts
    without superiority language.
    """

    if baseline_method not in BASELINE_METHODS:
        raise ValueError("baseline_method must name a published baseline.")
    canonical_budget = _canonical_budget(budget)
    ours = _caught_count(ours_caught)
    baseline = _caught_count(baseline_caught)
    prefix = (
        f"At budget {_format_currency(canonical_budget)}, `ours` caught {ours} "
        f"measured Safety_Drop outcomes and `{baseline_method}` caught "
        f"{baseline}."
    )

    if ours <= baseline:
        return prefix

    difference = ours - baseline
    if baseline == 0:
        return f"{prefix} Absolute difference: {difference}."
    return (
        f"{prefix} Difference: {difference} more declines caught; "
        f"ratio: {_format_ratio(ours, baseline)}."
    )


def prepare_backtest_comparisons(
    value: object,
    budget: object,
) -> tuple[str, ...]:
    """Create both baseline comparisons from the same budget row set."""

    counts = backtest_counts_at_budget(value, budget)
    return tuple(
        format_backtest_comparison(
            budget=budget,
            ours_caught=counts["ours"],
            baseline_caught=counts[method],
            baseline_method=method,
        )
        for method in BASELINE_METHODS
    )


def render_backtest(
    context: ViewContext,
    *,
    comparison_budget: object | None = None,
    ui: Any | None = None,
) -> pd.DataFrame:
    """Render measured backtest series and bounded same-budget copy."""

    ui = _ui_or_streamlit(ui)
    render_mode_notice(context, ui=ui)
    ui.subheader(BACKTEST_SECTION_LABEL)

    chart = prepare_backtest_chart(context)
    ui.line_chart(chart, use_container_width=True)

    budget = context.active_budget if comparison_budget is None else comparison_budget
    try:
        comparisons = prepare_backtest_comparisons(context, budget)
    except KeyError:
        ui.caption(
            "No measured backtest comparison is available at the active budget "
            f"{_format_currency(_canonical_budget(budget))}."
        )
    else:
        for comparison in comparisons:
            ui.markdown(comparison)
    return chart


def _source_records(context: ViewContext) -> pd.DataFrame:
    """Return an isolated source-facing panel from the context bundle."""

    _require_context(context)
    source = context.source
    if isinstance(source, pd.DataFrame):
        records = source
    elif isinstance(source, Mapping):
        records = source.get("source")
    else:
        records = getattr(source, "source", None)

    if not isinstance(records, pd.DataFrame):
        raise TypeError("source must expose a source DataFrame.")
    _require_columns(
        records,
        ("structure_id", "year", *BRIDGE_RATING_COLUMNS),
        frame_name="source records",
    )
    return records.copy(deep=True)


def _is_missing_cell(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def prepare_bridge_options(context: ViewContext) -> tuple[str, ...]:
    """Return every ranking identifier in deterministic score order."""

    records = _ranking_records(context)
    _require_columns(records, ("score",), frame_name="ranking records")
    for score_value in records["score"].tolist():
        if _finite_real(score_value) is None:
            raise ValueError("ranking scores must be finite numbers.")
    records.sort_values(
        by=["score", "structure_id"],
        ascending=[False, True],
        kind="mergesort",
        inplace=True,
    )
    return tuple(records["structure_id"].tolist())


def prepare_bridge_history(
    context: ViewContext,
    structure_id: str,
) -> pd.DataFrame:
    """Project exact source ratings onto the four required year slots.

    Missing bridge-years and null rating cells receive the display token
    ``Unavailable``.  Existing values are copied verbatim, and no interpolation
    or other replacement value is generated.
    """

    if not isinstance(structure_id, str) or not structure_id.strip():
        raise ValueError("structure_id must be a nonempty string.")
    records = _source_records(context)
    bridge_rows = records.loc[records["structure_id"].eq(structure_id)].copy()

    normalized_years: list[int] = []
    for value in bridge_rows["year"].tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError("Bridge history years must be integers.")
        normalized_years.append(int(value))
    bridge_rows["year"] = normalized_years

    required_rows = bridge_rows.loc[
        bridge_rows["year"].isin(BRIDGE_DETAIL_YEARS)
    ]
    if required_rows["year"].duplicated().any():
        raise ValueError("Bridge history must contain at most one row per year.")

    history = pd.DataFrame(
        UNAVAILABLE,
        index=pd.Index(BRIDGE_DETAIL_YEARS, name="year"),
        columns=list(BRIDGE_RATING_COLUMNS),
        dtype=object,
    )
    for row in required_rows.loc[
        :, ("year", *BRIDGE_RATING_COLUMNS)
    ].itertuples(index=False, name=None):
        year = int(row[0])
        for column, value in zip(BRIDGE_RATING_COLUMNS, row[1:]):
            history.at[year, column] = (
                UNAVAILABLE if _is_missing_cell(value) else value
            )
    return history


def build_infobridge_url(structure_id: str) -> str:
    """Associate the exact URL-encoded bridge identifier with InfoBridge."""

    if not isinstance(structure_id, str) or not structure_id.strip():
        raise ValueError("structure_id must be a nonempty string.")
    from urllib.parse import quote

    encoded_identifier = quote(structure_id, safe="")
    return f"{INFOBRIDGE_BASE_URL}?structure_id={encoded_identifier}"


def _required_detail_text(row: pd.Series, column: str) -> str:
    value = row[column]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Bridge detail {column} must be a nonempty string.")
    return value


def prepare_bridge_detail(
    context: ViewContext,
    structure_id: str,
) -> BridgeDetail:
    """Join sparse source history to one exact reconciled ranking record."""

    if not isinstance(structure_id, str) or not structure_id.strip():
        raise ValueError("structure_id must be a nonempty string.")
    records = _ranking_records(context)
    _require_columns(
        records,
        ("reason", "carries", "crosses", "county"),
        frame_name="ranking records",
    )
    matched = records.loc[records["structure_id"].eq(structure_id)]
    if len(matched) != 1:
        raise ValueError(
            "structure_id must identify exactly one reconciled ranking record."
        )
    row = matched.iloc[0]
    return BridgeDetail(
        structure_id=structure_id,
        history=prepare_bridge_history(context, structure_id),
        reason=_required_detail_text(row, "reason"),
        carries=_required_detail_text(row, "carries"),
        crosses=_required_detail_text(row, "crosses"),
        county=_required_detail_text(row, "county"),
        infobridge_url=build_infobridge_url(structure_id),
    )


def render_bridge_detail(
    context: ViewContext,
    *,
    structure_id: str | None = None,
    ui: Any | None = None,
) -> BridgeDetail | None:
    """Render one sparse bridge history and reconciled descriptors."""

    ui = _ui_or_streamlit(ui)
    render_mode_notice(context, ui=ui)
    ui.subheader(BRIDGE_DETAIL_SECTION_LABEL)

    options = prepare_bridge_options(context)
    if not options:
        ui.info("No bridge records are available.")
        return None

    if structure_id is None:
        structure_id = ui.selectbox("Bridge", options, index=0)
    if structure_id is None:
        return None

    detail = prepare_bridge_detail(context, structure_id)
    ui.dataframe(detail.history, use_container_width=True)
    ui.markdown(f"**Bridge:** {detail.structure_id}")
    ui.markdown(f"**Carries:** {detail.carries}")
    ui.markdown(f"**Crosses:** {detail.crosses}")
    ui.markdown(f"**County:** {detail.county}")
    ui.markdown(f"**Reason:** {detail.reason}")
    ui.link_button("Open in FHWA InfoBridge", detail.infobridge_url)
    return detail


HAZARDS_SECTION_LABEL: Final[str] = "Hazards"
EXPORT_SECTION_LABEL: Final[str] = "Export"
METHODS_SECTION_LABEL: Final[str] = "Methods"
HAZARD_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "structure_id",
    "county",
    "worst_rating",
    "score",
    "hazard",
    "underwater",
    "fracture_critical",
    "carries",
    "crosses",
    "reason",
)
EXPORT_COLUMNS: Final[tuple[str, ...]] = (
    "county",
    "structure_id",
    "worst_rating",
    "score",
    "insp_cost",
    "hazard",
    "carries",
    "crosses",
    "reason",
)
EXPORT_FILE_NAME: Final[str] = "selected_bridges.csv"
EXPORT_MIME_TYPE: Final[str] = "text/csv"
EXPORT_COST_DISCLOSURE: Final[str] = (
    "`insp_cost` contains modeled inspection-cost estimates; it does not "
    "contain `improvement_cost`, which is a repair-cost field."
)
PRODUCTION_SOURCE_LABELS: Final[tuple[str, ...]] = (
    "Real_Dataset",
    "Real_Model_Service",
    "Real_Planner",
    "Backtest_Engine",
)


def prepare_hazard_rows(context: ViewContext) -> pd.DataFrame:
    """Return exactly the selected 2022 hazard bridges in canonical order.

    Hazard status is recomputed from the two exact source Boolean fields rather
    than trusting a pre-existing display column.  Starting from the shared
    selected-record snapshot guarantees that map eligibility and county filters
    cannot alter this projection.
    """

    _require_context(context)
    ordered = order_selected_evidence(context.selected_records)
    _require_columns(
        ordered,
        HAZARD_TABLE_COLUMNS,
        frame_name="selected records",
    )
    hazards = ordered.loc[ordered["hazard"].eq(True)]  # noqa: E712
    return hazards.loc[:, HAZARD_TABLE_COLUMNS].reset_index(drop=True).copy(
        deep=True
    )


def render_hazards(
    context: ViewContext,
    *,
    ui: Any | None = None,
) -> pd.DataFrame:
    """Render the exact hazard intersection of the shared statewide selection."""

    ui = _ui_or_streamlit(ui)
    render_mode_notice(context, ui=ui)
    rows = prepare_hazard_rows(context)

    ui.subheader(HAZARDS_SECTION_LABEL)
    ui.caption(
        "Hazard bridges are selected bridges whose 2022 source record has "
        "`underwater` or `fracture_critical` equal to true."
    )
    if rows.empty:
        ui.info("No hazard bridges are present in the active statewide Selection.")
    ui.dataframe(rows, use_container_width=True, hide_index=True)
    return rows


def prepare_export_rows(context: ViewContext) -> pd.DataFrame:
    """Return the exact deterministic CSV projection of the shared selection."""

    _require_context(context)
    rows = order_selected_evidence(context.selected_records)
    _require_columns(rows, EXPORT_COLUMNS, frame_name="selected records")
    rows.sort_values(
        by=["county", "hazard", "score", "structure_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
        inplace=True,
    )
    rows.reset_index(drop=True, inplace=True)
    return rows.loc[:, EXPORT_COLUMNS].copy(deep=True)


def build_export_csv(context: ViewContext) -> bytes:
    """Encode selected-only export rows as deterministic UTF-8 CSV bytes."""

    return prepare_export_rows(context).to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")


def prepare_export_provenance(context: ViewContext) -> str:
    """Describe the active data/model/planner and exact shared selection."""

    _require_context(context)
    provenance = context.provenance
    return (
        f"Provenance — data: {provenance.data_source}; "
        f"scoring: {provenance.model_source}; "
        f"selection: {provenance.planner_source}; "
        f"Active_Budget: {_format_currency(context.active_budget)}; "
        f"current Selection: {context.selection_identity}."
    )


def render_export(
    context: ViewContext,
    *,
    ui: Any | None = None,
) -> bytes:
    """Render selected-only CSV preview, disclosures, and download control."""

    ui = _ui_or_streamlit(ui)
    render_mode_notice(context, ui=ui)
    rows = prepare_export_rows(context)
    csv_bytes = rows.to_csv(index=False, lineterminator="\n").encode("utf-8")

    ui.subheader(EXPORT_SECTION_LABEL)
    ui.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={"insp_cost": MODELED_COST_COLUMN_LABEL},
    )
    ui.caption(prepare_export_provenance(context))
    ui.caption(EXPORT_COST_DISCLOSURE)

    # Export is both a data-derived view and a download.  The second guard is
    # intentionally adjacent to the control so demo bytes cannot be mistaken
    # for production output even when the preview is long.
    render_mode_notice(context, ui=ui)
    ui.download_button(
        "Download selected bridges (CSV)",
        csv_bytes,
        file_name=EXPORT_FILE_NAME,
        mime=EXPORT_MIME_TYPE,
    )
    return csv_bytes


def prepare_methods_text(context: ViewContext) -> str:
    """Return the approved model, backtest, schema, and provenance narrative."""

    _require_context(context)
    provenance = context.provenance
    provenance_heading = (
        "Production provenance"
        if context.mode is RuntimeMode.PRODUCTION
        else "Active provenance"
    )
    return (
        "**NBI source and current-model timeline**\n\n"
        "The source is the Federal Highway Administration National Bridge "
        "Inventory (NBI) for Massachusetts, with snapshots from 2012, 2016, "
        "2018, and 2022. The current model fits the `2018` to `2022` interval "
        "and scores the complete 2022 ranking at Ranking_As_Of `2022-04-01`.\n\n"
        "**Backtest timeline**\n\n"
        "The historical backtest fits decline rates from `2012` to `2016`, "
        "uses only information available for the `2018` selection, and then "
        "measures observed outcomes in `2022`.\n\n"
        "**Safety_Drop, rebuilt records, and costs**\n\n"
        "Safety_Drop requires both endpoint ratings and a non-rebuilt ending "
        "record. It is true when the ending worst rating is 4 or lower and "
        "the starting rating is 5 or higher, or when the starting rating is "
        "4 or lower and the ending rating decreases by at least one. Rebuilt "
        "ending records are excluded from decline fitting and outcome "
        "evaluation. NBI does not provide a usable per-bridge inspection "
        "cost, so `insp_cost` uses a modeled inspection-cost estimate: $3,000 "
        "normally, $8,000 underwater, $6,000 fracture-critical, or $12,000 "
        "when both apply. These are estimates, not quotes or repair costs, and "
        "source `improvement_cost` is never used as inspection cost.\n\n"
        "**Schema mapping and reconciliation**\n\n"
        "At the sole model boundary, source `structure_id` maps to `id`, "
        "`insp_freq_months` maps to `insp_freq_mo`, and `year_built` is derived "
        "as `year - age`; other source meanings, including `improvement_cost`, "
        "are preserved. Model `id` values are reconciled one-to-one back to "
        "Ranking_Year `structure_id` values before presentation.\n\n"
        "**NBI → SNBI transition**\n\n"
        "The Specification for the National Bridge Inventory (SNBI) succeeds "
        "NBI. Because item identifiers and some fields change under SNBI, a "
        "future SNBI source requires an explicit ingestion mapping update; "
        "the application does not silently reinterpret NBI fields.\n\n"
        f"**{provenance_heading}**\n\n"
        f"- Data source: `{provenance.data_source}`\n"
        f"- Scoring source: `{provenance.model_source}`\n"
        f"- Selection source: `{provenance.planner_source}`\n"
        f"- Backtest source: `{provenance.backtest_source}`"
    )


def render_methods(
    context: ViewContext,
    *,
    ui: Any | None = None,
) -> str:
    """Render approved methodology and all active provenance labels."""

    ui = _ui_or_streamlit(ui)
    render_mode_notice(context, ui=ui)
    text = prepare_methods_text(context)
    ui.subheader(METHODS_SECTION_LABEL)
    ui.markdown(text)
    return text


def render_primary_views(
    context: ViewContext,
    *,
    county: str | None = None,
    ui: Any | None = None,
) -> tuple[CurrentRankingSummary, pd.DataFrame]:
    """Render the headline before Map, ahead of all secondary sections."""

    ui = _ui_or_streamlit(ui)
    summary = render_current_ranking(context, ui=ui)
    markers = render_map(context, county=county, ui=ui)
    return summary, markers


# Readable compatibility aliases for callers that use the design terminology.
prepare_headline_summary = prepare_current_ranking_summary
prepare_hazards = prepare_hazard_rows
prepare_export_csv = build_export_csv
export_csv_bytes = build_export_csv
render_headline = render_current_ranking


__all__ = [
    "BACKTEST_METHODS",
    "BACKTEST_RESULT_COLUMNS",
    "BACKTEST_SECTION_LABEL",
    "BASELINE_METHODS",
    "BRIDGE_DETAIL_SECTION_LABEL",
    "BRIDGE_DETAIL_YEARS",
    "BRIDGE_RATING_COLUMNS",
    "BUDGET_SECTION_LABEL",
    "BUDGET_TABLE_COLUMNS",
    "BridgeDetail",
    "CURRENT_RANKING_LABEL",
    "CurrentRankingSummary",
    "DEMO_INDICATOR",
    "EXPORT_COLUMNS",
    "EXPORT_COST_DISCLOSURE",
    "EXPORT_FILE_NAME",
    "EXPORT_MIME_TYPE",
    "EXPORT_SECTION_LABEL",
    "HAZARDS_SECTION_LABEL",
    "HAZARD_TABLE_COLUMNS",
    "INFOBRIDGE_BASE_URL",
    "MAP_RADIUS_BASE",
    "MAP_RADIUS_SCALE",
    "MAP_SECTION_LABEL",
    "MAP_TOOLTIP",
    "METHODS_SECTION_LABEL",
    "MODELED_COST_COLUMN_LABEL",
    "MODELED_COST_PHRASE",
    "PRIMARY_VIEW_ORDER",
    "PRODUCTION_SOURCE_LABELS",
    "SECTION_LABELS",
    "SELECTED_MARKER_COLOR",
    "UNAVAILABLE",
    "UNSELECTED_COLOR_HIGH",
    "UNSELECTED_COLOR_LOW",
    "adt_to_radius",
    "backtest_counts_at_budget",
    "build_export_csv",
    "build_infobridge_url",
    "export_csv_bytes",
    "format_backtest_comparison",
    "is_map_eligible",
    "normalize_score_values",
    "normalize_scores",
    "order_selected_evidence",
    "prepare_backtest_chart",
    "prepare_backtest_comparisons",
    "prepare_bridge_detail",
    "prepare_bridge_history",
    "prepare_bridge_options",
    "prepare_budget_rows",
    "prepare_current_ranking_summary",
    "prepare_export_csv",
    "prepare_export_provenance",
    "prepare_export_rows",
    "prepare_hazard_rows",
    "prepare_hazards",
    "prepare_headline_summary",
    "prepare_map_markers",
    "prepare_methods_text",
    "render_backtest",
    "render_bridge_detail",
    "render_budget",
    "render_current_ranking",
    "render_export",
    "render_hazards",
    "render_headline",
    "render_map",
    "render_methods",
    "render_mode_notice",
    "render_primary_views",
    "score_to_color",
]
