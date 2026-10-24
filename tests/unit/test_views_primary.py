"""Focused examples for task 7.3 current-ranking presentation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from app.config import Provenance, RuntimeMode
from app.state import SelectionContext, ViewContext
from app.views import (
    BUDGET_TABLE_COLUMNS,
    DEMO_INDICATOR,
    MAP_TOOLTIP,
    MODELED_COST_COLUMN_LABEL,
    PRIMARY_VIEW_ORDER,
    SECTION_LABELS,
    SELECTED_MARKER_COLOR,
    adt_to_radius,
    is_map_eligible,
    normalize_score_values,
    prepare_budget_rows,
    prepare_current_ranking_summary,
    prepare_map_markers,
    render_budget,
    render_current_ranking,
    render_map,
    render_mode_notice,
    render_primary_views,
)
from planner import Selection, SelectionMetrics


@dataclass
class _MetricColumn:
    events: list[tuple[Any, ...]]

    def metric(self, label: str, value: object) -> None:
        self.events.append(("metric", label, value))


class _RecordingUI:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def warning(self, message: str) -> None:
        self.events.append(("warning", message))

    def header(self, label: str) -> None:
        self.events.append(("header", label))

    def subheader(self, label: str) -> None:
        self.events.append(("subheader", label))

    def markdown(self, body: str, **kwargs: object) -> None:
        self.events.append(("markdown", body, kwargs))

    def caption(self, body: str) -> None:
        self.events.append(("caption", body))

    def columns(self, count: int) -> list[_MetricColumn]:
        self.events.append(("columns", count))
        return [_MetricColumn(self.events) for _ in range(count)]

    def pydeck_chart(self, deck: object, **kwargs: object) -> None:
        self.events.append(("pydeck_chart", deck, kwargs))

    def dataframe(self, frame: pd.DataFrame, **kwargs: object) -> None:
        self.events.append(("dataframe", frame.copy(deep=True), kwargs))


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "structure_id": "A",
                "county": "Alpha",
                "lat": 41.0,
                "lon": -74.0,
                "adt": 100,
                "worst_rating": 5.0,
                "underwater": False,
                "fracture_critical": False,
                "carries": "A ROAD",
                "crosses": "A RIVER",
                "p_worse": 0.25,
                "consequence": 10.0,
                "insp_cost": 100_000.0,
                "score": 0.25,
                "reason": "A reason; modeled inspection-cost estimate $100,000",
            },
            {
                "structure_id": "B",
                "county": "Beta",
                "lat": 43.0,
                "lon": -69.5,
                "adt": 400,
                "worst_rating": 4.0,
                "underwater": True,
                "fracture_critical": False,
                "carries": "B ROAD",
                "crosses": "B RIVER",
                "p_worse": 0.75,
                "consequence": 20.0,
                "insp_cost": 125_000.0,
                "score": 0.75,
                "reason": "B reason; modeled inspection-cost estimate $125,000",
            },
            {
                "structure_id": "C",
                "county": "Alpha",
                "lat": 42.2,
                "lon": -71.0,
                "adt": 400,
                "worst_rating": 6.0,
                "underwater": False,
                "fracture_critical": False,
                "carries": "C ROAD",
                "crosses": "C RAIL",
                "p_worse": 0.4,
                "consequence": 12.0,
                "insp_cost": 90_000.0,
                "score": 0.6,
                "reason": "C reason; modeled inspection-cost estimate $90,000",
            },
            {
                "structure_id": "D",
                "county": "Alpha",
                "lat": 42.4,
                "lon": -71.4,
                "adt": 0,
                "worst_rating": 7.0,
                "underwater": False,
                "fracture_critical": False,
                "carries": "D ROAD",
                "crosses": "D ROAD",
                "p_worse": 0.1,
                "consequence": 8.0,
                "insp_cost": 80_000.0,
                "score": 0.1,
                "reason": "D reason; modeled inspection-cost estimate $80,000",
            },
            {
                "structure_id": "E",
                "county": "Alpha",
                "lat": None,
                "lon": -71.5,
                "adt": 900,
                "worst_rating": 3.0,
                "underwater": True,
                "fracture_critical": True,
                "carries": "E ROAD",
                "crosses": "E RIVER",
                "p_worse": 0.9,
                "consequence": 30.0,
                "insp_cost": 130_000.0,
                "score": 0.9,
                "reason": "E reason; modeled inspection-cost estimate $130,000",
            },
            {
                "structure_id": "F",
                "county": "Beta",
                "lat": 43.0001,
                "lon": -70.0,
                "adt": 1_600,
                "worst_rating": 5.0,
                "underwater": False,
                "fracture_critical": True,
                "carries": "F ROAD",
                "crosses": "F RIVER",
                "p_worse": 0.5,
                "consequence": 11.0,
                "insp_cost": 95_000.0,
                "score": 0.5,
                "reason": "F reason; modeled inspection-cost estimate $95,000",
            },
        ]
    )


def _context(mode: RuntimeMode = RuntimeMode.PRODUCTION) -> ViewContext:
    records = _records()
    selected = records.loc[records["structure_id"].isin(("A", "B"))].copy()
    selection = Selection(
        mode="budget",
        constraint=Decimal("500000"),
        ids=("A", "B"),
    )
    metrics = SelectionMetrics(
        selected_count=2,
        selected_total_cost=Decimal("225000"),
        expected_safety_drop_coverage=1.0,
    )
    selection_context = SelectionContext(
        active_budget=500_000,
        selection=selection,
        metrics=metrics,
        selected_records=selected,
        score_identity="ranking-score-v1",
    )
    prefix = "Demo" if mode is RuntimeMode.DEMO else "Real"
    provenance = Provenance(
        mode=mode,
        data_source=f"{prefix}_Dataset",
        model_source=f"{prefix}_Model_Service",
        planner_source=f"{prefix}_Planner",
        backtest_source=f"{prefix}_Backtest_Engine",
    )
    return ViewContext(
        mode=mode,
        provenance=provenance,
        source=object(),
        current_ranking=SimpleNamespace(records=records),
        selection_context=selection_context,
    )


def test_demo_notice_is_exact_and_production_emits_nothing() -> None:
    demo_ui = _RecordingUI()
    production_ui = _RecordingUI()

    assert render_mode_notice(_context(RuntimeMode.DEMO), ui=demo_ui)
    assert demo_ui.events == [("warning", DEMO_INDICATOR)]
    assert not render_mode_notice(_context(), ui=production_ui)
    assert production_ui.events == []


def test_headline_uses_shared_metrics_and_complete_population() -> None:
    context = _context()
    summary = prepare_current_ranking_summary(context)

    assert summary.active_budget == 500_000
    assert summary.selected_count == 2
    assert summary.ranking_population_count == 6
    assert summary.selected_total_cost == Decimal("225000")
    assert summary.expected_safety_drop_coverage == 1.0
    assert summary.selection_identity == context.selection_identity

    ui = _RecordingUI()
    rendered = render_current_ranking(context, ui=ui)
    metric_events = [event for event in ui.events if event[0] == "metric"]

    assert rendered == summary
    assert ("metric", "Selected bridges", 2) in metric_events
    assert ("metric", "Ranking population", 6) in metric_events
    assert (
        "metric",
        "Selected total modeled inspection-cost estimate",
        "$225,000",
    ) in metric_events
    assert ("metric", "Expected Safety_Drop coverage", 1.0) in metric_events


def test_map_projection_applies_only_exact_coordinate_and_county_predicates() -> None:
    context = _context()
    ranking_before = context.current_ranking.records.copy(deep=True)
    selection_identity_before = context.selection_identity

    statewide = prepare_map_markers(context)
    alpha = prepare_map_markers(context, county="Alpha")

    assert statewide["structure_id"].tolist() == ["A", "B", "C", "D"]
    assert alpha["structure_id"].tolist() == ["A", "C", "D"]
    assert context.selection.ids == ("A", "B")
    assert context.selection_identity == selection_identity_before
    assert_frame_equal(context.current_ranking.records, ranking_before)

    assert is_map_eligible(41.0, -74.0)
    assert is_map_eligible(43.0, -69.5)
    assert not is_map_eligible(None, -71.0)
    assert not is_map_eligible(float("nan"), -71.0)
    assert not is_map_eligible(43.0001, -71.0)
    assert not is_map_eligible(42.0, -74.0001)


def test_map_encodings_are_monotonic_equal_preserving_and_tooltip_exact() -> None:
    markers = prepare_map_markers(_context())
    by_id = markers.set_index("structure_id")

    assert by_id.at["A", "fill_color"] == SELECTED_MARKER_COLOR
    assert by_id.at["B", "fill_color"] == SELECTED_MARKER_COLOR
    assert by_id.at["D", "score_color_value"] < by_id.at[
        "C", "score_color_value"
    ]
    assert normalize_score_values([0.2, 0.2]) == (0.5, 0.5)
    assert by_id.at["B", "radius"] == by_id.at["C", "radius"]
    assert adt_to_radius(0) < adt_to_radius(100) < adt_to_radius(400)

    tooltip = MAP_TOOLTIP["text"]
    for field in ("structure_id", "carries", "crosses", "county", "score"):
        assert "{" + field + "}" in tooltip
    assert by_id.at["C", "carries"] == "C ROAD"
    assert by_id.at["C", "crosses"] == "C RAIL"
    assert by_id.at["C", "county"] == "Alpha"
    assert by_id.at["C", "score"] == 0.6


def test_budget_rows_use_shared_selection_exact_values_and_canonical_order() -> None:
    context = _context()
    selected_before = context.selected_records

    rows = prepare_budget_rows(context)

    assert tuple(rows.columns) == BUDGET_TABLE_COLUMNS
    assert rows["structure_id"].tolist() == ["B", "A"]
    assert rows["hazard"].tolist() == [True, False]
    assert rows.set_index("structure_id").at["B", "insp_cost"] == 125_000.0
    assert rows.set_index("structure_id").at["A", "reason"] == (
        "A reason; modeled inspection-cost estimate $100,000"
    )
    assert_frame_equal(context.selected_records, selected_before)

    ui = _RecordingUI()
    rendered = render_budget(context, ui=ui)
    dataframe_event = next(event for event in ui.events if event[0] == "dataframe")
    caption = next(event[1] for event in ui.events if event[0] == "caption")

    assert_frame_equal(rendered, rows)
    assert dataframe_event[2]["column_config"] == {
        "insp_cost": MODELED_COST_COLUMN_LABEL
    }
    assert "modeled inspection-cost estimate" in caption
    assert "not a quote or repair cost" in caption


def test_each_renderer_guards_demo_output_and_primary_order_is_explicit() -> None:
    context = _context(RuntimeMode.DEMO)
    for renderer in (render_current_ranking, render_map, render_budget):
        ui = _RecordingUI()
        renderer(context, ui=ui)
        assert ui.events[0] == ("warning", DEMO_INDICATOR)

    ui = _RecordingUI()
    render_primary_views(context, county="Alpha", ui=ui)
    header_position = ui.events.index(("header", "Current ranking"))
    map_position = ui.events.index(("subheader", "Map"))

    assert PRIMARY_VIEW_ORDER == ("Current ranking", "Map")
    assert SECTION_LABELS == (
        "Map",
        "Budget",
        "Backtest",
        "Bridge detail",
        "Hazards",
        "Export",
        "Methods",
    )
    assert header_position < map_position
