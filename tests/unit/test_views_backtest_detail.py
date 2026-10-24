"""Focused examples for task 7.4 backtest and bridge-detail views."""

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
    BACKTEST_METHODS,
    BRIDGE_DETAIL_YEARS,
    DEMO_INDICATOR,
    UNAVAILABLE,
    backtest_counts_at_budget,
    build_infobridge_url,
    format_backtest_comparison,
    prepare_backtest_chart,
    prepare_backtest_comparisons,
    prepare_bridge_detail,
    prepare_bridge_history,
    render_backtest,
    render_bridge_detail,
)
from planner import Selection, SelectionMetrics


@dataclass
class _RecordingUI:
    events: list[tuple[Any, ...]]

    def warning(self, message: str) -> None:
        self.events.append(("warning", message))

    def subheader(self, label: str) -> None:
        self.events.append(("subheader", label))

    def line_chart(self, frame: pd.DataFrame, **kwargs: object) -> None:
        self.events.append(("line_chart", frame.copy(deep=True), kwargs))

    def caption(self, body: str) -> None:
        self.events.append(("caption", body))

    def markdown(self, body: str, **kwargs: object) -> None:
        self.events.append(("markdown", body, kwargs))

    def dataframe(self, frame: pd.DataFrame, **kwargs: object) -> None:
        self.events.append(("dataframe", frame.copy(deep=True), kwargs))

    def link_button(self, label: str, url: str) -> None:
        self.events.append(("link_button", label, url))

    def selectbox(
        self,
        label: str,
        options: tuple[str, ...],
        *,
        index: int,
    ) -> str:
        self.events.append(("selectbox", label, options, index))
        return options[index]

    def info(self, message: str) -> None:
        self.events.append(("info", message))


def _backtest_result() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"budget": 500_000, "method": "ours", "caught": 4},
            {"budget": 500_000, "method": "worst_first", "caught": 2},
            {"budget": 500_000, "method": "oldest_first", "caught": 0},
            {"budget": 550_000, "method": "ours", "caught": 1},
            {"budget": 550_000, "method": "worst_first", "caught": 1},
            {"budget": 550_000, "method": "oldest_first", "caught": 2},
        ],
        columns=["budget", "method", "caught"],
    )


def _context(mode: RuntimeMode = RuntimeMode.PRODUCTION) -> ViewContext:
    selected_id = "A/1 2"
    ranking = pd.DataFrame(
        [
            {
                "structure_id": selected_id,
                "score": 0.8,
                "reason": "Measured model reason",
                "carries": "MAIN STREET",
                "crosses": "PIONEER RIVER",
                "county": "Hampshire",
            },
            {
                "structure_id": "B",
                "score": 0.4,
                "reason": "Another measured reason",
                "carries": "SECOND STREET",
                "crosses": "RAILROAD",
                "county": "Berkshire",
            },
        ]
    )
    source = pd.DataFrame(
        [
            {
                "structure_id": selected_id,
                "year": 2012,
                "deck": 7,
                "super": None,
                "sub": 6,
            },
            {
                "structure_id": selected_id,
                "year": 2018,
                "deck": 5,
                "super": 4,
                "sub": None,
            },
            {
                "structure_id": selected_id,
                "year": 2022,
                "deck": 4,
                "super": 4,
                "sub": 3,
            },
        ]
    )
    selection = Selection(
        mode="budget",
        constraint=Decimal("500000"),
        ids=(selected_id,),
    )
    selection_context = SelectionContext(
        active_budget=500_000,
        selection=selection,
        metrics=SelectionMetrics(
            selected_count=1,
            selected_total_cost=Decimal("100000"),
            expected_safety_drop_coverage=0.5,
        ),
        selected_records=ranking.loc[
            ranking["structure_id"].eq(selected_id)
        ],
        score_identity="view-task-7.4",
    )
    prefix = "Demo" if mode is RuntimeMode.DEMO else "Real"
    return ViewContext(
        mode=mode,
        provenance=Provenance(
            mode=mode,
            data_source=f"{prefix}_Dataset",
            model_source=f"{prefix}_Model_Service",
            planner_source=f"{prefix}_Planner",
            backtest_source=f"{prefix}_Backtest_Engine",
        ),
        source=SimpleNamespace(source=source),
        current_ranking=SimpleNamespace(records=ranking),
        selection_context=selection_context,
        backtest=_backtest_result(),
    )


def test_backtest_chart_and_comparisons_use_unmodified_same_budget_counts() -> None:
    context = _context()

    chart = prepare_backtest_chart(context)

    assert chart.index.tolist() == [500_000, 550_000]
    assert tuple(chart.columns) == BACKTEST_METHODS
    assert chart.loc[500_000].to_dict() == {
        "ours": 4,
        "worst_first": 2,
        "oldest_first": 0,
    }
    assert chart.loc[550_000].to_dict() == {
        "ours": 1,
        "worst_first": 1,
        "oldest_first": 2,
    }
    assert backtest_counts_at_budget(context, 500_000) == {
        "ours": 4,
        "worst_first": 2,
        "oldest_first": 0,
    }

    positive, zero = prepare_backtest_comparisons(context, 500_000)
    assert "`ours` caught 4" in positive
    assert "`worst_first` caught 2" in positive
    assert "Difference: 2 more declines caught" in positive
    assert "ratio: 4/2 = 2×" in positive
    assert "`ours` caught 4" in zero
    assert "`oldest_first` caught 0" in zero
    assert "Absolute difference: 4" in zero
    assert "ratio" not in zero.lower()

    tied = format_backtest_comparison(
        budget=550_000,
        ours_caught=1,
        baseline_caught=1,
        baseline_method="worst_first",
    )
    lost = format_backtest_comparison(
        budget=550_000,
        ours_caught=1,
        baseline_caught=2,
        baseline_method="oldest_first",
    )
    assert "`ours` caught 1" in tied and "`worst_first` caught 1" in tied
    assert "`ours` caught 1" in lost and "`oldest_first` caught 2" in lost
    assert "more declines caught" not in tied
    assert "more declines caught" not in lost


def test_bridge_detail_preserves_sparse_history_and_encoded_identifier() -> None:
    context = _context()
    selected_id = "A/1 2"

    history = prepare_bridge_history(context, selected_id)
    detail = prepare_bridge_detail(context, selected_id)

    assert history.index.tolist() == list(BRIDGE_DETAIL_YEARS)
    assert history.columns.tolist() == ["deck", "super", "sub"]
    assert history.at[2012, "deck"] == 7
    assert history.at[2012, "super"] == UNAVAILABLE
    assert history.loc[2016].tolist() == [UNAVAILABLE, UNAVAILABLE, UNAVAILABLE]
    assert history.at[2018, "sub"] == UNAVAILABLE
    assert history.at[2022, "sub"] == 3
    assert_frame_equal(detail.history, history)
    assert detail.reason == "Measured model reason"
    assert detail.carries == "MAIN STREET"
    assert detail.crosses == "PIONEER RIVER"
    assert detail.county == "Hampshire"
    assert detail.infobridge_url == build_infobridge_url(selected_id)
    assert "A%2F1%202" in detail.infobridge_url
    assert selected_id not in detail.infobridge_url


def test_both_data_renderers_place_exact_demo_indicator_first() -> None:
    context = _context(RuntimeMode.DEMO)

    backtest_ui = _RecordingUI([])
    rendered_chart = render_backtest(context, ui=backtest_ui)
    assert backtest_ui.events[0] == ("warning", DEMO_INDICATOR)
    plotted = next(event[1] for event in backtest_ui.events if event[0] == "line_chart")
    assert_frame_equal(plotted, rendered_chart)

    detail_ui = _RecordingUI([])
    detail = render_bridge_detail(context, structure_id="A/1 2", ui=detail_ui)
    assert detail_ui.events[0] == ("warning", DEMO_INDICATOR)
    assert detail is not None
    assert (
        "link_button",
        "Open in FHWA InfoBridge",
        detail.infobridge_url,
    ) in detail_ui.events
