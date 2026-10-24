"""Focused checks for explicit, isolated fabricated demo providers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from app.config import ProviderModeMismatch, RuntimeMode
from app.demo import (
    DEMO_INDICATOR,
    DEMO_PROVENANCE,
    DemoBacktestResult,
    DemoCurrentRanking,
    DemoSelectionResult,
    DemoSourceBundle,
    build_demo_providers,
)
from app.views import render_mode_notice


def _demo_panel() -> pd.DataFrame:
    rows = [
        {
            "structure_id": "DEMO-A",
            "year": 2022,
            "county": "Hampshire",
            "material": "Steel",
            "age": 60,
            "adt": 12_000,
            "deck": 5,
            "super": 4,
            "sub": 5,
            "worst_rating": 4,
            "insp_freq_months": 24,
            "fracture_critical": True,
            "underwater": False,
            "year_recon": 0,
            "rebuilt": False,
            "lat": 42.31,
            "lon": -72.64,
            "carries": "Demo Route 1",
            "crosses": "Demo River",
            "improvement_cost": 2_000_000,
            "insp_date": pd.Timestamp("2022-04-01"),
        },
        {
            "structure_id": "DEMO-B",
            "year": 2022,
            "county": "Berkshire",
            "material": "Concrete",
            "age": 25,
            "adt": 3_500,
            "deck": 7,
            "super": 7,
            "sub": 6,
            "worst_rating": 6,
            "insp_freq_months": 24,
            "fracture_critical": False,
            "underwater": True,
            "year_recon": 2008,
            "rebuilt": False,
            "lat": 42.45,
            "lon": -73.2,
            "carries": "Demo Route 2",
            "crosses": "Demo Road",
            "improvement_cost": 500_000,
            "insp_date": pd.Timestamp("2022-03-01"),
        },
    ]
    return pd.DataFrame(rows)


class _RecordingUI:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def warning(self, value: object) -> None:
        self.events.append(("warning", value))


def _assert_demo_tag(value: object) -> None:
    assert value.mode is RuntimeMode.DEMO
    assert value.provenance == DEMO_PROVENANCE
    assert value.demo_indicator == DEMO_INDICATOR


def test_demo_factory_requires_exact_mode_and_construction_is_lazy() -> None:
    calls: list[Path] = []

    def loader(path: Path) -> pd.DataFrame:
        calls.append(path)
        return _demo_panel()

    for invalid_mode in (RuntimeMode.PRODUCTION, "demo", None):
        with pytest.raises(ProviderModeMismatch):
            build_demo_providers(invalid_mode, panel_loader=loader)  # type: ignore[arg-type]

    providers = build_demo_providers(RuntimeMode.DEMO, panel_loader=loader)
    _assert_demo_tag(providers)
    assert calls == []

    source = providers.load_source()
    _assert_demo_tag(source)
    assert calls == [providers.panel_path]
    assert source.source.attrs["provenance"] == DEMO_PROVENANCE
    assert source.adapted.attrs["runtime_mode"] is RuntimeMode.DEMO


def test_demo_values_build_the_shared_view_context_with_visible_indicator() -> None:
    providers = build_demo_providers(
        RuntimeMode.DEMO,
        panel_loader=lambda _path: _demo_panel(),
    )
    source = providers.load_source()
    ranking = providers.build_current_ranking(source)
    selected = providers.select_current(ranking, 100_000)
    backtest = providers.run_backtest((100_000, 150_000))

    assert isinstance(source, DemoSourceBundle)
    assert isinstance(ranking, DemoCurrentRanking)
    assert isinstance(selected, DemoSelectionResult)
    assert isinstance(backtest, DemoBacktestResult)
    for value in (source, ranking, selected, backtest):
        _assert_demo_tag(value)

    assert tuple(ranking.scored.columns) == (
        "id",
        "p_worse",
        "consequence",
        "insp_cost",
        "score",
        "reason",
    )
    assert ranking.scored["reason"].str.contains(
        "modeled inspection-cost estimate", regex=False
    ).all()
    assert tuple(backtest.result.columns) == ("budget", "method", "caught")

    context = providers.build_view_context(
        100_000,
        backtest_budgets=(100_000, 150_000),
    )
    assert context.mode is RuntimeMode.DEMO
    assert context.provenance == DEMO_PROVENANCE
    assert context.current_ranking.population_size == 2
    assert set(context.selected_records["structure_id"]) == {"DEMO-A", "DEMO-B"}

    ui = _RecordingUI()
    assert render_mode_notice(context, ui=ui)
    assert ui.events == [("warning", "DEMO MODE - FABRICATED DATA")]


def test_demo_scores_selections_and_backtests_are_deterministic() -> None:
    providers = build_demo_providers(
        RuntimeMode.DEMO,
        panel_loader=lambda _path: _demo_panel(),
    )
    source = providers.load_source()
    first_ranking = providers.build_current_ranking(source)
    second_ranking = providers.build_current_ranking(source)
    pd.testing.assert_frame_equal(first_ranking.scored, second_ranking.scored)

    first_selection = providers.select_current(first_ranking, 100_000)
    second_selection = providers.select_current(second_ranking, 100_000)
    assert first_selection.selection == second_selection.selection
    assert first_selection.metrics == second_selection.metrics

    first_backtest = providers.run_backtest((150_000, 100_000, 100_000))
    second_backtest = providers.run_backtest((100_000, 150_000))
    pd.testing.assert_frame_equal(first_backtest.result, second_backtest.result)


def test_import_and_provider_construction_do_not_import_production_packages() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc
import sys

class BlockProduction(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'model' or fullname.startswith('model.'):
            raise RuntimeError('production model import attempted')
        if fullname == 'planner' or fullname.startswith('planner.'):
            raise RuntimeError('production planner import attempted')
        return None

sys.meta_path.insert(0, BlockProduction())
from app.config import RuntimeMode
from app.demo import build_demo_providers
providers = build_demo_providers(RuntimeMode.DEMO, panel_loader=lambda path: None)
assert providers.mode is RuntimeMode.DEMO
assert 'model' not in sys.modules
assert 'planner' not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_legacy_modules_cannot_fallback_to_fabricated_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.data as legacy_data
    import app.stubs as retired_stubs

    real_path = tmp_path / "bridges" / "panel.parquet"
    fake_path = tmp_path / "bridges" / "fake.parquet"
    fake_path.parent.mkdir(parents=True)
    fake_path.write_bytes(b"fabricated")
    monkeypatch.setattr(legacy_data, "PANEL_PATH", real_path)

    assert legacy_data._panel_path() == real_path
    assert not legacy_data.using_fake()
    assert retired_stubs.__all__ == ()
    for name in ("score", "pick", "backtest"):
        with pytest.raises(AttributeError, match="exact demo-mode resolution"):
            getattr(retired_stubs, name)
