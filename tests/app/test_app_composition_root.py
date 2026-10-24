"""Focused tests for the task 8.1 Streamlit composition root."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app import app as composition
from app.config import Provenance, RuntimeMode
from app.errors import ErrorCategory
from app.state import SelectionContext
from planner import Selection, SelectionMetrics


@dataclass(frozen=True)
class _TaggedProvider:
    mode: RuntimeMode

    @property
    def provenance(self) -> Provenance:
        prefix = "Real" if self.mode is RuntimeMode.PRODUCTION else "Demo"
        return Provenance(
            mode=self.mode,
            data_source=f"{prefix}_Dataset",
            model_source=f"{prefix}_Model_Service",
            planner_source=f"{prefix}_Planner",
            backtest_source=f"{prefix}_Backtest",
        )


def _selection_context() -> SelectionContext:
    selection = Selection(
        mode="budget",
        constraint=Decimal("500000"),
        ids=("A",),
    )
    return SelectionContext(
        active_budget=500_000,
        selection=selection,
        metrics=SelectionMetrics(
            selected_count=1,
            selected_total_cost=Decimal("3000"),
            expected_safety_drop_coverage=0.25,
        ),
        selected_records=pd.DataFrame({"structure_id": ["A"]}),
        score_identity="score-v1",
    )


def test_repository_bootstrap_supports_foreign_caller_directory(
    tmp_path: Path,
) -> None:
    app_path = Path(composition.__file__).resolve()
    repository_root = app_path.parents[1]
    script = (
        "import pathlib, runpy, sys; "
        f"namespace = runpy.run_path({str(app_path)!r}, run_name='composition_import'); "
        f"expected = pathlib.Path({str(repository_root)!r}); "
        "assert namespace['REPOSITORY_ROOT'] == expected; "
        "assert str(expected) in sys.path"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_provider_factory_constructs_only_the_resolved_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.demo as demo
    import app.services as services

    production = _TaggedProvider(RuntimeMode.PRODUCTION)
    demo_provider = _TaggedProvider(RuntimeMode.DEMO)
    calls: list[str] = []

    monkeypatch.setattr(
        services,
        "build_production_provider",
        lambda: calls.append("production") or production,
    )
    monkeypatch.setattr(
        demo,
        "build_demo_providers",
        lambda mode: calls.append("demo") or demo_provider,
    )

    assert composition._build_provider(RuntimeMode.PRODUCTION) is production
    assert calls == ["production"]

    calls.clear()
    assert composition._build_provider(RuntimeMode.DEMO) is demo_provider
    assert calls == ["demo"]


def test_county_projection_does_not_request_another_selection() -> None:
    selection_context = _selection_context()
    calls: list[tuple[object, int]] = []

    class Provider:
        def build_selection_context(
            self,
            ranking: object,
            budget: int,
        ) -> SelectionContext:
            calls.append((ranking, budget))
            return selection_context

    ranking = SimpleNamespace(
        records=pd.DataFrame(
            {"structure_id": ["A", "B"], "county": ["Worcester", "Berkshire"]}
        )
    )
    selected = composition._build_statewide_selection(
        Provider(),
        RuntimeMode.PRODUCTION,
        ranking,
        500_000,
    )

    assert selected is selection_context
    assert composition._county_options(ranking) == (
        "All",
        "Berkshire",
        "Worcester",
    )
    assert calls == [(ranking, 500_000)]


def test_compound_source_boundary_maps_schema_failure_to_safe_diagnostic() -> None:
    class SourceSchemaError(Exception):
        pass

    def fail() -> object:
        raise SourceSchemaError("secret path C:/private/panel.parquet")

    outcome = composition._run_classified_component(
        fail,
        ErrorCategory.DATA_LOAD,
        composition._SOURCE_ERROR_CATEGORIES,
    )

    assert outcome.is_failure
    assert outcome.error is not None
    assert outcome.error.as_public_fields() == {
        "component": "Schema_Adapter",
        "summary": "Production bridge data failed validation.",
        "code": "SCHEMA-001",
    }
    assert "private" not in str(outcome.error.as_public_fields())
