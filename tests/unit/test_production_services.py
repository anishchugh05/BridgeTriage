"""Focused production-service examples for task 6.1."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.config import RuntimeMode
from app.services import (
    BacktestResultCacheValidationError,
    BacktestService,
    ModelCache,
    ModelStateCacheIdentity,
    PRODUCTION_PROVENANCE,
    PRODUCTION_SOURCE_PATH,
    PRODUCTION_SOURCE_PATH_IDENTITY,
    PanelBundle,
    ProductionProvider,
    SOURCE_SCHEMA_VERSION,
    CurrentRanking,
    build_production_provider,
    bytes_content_digest,
    configuration_fingerprint,
)
from model import SOURCE_COLUMNS
from planner import BACKTEST_METHODS


class MemoryCachedFunction:
    """Minimal CachedFunc-compatible backend with observable computations."""

    def __init__(self) -> None:
        self.values: dict[object, Any] = {}
        self.compute_calls = 0

    def __call__(self, identity: object, _operation=None):
        if identity in self.values:
            return self.values[identity]
        self.compute_calls += 1
        value = _operation()
        self.values[identity] = value
        return value

    def clear(self, identity: object) -> None:
        self.values.pop(identity, None)


def _cache() -> tuple[ModelCache[Any], MemoryCachedFunction, MemoryCachedFunction]:
    data = MemoryCachedFunction()
    resource = MemoryCachedFunction()
    return (
        ModelCache(
            data_cache_function=data,
            resource_cache_function=resource,
        ),
        data,
        resource,
    )


def _valid_source() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, rating in ((2012, 8), (2016, 7), (2018, 6), (2022, 5)):
        rows.append(
            {
                "structure_id": "A",
                "year": year,
                "county": "Test County",
                "material": "steel",
                "age": year - 2000,
                "adt": 1000,
                "deck": rating,
                "super": rating,
                "sub": rating,
                "worst_rating": rating,
                "insp_freq_months": 24,
                "fracture_critical": False,
                "underwater": False,
                "year_recon": 0,
                "rebuilt": False,
                "lat": 42.0,
                "lon": -71.0,
                "carries": "Route 1",
                "crosses": "Test River",
                "improvement_cost": None,
                "insp_date": pd.Timestamp(year=year, month=4, day=1),
            }
        )
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def _bundle() -> PanelBundle:
    return PanelBundle(
        source=_valid_source(),
        dataset_sha256=bytes_content_digest(b"validated-source"),
        mode=RuntimeMode.PRODUCTION,
    )


def _model_identity() -> ModelStateCacheIdentity:
    return ModelStateCacheIdentity(
        mode=RuntimeMode.PRODUCTION,
        source_sha256=bytes_content_digest(b"validated-source"),
        start_year=2018,
        end_year=2022,
        model_family="fit_decline",
        model_config_fingerprint=configuration_fingerprint(
            {"contract": "test-current-model"}
        ),
    )


def _scored() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["A", "B", "C"],
            "p_worse": [0.8, 0.5, 0.2],
            "consequence": [600_000.0, 300_000.0, 1_000_000.0],
            "insp_cost": [60_000.0, 60_000.0, 50_000.0],
            "score": [8.0, 2.5, 4.0],
            "reason": [
                "modeled inspection-cost estimate" for _ in range(3)
            ],
        }
    )


def test_panel_bundle_validates_once_and_exposes_only_defensive_frame_copies() -> None:
    source = _valid_source()
    bundle = PanelBundle(
        source=source,
        dataset_sha256=bytes_content_digest(b"panel"),
        mode=RuntimeMode.PRODUCTION,
    )

    source.loc[:, "county"] = "mutated input"
    source_copy = bundle.source
    adapted_copy = bundle.adapted
    source_copy.loc[:, "county"] = "mutated output"
    adapted_copy.loc[:, "county"] = "mutated output"

    assert bundle.source["county"].eq("Test County").all()
    assert bundle.adapted["county"].eq("Test County").all()
    assert "id" in bundle.adapted.columns
    assert "structure_id" not in bundle.adapted.columns
    assert "insp_freq_mo" in bundle.adapted.columns
    assert "year_built" in bundle.adapted.columns
    with pytest.raises(FrozenInstanceError):
        bundle.mode = RuntimeMode.DEMO  # type: ignore[misc]


def test_production_provider_is_lazy_and_carries_only_exact_real_provenance() -> None:
    cache, data_cache, resource_cache = _cache()
    provider = build_production_provider(cache)

    assert isinstance(provider, ProductionProvider)
    assert provider.mode is RuntimeMode.PRODUCTION
    assert provider.provenance is PRODUCTION_PROVENANCE
    assert provider.provenance.data_source == "Real_Dataset"
    assert provider.provenance.model_source == "Real_Model_Service"
    assert provider.provenance.planner_source == "Real_Planner"
    assert provider.provenance.backtest_source == "Backtest_Engine"
    assert data_cache.compute_calls == 0
    assert resource_cache.compute_calls == 0


def test_real_panel_source_loading_is_repository_relative_and_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache, data_cache, resource_cache = _cache()
    provider = ProductionProvider(cache)
    monkeypatch.chdir(tmp_path)

    first = provider.load_source_bundle()
    second = provider.load_source_bundle()

    assert PRODUCTION_SOURCE_PATH.is_file()
    assert first.mode is RuntimeMode.PRODUCTION
    assert first.source_path_identity == PRODUCTION_SOURCE_PATH_IDENTITY
    assert first.schema_version == SOURCE_SCHEMA_VERSION
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.row_count == len(first.adapted) > 0
    assert set(first.source["year"]) == {2012, 2016, 2018, 2022}
    assert data_cache.compute_calls == 1
    assert resource_cache.compute_calls == 0


def test_production_selection_builds_one_deterministic_shared_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scored = _scored()
    records = pd.DataFrame(
        {
            "structure_id": ["C", "A", "B"],
            "county": ["Three", "One", "Two"],
        }
    )
    ranking = CurrentRanking(_model_identity(), scored, records)
    provider = ProductionProvider(_cache()[0])

    import planner

    real_select = planner.select
    calls = 0

    def recording_select(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return real_select(*args, **kwargs)

    monkeypatch.setattr(planner, "select", recording_select)
    context = provider.build_selection_context(ranking, 100_000)

    assert calls == 1
    assert context.selection.ids == ("A",)
    assert context.metrics.selected_count == 1
    assert context.metrics.selected_total_cost == Decimal("60000.0")
    assert context.selected_records["structure_id"].tolist() == ["A"]
    assert context.active_budget == 100_000


def test_backtest_service_reuses_canonical_identity_and_rejects_invalid_output() -> None:
    cache, data_cache, _resource_cache = _cache()
    calls: list[tuple[Decimal, ...]] = []

    def valid_run(
        _panel: pd.DataFrame,
        budgets: object,
    ) -> pd.DataFrame:
        canonical = tuple(budgets)  # type: ignore[arg-type]
        calls.append(canonical)
        return pd.DataFrame(
            [
                {"budget": budget, "method": method, "caught": 0}
                for budget in canonical
                for method in BACKTEST_METHODS
            ],
            columns=["budget", "method", "caught"],
        )

    service = BacktestService(cache, run_fn=valid_run)
    bundle = _bundle()
    first_identity = service.create_identity(
        bundle, [500_000, 100_000, 500_000]
    )

    first = service.run(bundle, [500_000, 100_000, 500_000])
    second = service.run(bundle, [100_000.0, 500_000])

    assert first.equals(second)
    assert calls == [(Decimal("1E+5"), Decimal("5E+5"))]
    assert data_cache.compute_calls == 1
    assert first_identity.mode is RuntimeMode.PRODUCTION
    assert first_identity.fit_start_year == 2012
    assert first_identity.fit_end_year == 2016
    assert first_identity.candidate_year == 2018
    assert first_identity.outcome_year == 2022
    assert first_identity.as_of.isoformat() == "2018-04-01"

    invalid_cache, invalid_data_cache, _ = _cache()
    invalid_service = BacktestService(
        invalid_cache,
        run_fn=lambda _panel, _budgets: pd.DataFrame(
            {"budget": [100_000], "method": ["ours"], "wrong": [0]}
        ),
    )
    invalid_identity = invalid_service.create_identity(bundle, [100_000])
    with pytest.raises(BacktestResultCacheValidationError):
        invalid_service.run(bundle, [100_000])
    assert invalid_identity not in invalid_data_cache.values
