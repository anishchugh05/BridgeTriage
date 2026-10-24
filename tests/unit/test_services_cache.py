"""Focused examples for task 3.3 cache identities and transactions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from typing import Any

import pandas as pd
import pytest

from app.config import RuntimeMode
from app.services import (
    BacktestCacheIdentity,
    CacheValidationError,
    ModelCache,
    ModelStateCacheIdentity,
    ScoredTableCacheIdentity,
    SourceCacheIdentity,
    bytes_content_digest,
    configuration_fingerprint,
    dataframe_content_digest,
)


class MemoryCachedFunction:
    """Small CachedFunc-compatible backend used to observe transitions."""

    def __init__(self) -> None:
        self.values: dict[object, Any] = {}
        self.compute_calls = 0
        self.cleared: list[object] = []

    def __call__(self, identity: object, _operation=None):
        if identity in self.values:
            return self.values[identity]
        self.compute_calls += 1
        value = _operation()
        self.values[identity] = value
        return value

    def clear(self, identity: object) -> None:
        self.cleared.append(identity)
        self.values.pop(identity, None)


def source_identity(
    mode: RuntimeMode = RuntimeMode.PRODUCTION,
    *,
    label: bytes = b"source-a",
) -> SourceCacheIdentity:
    return SourceCacheIdentity.from_content(
        mode,
        "bridges/panel.parquet",
        label,
        "source-schema-v1",
    )


def model_identity(
    mode: RuntimeMode = RuntimeMode.PRODUCTION,
) -> ModelStateCacheIdentity:
    return ModelStateCacheIdentity.create(
        mode=mode,
        source_sha256=source_identity(mode).source_sha256,
        start_year=2018,
        end_year=2022,
        model_configuration={"horizon_months": 48, "smoothing": 1.0},
    )


def test_identities_canonicalize_every_semantic_input_and_are_frozen() -> None:
    assert configuration_fingerprint({"a": 1, "b": [2, 3]}) == (
        configuration_fingerprint({"b": [2, 3], "a": 1})
    )

    ranking = pd.DataFrame(
        {
            "id": ["B", "A"],
            "year": [2022, 2022],
            "score_input": [2.0, None],
        }
    )
    reordered = ranking.iloc[::-1].reset_index(drop=True)
    score_a = ScoredTableCacheIdentity.create(
        mode=RuntimeMode.PRODUCTION,
        ranking_population=ranking,
        model_state_identity=model_identity(),
        as_of="2022-04-01",
        scoring_configuration={"cost_version": 1},
    )
    score_b = ScoredTableCacheIdentity.create(
        mode=RuntimeMode.PRODUCTION,
        ranking_population=reordered,
        model_state_identity=model_identity(),
        as_of=date(2022, 4, 1),
        scoring_configuration={"cost_version": 1},
    )

    assert score_a == score_b
    assert score_a.as_of == date(2022, 4, 1)
    assert dataframe_content_digest(ranking, sort_by="id") == (
        dataframe_content_digest(reordered, sort_by="id")
    )
    with pytest.raises(FrozenInstanceError):
        score_a.as_of = date(2022, 5, 1)  # type: ignore[misc]


def test_modes_intervals_configuration_and_budgets_isolate_identities() -> None:
    production = model_identity(RuntimeMode.PRODUCTION)
    demo = model_identity(RuntimeMode.DEMO)
    changed_interval = ModelStateCacheIdentity.create(
        mode=RuntimeMode.PRODUCTION,
        source_sha256=production.source_sha256,
        start_year=2012,
        end_year=2016,
        model_configuration={"horizon_months": 48, "smoothing": 1.0},
    )

    assert production != demo
    assert production != changed_interval

    common = dict(
        mode=RuntimeMode.PRODUCTION,
        source_sha256=bytes_content_digest(b"panel"),
        fit_start_year=2012,
        fit_end_year=2016,
        candidate_year=2018,
        outcome_year=2022,
        as_of="2018-04-01",
        model_configuration={"model": 1},
        planner_configuration={"planner": 1},
    )
    first = BacktestCacheIdentity.create(budgets=[500_000, 100_000, 500_000], **common)
    same = BacktestCacheIdentity.create(budgets=[100_000.0, 500_000], **common)
    changed = BacktestCacheIdentity.create(budgets=[100_000, 550_000], **common)

    assert first == same
    assert first != changed
    assert tuple(map(str, first.budgets)) == ("1E+5", "5E+5")


def test_cache_reuses_only_the_validated_success_for_an_identity() -> None:
    data_cache = MemoryCachedFunction()
    resource_cache = MemoryCachedFunction()
    cache = ModelCache(
        data_cache_function=data_cache,
        resource_cache_function=resource_cache,
    )
    identity = source_identity()
    events: list[str] = []

    def compute() -> dict[str, int]:
        events.append("compute")
        return {"rows": 10}

    def validate(value: dict[str, int]) -> None:
        events.append("validate")
        assert value["rows"] > 0

    first = cache.get_data(identity, compute, validate)
    second = cache.get_data(identity, compute, validate)

    assert first == second == {"rows": 10}
    assert events == ["compute", "validate"]
    assert data_cache.compute_calls == 1
    assert resource_cache.compute_calls == 0


def test_invalid_or_failed_computations_never_remain_cached() -> None:
    data_cache = MemoryCachedFunction()
    cache = ModelCache(
        data_cache_function=data_cache,
        resource_cache_function=MemoryCachedFunction(),
    )
    identity = source_identity()

    with pytest.raises(CacheValidationError):
        cache.get_data(identity, lambda: -1, lambda value: value >= 0)
    assert identity not in data_cache.values

    with pytest.raises(RuntimeError, match="failed"):
        cache.get_data(
            identity,
            lambda: (_ for _ in ()).throw(RuntimeError("failed")),
            lambda value: True,
        )
    assert identity not in data_cache.values


def test_forced_recomputation_evicts_stale_success_before_user_code_runs() -> None:
    data_cache = MemoryCachedFunction()
    cache = ModelCache(
        data_cache_function=data_cache,
        resource_cache_function=MemoryCachedFunction(),
    )
    identity = source_identity()
    assert cache.get_data(identity, lambda: "old", lambda value: True) == "old"

    def fail_after_observing_eviction() -> str:
        assert identity not in data_cache.values
        raise RuntimeError("replacement failed")

    with pytest.raises(RuntimeError, match="replacement failed"):
        cache.force_recompute(identity, fail_after_observing_eviction, lambda value: True)

    assert identity not in data_cache.values
    assert cache.get_data(identity, lambda: "new", lambda value: True) == "new"


def test_exact_eviction_preserves_other_keys_and_model_state_uses_resource_cache() -> None:
    data_cache = MemoryCachedFunction()
    resource_cache = MemoryCachedFunction()
    cache = ModelCache(
        data_cache_function=data_cache,
        resource_cache_function=resource_cache,
    )
    first = source_identity(label=b"first")
    second = source_identity(label=b"second")
    model = model_identity()

    cache.get_data(first, lambda: "first", lambda value: True)
    cache.get_data(second, lambda: "second", lambda value: True)
    resource = object()
    assert cache.get_resource(model, lambda: resource, lambda value: True) is resource

    cache.evict(first)

    assert first not in data_cache.values
    assert second in data_cache.values
    assert model in resource_cache.values
    assert data_cache.cleared[-1] == first
