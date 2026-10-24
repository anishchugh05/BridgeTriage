"""Validated caching and bounded real-model orchestration.

The cache layer exposes immutable semantic identities and a transactional
:class:`ModelCache`. :class:`CurrentModelService` composes the public model
package without copying model logic: it bounds fitting to 2018/2022, scores the
complete 2022 population, validates both cached values, and reconciles model
identifiers to source records before returning an immutable ranking snapshot.
The production provider below composes those services with the real planner;
presentation and runtime error rendering remain outside this module.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast

import numpy as np
import pandas as pd
import streamlit as st

from app.config import Provenance, RuntimeMode, validate_provider_mode


T = TypeVar("T")
ValidationResult = bool | np.bool_ | None
Validator = Callable[[T], ValidationResult]

_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


class CacheNamespace(str, Enum):
    """The four independently keyed value namespaces."""

    SOURCE = "source"
    MODEL_STATE = "model_state"
    SCORED_TABLE = "scored_table"
    BACKTEST = "backtest"


class CacheStorage(str, Enum):
    """The Streamlit cache appropriate for a value category."""

    DATA = "data"
    RESOURCE = "resource"


def _packed(tag: bytes, payload: bytes = b"") -> bytes:
    """Return an unambiguous type-tagged byte segment."""

    return tag + len(payload).to_bytes(8, byteorder="big", signed=False) + payload


def _type_name(value: object) -> bytes:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}".encode("utf-8")


def _canonical_bytes(value: object) -> bytes:
    """Encode supported values deterministically without using ``repr``.

    The encoding is type-aware, mappings and sets are order-independent, and
    sequences preserve order. Unsupported objects are rejected rather than
    risking process-specific representations such as memory addresses.
    """

    if value is None:
        return _packed(b"none")
    if isinstance(value, np.generic):
        return _canonical_bytes(value.item())
    if isinstance(value, Enum):
        payload = _packed(b"type", _type_name(value)) + _canonical_bytes(value.value)
        return _packed(b"enum", payload)
    if isinstance(value, bool):
        return _packed(b"bool", b"1" if value else b"0")
    if isinstance(value, Integral):
        return _packed(b"int", str(int(value)).encode("ascii"))
    if isinstance(value, Decimal):
        if value.is_nan():
            payload = b"nan"
        elif value.is_infinite():
            payload = b"+inf" if value > 0 else b"-inf"
        else:
            payload = str(value.normalize()).encode("ascii")
        return _packed(b"decimal", payload)
    if isinstance(value, Real):
        number = float(value)
        if math.isnan(number):
            payload = b"nan"
        elif math.isinf(number):
            payload = b"+inf" if number > 0 else b"-inf"
        else:
            payload = number.hex().encode("ascii")
        return _packed(b"float", payload)
    if isinstance(value, str):
        return _packed(b"str", value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _packed(b"bytes", bytes(value))
    if isinstance(value, pd.Timestamp):
        return _packed(b"timestamp", value.isoformat().encode("utf-8"))
    if isinstance(value, datetime):
        return _packed(b"datetime", value.isoformat().encode("utf-8"))
    if isinstance(value, date):
        return _packed(b"date", value.isoformat().encode("ascii"))
    if isinstance(value, (pd.Timedelta, timedelta)):
        return _packed(b"timedelta_ns", str(pd.Timedelta(value).value).encode("ascii"))
    if isinstance(value, Path):
        return _packed(b"path", value.as_posix().encode("utf-8"))
    if isinstance(value, pd.DataFrame):
        return _packed(b"dataframe_sha256", bytes.fromhex(dataframe_content_digest(value)))
    if isinstance(value, Mapping):
        encoded_items = [
            _packed(b"key", _canonical_bytes(key))
            + _packed(b"value", _canonical_bytes(item_value))
            for key, item_value in value.items()
        ]
        encoded_items.sort()
        return _packed(b"mapping", b"".join(encoded_items))
    if isinstance(value, tuple):
        return _packed(b"tuple", b"".join(_packed(b"item", _canonical_bytes(item)) for item in value))
    if isinstance(value, list):
        return _packed(b"list", b"".join(_packed(b"item", _canonical_bytes(item)) for item in value))
    if isinstance(value, (set, frozenset)):
        encoded_items = sorted(_canonical_bytes(item) for item in value)
        tag = b"frozenset" if isinstance(value, frozenset) else b"set"
        return _packed(tag, b"".join(_packed(b"item", item) for item in encoded_items))
    if is_dataclass(value) and not isinstance(value, type):
        payload = _packed(b"type", _type_name(value))
        for field in fields(value):
            payload += _packed(b"field", field.name.encode("utf-8"))
            payload += _packed(b"value", _canonical_bytes(getattr(value, field.name)))
        return _packed(b"dataclass", payload)

    raise TypeError(
        "Unsupported value type for deterministic digest: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def deterministic_content_digest(value: object) -> str:
    """Return a process-independent SHA-256 digest for a supported value."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def configuration_fingerprint(configuration: object) -> str:
    """Fingerprint complete model, scoring, or planner configuration.

    Mapping order and set iteration order do not affect the result; sequence
    order and concrete value types do. Callers must include every setting that
    can change the corresponding computation.
    """

    return deterministic_content_digest(configuration)


def bytes_content_digest(content: bytes | bytearray | memoryview) -> str:
    """Return the SHA-256 digest of exact source bytes."""

    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("content must be bytes-like.")
    return hashlib.sha256(bytes(content)).hexdigest()


def file_content_digest(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into a SHA-256 digest without changing its contents."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
        raise TypeError("chunk_size must be a positive integer.")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(int(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def dataframe_content_digest(
    frame: pd.DataFrame,
    *,
    sort_by: str | Sequence[str] | None = None,
) -> str:
    """Digest ordered columns, dtypes, null markers, and cell values.

    The DataFrame index is intentionally excluded. When ``sort_by`` is given,
    rows are stably canonicalized by unique key columns before hashing. This is
    the intended form for complete ranking-population identities.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if sort_by is None:
        sort_columns: tuple[str, ...] = ()
    elif isinstance(sort_by, str):
        sort_columns = (sort_by,)
    else:
        sort_columns = tuple(sort_by)

    ordered = frame
    if sort_columns:
        if not sort_columns or any(
            not isinstance(column, str) or not column for column in sort_columns
        ):
            raise ValueError("sort_by must contain nonempty column names.")
        missing_columns = [
            column for column in sort_columns if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError("sort_by contains a column absent from the frame.")
        if frame.duplicated(subset=list(sort_columns), keep=False).any():
            raise ValueError("sort_by columns must uniquely identify every row.")
        ordered = frame.sort_values(
            list(sort_columns), kind="mergesort", na_position="last"
        )

    digest = hashlib.sha256()
    digest.update(_packed(b"rows", str(len(ordered)).encode("ascii")))
    digest.update(_packed(b"columns", str(len(ordered.columns)).encode("ascii")))
    for column, dtype in zip(ordered.columns, ordered.dtypes):
        digest.update(_packed(b"column", _canonical_bytes(column)))
        digest.update(_packed(b"dtype", str(dtype).encode("utf-8")))

    for row in ordered.itertuples(index=False, name=None):
        digest.update(_packed(b"row"))
        for value in row:
            encoded = _packed(b"missing") if _is_missing_scalar(value) else _canonical_bytes(value)
            digest.update(_packed(b"cell", encoded))

    return digest.hexdigest()


def _normalize_digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest.")
    return value.lower()


def _normalize_token(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a nonempty unpadded string.")
    return value


def _normalize_relative_path_identity(value: object) -> str:
    if not isinstance(value, (str, Path)):
        raise TypeError("source_path_identity must be a repository-relative path.")
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or path.as_posix() == "."
        or ".." in path.parts
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise ValueError("source_path_identity must be a repository-relative path.")
    return path.as_posix()


def _normalize_year(value: object, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer year.")
    return int(value)


def _normalize_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO calendar date.") from exc
    raise TypeError(f"{field_name} must be a date or ISO date string.")


def _normalize_budgets(values: Iterable[object]) -> tuple[Decimal, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("budgets must be an iterable of numeric values.")

    normalized: set[Decimal] = set()
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Real, Decimal)
        ):
            raise TypeError("Every budget must be numeric and not Boolean.")
        try:
            budget = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("Every budget must be finite and nonnegative.") from exc
        if not budget.is_finite() or budget < 0:
            raise ValueError("Every budget must be finite and nonnegative.")
        normalized.add(Decimal(0) if budget == 0 else budget.normalize())
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class CacheIdentity(ABC):
    """Base contract shared by every immutable cache identity."""

    mode: RuntimeMode

    namespace: ClassVar[CacheNamespace | None] = None
    storage: ClassVar[CacheStorage | None] = None

    def __post_init__(self) -> None:
        if type(self) is CacheIdentity:
            raise TypeError("CacheIdentity is abstract; use a concrete identity.")
        if not isinstance(self.mode, RuntimeMode):
            raise TypeError("mode must be a RuntimeMode.")
        if self.namespace is None or self.storage is None:
            raise TypeError("Concrete cache identities must declare a namespace and storage.")


@dataclass(frozen=True, slots=True)
class SourceCacheIdentity(CacheIdentity):
    """Identity for one validated source bundle."""

    source_path_identity: str
    source_sha256: str
    schema_version: str

    namespace: ClassVar[CacheNamespace] = CacheNamespace.SOURCE
    storage: ClassVar[CacheStorage] = CacheStorage.DATA

    def __post_init__(self) -> None:
        super(SourceCacheIdentity, self).__post_init__()
        object.__setattr__(
            self,
            "source_path_identity",
            _normalize_relative_path_identity(self.source_path_identity),
        )
        object.__setattr__(
            self, "source_sha256", _normalize_digest(self.source_sha256, "source_sha256")
        )
        object.__setattr__(
            self, "schema_version", _normalize_token(self.schema_version, "schema_version")
        )

    @classmethod
    def from_content(
        cls,
        mode: RuntimeMode,
        source_path_identity: str | Path,
        content: bytes | bytearray | memoryview,
        schema_version: str,
    ) -> "SourceCacheIdentity":
        """Build an identity from exact source bytes."""

        return cls(
            mode=mode,
            source_path_identity=str(source_path_identity),
            source_sha256=bytes_content_digest(content),
            schema_version=schema_version,
        )


@dataclass(frozen=True, slots=True)
class ModelStateCacheIdentity(CacheIdentity):
    """Identity for a fitted decline-model resource."""

    source_sha256: str
    start_year: int
    end_year: int
    model_family: str
    model_config_fingerprint: str

    namespace: ClassVar[CacheNamespace] = CacheNamespace.MODEL_STATE
    storage: ClassVar[CacheStorage] = CacheStorage.RESOURCE

    def __post_init__(self) -> None:
        super(ModelStateCacheIdentity, self).__post_init__()
        object.__setattr__(
            self, "source_sha256", _normalize_digest(self.source_sha256, "source_sha256")
        )
        start_year = _normalize_year(self.start_year, "start_year")
        end_year = _normalize_year(self.end_year, "end_year")
        if start_year >= end_year:
            raise ValueError("start_year must be earlier than end_year.")
        object.__setattr__(self, "start_year", start_year)
        object.__setattr__(self, "end_year", end_year)
        object.__setattr__(
            self, "model_family", _normalize_token(self.model_family, "model_family")
        )
        object.__setattr__(
            self,
            "model_config_fingerprint",
            _normalize_digest(
                self.model_config_fingerprint, "model_config_fingerprint"
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        mode: RuntimeMode,
        source_sha256: str,
        start_year: int,
        end_year: int,
        model_configuration: object,
        model_family: str = "fit_decline",
    ) -> "ModelStateCacheIdentity":
        """Build a model-state identity from complete semantic inputs."""

        return cls(
            mode=mode,
            source_sha256=source_sha256,
            start_year=start_year,
            end_year=end_year,
            model_family=model_family,
            model_config_fingerprint=configuration_fingerprint(model_configuration),
        )


@dataclass(frozen=True, slots=True)
class ScoredTableCacheIdentity(CacheIdentity):
    """Identity for a complete validated scored ranking table."""

    ranking_population_digest: str
    model_state_identity: ModelStateCacheIdentity
    as_of: date
    scoring_config_fingerprint: str

    namespace: ClassVar[CacheNamespace] = CacheNamespace.SCORED_TABLE
    storage: ClassVar[CacheStorage] = CacheStorage.DATA

    def __post_init__(self) -> None:
        super(ScoredTableCacheIdentity, self).__post_init__()
        object.__setattr__(
            self,
            "ranking_population_digest",
            _normalize_digest(
                self.ranking_population_digest, "ranking_population_digest"
            ),
        )
        if not isinstance(self.model_state_identity, ModelStateCacheIdentity):
            raise TypeError("model_state_identity must be a ModelStateCacheIdentity.")
        if self.model_state_identity.mode is not self.mode:
            raise ValueError("Nested model-state mode must match the scoring mode.")
        object.__setattr__(self, "as_of", _normalize_date(self.as_of, "as_of"))
        object.__setattr__(
            self,
            "scoring_config_fingerprint",
            _normalize_digest(
                self.scoring_config_fingerprint, "scoring_config_fingerprint"
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        mode: RuntimeMode,
        ranking_population: pd.DataFrame,
        model_state_identity: ModelStateCacheIdentity,
        as_of: date | datetime | str,
        scoring_configuration: object,
        identifier_column: str = "id",
    ) -> "ScoredTableCacheIdentity":
        """Build a score identity from a canonically ID-ordered population."""

        return cls(
            mode=mode,
            ranking_population_digest=dataframe_content_digest(
                ranking_population, sort_by=identifier_column
            ),
            model_state_identity=model_state_identity,
            as_of=_normalize_date(as_of, "as_of"),
            scoring_config_fingerprint=configuration_fingerprint(
                scoring_configuration
            ),
        )


@dataclass(frozen=True, slots=True)
class BacktestCacheIdentity(CacheIdentity):
    """Identity for a validated historical backtest table."""

    source_sha256: str
    budgets: tuple[Decimal, ...]
    fit_start_year: int
    fit_end_year: int
    candidate_year: int
    outcome_year: int
    as_of: date
    model_config_fingerprint: str
    planner_config_fingerprint: str

    namespace: ClassVar[CacheNamespace] = CacheNamespace.BACKTEST
    storage: ClassVar[CacheStorage] = CacheStorage.DATA

    def __post_init__(self) -> None:
        super(BacktestCacheIdentity, self).__post_init__()
        object.__setattr__(
            self, "source_sha256", _normalize_digest(self.source_sha256, "source_sha256")
        )
        object.__setattr__(self, "budgets", _normalize_budgets(self.budgets))

        timeline = (
            _normalize_year(self.fit_start_year, "fit_start_year"),
            _normalize_year(self.fit_end_year, "fit_end_year"),
            _normalize_year(self.candidate_year, "candidate_year"),
            _normalize_year(self.outcome_year, "outcome_year"),
        )
        if not timeline[0] < timeline[1] < timeline[2] < timeline[3]:
            raise ValueError("Backtest years must be strictly chronological.")
        object.__setattr__(self, "fit_start_year", timeline[0])
        object.__setattr__(self, "fit_end_year", timeline[1])
        object.__setattr__(self, "candidate_year", timeline[2])
        object.__setattr__(self, "outcome_year", timeline[3])
        object.__setattr__(self, "as_of", _normalize_date(self.as_of, "as_of"))
        object.__setattr__(
            self,
            "model_config_fingerprint",
            _normalize_digest(
                self.model_config_fingerprint, "model_config_fingerprint"
            ),
        )
        object.__setattr__(
            self,
            "planner_config_fingerprint",
            _normalize_digest(
                self.planner_config_fingerprint, "planner_config_fingerprint"
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        mode: RuntimeMode,
        source_sha256: str,
        budgets: Iterable[object],
        fit_start_year: int,
        fit_end_year: int,
        candidate_year: int,
        outcome_year: int,
        as_of: date | datetime | str,
        model_configuration: object,
        planner_configuration: object,
    ) -> "BacktestCacheIdentity":
        """Build a canonical backtest identity from every semantic input."""

        return cls(
            mode=mode,
            source_sha256=source_sha256,
            budgets=_normalize_budgets(budgets),
            fit_start_year=fit_start_year,
            fit_end_year=fit_end_year,
            candidate_year=candidate_year,
            outcome_year=outcome_year,
            as_of=_normalize_date(as_of, "as_of"),
            model_config_fingerprint=configuration_fingerprint(model_configuration),
            planner_config_fingerprint=configuration_fingerprint(
                planner_configuration
            ),
        )


class CacheValidationError(ValueError):
    """Raised when a validator explicitly rejects a computed value."""


def _compute_validated(
    operation: Callable[[], T], validator: Validator[T]
) -> T:
    value = operation()
    validation_result = validator(value)
    if validation_result is not None and not isinstance(
        validation_result, (bool, np.bool_)
    ):
        raise TypeError("A cache validator must return None or a Boolean.")
    if validation_result is not None and not bool(validation_result):
        raise CacheValidationError("Computed cache value failed validation.")
    return value


@st.cache_data(show_spinner=False)
def _streamlit_data_value(
    identity: CacheIdentity,
    _operation: Callable[[], Any] | None = None,
) -> Any:
    """Compute one data value; ``_operation`` is excluded from cache hashing."""

    if _operation is None:  # pragma: no cover - only direct misuse can reach this
        raise RuntimeError("A cache computation operation is required.")
    return _operation()


@st.cache_resource(show_spinner=False)
def _streamlit_resource_value(
    identity: CacheIdentity,
    _operation: Callable[[], Any] | None = None,
) -> Any:
    """Compute one resource; ``_operation`` is excluded from cache hashing."""

    if _operation is None:  # pragma: no cover - only direct misuse can reach this
        raise RuntimeError("A cache computation operation is required.")
    return _operation()


class CachedFunction(Protocol):
    """Injectable subset of Streamlit's cached-function interface."""

    def __call__(
        self,
        identity: CacheIdentity,
        _operation: Callable[[], Any] | None = None,
    ) -> Any: ...

    def clear(self, *args: Any, **kwargs: Any) -> None: ...


class ModelCache(Generic[T]):
    """Transactional facade over ``st.cache_data`` and ``st.cache_resource``.

    Computation and validation both execute inside the cached function, so a
    failing operation or validator cannot create a cache entry. Exact-key
    eviction delegates to the cached function's argument-specific ``clear``.
    Forced recomputation always evicts before invoking user code; if that code
    fails, the identity remains empty instead of exposing its stale success.

    Cached functions are injectable to make state transitions testable without
    a running Streamlit application.
    """

    def __init__(
        self,
        *,
        data_cache_function: CachedFunction | None = None,
        resource_cache_function: CachedFunction | None = None,
    ) -> None:
        self._data_cache = data_cache_function or cast(
            CachedFunction, _streamlit_data_value
        )
        self._resource_cache = resource_cache_function or cast(
            CachedFunction, _streamlit_resource_value
        )
        for cache_function in (self._data_cache, self._resource_cache):
            if not callable(cache_function) or not callable(
                getattr(cache_function, "clear", None)
            ):
                raise TypeError("Cache functions must be callable and support clear().")
        self._lock = RLock()

    @staticmethod
    def _require_identity(identity: CacheIdentity) -> None:
        if not isinstance(identity, CacheIdentity):
            raise TypeError("identity must be a concrete CacheIdentity.")

    def _cache_for(self, identity: CacheIdentity) -> CachedFunction:
        self._require_identity(identity)
        if identity.storage is CacheStorage.RESOURCE:
            return self._resource_cache
        if identity.storage is CacheStorage.DATA:
            return self._data_cache
        raise TypeError("identity declares an unsupported cache storage kind.")

    @staticmethod
    def _evict_from(
        cache_function: CachedFunction, identity: CacheIdentity
    ) -> None:
        # ``_operation`` is underscore-prefixed and therefore excluded from the
        # Streamlit key. Supplying only the identity clears exactly that entry.
        cache_function.clear(identity)

    def get_or_compute(
        self,
        identity: CacheIdentity,
        operation: Callable[[], T],
        validator: Validator[T],
        *,
        force_recompute: bool = False,
    ) -> T:
        """Return a validated cached value or transactionally compute one."""

        self._require_identity(identity)
        if not callable(operation):
            raise TypeError("operation must be callable.")
        if not callable(validator):
            raise TypeError("validator must be callable.")
        if not isinstance(force_recompute, bool):
            raise TypeError("force_recompute must be Boolean.")

        cache_function = self._cache_for(identity)
        validated_operation = lambda: _compute_validated(operation, validator)

        with self._lock:
            if force_recompute:
                self._evict_from(cache_function, identity)
            try:
                return cast(
                    T,
                    cache_function(identity, _operation=validated_operation),
                )
            except Exception:
                # Streamlit does not cache raised computations. Clearing again
                # makes the invariant explicit for injected backends as well.
                self._evict_from(cache_function, identity)
                raise

    def get_data(
        self,
        identity: CacheIdentity,
        operation: Callable[[], T],
        validator: Validator[T],
        *,
        force_recompute: bool = False,
    ) -> T:
        """Get a data-cached source, scored-table, or backtest value."""

        self._require_identity(identity)
        if identity.storage is not CacheStorage.DATA:
            raise TypeError("This identity belongs in the resource cache.")
        return self.get_or_compute(
            identity,
            operation,
            validator,
            force_recompute=force_recompute,
        )

    def get_resource(
        self,
        identity: CacheIdentity,
        operation: Callable[[], T],
        validator: Validator[T],
        *,
        force_recompute: bool = False,
    ) -> T:
        """Get a resource-cached fitted model-state value."""

        self._require_identity(identity)
        if identity.storage is not CacheStorage.RESOURCE:
            raise TypeError("This identity belongs in the data cache.")
        return self.get_or_compute(
            identity,
            operation,
            validator,
            force_recompute=force_recompute,
        )

    def force_recompute(
        self,
        identity: CacheIdentity,
        operation: Callable[[], T],
        validator: Validator[T],
    ) -> T:
        """Evict one identity, then compute and admit only a valid replacement."""

        return self.get_or_compute(
            identity, operation, validator, force_recompute=True
        )

    def evict(self, identity: CacheIdentity) -> None:
        """Remove exactly one semantic identity from its assigned cache."""

        cache_function = self._cache_for(identity)
        with self._lock:
            self._evict_from(cache_function, identity)

    invalidate = evict


# ---------------------------------------------------------------------------
# Bounded current-model orchestration
# ---------------------------------------------------------------------------

CURRENT_FIT_START_YEAR = 2018
CURRENT_FIT_END_YEAR = 2022
CURRENT_RANKING_YEAR = 2022
CURRENT_RANKING_AS_OF = pd.Timestamp("2022-04-01")
CURRENT_MODEL_FAMILY = "fit_decline"
CURRENT_MODEL_HORIZON_MONTHS = 48.0


def _default_current_model_configuration() -> tuple[tuple[str, object], ...]:
    """Capture every fitting knob used by the grouped decline model."""

    from model.score import (
        AGE_BAND_EDGES,
        AGE_BAND_LABELS,
        CONDITION_COLS,
        CULVERT_COL,
        NBI_SNAPSHOT_DAY,
        NBI_SNAPSHOT_MONTH,
        SHRINKAGE_K,
    )

    return (
        ("contract_version", "grouped-decline-current-v1"),
        ("condition_columns", tuple(CONDITION_COLS)),
        ("culvert_column", CULVERT_COL),
        ("age_band_edges", tuple(AGE_BAND_EDGES)),
        ("age_band_labels", tuple(AGE_BAND_LABELS)),
        ("shrinkage_k", SHRINKAGE_K),
        ("snapshot_month", NBI_SNAPSHOT_MONTH),
        ("snapshot_day", NBI_SNAPSHOT_DAY),
        ("target_contract", "safety-drop-v1"),
        ("pair_contract", "shared-ids-null-and-rebuilt-excluded-v1"),
    )


def _default_current_scoring_configuration() -> tuple[tuple[str, object], ...]:
    """Capture every scoring knob that can change the current ranking."""

    from model.data import MODELED_COST_DISCLOSURE
    from model.score import (
        CONSEQUENCE_POOR_MULT,
        CONSEQUENCE_RAIL_MULT,
        CONSEQUENCE_WATER_MULT,
        COST_BASE,
        COST_FRACTURE_CRITICAL,
        COST_UNDERWATER,
        COST_UNDERWATER_AND_FC,
        DEFAULT_MONTHS_NO_DATE,
        LOOKAHEAD_MONTHS,
        MONTHS_SINCE_INSP_CAP,
        OUTPUT_COLUMNS,
        POOR_CONDITION_THRESHOLD,
        RAIL_PATTERN,
        WATER_PATTERN,
    )

    return (
        ("contract_version", "current-score-v1"),
        ("output_columns", tuple(OUTPUT_COLUMNS)),
        ("cost_base", COST_BASE),
        ("cost_underwater", COST_UNDERWATER),
        ("cost_fracture_critical", COST_FRACTURE_CRITICAL),
        ("cost_underwater_and_fracture_critical", COST_UNDERWATER_AND_FC),
        ("consequence_rail_multiplier", CONSEQUENCE_RAIL_MULT),
        ("consequence_water_multiplier", CONSEQUENCE_WATER_MULT),
        ("consequence_poor_multiplier", CONSEQUENCE_POOR_MULT),
        ("poor_condition_threshold", POOR_CONDITION_THRESHOLD),
        ("lookahead_months", LOOKAHEAD_MONTHS),
        ("default_months_without_date", DEFAULT_MONTHS_NO_DATE),
        ("months_since_inspection_cap", MONTHS_SINCE_INSP_CAP),
        ("rail_pattern", RAIL_PATTERN.pattern),
        ("water_pattern", WATER_PATTERN.pattern),
        ("modeled_cost_disclosure", MODELED_COST_DISCLOSURE),
        ("repair_cost_input", "excluded"),
    )


# Immutable tuples make the default semantic configurations safe to share.
CURRENT_MODEL_CONFIGURATION = _default_current_model_configuration()
CURRENT_SCORING_CONFIGURATION = _default_current_scoring_configuration()


class CurrentModelValidationError(ValueError):
    """Raised when current-model inputs or fitted state violate the contract."""


def _is_finite_real(value: object) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (Real, Decimal))
        and math.isfinite(float(value))
    )


def validate_current_model_state(model_state: object) -> None:
    """Validate the cached grouped decline state before it can be exposed.

    The real ``fit_decline`` result may legitimately contain no fine-grained
    groups when no eligible pairs exist, but it must always expose a callable
    probability interface, a 48-month horizon, and finite probabilities in the
    closed unit interval.
    """

    issues: list[str] = []
    if not callable(getattr(model_state, "interval_prob", None)):
        issues.append("model_state.interval_prob_callable")

    horizon = getattr(model_state, "horizon_months", None)
    if (
        not _is_finite_real(horizon)
        or float(horizon) != CURRENT_MODEL_HORIZON_MONTHS
    ):
        issues.append("model_state.horizon_48_months")

    global_probability = getattr(model_state, "global_prob", None)
    if (
        not _is_finite_real(global_probability)
        or not 0.0 <= float(global_probability) <= 1.0
    ):
        issues.append("model_state.global_probability")

    group_probabilities = getattr(model_state, "group_prob", None)
    if not isinstance(group_probabilities, Mapping):
        issues.append("model_state.group_probabilities_mapping")
    elif any(
        not _is_finite_real(probability)
        or not 0.0 <= float(probability) <= 1.0
        for probability in group_probabilities.values()
    ):
        issues.append("model_state.group_probabilities")

    if issues:
        raise CurrentModelValidationError(
            "Current model state failed validation: " + ", ".join(issues)
        )


class CurrentPanelBundle(Protocol):
    """Structural input expected from the later production source provider."""

    source: pd.DataFrame
    adapted: pd.DataFrame
    dataset_sha256: str
    mode: RuntimeMode


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CurrentRanking:
    """Immutable snapshot of one validated, reconciled current ranking.

    Pandas frames are copied on admission and every public read, preventing a
    renderer or planner from mutating the cached score table or another view's
    statewide population.
    """

    rates_identity: ModelStateCacheIdentity
    _scored: pd.DataFrame
    _records: pd.DataFrame
    _ranking_ids: tuple[str, ...]

    def __init__(
        self,
        rates_identity: ModelStateCacheIdentity,
        scored: pd.DataFrame,
        records: pd.DataFrame,
    ) -> None:
        if not isinstance(rates_identity, ModelStateCacheIdentity):
            raise TypeError("rates_identity must be a ModelStateCacheIdentity.")
        if not isinstance(scored, pd.DataFrame):
            raise TypeError("scored must be a pandas DataFrame.")
        if not isinstance(records, pd.DataFrame):
            raise TypeError("records must be a pandas DataFrame.")
        if "structure_id" not in records.columns:
            raise CurrentModelValidationError(
                "Current ranking records require source identifiers."
            )

        scored_ids = tuple(scored["id"].tolist()) if "id" in scored.columns else ()
        record_ids = tuple(records["structure_id"].tolist())
        if (
            len(scored_ids) != len(record_ids)
            or len(set(scored_ids)) != len(scored_ids)
            or len(set(record_ids)) != len(record_ids)
            or set(scored_ids) != set(record_ids)
        ):
            raise CurrentModelValidationError(
                "Current ranking identifiers are not one-to-one."
            )

        object.__setattr__(self, "rates_identity", rates_identity)
        object.__setattr__(self, "_scored", scored.copy(deep=True))
        object.__setattr__(self, "_records", records.copy(deep=True))
        object.__setattr__(self, "_ranking_ids", cast(tuple[str, ...], record_ids))

    @property
    def model_state_identity(self) -> ModelStateCacheIdentity:
        """Alias exposing the semantic meaning of ``rates_identity``."""

        return self.rates_identity

    @property
    def scored(self) -> pd.DataFrame:
        """Return an isolated copy of the exact validated score table."""

        return self._scored.copy(deep=True)

    @property
    def records(self) -> pd.DataFrame:
        """Return an isolated copy of the reconciled source-facing records."""

        return self._records.copy(deep=True)

    @property
    def ranking_ids(self) -> tuple[str, ...]:
        """Return all source-facing ranking identifiers in source order."""

        return self._ranking_ids

    @property
    def population_size(self) -> int:
        return len(self._ranking_ids)

    def __repr__(self) -> str:
        return (
            "CurrentRanking("
            f"rates_identity={self.rates_identity!r}, "
            f"population_size={self.population_size})"
        )


class CurrentModelService:
    """Bound and cache the real current-ranking model workflow.

    ``fit_fn`` and ``score_fn`` default to the public :mod:`model` package and
    are injectable solely so boundary tests can observe their exact inputs.
    The service neither filters coordinates nor reimplements any fitting or
    scoring rule.
    """

    def __init__(
        self,
        cache: ModelCache[Any] | None = None,
        *,
        fit_fn: Callable[[pd.DataFrame, int, int], object] | None = None,
        score_fn: Callable[[pd.DataFrame, object, object], pd.DataFrame]
        | None = None,
        model_configuration: object = CURRENT_MODEL_CONFIGURATION,
        scoring_configuration: object = CURRENT_SCORING_CONFIGURATION,
    ) -> None:
        if cache is not None and not isinstance(cache, ModelCache):
            raise TypeError("cache must be a ModelCache.")

        if fit_fn is None or score_fn is None:
            import model as model_package

            fit_fn = fit_fn or model_package.fit_decline
            score_fn = score_fn or model_package.score
        if not callable(fit_fn):
            raise TypeError("fit_fn must be callable.")
        if not callable(score_fn):
            raise TypeError("score_fn must be callable.")

        self._cache = cache or ModelCache[Any]()
        self._fit_fn = fit_fn
        self._score_fn = score_fn
        # Fingerprint once so later mutation of a caller-owned configuration
        # cannot silently change this service instance's cache semantics.
        self._model_config_fingerprint = configuration_fingerprint(
            model_configuration
        )
        self._scoring_config_fingerprint = configuration_fingerprint(
            scoring_configuration
        )

    def create_model_state_identity(
        self,
        source_sha256: str,
        *,
        mode: RuntimeMode = RuntimeMode.PRODUCTION,
    ) -> ModelStateCacheIdentity:
        """Create the complete current fitted-state cache identity."""

        return ModelStateCacheIdentity(
            mode=mode,
            source_sha256=source_sha256,
            start_year=CURRENT_FIT_START_YEAR,
            end_year=CURRENT_FIT_END_YEAR,
            model_family=CURRENT_MODEL_FAMILY,
            model_config_fingerprint=self._model_config_fingerprint,
        )

    def create_scored_table_identity(
        self,
        ranking_population: pd.DataFrame,
        model_state_identity: ModelStateCacheIdentity,
        *,
        mode: RuntimeMode | None = None,
    ) -> ScoredTableCacheIdentity:
        """Create the complete current score cache identity."""

        self._require_model_identity(model_state_identity)
        active_mode = model_state_identity.mode if mode is None else mode
        return ScoredTableCacheIdentity(
            mode=active_mode,
            ranking_population_digest=dataframe_content_digest(
                ranking_population, sort_by="id"
            ),
            model_state_identity=model_state_identity,
            as_of=CURRENT_RANKING_AS_OF.date(),
            scoring_config_fingerprint=self._scoring_config_fingerprint,
        )

    def _require_model_identity(
        self, identity: ModelStateCacheIdentity
    ) -> None:
        if not isinstance(identity, ModelStateCacheIdentity):
            raise TypeError("cache_key must be a ModelStateCacheIdentity.")
        if (
            identity.start_year != CURRENT_FIT_START_YEAR
            or identity.end_year != CURRENT_FIT_END_YEAR
            or identity.model_family != CURRENT_MODEL_FAMILY
            or identity.model_config_fingerprint
            != self._model_config_fingerprint
        ):
            raise CurrentModelValidationError(
                "Model cache identity does not describe this current model."
            )

    def _require_score_identity(
        self,
        identity: ScoredTableCacheIdentity,
        ranking_population: pd.DataFrame,
    ) -> None:
        if not isinstance(identity, ScoredTableCacheIdentity):
            raise TypeError("cache_key must be a ScoredTableCacheIdentity.")
        self._require_model_identity(identity.model_state_identity)
        expected_population_digest = dataframe_content_digest(
            ranking_population, sort_by="id"
        )
        if (
            identity.as_of != CURRENT_RANKING_AS_OF.date()
            or identity.scoring_config_fingerprint
            != self._scoring_config_fingerprint
            or identity.ranking_population_digest != expected_population_digest
        ):
            raise CurrentModelValidationError(
                "Score cache identity does not describe this ranking population."
            )

    @staticmethod
    def _bounded_fit_panel(adapted_panel: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(adapted_panel, pd.DataFrame):
            raise TypeError("adapted_panel must be a pandas DataFrame.")
        if "year" not in adapted_panel.columns or "id" not in adapted_panel.columns:
            raise CurrentModelValidationError(
                "Adapted panel requires year and id columns."
            )

        bounded = adapted_panel.loc[
            adapted_panel["year"].isin(
                (CURRENT_FIT_START_YEAR, CURRENT_FIT_END_YEAR)
            )
        ].copy(deep=True)
        if set(bounded["year"].tolist()) != {
            CURRENT_FIT_START_YEAR,
            CURRENT_FIT_END_YEAR,
        }:
            raise CurrentModelValidationError(
                "Adapted panel must contain both current fitting years."
            )
        if bounded.duplicated(subset=["id", "year"], keep=False).any():
            raise CurrentModelValidationError(
                "Adapted current fitting identifiers must be unique by year."
            )
        return bounded

    @staticmethod
    def _ranking_population(adapted_panel: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(adapted_panel, pd.DataFrame):
            raise TypeError("adapted_panel must be a pandas DataFrame.")
        if "year" not in adapted_panel.columns or "id" not in adapted_panel.columns:
            raise CurrentModelValidationError(
                "Adapted panel requires year and id columns."
            )

        ranking = adapted_panel.loc[
            adapted_panel["year"].eq(CURRENT_RANKING_YEAR)
        ].copy(deep=True)
        ranking_ids = tuple(ranking["id"].tolist())
        if not ranking_ids:
            raise CurrentModelValidationError(
                "Adapted panel has no current ranking population."
            )
        if not all(
            isinstance(identifier, str) and bool(identifier.strip())
            for identifier in ranking_ids
        ):
            raise CurrentModelValidationError(
                "Current ranking identifiers must be nonempty strings."
            )
        if len(set(ranking_ids)) != len(ranking_ids):
            raise CurrentModelValidationError(
                "Current ranking identifiers must be unique."
            )
        return ranking

    @staticmethod
    def _source_ranking_population(source_panel: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(source_panel, pd.DataFrame):
            raise TypeError("bundle.source must be a pandas DataFrame.")
        if "year" not in source_panel.columns:
            raise CurrentModelValidationError("Source panel requires a year column.")
        source_2022 = source_panel.loc[
            source_panel["year"].eq(CURRENT_RANKING_YEAR)
        ].copy(deep=True)
        if source_2022.empty:
            raise CurrentModelValidationError(
                "Source panel has no current ranking population."
            )
        return source_2022

    def fit_current(
        self,
        adapted_panel: pd.DataFrame,
        cache_key: ModelStateCacheIdentity,
        *,
        force_recompute: bool = False,
    ) -> object:
        """Fit and cache the validated 2018-to-2022 decline state."""

        self._require_model_identity(cache_key)
        bounded_panel = self._bounded_fit_panel(adapted_panel)

        def operation() -> object:
            return self._fit_fn(
                bounded_panel.copy(deep=True),
                CURRENT_FIT_START_YEAR,
                CURRENT_FIT_END_YEAR,
            )

        return self._cache.get_resource(
            cache_key,
            operation,
            validate_current_model_state,
            force_recompute=force_recompute,
        )

    def score_current(
        self,
        adapted_panel: pd.DataFrame,
        rates: object,
        cache_key: ScoredTableCacheIdentity,
        *,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Score and cache every 2022 bridge before any map projection."""

        from model import validate_scored_table

        ranking_population = self._ranking_population(adapted_panel)
        self._require_score_identity(cache_key, ranking_population)
        validate_current_model_state(rates)
        ranking_ids = tuple(ranking_population["id"].tolist())

        def operation() -> pd.DataFrame:
            return self._score_fn(
                ranking_population.copy(deep=True),
                rates,
                pd.Timestamp(CURRENT_RANKING_AS_OF),
            )

        def validator(scored: pd.DataFrame) -> None:
            validate_scored_table(scored, ranking_ids)

        scored = self._cache.get_data(
            cache_key,
            operation,
            validator,
            force_recompute=force_recompute,
        )
        if not isinstance(scored, pd.DataFrame):  # defensive cached-backend guard
            raise CurrentModelValidationError(
                "Validated score cache returned a non-table value."
            )
        return scored.copy(deep=True)

    def build_current_ranking(
        self,
        bundle: CurrentPanelBundle,
        *,
        force_recompute: bool = False,
    ) -> CurrentRanking:
        """Build one complete validated and source-reconciled current ranking."""

        try:
            source_panel = bundle.source
            adapted_panel = bundle.adapted
            source_sha256 = bundle.dataset_sha256
            mode = bundle.mode
        except AttributeError as exc:
            raise TypeError(
                "bundle must provide source, adapted, dataset_sha256, and mode."
            ) from exc
        if not isinstance(mode, RuntimeMode):
            raise TypeError("bundle.mode must be a RuntimeMode.")

        model_identity = self.create_model_state_identity(
            source_sha256, mode=mode
        )
        rates = self.fit_current(
            adapted_panel,
            model_identity,
            force_recompute=force_recompute,
        )

        ranking_population = self._ranking_population(adapted_panel)
        score_identity = self.create_scored_table_identity(
            ranking_population,
            model_identity,
            mode=mode,
        )
        scored = self.score_current(
            adapted_panel,
            rates,
            score_identity,
            force_recompute=force_recompute,
        )

        from model import reconcile_scored_table

        source_2022 = self._source_ranking_population(source_panel)
        records = reconcile_scored_table(source_2022, scored)
        return CurrentRanking(model_identity, scored, records)


# ---------------------------------------------------------------------------
# Production-only source, planner, and backtest composition
# ---------------------------------------------------------------------------

PRODUCTION_SOURCE_PATH_IDENTITY = "bridges/panel.parquet"
PRODUCTION_SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "bridges" / "panel.parquet"
).resolve()
SOURCE_SCHEMA_VERSION = "source-schema-v1"

PRODUCTION_PROVENANCE = Provenance(
    mode=RuntimeMode.PRODUCTION,
    data_source="Real_Dataset",
    model_source="Real_Model_Service",
    planner_source="Real_Planner",
    backtest_source="Backtest_Engine",
)

PRODUCTION_BACKTEST_FIT_START_YEAR = 2012
PRODUCTION_BACKTEST_FIT_END_YEAR = 2016
PRODUCTION_BACKTEST_CANDIDATE_YEAR = 2018
PRODUCTION_BACKTEST_OUTCOME_YEAR = 2022
PRODUCTION_BACKTEST_AS_OF = date(2018, 4, 1)


def _default_backtest_model_configuration() -> tuple[tuple[str, object], ...]:
    """Capture all real fitting and scoring semantics used by the backtest."""

    return (
        ("contract_version", "real-backtest-model-v1"),
        ("fitting", CURRENT_MODEL_CONFIGURATION),
        ("scoring", CURRENT_SCORING_CONFIGURATION),
        ("target", "model.is_safety_drop-v1"),
    )


def _default_backtest_planner_configuration() -> tuple[tuple[str, object], ...]:
    """Capture deterministic method ordering and shared affordability rules."""

    return (
        ("contract_version", "real-backtest-planner-v1"),
        ("methods", ("ours", "worst_first", "oldest_first")),
        ("ours_order", ("score_desc", "id_asc")),
        (
            "worst_first_order",
            ("worst_rating_asc", "null_last", "id_asc"),
        ),
        ("oldest_first_order", ("age_desc", "id_asc")),
        ("affordability", "decimal-greedy-continue-v1"),
        (
            "outcome_eligibility",
            ("both_endpoints", "ratings_present", "ending_not_rebuilt"),
        ),
    )


BACKTEST_MODEL_CONFIGURATION = _default_backtest_model_configuration()
BACKTEST_PLANNER_CONFIGURATION = _default_backtest_planner_configuration()


class SourceBundleValidationError(ValueError):
    """Raised when a cached source bundle does not match its identity."""

    def __init__(self, issues: Iterable[str]):
        unique_issues = tuple(dict.fromkeys(issues)) or ("source_bundle.unknown",)
        self.issues = unique_issues
        super().__init__(
            "Production source bundle failed validation: "
            + ", ".join(unique_issues)
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class PanelBundle:
    """Immutable validated source/adapted snapshots and source provenance.

    Frames are copied when admitted and whenever read.  The adapted snapshot is
    produced only by :func:`model.adapt_source_panel`, so constructing this
    value cannot introduce a second schema mapping.  Metadata is normalized by
    the same helpers used for cache identities.
    """

    mode: RuntimeMode
    dataset_sha256: str
    source_path_identity: str
    schema_version: str
    _source: pd.DataFrame
    _adapted: pd.DataFrame

    def __init__(
        self,
        *,
        source: pd.DataFrame,
        dataset_sha256: str,
        mode: RuntimeMode,
        source_path_identity: str = PRODUCTION_SOURCE_PATH_IDENTITY,
        schema_version: str = SOURCE_SCHEMA_VERSION,
    ) -> None:
        if not isinstance(source, pd.DataFrame):
            raise TypeError("source must be a pandas DataFrame.")
        if not isinstance(mode, RuntimeMode):
            raise TypeError("mode must be a RuntimeMode.")

        from model import adapt_source_panel

        source_snapshot = source.copy(deep=True)
        adapted_snapshot = adapt_source_panel(source_snapshot)

        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "dataset_sha256",
            _normalize_digest(dataset_sha256, "dataset_sha256"),
        )
        object.__setattr__(
            self,
            "source_path_identity",
            _normalize_relative_path_identity(source_path_identity),
        )
        object.__setattr__(
            self,
            "schema_version",
            _normalize_token(schema_version, "schema_version"),
        )
        object.__setattr__(self, "_source", source_snapshot)
        object.__setattr__(self, "_adapted", adapted_snapshot.copy(deep=True))

    @property
    def source(self) -> pd.DataFrame:
        """Return an isolated source-schema snapshot."""

        return self._source.copy(deep=True)

    @property
    def adapted(self) -> pd.DataFrame:
        """Return an isolated model-facing snapshot."""

        return self._adapted.copy(deep=True)

    @property
    def row_count(self) -> int:
        return len(self._source)

    def __repr__(self) -> str:
        return (
            "PanelBundle("
            f"mode={self.mode!r}, "
            f"dataset_sha256={self.dataset_sha256!r}, "
            f"source_path_identity={self.source_path_identity!r}, "
            f"schema_version={self.schema_version!r}, "
            f"row_count={self.row_count})"
        )


def validate_panel_bundle(
    bundle: object,
    identity: SourceCacheIdentity | None = None,
) -> None:
    """Validate a bundle before source-cache admission or provider use."""

    if not isinstance(bundle, PanelBundle):
        raise SourceBundleValidationError(("source_bundle.type",))
    if identity is not None and not isinstance(identity, SourceCacheIdentity):
        raise TypeError("identity must be a SourceCacheIdentity.")

    issues: list[str] = []
    if identity is not None:
        if bundle.mode is not identity.mode:
            issues.append("source_bundle.mode")
        if bundle.dataset_sha256 != identity.source_sha256:
            issues.append("source_bundle.digest")
        if bundle.source_path_identity != identity.source_path_identity:
            issues.append("source_bundle.path_identity")
        if bundle.schema_version != identity.schema_version:
            issues.append("source_bundle.schema_version")
    if issues:
        raise SourceBundleValidationError(issues)

    from model import adapt_source_panel, validate_source_panel

    source = bundle.source
    validate_source_panel(source)
    expected_adapted = adapt_source_panel(source)
    try:
        pd.testing.assert_frame_equal(
            bundle.adapted,
            expected_adapted,
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise SourceBundleValidationError(
            ("source_bundle.adaptation_exact",)
        ) from exc


def _require_production_bundle(bundle: object) -> PanelBundle:
    """Accept only a validated bundle tagged for the real production branch."""

    if not isinstance(bundle, PanelBundle):
        raise TypeError("bundle must be a validated PanelBundle.")
    validate_provider_mode(bundle, RuntimeMode.PRODUCTION)
    if (
        bundle.source_path_identity != PRODUCTION_SOURCE_PATH_IDENTITY
        or bundle.schema_version != SOURCE_SCHEMA_VERSION
    ):
        raise SourceBundleValidationError(
            ("source_bundle.production_identity",)
        )
    return bundle


class BacktestResultCacheValidationError(ValueError):
    """Raised when a cached public backtest table violates its contract."""

    def __init__(self, issues: Iterable[str]):
        unique_issues = tuple(dict.fromkeys(issues)) or ("backtest_cache.unknown",)
        self.issues = unique_issues
        super().__init__(
            "Cached backtest result failed validation: "
            + ", ".join(unique_issues)
        )


def _normalize_backtest_result_budget(value: object) -> Decimal | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, Decimal)
    ):
        return None
    try:
        budget = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not budget.is_finite() or budget < 0:
        return None
    return Decimal(0) if budget == 0 else budget.normalize()


def validate_cached_backtest_result(
    result: object,
    identity: BacktestCacheIdentity,
    *,
    candidate_count: int,
) -> None:
    """Validate the public backtest projection against its cache request.

    ``planner.run_backtest`` validates each ``caught`` value against the
    private eligible-selection audit before returning.  This cache boundary
    independently rechecks all public predicates plus the complete candidate
    population bound, protecting against a corrupt or substituted cache value.
    """

    if not isinstance(identity, BacktestCacheIdentity):
        raise TypeError("identity must be a BacktestCacheIdentity.")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, Integral):
        raise TypeError("candidate_count must be a nonnegative integer.")
    normalized_candidate_count = int(candidate_count)
    if normalized_candidate_count < 0:
        raise ValueError("candidate_count must be a nonnegative integer.")

    if not isinstance(result, pd.DataFrame):
        raise BacktestResultCacheValidationError(("backtest_cache.frame_type",))

    from planner import BACKTEST_METHODS, BACKTEST_RESULT_COLUMNS

    if tuple(result.columns) != tuple(BACKTEST_RESULT_COLUMNS):
        raise BacktestResultCacheValidationError(
            ("backtest_cache.columns.exact_order",)
        )

    expected_keys = tuple(
        (budget, method)
        for budget in identity.budgets
        for method in BACKTEST_METHODS
    )
    records = tuple(result.itertuples(index=False, name=None))
    issues: list[str] = []
    if len(records) != len(expected_keys):
        issues.append("backtest_cache.row_cardinality")

    actual_keys: list[tuple[Decimal, str]] = []
    for raw_budget, raw_method, raw_caught in records:
        budget = _normalize_backtest_result_budget(raw_budget)
        if budget is None:
            issues.append("backtest_cache.budget")
        if raw_method not in BACKTEST_METHODS:
            issues.append("backtest_cache.method")
        if (
            isinstance(raw_caught, (bool, np.bool_))
            or not isinstance(raw_caught, Integral)
            or int(raw_caught) < 0
            or int(raw_caught) > normalized_candidate_count
        ):
            issues.append("backtest_cache.caught")
        if budget is not None and raw_method in BACKTEST_METHODS:
            actual_keys.append((budget, raw_method))

    if tuple(actual_keys) != expected_keys:
        issues.append("backtest_cache.canonical_keys")
    if issues:
        raise BacktestResultCacheValidationError(issues)


class BacktestService:
    """Cache validated executions of the real leakage-free backtest."""

    def __init__(
        self,
        cache: ModelCache[Any] | None = None,
        *,
        run_fn: Callable[[pd.DataFrame, Iterable[object]], pd.DataFrame]
        | None = None,
        model_configuration: object = BACKTEST_MODEL_CONFIGURATION,
        planner_configuration: object = BACKTEST_PLANNER_CONFIGURATION,
    ) -> None:
        if cache is not None and not isinstance(cache, ModelCache):
            raise TypeError("cache must be a ModelCache.")
        if run_fn is None:
            import planner as planner_package

            run_fn = planner_package.run_backtest
        if not callable(run_fn):
            raise TypeError("run_fn must be callable.")

        self._cache = cache or ModelCache[Any]()
        self._run_fn = run_fn
        self._model_config_fingerprint = configuration_fingerprint(
            model_configuration
        )
        self._planner_config_fingerprint = configuration_fingerprint(
            planner_configuration
        )

    def create_identity(
        self,
        bundle: PanelBundle,
        budgets: Iterable[object],
    ) -> BacktestCacheIdentity:
        """Create the complete production backtest semantic identity."""

        production_bundle = _require_production_bundle(bundle)
        if isinstance(budgets, (str, bytes, bytearray)):
            raise TypeError("budgets must be an iterable of numeric values.")
        try:
            requested_budgets = tuple(budgets)
        except TypeError as exc:
            raise TypeError(
                "budgets must be an iterable of numeric values."
            ) from exc
        if not requested_budgets:
            raise ValueError("budgets must contain at least one value.")

        return BacktestCacheIdentity(
            mode=RuntimeMode.PRODUCTION,
            source_sha256=production_bundle.dataset_sha256,
            budgets=_normalize_budgets(requested_budgets),
            fit_start_year=PRODUCTION_BACKTEST_FIT_START_YEAR,
            fit_end_year=PRODUCTION_BACKTEST_FIT_END_YEAR,
            candidate_year=PRODUCTION_BACKTEST_CANDIDATE_YEAR,
            outcome_year=PRODUCTION_BACKTEST_OUTCOME_YEAR,
            as_of=PRODUCTION_BACKTEST_AS_OF,
            model_config_fingerprint=self._model_config_fingerprint,
            planner_config_fingerprint=self._planner_config_fingerprint,
        )

    def run(
        self,
        bundle: PanelBundle,
        budgets: Iterable[object],
        *,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Return one validated cached production backtest result."""

        production_bundle = _require_production_bundle(bundle)
        identity = self.create_identity(production_bundle, budgets)
        adapted_panel = production_bundle.adapted
        candidate_count = int(
            adapted_panel["year"].eq(PRODUCTION_BACKTEST_CANDIDATE_YEAR).sum()
        )

        def operation() -> pd.DataFrame:
            return self._run_fn(
                adapted_panel.copy(deep=True),
                identity.budgets,
            )

        def validator(result: pd.DataFrame) -> None:
            validate_cached_backtest_result(
                result,
                identity,
                candidate_count=candidate_count,
            )

        result = self._cache.get_data(
            identity,
            operation,
            validator,
            force_recompute=force_recompute,
        )
        if not isinstance(result, pd.DataFrame):
            raise BacktestResultCacheValidationError(
                ("backtest_cache.frame_type",)
            )
        return result.copy(deep=True)


class ProductionProvider:
    """Lazy production-only composition over real repository components.

    Each public method is an independent exception boundary for the later
    Streamlit composition root.  Loading source data never initializes the
    model or backtest; ranking does not run selection or backtest; selection
    does not run the backtest; and a backtest failure cannot erase a validated
    current ranking held by its caller.
    """

    __slots__ = ("_cache", "_current_model_service", "_backtest_service")

    def __init__(self, cache: ModelCache[Any] | None = None) -> None:
        if cache is not None and not isinstance(cache, ModelCache):
            raise TypeError("cache must be a ModelCache.")
        shared_cache = cache or ModelCache[Any]()
        self._cache = shared_cache
        self._current_model_service = CurrentModelService(shared_cache)
        self._backtest_service = BacktestService(shared_cache)
        validate_provider_mode(self, RuntimeMode.PRODUCTION)

    @property
    def mode(self) -> RuntimeMode:
        return RuntimeMode.PRODUCTION

    @property
    def provenance(self) -> Provenance:
        return PRODUCTION_PROVENANCE

    def create_source_identity(self) -> SourceCacheIdentity:
        """Hash the exact checked-in production asset for cache lookup."""

        return SourceCacheIdentity(
            mode=RuntimeMode.PRODUCTION,
            source_path_identity=PRODUCTION_SOURCE_PATH_IDENTITY,
            source_sha256=file_content_digest(PRODUCTION_SOURCE_PATH),
            schema_version=SOURCE_SCHEMA_VERSION,
        )

    def load_source_bundle(
        self,
        *,
        force_recompute: bool = False,
    ) -> PanelBundle:
        """Load, validate, adapt, and cache only ``bridges/panel.parquet``."""

        identity = self.create_source_identity()

        def operation() -> PanelBundle:
            from model import read_source_panel

            source = read_source_panel(PRODUCTION_SOURCE_PATH)
            # Reject a mid-read replacement rather than storing bytes under the
            # digest of a different file revision.
            if file_content_digest(PRODUCTION_SOURCE_PATH) != identity.source_sha256:
                raise SourceBundleValidationError(
                    ("source_bundle.changed_during_load",)
                )
            return PanelBundle(
                source=source,
                dataset_sha256=identity.source_sha256,
                mode=RuntimeMode.PRODUCTION,
                source_path_identity=identity.source_path_identity,
                schema_version=identity.schema_version,
            )

        def validator(bundle: PanelBundle) -> None:
            validate_panel_bundle(bundle, identity)

        bundle = self._cache.get_data(
            identity,
            operation,
            validator,
            force_recompute=force_recompute,
        )
        _require_production_bundle(bundle)
        return bundle

    # Concise source alias for composition roots that name operations rather
    # than value types.
    load_source = load_source_bundle

    def build_current_ranking(
        self,
        bundle: PanelBundle,
        *,
        force_recompute: bool = False,
    ) -> CurrentRanking:
        """Build the real complete current ranking through CurrentModelService."""

        production_bundle = _require_production_bundle(bundle)
        return self._current_model_service.build_current_ranking(
            production_bundle,
            force_recompute=force_recompute,
        )

    def build_selection_context(
        self,
        current_ranking: CurrentRanking,
        active_budget: object,
    ) -> object:
        """Create one real statewide selection, metrics, and record snapshot."""

        if not isinstance(current_ranking, CurrentRanking):
            raise TypeError("current_ranking must be a CurrentRanking.")
        if current_ranking.model_state_identity.mode is not RuntimeMode.PRODUCTION:
            raise ValueError("Current ranking must have production provenance.")

        from app.state import build_selection_context as make_selection_context
        from planner import select, summarize_selection

        scored = current_ranking.scored
        selection = select(scored, budget=active_budget)
        metrics = summarize_selection(scored, selection)

        records = current_ranking.records
        if selection.ids:
            by_identifier = records.set_index("structure_id", drop=False)
            selected_records = by_identifier.loc[
                list(selection.ids)
            ].reset_index(drop=True)
        else:
            selected_records = records.iloc[0:0].copy(deep=True)

        score_identity = deterministic_content_digest(
            (
                "validated-current-ranking-v1",
                current_ranking.model_state_identity,
                dataframe_content_digest(scored, sort_by="id"),
            )
        )
        return make_selection_context(
            active_budget=active_budget,
            selection=selection,
            metrics=metrics,
            selected_records=selected_records,
            score_identity=score_identity,
        )

    def run_backtest(
        self,
        bundle: PanelBundle,
        budgets: Iterable[object],
        *,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Run the independently cached real Backtest_Engine branch."""

        return self._backtest_service.run(
            bundle,
            budgets,
            force_recompute=force_recompute,
        )


def build_production_provider(
    cache: ModelCache[Any] | None = None,
) -> ProductionProvider:
    """Construct the sole real provider branch without loading any data."""

    provider = ProductionProvider(cache)
    validate_provider_mode(provider, RuntimeMode.PRODUCTION)
    if provider.provenance != PRODUCTION_PROVENANCE:
        raise ValueError("Production provider provenance is invalid.")
    return provider


# Concise aliases preserve the domain terminology used by callers and tests.
SourceIdentity = SourceCacheIdentity
ModelStateIdentity = ModelStateCacheIdentity
ScoredTableIdentity = ScoredTableCacheIdentity
BacktestIdentity = BacktestCacheIdentity
ValidatedSourceBundle = PanelBundle
ProductionProviders = ProductionProvider
build_production_providers = build_production_provider


__all__ = [
    "BACKTEST_MODEL_CONFIGURATION",
    "BACKTEST_PLANNER_CONFIGURATION",
    "BacktestCacheIdentity",
    "BacktestIdentity",
    "BacktestResultCacheValidationError",
    "BacktestService",
    "CURRENT_FIT_END_YEAR",
    "CURRENT_FIT_START_YEAR",
    "CURRENT_MODEL_CONFIGURATION",
    "CURRENT_MODEL_FAMILY",
    "CURRENT_MODEL_HORIZON_MONTHS",
    "CURRENT_RANKING_AS_OF",
    "CURRENT_RANKING_YEAR",
    "CURRENT_SCORING_CONFIGURATION",
    "CacheIdentity",
    "CacheNamespace",
    "CacheStorage",
    "CacheValidationError",
    "CachedFunction",
    "CurrentModelService",
    "CurrentModelValidationError",
    "CurrentPanelBundle",
    "CurrentRanking",
    "ModelCache",
    "ModelStateCacheIdentity",
    "ModelStateIdentity",
    "PRODUCTION_BACKTEST_AS_OF",
    "PRODUCTION_BACKTEST_CANDIDATE_YEAR",
    "PRODUCTION_BACKTEST_FIT_END_YEAR",
    "PRODUCTION_BACKTEST_FIT_START_YEAR",
    "PRODUCTION_BACKTEST_OUTCOME_YEAR",
    "PRODUCTION_PROVENANCE",
    "PRODUCTION_SOURCE_PATH",
    "PRODUCTION_SOURCE_PATH_IDENTITY",
    "PanelBundle",
    "ProductionProvider",
    "ProductionProviders",
    "SOURCE_SCHEMA_VERSION",
    "ScoredTableCacheIdentity",
    "ScoredTableIdentity",
    "SourceBundleValidationError",
    "SourceCacheIdentity",
    "SourceIdentity",
    "ValidatedSourceBundle",
    "build_production_provider",
    "build_production_providers",
    "bytes_content_digest",
    "configuration_fingerprint",
    "dataframe_content_digest",
    "deterministic_content_digest",
    "file_content_digest",
    "validate_cached_backtest_result",
    "validate_current_model_state",
    "validate_panel_bundle",
]
