"""Validated cache identities and transactional Streamlit cache wrappers.

This module deliberately contains no model orchestration or provider composition.
It defines the cache boundary used by those later services: immutable semantic
identities, deterministic digest helpers, and a :class:`ModelCache` that admits
only values which pass their validator.
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

from app.config import RuntimeMode


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


# Concise aliases preserve the domain terminology used by callers and tests.
SourceIdentity = SourceCacheIdentity
ModelStateIdentity = ModelStateCacheIdentity
ScoredTableIdentity = ScoredTableCacheIdentity
BacktestIdentity = BacktestCacheIdentity


__all__ = [
    "BacktestCacheIdentity",
    "BacktestIdentity",
    "CacheIdentity",
    "CacheNamespace",
    "CacheStorage",
    "CacheValidationError",
    "CachedFunction",
    "ModelCache",
    "ModelStateCacheIdentity",
    "ModelStateIdentity",
    "ScoredTableCacheIdentity",
    "ScoredTableIdentity",
    "SourceCacheIdentity",
    "SourceIdentity",
    "bytes_content_digest",
    "configuration_fingerprint",
    "dataframe_content_digest",
    "deterministic_content_digest",
    "file_content_digest",
]
