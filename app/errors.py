"""Safe, dependency-scoped error boundaries for the application.

Internal exceptions may contain diagnostic detail for server logs.  They are
never retained in :class:`Outcome` or passed to the renderer; public errors are
created exclusively from the fixed registry below.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Lock
from types import MappingProxyType
from typing import Any, ClassVar, Generic, Mapping, TypeVar, cast


T = TypeVar("T")


class ErrorCategory(str, Enum):
    """Stable categories used to select public diagnostics."""

    RUNTIME_MODE = "MODE-001"
    DATA_LOAD = "DATA-001"
    SOURCE_SCHEMA = "SCHEMA-001"
    IDENTIFIER_RECONCILIATION = "RECON-001"
    MODEL_FIT = "MODEL-FIT-001"
    MODEL_SCORE = "MODEL-SCORE-001"
    PLANNER = "PLANNER-001"
    BACKTEST = "BACKTEST-001"
    BRIDGE_DETAIL = "DETAIL-001"
    EXPORT = "EXPORT-001"


@dataclass(frozen=True, slots=True)
class ErrorDefinition:
    """One immutable entry in the public error registry."""

    code: str
    component: str
    summary: str


_ERROR_DEFINITIONS = (
    ErrorDefinition(
        "MODE-001",
        "Runtime_Mode_Controller",
        "Runtime mode configuration is invalid.",
    ),
    ErrorDefinition(
        "DATA-001",
        "Real_Dataset",
        "Production bridge data could not be loaded.",
    ),
    ErrorDefinition(
        "SCHEMA-001",
        "Schema_Adapter",
        "Production bridge data failed validation.",
    ),
    ErrorDefinition(
        "RECON-001",
        "Schema_Adapter",
        "Model results could not be matched to bridge records.",
    ),
    ErrorDefinition(
        "MODEL-FIT-001",
        "Real_Model_Service",
        "The decline model could not be initialized.",
    ),
    ErrorDefinition(
        "MODEL-SCORE-001",
        "Real_Model_Service",
        "Bridge scores could not be validated.",
    ),
    ErrorDefinition(
        "PLANNER-001",
        "Real_Planner",
        "The inspection selection could not be created.",
    ),
    ErrorDefinition(
        "BACKTEST-001",
        "Backtest_Engine",
        "Historical comparison results are unavailable.",
    ),
    ErrorDefinition(
        "DETAIL-001",
        "Bridge detail",
        "The selected bridge detail is unavailable.",
    ),
    ErrorDefinition(
        "EXPORT-001",
        "Export",
        "The selected plan could not be exported.",
    ),
)

ERROR_REGISTRY: Mapping[str, ErrorDefinition] = MappingProxyType(
    {definition.code: definition for definition in _ERROR_DEFINITIONS}
)
# Explicit alias for callers that prefer the longer contract name.
ERROR_CODE_REGISTRY = ERROR_REGISTRY

if set(ERROR_REGISTRY) != {category.value for category in ErrorCategory}:
    raise RuntimeError("The public error registry does not match ErrorCategory.")


class BridgeAppError(Exception):
    """Base class for typed internal application failures."""

    category: ClassVar[ErrorCategory]


class InvalidRuntimeMode(BridgeAppError):
    """The configured runtime mode was not an accepted exact value."""

    category = ErrorCategory.RUNTIME_MODE


class DataLoadError(BridgeAppError):
    """The production dataset could not be read."""

    category = ErrorCategory.DATA_LOAD


class SourceSchemaError(BridgeAppError):
    """The source panel violated its published schema."""

    category = ErrorCategory.SOURCE_SCHEMA


class IdentifierReconciliationError(BridgeAppError):
    """Model and source identifiers could not be reconciled one-to-one."""

    category = ErrorCategory.IDENTIFIER_RECONCILIATION


class ModelFitError(BridgeAppError):
    """The decline model could not be initialized or validated."""

    category = ErrorCategory.MODEL_FIT


class ModelScoreError(BridgeAppError):
    """Model scoring failed or returned an invalid scored table."""

    category = ErrorCategory.MODEL_SCORE


class PlannerError(BridgeAppError):
    """Planner construction, validation, or selection failed."""

    category = ErrorCategory.PLANNER


class BacktestError(BridgeAppError):
    """Backtest construction, execution, or validation failed."""

    category = ErrorCategory.BACKTEST


class BridgeDetailError(BridgeAppError):
    """A bridge-detail projection could not be produced."""

    category = ErrorCategory.BRIDGE_DETAIL


class ExportError(BridgeAppError):
    """A selected-plan export could not be produced."""

    category = ErrorCategory.EXPORT


# Compatibility names describe the same typed boundaries without creating a
# second category or public mapping.
InvalidRuntimeModeError = InvalidRuntimeMode
DatasetLoadError = DataLoadError
ModelInitializationError = ModelFitError
ScoredTableValidationError = ModelScoreError
BacktestValidationError = BacktestError


@dataclass(frozen=True, slots=True)
class UserSafeError:
    """The complete public diagnostic model—exactly three fixed fields."""

    component: str
    summary: str
    code: str

    def __post_init__(self) -> None:
        definition = ERROR_REGISTRY.get(self.code)
        if definition is None or (
            self.component != definition.component
            or self.summary != definition.summary
        ):
            raise ValueError("UserSafeError must match the fixed error registry.")

    def as_public_fields(self) -> dict[str, str]:
        """Return only the three fields permitted in public output."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class Outcome(Generic[T]):
    """A component result containing either a value or a safe error."""

    value: T | None = None
    error: UserSafeError | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.error is None):
            raise ValueError("Outcome must contain exactly one of value or error.")

    @classmethod
    def success(cls, value: T) -> "Outcome[T]":
        if value is None:
            raise ValueError("A successful Outcome requires a non-None value.")
        return cls(value=value)

    @classmethod
    def failure(cls, error: UserSafeError) -> "Outcome[T]":
        return cls(error=error)

    @property
    def is_success(self) -> bool:
        return self.error is None

    @property
    def is_failure(self) -> bool:
        return self.error is not None


ErrorSelector = ErrorCategory | ErrorDefinition | str | type[BridgeAppError] | BridgeAppError


def _definition_for(selector: ErrorSelector) -> ErrorDefinition:
    if isinstance(selector, BridgeAppError):
        code = selector.category.value
    elif isinstance(selector, type) and issubclass(selector, BridgeAppError):
        code = selector.category.value
    elif isinstance(selector, ErrorDefinition):
        code = selector.code
    elif isinstance(selector, ErrorCategory):
        code = selector.value
    elif isinstance(selector, str):
        code = selector
    else:  # pragma: no cover - guarded by the public type contract
        raise TypeError("Unsupported error category selector.")

    try:
        definition = ERROR_REGISTRY[code]
    except KeyError as exc:
        raise KeyError("Unknown error category.") from exc

    if isinstance(selector, ErrorDefinition) and selector != definition:
        raise ValueError("Error definition must match the fixed registry.")
    return definition


def user_safe_error(selector: ErrorSelector) -> UserSafeError:
    """Create a safe diagnostic solely from a registered category."""

    definition = _definition_for(selector)
    return UserSafeError(
        component=definition.component,
        summary=definition.summary,
        code=definition.code,
    )


_LOGGER = logging.getLogger("bridge_inspection_planner")
_LOGGED_MARKER = "_bridge_app_failure_logged"
_LOGGED_LOCK = Lock()
# BaseException instances normally accept marker attributes.  This identity
# list covers unusual immutable exception implementations without using IDs,
# which could later be reused for a different exception.
_UNMARKABLE_LOGGED: list[BaseException] = []


def _claim_for_logging(exception: BaseException) -> bool:
    """Atomically claim an exception object for its sole log event."""

    with _LOGGED_LOCK:
        if getattr(exception, _LOGGED_MARKER, False):
            return False
        try:
            setattr(exception, _LOGGED_MARKER, True)
        except (AttributeError, TypeError):
            if any(item is exception for item in _UNMARKABLE_LOGGED):
                return False
            _UNMARKABLE_LOGGED.append(exception)
        return True


def log_failure_once(
    exception: BaseException,
    public_error: UserSafeError,
    logger: logging.Logger | None = None,
) -> None:
    """Emit one structured server event for this exception object.

    Exception class and message are intentionally server-only.  The associated
    :class:`UserSafeError` remains the sole value returned to presentation code.
    """

    if not isinstance(exception, BaseException):
        raise TypeError("exception must derive from BaseException.")
    if not isinstance(public_error, UserSafeError):
        raise TypeError("public_error must be a UserSafeError.")
    if not _claim_for_logging(exception):
        return

    try:
        exception_message = str(exception)
    except Exception:  # an exception's custom __str__ must not break handling
        exception_message = "<unprintable exception>"

    payload = {
        "error_code": public_error.code,
        "component": public_error.component,
        "exception_class": type(exception).__name__,
        "exception_message": exception_message,
    }
    target_logger = logger if logger is not None else _LOGGER
    target_logger.error(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        extra=payload,
    )


def run_component(
    component: str,
    operation: Callable[[], T],
    error_category: ErrorSelector,
    logger: logging.Logger | None = None,
) -> Outcome[T]:
    """Run one dependency boundary and return a value or fixed safe error.

    The declared component must match the registry entry.  This prevents caller
    data—including paths, environment values, credentials, or raw rows—from
    becoming a public component label.
    """

    definition = _definition_for(error_category)
    if component != definition.component:
        raise ValueError("Component must match the fixed error registry.")
    if not callable(operation):
        raise TypeError("operation must be callable.")

    try:
        return Outcome.success(operation())
    except Exception as exception:
        public_error = user_safe_error(definition)
        log_failure_once(exception, public_error, logger)
        return Outcome.failure(public_error)


def render_user_safe_error(
    public_error: UserSafeError,
    renderer: Callable[[str], Any] | None = None,
) -> None:
    """Render exactly component, fixed summary, and code.

    ``renderer`` is injectable for tests and adapters.  When omitted, Streamlit
    is imported lazily so non-UI consumers can use error boundaries without
    importing the presentation framework.
    """

    if not isinstance(public_error, UserSafeError):
        raise TypeError("public_error must be a UserSafeError.")

    if renderer is None:
        import streamlit as st

        renderer = cast(Callable[[str], Any], st.error)

    renderer(
        f"component: {public_error.component}\n\n"
        f"summary: {public_error.summary}\n\n"
        f"code: {public_error.code}"
    )


__all__ = [
    "BacktestError",
    "BacktestValidationError",
    "BridgeAppError",
    "BridgeDetailError",
    "DataLoadError",
    "DatasetLoadError",
    "ERROR_CODE_REGISTRY",
    "ERROR_REGISTRY",
    "ErrorCategory",
    "ErrorDefinition",
    "ExportError",
    "IdentifierReconciliationError",
    "InvalidRuntimeMode",
    "InvalidRuntimeModeError",
    "ModelFitError",
    "ModelInitializationError",
    "ModelScoreError",
    "Outcome",
    "PlannerError",
    "ScoredTableValidationError",
    "SourceSchemaError",
    "UserSafeError",
    "log_failure_once",
    "render_user_safe_error",
    "run_component",
    "user_safe_error",
]
