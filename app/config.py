"""Runtime-mode and provenance contracts for the application.

Mode resolution is intentionally exact: only a missing ``BRIDGE_APP_MODE``
setting or one of the two documented lowercase literals is accepted.  This
module does not construct providers or access application data, allowing the
composition root to resolve mode before either operation occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable

RUNTIME_MODE_ENV_VAR = "BRIDGE_APP_MODE"


class RuntimeMode(str, Enum):
    """The two mutually exclusive application runtime modes."""

    PRODUCTION = "production"
    DEMO = "demo"


class InvalidRuntimeMode(ValueError):
    """Raised when a present runtime-mode setting is not an exact literal."""

    def __init__(self) -> None:
        # Keep the failure stable and avoid reflecting an environment value.
        super().__init__(
            "BRIDGE_APP_MODE must be exactly 'production' or 'demo' when set."
        )


class ProviderModeMismatch(ValueError):
    """Raised when a provider's mode tag differs from the active mode."""

    def __init__(self) -> None:
        super().__init__("Provider mode does not match the active runtime mode.")


@dataclass(frozen=True, slots=True)
class Provenance:
    """Immutable provenance labels associated with one provider branch."""

    mode: RuntimeMode
    data_source: str
    model_source: str
    planner_source: str
    backtest_source: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RuntimeMode):
            raise TypeError("Provenance mode must be a RuntimeMode.")

        labels = (
            self.data_source,
            self.model_source,
            self.planner_source,
            self.backtest_source,
        )
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("Provenance source labels must be nonempty strings.")


@runtime_checkable
class ModeTaggedProvider(Protocol):
    """Structural contract implemented by every production or demo provider."""

    @property
    def mode(self) -> RuntimeMode:
        """Return the provider branch's immutable runtime-mode tag."""
        ...


def validate_provider_mode(
    provider: ModeTaggedProvider,
    active_mode: RuntimeMode,
) -> None:
    """Reject a provider whose explicit tag is absent, invalid, or mismatched."""

    if not isinstance(active_mode, RuntimeMode):
        raise TypeError("Active mode must be a RuntimeMode.")

    provider_mode = getattr(provider, "mode", None)
    if not isinstance(provider_mode, RuntimeMode) or provider_mode is not active_mode:
        raise ProviderModeMismatch()


def resolve_runtime_mode(env: Mapping[str, str]) -> RuntimeMode:
    """Resolve the active mode without trimming, coercion, or case folding.

    A missing ``BRIDGE_APP_MODE`` key selects production.  A present key is
    accepted only when its value is exactly ``"production"`` or ``"demo"``;
    every other present value raises before callers can construct a provider
    branch or access data.
    """

    if RUNTIME_MODE_ENV_VAR not in env:
        return RuntimeMode.PRODUCTION

    setting = env[RUNTIME_MODE_ENV_VAR]
    if setting == RuntimeMode.PRODUCTION.value:
        return RuntimeMode.PRODUCTION
    if setting == RuntimeMode.DEMO.value:
        return RuntimeMode.DEMO

    raise InvalidRuntimeMode()


__all__ = [
    "InvalidRuntimeMode",
    "ModeTaggedProvider",
    "Provenance",
    "ProviderModeMismatch",
    "RUNTIME_MODE_ENV_VAR",
    "RuntimeMode",
    "resolve_runtime_mode",
    "validate_provider_mode",
]
