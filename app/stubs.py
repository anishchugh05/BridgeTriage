"""Retired automatic fallback module.

Fabricated data and calculations are available only from the explicit
``RuntimeMode.DEMO`` factory in :mod:`app.demo`.  This compatibility module
intentionally exports no scoring, selection, or backtest callable, so a
production import failure cannot silently activate demo behavior.
"""

from __future__ import annotations

_RETIRED_NAMES = frozenset({"score", "pick", "backtest", "METHODS"})


def __getattr__(name: str) -> object:
    if name in _RETIRED_NAMES:
        raise AttributeError(
            f"{name!r} is not available from app.stubs; "
            "construct app.demo providers only after exact demo-mode resolution."
        )
    raise AttributeError(name)


__all__: tuple[str, ...] = ()
