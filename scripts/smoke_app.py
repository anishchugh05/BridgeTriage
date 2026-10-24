"""Run one bounded, network-isolated production Streamlit smoke check.

The script is intentionally standalone and noninteractive.  It executes the
checked-in Streamlit entry point exactly once through ``AppTest``, then checks
that the initial render proves the real source, model, planner, and backtest
branches initialized without exposing internal exception details on failure.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Final
from unittest.mock import patch

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
APP_ENTRYPOINT: Final[str] = "app/app.py"
PRODUCTION_PANEL: Final[Path] = REPOSITORY_ROOT / "bridges" / "panel.parquet"
PRODUCTION_MODE_VARIABLE: Final[str] = "BRIDGE_APP_MODE"
PRODUCTION_MODE_VALUE: Final[str] = "production"
SMOKE_TIMEOUT_SECONDS: Final[int] = 120

PRODUCTION_PROVENANCE_LABELS: Final[tuple[str, ...]] = (
    "Real_Dataset",
    "Real_Model_Service",
    "Real_Planner",
    "Backtest_Engine",
)
DEMO_TEXT_MARKERS: Final[tuple[str, ...]] = (
    "DEMO MODE - FABRICATED DATA",
    "Demo_Data",
    "Demo_Model_Service",
    "Demo_Planner",
    "Demo_Backtest",
)
CURRENT_RANKING_LABEL: Final[str] = "Current ranking"
MAP_SECTION_LABEL: Final[str] = "Map"
EXPECTED_TAB_LABELS: Final[tuple[str, ...]] = (
    "Budget",
    "Backtest",
    "Bridge detail",
    "Hazards",
    "Export",
    "Methods",
)
EXPECTED_SECTION_LABELS: Final[tuple[str, ...]] = (
    MAP_SECTION_LABEL,
    *EXPECTED_TAB_LABELS,
)
EXPECTED_SUMMARY_METRICS: Final[frozenset[str]] = frozenset(
    {
        "Active budget",
        "Selected bridges",
        "Ranking population",
        "Selected total modeled inspection-cost estimate",
        "Expected Safety_Drop coverage",
    }
)


class SmokeCheckFailure(RuntimeError):
    """A deterministic, non-sensitive production smoke-check failure."""


class UnexpectedExternalNetworkAccess(RuntimeError):
    """Raised when code attempts an external connection during AppTest."""

    def __init__(self) -> None:
        super().__init__("External network access is disabled during smoke checks.")


@dataclass(slots=True)
class _NetworkGuard:
    attempted_external_access: bool = False

    def reject(self) -> None:
        self.attempted_external_access = True
        raise UnexpectedExternalNetworkAccess()


def _host_from_address(address: object) -> object | None:
    if isinstance(address, tuple) and address:
        return address[0]
    return None


def _is_loopback_host(host: object) -> bool:
    """Return whether a socket host is explicitly local to this process."""

    if host is None:
        return True
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError:
            return False
    if not isinstance(host, str):
        return False

    normalized = host.rstrip(".").casefold()
    if normalized == "localhost":
        return True
    try:
        address = ipaddress.ip_address(normalized.split("%", 1)[0])
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped is not None and mapped.is_loopback)


def _is_local_socket_destination(family: int, address: object) -> bool:
    unix_family = getattr(socket, "AF_UNIX", None)
    if unix_family is not None and family == unix_family:
        return True
    return _is_loopback_host(_host_from_address(address))


@contextmanager
def _block_external_network() -> Iterator[_NetworkGuard]:
    """Deny and record outbound non-loopback socket activity.

    AppTest runs in process, so guarding socket connection, resolution, and
    connectionless-send entry points covers ordinary HTTP clients without
    starting a server or requiring a test-only dependency.  Loopback and Unix
    sockets remain available for framework internals, but no external hostname
    is resolved and no external address is contacted.
    """

    guard = _NetworkGuard()
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_sendto = socket.socket.sendto

    def guarded_connect(sock: socket.socket, address: object) -> Any:
        if not _is_local_socket_destination(sock.family, address):
            guard.reject()
        return original_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: object) -> int:
        if not _is_local_socket_destination(sock.family, address):
            guard.reject()
        return original_connect_ex(sock, address)

    def guarded_create_connection(
        address: tuple[str, int],
        timeout: float | object = socket._GLOBAL_DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
        *,
        all_errors: bool = False,
    ) -> socket.socket:
        if not _is_loopback_host(_host_from_address(address)):
            guard.reject()
        return original_create_connection(
            address,
            timeout,
            source_address,
            all_errors=all_errors,
        )

    def guarded_getaddrinfo(
        host: object,
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[Any, ...]]:
        if not _is_loopback_host(host):
            guard.reject()
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_sendto(sock: socket.socket, data: bytes, *args: Any) -> int:
        # sendto(data, address) and sendto(data, flags, address) both place the
        # destination last.  A connected datagram with no destination was
        # necessarily connected through the guarded connect entry point.
        if args:
            destination = args[-1]
            if not _is_local_socket_destination(sock.family, destination):
                guard.reject()
        return original_sendto(sock, data, *args)

    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket, "connect", guarded_connect))
        stack.enter_context(
            patch.object(socket.socket, "connect_ex", guarded_connect_ex)
        )
        stack.enter_context(
            patch.object(socket, "create_connection", guarded_create_connection)
        )
        stack.enter_context(patch.object(socket, "getaddrinfo", guarded_getaddrinfo))
        stack.enter_context(patch.object(socket.socket, "sendto", guarded_sendto))
        yield guard


@contextmanager
def _repository_working_directory() -> Iterator[None]:
    """Run the AppTest process context from the repository root."""

    previous = Path.cwd()
    os.chdir(REPOSITORY_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def _validate_local_inputs() -> None:
    app_path = REPOSITORY_ROOT / APP_ENTRYPOINT
    if not app_path.is_file():
        raise SmokeCheckFailure(
            "the required Streamlit entry point is missing: app/app.py"
        )
    try:
        panel_is_usable = PRODUCTION_PANEL.is_file() and PRODUCTION_PANEL.stat().st_size > 0
    except OSError:
        panel_is_usable = False
    if not panel_is_usable:
        raise SmokeCheckFailure(
            "the required production asset is missing or unreadable: "
            "bridges/panel.parquet"
        )


def _string_values(elements: Sequence[Any]) -> tuple[str, ...]:
    values: list[str] = []
    for element in elements:
        value = getattr(element, "value", None)
        if isinstance(value, str):
            values.append(value)
    return tuple(values)


def _rendered_text(app_test: Any) -> str:
    """Collect user-visible scalar text without serializing tables or errors."""

    collection_names = (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "text",
        "info",
        "warning",
        "success",
    )
    values: list[str] = []
    for name in collection_names:
        values.extend(_string_values(getattr(app_test, name)))
    values.extend(
        label
        for label in (getattr(tab, "label", None) for tab in app_test.tabs)
        if isinstance(label, str)
    )
    values.extend(
        label
        for label in (getattr(metric, "label", None) for metric in app_test.metric)
        if isinstance(label, str)
    )
    return "\n".join(values)


def _assert_initial_render(app_test: Any) -> None:
    """Validate the complete initial production render from one AppTest tree."""

    if len(app_test.exception) != 0:
        raise SmokeCheckFailure("the application reported an uncaught exception")
    if len(app_test.error) != 0:
        raise SmokeCheckFailure("the application rendered a production error state")

    rendered_text = _rendered_text(app_test)
    missing_provenance = tuple(
        label
        for label in PRODUCTION_PROVENANCE_LABELS
        if label not in rendered_text
    )
    if missing_provenance:
        raise SmokeCheckFailure(
            "production provenance was incomplete; missing: "
            + ", ".join(missing_provenance)
        )

    present_demo_markers = tuple(
        marker for marker in DEMO_TEXT_MARKERS if marker in rendered_text
    )
    if present_demo_markers:
        raise SmokeCheckFailure(
            "demo indicator or demo provenance appeared in production output"
        )

    headers = _string_values(app_test.header)
    if CURRENT_RANKING_LABEL not in headers:
        raise SmokeCheckFailure("the current-ranking headline did not render")
    if not any("bridges selected" in value for value in _string_values(app_test.markdown)):
        raise SmokeCheckFailure("the current-ranking selection headline was incomplete")

    subheaders = _string_values(app_test.subheader)
    missing_sections = tuple(
        label for label in EXPECTED_SECTION_LABELS if label not in subheaders
    )
    if missing_sections:
        raise SmokeCheckFailure(
            "required application sections did not render; missing: "
            + ", ".join(missing_sections)
        )

    tab_labels = tuple(
        label
        for label in (getattr(tab, "label", None) for tab in app_test.tabs)
        if isinstance(label, str)
    )
    if tab_labels != EXPECTED_TAB_LABELS:
        raise SmokeCheckFailure("secondary application sections were incomplete or reordered")

    if len(app_test.get("deck_gl_json_chart")) != 1:
        raise SmokeCheckFailure("the statewide PyDeck map did not render exactly once")
    if len(app_test.get("vega_lite_chart")) < 1:
        raise SmokeCheckFailure("the measured backtest chart did not render")

    metric_labels = {
        label
        for label in (getattr(metric, "label", None) for metric in app_test.metric)
        if isinstance(label, str)
    }
    missing_metrics = EXPECTED_SUMMARY_METRICS - metric_labels
    if missing_metrics:
        raise SmokeCheckFailure(
            "the current-ranking model/planner summary was incomplete; missing: "
            + ", ".join(sorted(missing_metrics))
        )
    if len(app_test.download_button) != 1:
        raise SmokeCheckFailure("the selected-plan export control did not render")


def run_smoke_check() -> float:
    """Execute exactly one production AppTest and return elapsed seconds."""

    _validate_local_inputs()
    try:
        from streamlit.testing.v1 import AppTest
    except (ImportError, AttributeError):
        raise SmokeCheckFailure("Streamlit AppTest is unavailable") from None

    started = monotonic()
    network_guard: _NetworkGuard | None = None
    try:
        with (
            _repository_working_directory(),
            patch.dict(
                os.environ,
                {PRODUCTION_MODE_VARIABLE: PRODUCTION_MODE_VALUE},
                clear=False,
            ),
            _block_external_network() as active_network_guard,
        ):
            network_guard = active_network_guard
            # AppTest resolves relative paths against this runner file rather
            # than the process working directory, so anchor the approved
            # ``app/app.py`` entry point to the checked-out repository.
            app_test = AppTest.from_file(REPOSITORY_ROOT / APP_ENTRYPOINT)
            app_test.run(timeout=SMOKE_TIMEOUT_SECONDS)
            if network_guard.attempted_external_access:
                raise SmokeCheckFailure(
                    "unexpected external network access was attempted"
                )
            _assert_initial_render(app_test)
    except SmokeCheckFailure:
        raise
    except UnexpectedExternalNetworkAccess:
        raise SmokeCheckFailure(
            "unexpected external network access was attempted"
        ) from None
    except TimeoutError:
        raise SmokeCheckFailure(
            "the initial production render exceeded 120 seconds"
        ) from None
    except Exception as exc:
        # Exception details may include local paths, environment content, or
        # source rows.  Report only the stable exception class to the caller.
        raise SmokeCheckFailure(
            "AppTest could not complete the production render "
            f"({type(exc).__name__})"
        ) from None

    elapsed = monotonic() - started
    if elapsed > SMOKE_TIMEOUT_SECONDS:
        raise SmokeCheckFailure(
            "the initial production render and assertions exceeded 120 seconds"
        )
    return elapsed


def main() -> int:
    """Return a process-compatible success or failure status."""

    try:
        elapsed = run_smoke_check()
    except SmokeCheckFailure as exc:
        print(f"Production smoke check failed: {exc}.", file=sys.stderr)
        return 1
    except Exception as exc:  # defensive runner failure, still sanitized
        print(
            "Production smoke check failed: internal runner failure "
            f"({type(exc).__name__}).",
            file=sys.stderr,
        )
        return 1

    print(f"Production smoke check passed in {elapsed:.2f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
