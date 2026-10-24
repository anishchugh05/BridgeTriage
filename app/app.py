from pathlib import Path
import sys

# Streamlit executes this nested file with ``app/`` as the script directory,
# so ``app/`` lands on ``sys.path`` ahead of the repository root. That lets the
# bare name ``app`` resolve to this directory as a plain (non-package) module,
# after which ``from app.config import ...`` raises
# "No module named 'app.config'; 'app' is not a package". Establish the
# checked-out repository as the import root, demote the script directory, and
# drop any ``app`` binding that resolved from the wrong location before any
# absolute application, model, or planner package import runs.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT_TEXT = str(REPOSITORY_ROOT)
_SCRIPT_DIRECTORY_TEXT = str(Path(__file__).resolve().parent)

sys.path[:] = [entry for entry in sys.path if entry != _SCRIPT_DIRECTORY_TEXT]
if _REPOSITORY_ROOT_TEXT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT_TEXT)

_app_module = sys.modules.get("app")
if _app_module is not None and not hasattr(_app_module, "__path__"):
    # ``app`` was bound to the script directory as a plain module; evict it so
    # the import below re-resolves ``app`` as a package under the repository root.
    del sys.modules["app"]

import os
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, cast

import pandas as pd
import streamlit as st

from app.config import RuntimeMode, resolve_runtime_mode, validate_provider_mode
from app.errors import (
    ERROR_REGISTRY,
    ErrorCategory,
    Outcome,
    log_failure_once,
    render_user_safe_error,
    run_component,
    user_safe_error,
)
from app.state import (
    ACTIVE_BUDGET_STEP,
    ACTIVE_BUDGET_VALUES,
    INITIAL_ACTIVE_BUDGET,
    MAX_ACTIVE_BUDGET,
    MIN_ACTIVE_BUDGET,
    SelectionContext,
    ViewContext,
    build_selection_context,
    get_active_budget,
)
from app.style import inject_css
from app.views import (
    BACKTEST_SECTION_LABEL,
    render_backtest,
    render_bridge_detail,
    render_budget,
    render_current_ranking,
    render_export,
    render_hazards,
    render_map,
    render_methods,
)


T = TypeVar("T")
PAGE_TITLE = "Massachusetts Bridge Inspection Planner"
BUDGET_WIDGET_KEY = "active_budget_widget"
COUNTY_WIDGET_KEY = "map_county_widget"
ALL_COUNTIES = "All"
BACKTEST_BUDGETS = tuple(ACTIVE_BUDGET_VALUES)

_SOURCE_ERROR_CATEGORIES: Mapping[str, ErrorCategory] = {
    "SourceReadError": ErrorCategory.DATA_LOAD,
    "SourceSchemaError": ErrorCategory.SOURCE_SCHEMA,
    "SourceAdaptationError": ErrorCategory.SOURCE_SCHEMA,
    "SourceBundleValidationError": ErrorCategory.SOURCE_SCHEMA,
    "DemoDataError": ErrorCategory.DATA_LOAD,
}
_CURRENT_RANKING_ERROR_CATEGORIES: Mapping[str, ErrorCategory] = {
    "SourceSchemaError": ErrorCategory.SOURCE_SCHEMA,
    "IdentifierReconciliationError": ErrorCategory.IDENTIFIER_RECONCILIATION,
    "ScoredDataError": ErrorCategory.MODEL_SCORE,
    "ScoredTableValidationError": ErrorCategory.MODEL_SCORE,
    "DemoCalculationError": ErrorCategory.MODEL_SCORE,
}


def _component_name(category: ErrorCategory) -> str:
    """Return the fixed registry component for one safe boundary."""

    return ERROR_REGISTRY[category.value].component


def _run_component(
    operation: Callable[[], T],
    category: ErrorCategory,
) -> Outcome[T]:
    """Run a fixed-category component through the shared error boundary."""

    return run_component(_component_name(category), operation, category)


def _run_classified_component(
    operation: Callable[[], T],
    default_category: ErrorCategory,
    categories_by_exception_name: Mapping[str, ErrorCategory],
) -> Outcome[T]:
    """Capture a compound provider operation under its precise safe category.

    Source loading includes read, schema, and adaptation stages, while current
    ranking includes fit, score validation, and identifier reconciliation. The
    provider APIs intentionally expose each workflow atomically, so this root
    classifies their typed exceptions without exposing exception text publicly.
    """

    try:
        return Outcome.success(operation())
    except Exception as exception:
        declared_category = getattr(exception, "category", None)
        if isinstance(declared_category, ErrorCategory):
            category = declared_category
        else:
            category = default_category
            for exception_type in type(exception).__mro__:
                matched = categories_by_exception_name.get(exception_type.__name__)
                if matched is not None:
                    category = matched
                    break

        public_error = user_safe_error(category)
        log_failure_once(exception, public_error)
        return Outcome.failure(public_error)


def _render_failure(outcome: Outcome[Any]) -> None:
    """Render the fixed three-field diagnostic from a failed outcome."""

    if outcome.error is None:
        raise ValueError("A successful outcome has no failure to render.")
    render_user_safe_error(outcome.error)


def _build_provider(mode: RuntimeMode) -> object:
    """Construct exactly one lazy provider branch for the resolved mode."""

    if mode is RuntimeMode.PRODUCTION:
        from app.services import build_production_provider

        provider = build_production_provider()
    elif mode is RuntimeMode.DEMO:
        # This import is intentionally unreachable from the production branch.
        from app.demo import build_demo_providers

        provider = build_demo_providers(mode)
    else:  # pragma: no cover - RuntimeMode is exhaustive
        raise ValueError("Unsupported resolved runtime mode.")

    validate_provider_mode(cast(Any, provider), mode)
    return provider


def _load_source(provider: object, mode: RuntimeMode) -> object:
    """Load only the source owned by the already-selected provider branch."""

    if mode is RuntimeMode.PRODUCTION:
        return cast(Any, provider).load_source_bundle()
    return cast(Any, provider).load_source()


def _build_current_ranking(
    provider: object,
    source: object,
) -> object:
    """Request one complete current ranking from the active provider."""

    return cast(Any, provider).build_current_ranking(source)


def _build_statewide_selection(
    provider: object,
    mode: RuntimeMode,
    ranking: object,
    active_budget: int,
) -> SelectionContext:
    """Create exactly one statewide selection for this Streamlit rerun."""

    if mode is RuntimeMode.PRODUCTION:
        context = cast(Any, provider).build_selection_context(
            ranking,
            active_budget,
        )
        if not isinstance(context, SelectionContext):
            raise TypeError("Production selection did not return SelectionContext.")
        return context

    selected = cast(Any, provider).select_current(ranking, active_budget)
    validate_provider_mode(selected, RuntimeMode.DEMO)
    return build_selection_context(
        active_budget=active_budget,
        selection=selected.selection,
        metrics=selected.metrics,
        selected_records=selected.selected_records,
        score_identity=selected.score_identity,
    )


def _run_backtest(
    provider: object,
    mode: RuntimeMode,
    source: object,
) -> object:
    """Request the independently computed backtest from the active branch."""

    if mode is RuntimeMode.PRODUCTION:
        return cast(Any, provider).run_backtest(source, BACKTEST_BUDGETS)

    result = cast(Any, provider).run_backtest(BACKTEST_BUDGETS)
    validate_provider_mode(result, RuntimeMode.DEMO)
    return result


def _county_options(ranking: object) -> tuple[str, ...]:
    """Return deterministic county choices from the complete ranking only."""

    records = getattr(ranking, "records", None)
    if not isinstance(records, pd.DataFrame) or "county" not in records.columns:
        raise TypeError("Current ranking must expose county records.")

    counties = tuple(records["county"].tolist())
    if any(not isinstance(county, str) or not county for county in counties):
        raise ValueError("Ranking counties must be nonempty strings.")
    return (ALL_COUNTIES, *sorted(set(counties)))


def _render_with_boundary(
    operation: Callable[[], object],
    category: ErrorCategory,
) -> None:
    """Keep a presentation failure inside its dependency-scoped region."""

    outcome = _run_component(operation, category)
    if outcome.is_failure:
        _render_failure(outcome)


def _active_budget_control() -> int:
    """Validate shared budget state before any statewide selection is made."""

    remembered_budget = get_active_budget(st.session_state)
    requested_budget = st.sidebar.slider(
        "Inspection budget (dollars)",
        min_value=MIN_ACTIVE_BUDGET,
        max_value=MAX_ACTIVE_BUDGET,
        value=remembered_budget,
        step=ACTIVE_BUDGET_STEP,
        key=BUDGET_WIDGET_KEY,
    )
    return get_active_budget(st.session_state, requested_budget)


def main() -> None:
    """Compose one mode-isolated, dependency-scoped application rerun."""

    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    inject_css()
    st.title(PAGE_TITLE)

    # Exact mode resolution precedes provider construction, cache access, file
    # reads, or any other data-derived operation.
    mode_outcome = _run_component(
        lambda: resolve_runtime_mode(os.environ),
        ErrorCategory.RUNTIME_MODE,
    )
    if mode_outcome.is_failure:
        _render_failure(mode_outcome)
        return
    mode = cast(RuntimeMode, mode_outcome.value)

    provider_category = (
        ErrorCategory.MODEL_FIT
        if mode is RuntimeMode.PRODUCTION
        else ErrorCategory.RUNTIME_MODE
    )
    provider_outcome = _run_component(
        lambda: _build_provider(mode),
        provider_category,
    )
    if provider_outcome.is_failure:
        _render_failure(provider_outcome)
        return
    provider = provider_outcome.value

    active_budget = _active_budget_control()

    source_outcome = _run_classified_component(
        lambda: _load_source(provider, mode),
        ErrorCategory.DATA_LOAD,
        _SOURCE_ERROR_CATEGORIES,
    )
    if source_outcome.is_failure:
        _render_failure(source_outcome)
        return
    source = source_outcome.value

    # Current ranking and historical backtest share only the validated source.
    # Capture both independently so a backtest failure cannot erase current
    # ranking output and no current-ranking value enters backtest computation.
    ranking_outcome = _run_classified_component(
        lambda: _build_current_ranking(provider, source),
        ErrorCategory.MODEL_FIT,
        _CURRENT_RANKING_ERROR_CATEGORIES,
    )
    backtest_outcome = _run_component(
        lambda: _run_backtest(provider, mode, source),
        ErrorCategory.BACKTEST,
    )

    if ranking_outcome.is_failure:
        _render_failure(ranking_outcome)
        if backtest_outcome.is_failure:
            _render_failure(backtest_outcome)
        return
    ranking = ranking_outcome.value

    selection_outcome = _run_component(
        lambda: _build_statewide_selection(
            provider,
            mode,
            ranking,
            active_budget,
        ),
        ErrorCategory.PLANNER,
    )
    if selection_outcome.is_failure:
        _render_failure(selection_outcome)
        if backtest_outcome.is_failure:
            _render_failure(backtest_outcome)
        return
    selection_context = cast(SelectionContext, selection_outcome.value)

    context_outcome = _run_component(
        lambda: ViewContext(
            mode=mode,
            provenance=cast(Any, provider).provenance,
            source=source,
            current_ranking=ranking,
            selection_context=selection_context,
            backtest=(
                backtest_outcome.value if backtest_outcome.is_success else None
            ),
        ),
        ErrorCategory.PLANNER,
    )
    if context_outcome.is_failure:
        _render_failure(context_outcome)
        return
    context = cast(ViewContext, context_outcome.value)

    # County is deliberately read only after the statewide SelectionContext is
    # frozen. It is passed only to render_map and cannot affect any planner key.
    county_outcome = _run_component(
        lambda: _county_options(ranking),
        ErrorCategory.MODEL_SCORE,
    )
    if county_outcome.is_success:
        county_choice = st.sidebar.selectbox(
            "County",
            cast(tuple[str, ...], county_outcome.value),
            index=0,
            key=COUNTY_WIDGET_KEY,
        )
        map_county = None if county_choice == ALL_COUNTIES else county_choice
    else:
        _render_failure(county_outcome)
        map_county = None

    # The required primary evidence is rendered before every secondary tab.
    _render_with_boundary(
        lambda: render_current_ranking(context),
        ErrorCategory.PLANNER,
    )
    _render_with_boundary(
        lambda: render_map(context, county=map_county),
        ErrorCategory.MODEL_SCORE,
    )

    budget_tab, backtest_tab, detail_tab, hazards_tab, export_tab, methods_tab = (
        st.tabs(
            [
                "Budget",
                BACKTEST_SECTION_LABEL,
                "Bridge detail",
                "Hazards",
                "Export",
                "Methods",
            ]
        )
    )

    with budget_tab:
        _render_with_boundary(
            lambda: render_budget(context),
            ErrorCategory.PLANNER,
        )

    with backtest_tab:
        if backtest_outcome.is_failure:
            st.subheader(BACKTEST_SECTION_LABEL)
            _render_failure(backtest_outcome)
        else:
            _render_with_boundary(
                lambda: render_backtest(context),
                ErrorCategory.BACKTEST,
            )

    with detail_tab:
        _render_with_boundary(
            lambda: render_bridge_detail(context),
            ErrorCategory.BRIDGE_DETAIL,
        )

    with hazards_tab:
        _render_with_boundary(
            lambda: render_hazards(context),
            ErrorCategory.PLANNER,
        )

    with export_tab:
        _render_with_boundary(
            lambda: render_export(context),
            ErrorCategory.EXPORT,
        )

    with methods_tab:
        _render_with_boundary(
            lambda: render_methods(context),
            ErrorCategory.MODEL_SCORE,
        )


if __name__ == "__main__":
    main()
