"""Explicit, isolated providers for the fabricated application demo.

Nothing in this module is selected implicitly.  Callers must first resolve the
runtime setting and pass the exact :class:`~app.config.RuntimeMode.DEMO` enum to
:func:`build_demo_providers`.  Construction is lazy: it does not read a file or
compute a value.  The production composition branch must never import this
module as an exception fallback.

Demo scoring, selection, and backtest calculations live here rather than in
``app.stubs``.  They intentionally do not import or call the production
``model`` or ``planner`` implementations.  Only ``build_view_context`` imports
shared immutable application/planner value contracts, after the explicit demo
branch is already active, so the result has the same shape consumed by views.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from pathlib import Path
from typing import Final, Protocol, TypeAlias, cast

import numpy as np
import pandas as pd

from app.config import Provenance, ProviderModeMismatch, RuntimeMode

DEMO_INDICATOR: Final[str] = "DEMO MODE - FABRICATED DATA"
REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_PANEL_PATH: Final[Path] = REPOSITORY_ROOT / "bridges" / "fake.parquet"
DEFAULT_DEMO_BACKTEST_BUDGETS: Final[tuple[int, ...]] = tuple(
    range(100_000, 3_000_001, 50_000)
)
DEMO_BACKTEST_METHODS: Final[tuple[str, ...]] = (
    "ours",
    "worst_first",
    "oldest_first",
)
DEMO_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "structure_id",
    "year",
    "county",
    "material",
    "age",
    "adt",
    "deck",
    "super",
    "sub",
    "worst_rating",
    "insp_freq_months",
    "fracture_critical",
    "underwater",
    "year_recon",
    "rebuilt",
    "lat",
    "lon",
    "carries",
    "crosses",
    "improvement_cost",
    "insp_date",
)
DEMO_SCORED_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "p_worse",
    "consequence",
    "insp_cost",
    "score",
    "reason",
)
DEMO_PROVENANCE: Final[Provenance] = Provenance(
    mode=RuntimeMode.DEMO,
    data_source="Demo_Data",
    model_source="Demo_Model_Service",
    planner_source="Demo_Planner",
    backtest_source="Demo_Backtest",
)

DemoPanelLoader: TypeAlias = Callable[[Path], pd.DataFrame]


class DemoProviderError(RuntimeError):
    """Base class for explicit demo-provider failures."""


class DemoDataError(DemoProviderError):
    """Raised when the configured fabricated panel cannot supply a demo."""


class DemoCalculationError(DemoProviderError):
    """Raised when a demo calculation cannot produce a complete value."""


class DemoTagged(Protocol):
    """Structural provenance contract for every public demo wrapper."""

    @property
    def mode(self) -> RuntimeMode: ...

    @property
    def provenance(self) -> Provenance: ...

    @property
    def demo_indicator(self) -> str: ...


class _DemoTag:
    """Shared immutable provenance properties for demo-only values."""

    __slots__ = ()

    @property
    def mode(self) -> RuntimeMode:
        return RuntimeMode.DEMO

    @property
    def provenance(self) -> Provenance:
        return DEMO_PROVENANCE

    @property
    def demo_indicator(self) -> str:
        return DEMO_INDICATOR


def _tag_frame(frame: pd.DataFrame, component: str) -> pd.DataFrame:
    """Return a defensive frame copy carrying explicit demo metadata."""

    tagged = frame.copy(deep=True)
    tagged.attrs = {
        "runtime_mode": RuntimeMode.DEMO,
        "provenance": DEMO_PROVENANCE,
        "demo_indicator": DEMO_INDICATOR,
        "demo_component": component,
    }
    return tagged


def _frame_digest(frame: pd.DataFrame) -> str:
    """Build a deterministic identity without importing production caches."""

    value = frame.copy(deep=True)
    value.attrs = {}
    digest = hashlib.sha256()
    for column in value.columns:
        encoded = str(column).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    for dtype in value.dtypes.astype(str).tolist():
        encoded = dtype.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    try:
        hashed = pd.util.hash_pandas_object(value, index=True, categorize=True)
    except (TypeError, ValueError) as exc:
        raise DemoDataError("Fabricated demo data cannot be identified.") from exc
    digest.update(hashed.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


def _require_demo_tag(value: object, *, name: str) -> None:
    if (
        getattr(value, "mode", None) is not RuntimeMode.DEMO
        or getattr(value, "provenance", None) != DEMO_PROVENANCE
    ):
        raise ProviderModeMismatch() from None


def _validate_demo_source(source: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(source, pd.DataFrame):
        raise DemoDataError("The fabricated demo source must be a table.")
    if tuple(source.columns) != DEMO_SOURCE_COLUMNS:
        raise DemoDataError("The fabricated demo source has an invalid schema.")
    if source.empty:
        raise DemoDataError("The fabricated demo source is empty.")

    current = source.loc[source["year"].eq(2022)]
    identifiers = tuple(current["structure_id"].tolist())
    if not identifiers:
        raise DemoDataError("The fabricated demo source has no 2022 records.")
    if not all(
        isinstance(identifier, str) and bool(identifier.strip())
        for identifier in identifiers
    ):
        raise DemoDataError("Fabricated demo identifiers must be nonempty strings.")
    if len(set(identifiers)) != len(identifiers):
        raise DemoDataError("Fabricated 2022 demo identifiers must be unique.")

    for column in ("fracture_critical", "underwater", "rebuilt"):
        if not all(
            isinstance(value, (bool, np.bool_)) for value in source[column].tolist()
        ):
            raise DemoDataError("Fabricated demo flags must be Boolean values.")
    return source.copy(deep=True)


def _adapt_demo_source(source: pd.DataFrame) -> pd.DataFrame:
    """Create a demo-local model-shaped copy without a production adapter."""

    adapted = source.copy(deep=True).rename(
        columns={"structure_id": "id", "insp_freq_months": "insp_freq_mo"}
    )
    adapted["year_built"] = adapted["year"] - adapted["age"]
    return _tag_frame(adapted, "Demo_Data_Adapted")


@dataclass(frozen=True, slots=True, init=False)
class DemoSourceBundle(_DemoTag):
    """Fabricated source and demo-local adapted snapshots."""

    dataset_sha256: str
    _source: pd.DataFrame = field(repr=False, compare=False)
    _adapted: pd.DataFrame = field(repr=False, compare=False)

    def __init__(self, source: pd.DataFrame) -> None:
        validated = _validate_demo_source(source)
        object.__setattr__(self, "dataset_sha256", _frame_digest(validated))
        object.__setattr__(self, "_source", _tag_frame(validated, "Demo_Data"))
        object.__setattr__(
            self,
            "_adapted",
            _adapt_demo_source(validated),
        )

    @property
    def source(self) -> pd.DataFrame:
        return _tag_frame(self._source, "Demo_Data")

    @property
    def adapted(self) -> pd.DataFrame:
        return _tag_frame(self._adapted, "Demo_Data_Adapted")


@dataclass(frozen=True, slots=True, init=False)
class DemoCurrentRanking(_DemoTag):
    """Complete fabricated 2022 ranking with source-facing records."""

    score_identity: str
    _scored: pd.DataFrame = field(repr=False, compare=False)
    _records: pd.DataFrame = field(repr=False, compare=False)
    _ranking_ids: tuple[str, ...]

    def __init__(self, scored: pd.DataFrame, records: pd.DataFrame) -> None:
        if not isinstance(scored, pd.DataFrame) or tuple(scored.columns) != DEMO_SCORED_COLUMNS:
            raise DemoCalculationError("The fabricated score table is invalid.")
        if not isinstance(records, pd.DataFrame) or "structure_id" not in records:
            raise DemoCalculationError("The fabricated ranking records are invalid.")

        scored_ids = tuple(scored["id"].tolist())
        record_ids = tuple(records["structure_id"].tolist())
        if (
            not scored_ids
            or len(set(scored_ids)) != len(scored_ids)
            or len(set(record_ids)) != len(record_ids)
            or set(scored_ids) != set(record_ids)
        ):
            raise DemoCalculationError(
                "Fabricated ranking identifiers are not one-to-one."
            )

        tagged_scores = _tag_frame(scored, "Demo_Model_Service")
        tagged_records = _tag_frame(records, "Demo_Current_Ranking")
        object.__setattr__(self, "score_identity", _frame_digest(tagged_scores))
        object.__setattr__(self, "_scored", tagged_scores)
        object.__setattr__(self, "_records", tagged_records)
        object.__setattr__(self, "_ranking_ids", cast(tuple[str, ...], record_ids))

    @property
    def scored(self) -> pd.DataFrame:
        return _tag_frame(self._scored, "Demo_Model_Service")

    @property
    def records(self) -> pd.DataFrame:
        return _tag_frame(self._records, "Demo_Current_Ranking")

    @property
    def ranking_ids(self) -> tuple[str, ...]:
        return self._ranking_ids

    @property
    def population_size(self) -> int:
        return len(self._ranking_ids)


@dataclass(frozen=True, slots=True, init=False)
class DemoSelectionResult(_DemoTag):
    """One demo-calculated selection and exact aggregate projections."""

    selection: object
    metrics: object
    score_identity: str
    _selected_records: pd.DataFrame = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        selection: object,
        metrics: object,
        selected_records: pd.DataFrame,
        score_identity: str,
    ) -> None:
        identifiers = tuple(getattr(selection, "ids", ()))
        if not isinstance(selected_records, pd.DataFrame) or "structure_id" not in selected_records:
            raise DemoCalculationError("Fabricated selected records are invalid.")
        record_ids = tuple(selected_records["structure_id"].tolist())
        if len(record_ids) != len(identifiers) or set(record_ids) != set(identifiers):
            raise DemoCalculationError(
                "Fabricated selected records do not match the selection."
            )
        if not isinstance(score_identity, str) or not score_identity:
            raise DemoCalculationError("Fabricated score identity is invalid.")

        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "score_identity", score_identity)
        object.__setattr__(
            self,
            "_selected_records",
            _tag_frame(selected_records, "Demo_Planner"),
        )

    @property
    def selected_records(self) -> pd.DataFrame:
        return _tag_frame(self._selected_records, "Demo_Planner")


@dataclass(frozen=True, slots=True, init=False)
class DemoBacktestResult(_DemoTag):
    """Fabricated comparison values wrapped with explicit demo provenance."""

    _result: pd.DataFrame = field(repr=False, compare=False)

    def __init__(self, result: pd.DataFrame) -> None:
        if not isinstance(result, pd.DataFrame) or tuple(result.columns) != (
            "budget",
            "method",
            "caught",
        ):
            raise DemoCalculationError("The fabricated backtest result is invalid.")
        if result.empty:
            raise DemoCalculationError("The fabricated backtest result is empty.")
        object.__setattr__(
            self,
            "_result",
            _tag_frame(result, "Demo_Backtest"),
        )

    @property
    def result(self) -> pd.DataFrame:
        return _tag_frame(self._result, "Demo_Backtest")

    @property
    def value(self) -> pd.DataFrame:
        """Alias used by data-only presentation wrappers."""

        return self.result


def _read_demo_panel(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise DemoDataError("The explicit fabricated demo source could not be read.") from exc


def _finite_or(value: object, default: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _stable_demo_numbers(identifier: str) -> tuple[float, int, int]:
    payload = hashlib.sha256(identifier.encode("utf-8")).digest()
    unit = int.from_bytes(payload[:8], "big") / float((1 << 64) - 1)
    cost_offset = int.from_bytes(payload[8:12], "big") % 2_000
    jitter = int.from_bytes(payload[12:14], "big") % 3 - 1
    return unit, cost_offset, jitter


def _score_demo_source(source: DemoSourceBundle) -> DemoCurrentRanking:
    _require_demo_tag(source, name="source")
    current = source.source.loc[lambda frame: frame["year"].eq(2022)].copy(deep=True)
    rows: list[dict[str, object]] = []

    for row in current.itertuples(index=False):
        identifier = str(row.structure_id)
        random_unit, cost_offset, _ = _stable_demo_numbers(identifier)
        worst = _finite_or(row.worst_rating, 5.0)
        age = max(0.0, _finite_or(row.age, 50.0))
        adt = max(0.0, _finite_or(row.adt, 5_000.0))
        hazard = bool(row.fracture_critical) or bool(row.underwater)

        p_worse = round(
            min(
                0.98,
                max(
                    0.02,
                    0.5 * (9.0 - worst) / 9.0
                    + 0.3 * min(age, 100.0) / 100.0
                    + 0.2 * random_unit,
                ),
            ),
            6,
        )
        consequence = round(
            max(0.001, (1.0 + math.log1p(adt)) * (1.4 if hazard else 1.0)),
            6,
        )
        inspection_cost = 4_000 + (600 if hazard else 0) + cost_offset
        score = p_worse * consequence / inspection_cost
        reason = (
            f"Fabricated demo calculation; modeled inspection-cost estimate "
            f"${inspection_cost}; condition {worst:g}, age {age:g}, ADT {adt:g}"
        )
        if hazard:
            reason = "Fabricated hazard scenario; " + reason
        rows.append(
            {
                "id": identifier,
                "p_worse": p_worse,
                "consequence": consequence,
                "insp_cost": inspection_cost,
                "score": score,
                "reason": reason,
            }
        )

    scored = pd.DataFrame(rows, columns=DEMO_SCORED_COLUMNS)
    records = current.merge(
        scored.rename(columns={"id": "structure_id"}),
        on="structure_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    return DemoCurrentRanking(scored, records)


def _decimal_budget(value: object) -> Decimal:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, Decimal)):
        raise DemoCalculationError("The demo budget must be finite and nonnegative.")
    try:
        budget = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise DemoCalculationError(
            "The demo budget must be finite and nonnegative."
        ) from exc
    if not budget.is_finite() or budget < 0:
        raise DemoCalculationError("The demo budget must be finite and nonnegative.")
    return budget


def _select_demo_ranking(
    ranking: DemoCurrentRanking,
    budget_value: object,
) -> DemoSelectionResult:
    _require_demo_tag(ranking, name="ranking")
    budget = _decimal_budget(budget_value)
    scored = ranking.scored
    candidates = list(
        scored.loc[:, ["id", "score", "insp_cost", "p_worse"]].itertuples(
            index=False, name=None
        )
    )
    candidates.sort(key=lambda candidate: str(candidate[0]))
    candidates.sort(key=lambda candidate: float(candidate[1]), reverse=True)

    remaining = budget
    chosen: list[str] = []
    selected_cost = Decimal(0)
    probabilities: list[float] = []
    for identifier, _score, cost_value, p_worse in candidates:
        cost = Decimal(str(cost_value))
        if cost <= remaining:
            chosen.append(str(identifier))
            selected_cost += cost
            probabilities.append(float(p_worse))
            remaining -= cost

    # Shared immutable value classes are imported only after exact demo
    # activation.  No real planner function is called by this module.
    from planner import Selection, SelectionMetrics

    selection = Selection(mode="budget", constraint=budget, ids=tuple(chosen))
    metrics = SelectionMetrics(
        selected_count=len(chosen),
        selected_total_cost=selected_cost,
        expected_safety_drop_coverage=math.fsum(probabilities),
    )
    records = ranking.records
    if chosen:
        selected_records = (
            records.set_index("structure_id", drop=False)
            .loc[chosen]
            .reset_index(drop=True)
        )
    else:
        selected_records = records.iloc[0:0].copy(deep=True)
    return DemoSelectionResult(
        selection=selection,
        metrics=metrics,
        selected_records=selected_records,
        score_identity=ranking.score_identity,
    )


def _normalize_backtest_budgets(values: Iterable[object]) -> tuple[Decimal, ...]:
    try:
        budgets = tuple(_decimal_budget(value) for value in values)
    except TypeError as exc:
        raise DemoCalculationError("Demo backtest budgets must be iterable.") from exc
    if not budgets:
        raise DemoCalculationError("Demo backtest requires at least one budget.")
    return tuple(sorted(set(budgets)))


def _demo_backtest(values: Iterable[object]) -> DemoBacktestResult:
    budgets = _normalize_backtest_budgets(values)
    rows: list[dict[str, object]] = []
    for budget in budgets:
        base = 8.0 + float(budget) / 60_000.0
        _, _, jitter = _stable_demo_numbers(format(budget, "f"))
        counts = {
            "ours": max(0, int(round(base + jitter))),
            "worst_first": max(0, int(round(base * 0.72 + jitter))),
            "oldest_first": max(0, int(round(base * 0.55 + jitter))),
        }
        displayed_budget: object = (
            int(budget) if budget == budget.to_integral_value() else float(budget)
        )
        for method in DEMO_BACKTEST_METHODS:
            rows.append(
                {
                    "budget": displayed_budget,
                    "method": method,
                    "caught": counts[method],
                }
            )
    return DemoBacktestResult(
        pd.DataFrame(rows, columns=("budget", "method", "caught"))
    )


@dataclass(frozen=True, slots=True, init=False)
class DemoProviders(_DemoTag):
    """Lazy provider set available only after exact demo-mode resolution."""

    _panel_path: Path
    _panel_loader: DemoPanelLoader = field(repr=False, compare=False)

    def __init__(
        self,
        mode: RuntimeMode,
        *,
        panel_path: str | Path | None = None,
        panel_loader: DemoPanelLoader | None = None,
    ) -> None:
        if mode is not RuntimeMode.DEMO:
            raise ProviderModeMismatch()
        configured_path = DEFAULT_DEMO_PANEL_PATH if panel_path is None else Path(panel_path)
        if not configured_path.is_absolute():
            configured_path = REPOSITORY_ROOT / configured_path
        loader = _read_demo_panel if panel_loader is None else panel_loader
        if not callable(loader):
            raise TypeError("panel_loader must be callable.")

        # Deliberately retain only configuration.  No source access or
        # calculation occurs during provider construction.
        object.__setattr__(self, "_panel_path", configured_path)
        object.__setattr__(self, "_panel_loader", loader)

    @property
    def panel_path(self) -> Path:
        return self._panel_path

    def load_source(self) -> DemoSourceBundle:
        source = self._panel_loader(self._panel_path)
        return DemoSourceBundle(source)

    def build_current_ranking(
        self,
        source: DemoSourceBundle,
    ) -> DemoCurrentRanking:
        _require_demo_tag(source, name="source")
        if not isinstance(source, DemoSourceBundle):
            raise DemoDataError("Demo ranking requires a demo source bundle.")
        return _score_demo_source(source)

    def select_current(
        self,
        ranking: DemoCurrentRanking,
        budget: object,
    ) -> DemoSelectionResult:
        _require_demo_tag(ranking, name="ranking")
        if not isinstance(ranking, DemoCurrentRanking):
            raise DemoCalculationError("Demo selection requires a demo ranking.")
        return _select_demo_ranking(ranking, budget)

    def run_backtest(
        self,
        budgets: Iterable[object] = DEFAULT_DEMO_BACKTEST_BUDGETS,
    ) -> DemoBacktestResult:
        return _demo_backtest(budgets)

    def build_view_context(
        self,
        active_budget: object,
        *,
        backtest_budgets: Iterable[object] = DEFAULT_DEMO_BACKTEST_BUDGETS,
    ) -> object:
        """Build the shared presentation context on the explicit demo branch."""

        source = self.load_source()
        ranking = self.build_current_ranking(source)
        selected = self.select_current(ranking, active_budget)
        backtest = self.run_backtest(backtest_budgets)

        from app.state import SelectionContext, ViewContext

        selection_context = SelectionContext(
            active_budget=active_budget,
            selection=selected.selection,
            metrics=selected.metrics,
            selected_records=selected.selected_records,
            score_identity=selected.score_identity,
        )
        return ViewContext(
            mode=RuntimeMode.DEMO,
            provenance=DEMO_PROVENANCE,
            source=source,
            current_ranking=ranking,
            selection_context=selection_context,
            backtest=backtest,
        )


def build_demo_providers(
    active_mode: RuntimeMode,
    *,
    panel_path: str | Path | None = None,
    panel_loader: DemoPanelLoader | None = None,
) -> DemoProviders:
    """Construct lazy providers only for the exact resolved demo enum.

    Strings such as ``"demo"`` are deliberately rejected.  The runtime-mode
    controller must first perform exact setting resolution, then call this
    factory only from its demo branch.
    """

    if active_mode is not RuntimeMode.DEMO:
        raise ProviderModeMismatch()
    return DemoProviders(
        active_mode,
        panel_path=panel_path,
        panel_loader=panel_loader,
    )


__all__ = [
    "DEFAULT_DEMO_BACKTEST_BUDGETS",
    "DEFAULT_DEMO_PANEL_PATH",
    "DEMO_BACKTEST_METHODS",
    "DEMO_INDICATOR",
    "DEMO_PROVENANCE",
    "DemoBacktestResult",
    "DemoCalculationError",
    "DemoCurrentRanking",
    "DemoDataError",
    "DemoProviderError",
    "DemoProviders",
    "DemoSelectionResult",
    "DemoSourceBundle",
    "DemoTagged",
    "build_demo_providers",
]
