"""Streamlit frontend for the real Massachusetts bridge inspection pipeline.

The app consumes the checked-in NBI panel, fits the repository model, and
renders its real score and backtest artifacts. It intentionally has no
fabricated-data fallback: a failed production dependency must be fixed, not
silently replaced with believable-looking output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ``streamlit run app/app.py`` adds ``app/`` to sys.path. Resolve imports from
# the repository root so ``app`` remains a package in every launch mode.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
sys.path[:] = [entry for entry in sys.path if entry != _SCRIPT_DIRECTORY]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd
import pydeck as pdk
import streamlit as st

from app.data import counties, history, latest_year, load_panel
from app.style import inject_css
from model.data import load_panel as load_model_panel
from model.score import fit_logistic, score


PAGE_TITLE = "MA Bridge Inspection Planner"
INFOBRIDGE = "https://infobridge.fhwa.dot.gov/Data"
DEFAULT_BUDGET = 500_000
BACKTEST_PATH = REPOSITORY_ROOT / "results" / "backtest.csv"
BACKTEST_SUMMARY_PATH = REPOSITORY_ROOT / "results" / "summary.json"


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Raise a clear error when a real artifact has an incomplete contract."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {', '.join(missing)}")


@st.cache_resource(show_spinner=False)
def fitted_model():
    """Fit the selected real model on the approved historical interval."""

    panel = load_model_panel()
    return fit_logistic(panel, 2012, 2016, use_months_since_insp=False)


@st.cache_data(show_spinner=False)
def scored_latest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the map-ready latest panel slice joined to real model scores."""

    source_panel = load_panel()
    latest = latest_year(source_panel)
    if latest.empty:
        raise ValueError("The production panel has no mappable latest-year rows.")

    ranking_year = int(latest["year"].max())
    model_panel = load_model_panel()
    model_latest = model_panel[model_panel["year"] == ranking_year].copy()
    if model_latest.empty:
        raise ValueError("The model panel has no rows for the latest source year.")

    scored = score(
        model_latest,
        fitted_model(),
        asof=f"{ranking_year}-04-01",
    ).rename(columns={"id": "structure_id"})
    _require_columns(scored, ("structure_id", "p_worse", "insp_cost", "score"))

    bridges = latest.merge(
        scored,
        on="structure_id",
        how="inner",
        validate="one_to_one",
    )
    if bridges.empty:
        raise ValueError("Real model scoring produced no map-ready bridge records.")

    bridges["hazard"] = (
        bridges["fracture_critical"].fillna(False).astype(bool)
        | bridges["underwater"].fillna(False).astype(bool)
    )
    return source_panel, bridges.reset_index(drop=True)


def select_for_budget(bridges: pd.DataFrame, budget: int) -> list[str]:
    """Choose real scored bridges greedily without exceeding a dollar budget.

    ``score`` is already the model's expected-value-per-dollar ranking, so
    candidates are evaluated in descending score order. Ties are deterministic
    by structure ID and only valid, positive modeled inspection costs can be
    selected.
    """

    _require_columns(bridges, ("structure_id", "score", "insp_cost"))
    ranked = bridges[["structure_id", "score", "insp_cost"]].copy()
    ranked["score"] = pd.to_numeric(ranked["score"], errors="coerce")
    ranked["insp_cost"] = pd.to_numeric(ranked["insp_cost"], errors="coerce")
    ranked = ranked.dropna(subset=["structure_id", "score", "insp_cost"])
    ranked = ranked[ranked["insp_cost"] > 0]
    ranked = ranked.sort_values(
        ["score", "structure_id"], ascending=[False, True], kind="mergesort"
    )

    selected: list[str] = []
    spent = 0.0
    for row in ranked.itertuples(index=False):
        cost = float(row.insp_cost)
        if spent + cost <= budget:
            selected.append(str(row.structure_id))
            spent += cost
    return selected


@st.cache_data(show_spinner=False)
def load_real_backtest() -> pd.DataFrame:
    """Adapt the checked-in real backtest artifact for the historical chart."""

    backtest = pd.read_csv(BACKTEST_PATH)
    summary = json.loads(BACKTEST_SUMMARY_PATH.read_text(encoding="utf-8"))
    _require_columns(backtest, ("method", "budget", "recall"))
    decliners = summary.get("decliners")
    if not isinstance(decliners, int) or decliners < 0:
        raise ValueError("The backtest summary has no valid decliner count.")

    displayed = backtest[backtest["method"].isin(("model", "worst_first", "oldest_first"))].copy()
    if displayed.empty:
        raise ValueError("The real backtest artifact has no displayable methods.")
    displayed["method"] = displayed["method"].replace({"model": "ours"})
    displayed["caught"] = (displayed["recall"] * decliners).round().astype(int)
    return displayed[["budget", "method", "caught"]]


def color_for(row: pd.Series, chosen_set: set[str], max_score: float) -> list[int]:
    """Return the map color for one real scored bridge."""

    if row["structure_id"] in chosen_set:
        return [229, 72, 77]
    score_value = row.get("score")
    if pd.isna(score_value):
        return [160, 160, 160]
    normalized = min(1.0, float(score_value) / max(max_score, 1e-9))
    return [int(120 + 109 * normalized), int(120 - 48 * normalized), int(120 - 43 * normalized)]


st.set_page_config(page_title=PAGE_TITLE, layout="wide")
inject_css()

try:
    panel, bridges = scored_latest()
except Exception as error:
    st.error("The real bridge model could not be loaded. No substitute data was used.")
    st.exception(error)
    st.stop()

year = int(panel["year"].max())
chosen_ids = select_for_budget(bridges, DEFAULT_BUDGET)
chosen = bridges[bridges["structure_id"].isin(chosen_ids)].copy()
total = len(bridges)
hazards = int(bridges["hazard"].sum())
caught_estimate = int(round(chosen["p_worse"].sum())) if not chosen.empty else 0

left, right = st.columns([2, 1])
with left:
    st.markdown(
        f'<div class="headline"><span class="accent">{caught_estimate}</span> '
        f'likely declines caught</div>'
        f'<div class="headline-sub">at a ${DEFAULT_BUDGET:,} budget, '
        f'{len(chosen_ids)} of {total} {year} model-eligible bridges chosen</div>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        '<div class="card"><span class="pill">real model</span>'
        f'<div class="headline-sub">data: real NBI panel<br>'
        f'{hazards} hazard bridges flagged</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

tab_map, tab_budget, tab_back, tab_detail, tab_hazard, tab_export, tab_methods = st.tabs(
    ["Map", "Budget", "Backtest", "Bridge detail", "Hazards", "Export", "Methods"]
)

with tab_map:
    choices = ["All", *counties(bridges)]
    county = st.selectbox("County", choices, index=0)
    view = bridges if county == "All" else bridges[bridges["county"] == county]
    chosen_set = set(chosen_ids)
    mapped = view.copy()
    max_score = float(bridges["score"].max())
    mapped["fill"] = mapped.apply(
        lambda row: color_for(row, chosen_set, max_score), axis=1
    )
    mapped["radius"] = (mapped["adt"].fillna(1000).clip(lower=100) ** 0.5) * 3
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=mapped,
        get_position=["lon", "lat"],
        get_fill_color="fill",
        get_radius="radius",
        pickable=True,
        opacity=0.75,
    )
    view_state = pdk.ViewState(latitude=42.2, longitude=-71.8, zoom=7.2)
    tooltip = {"text": "{carries}\n{crosses}\n{county}\nscore {score}"}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
    st.caption("Red is chosen for inspection. Warmer means higher real-model score. Dot size is traffic.")

with tab_budget:
    budget = st.slider(
        "Inspection budget (dollars)", 100_000, 3_000_000, DEFAULT_BUDGET, 50_000
    )
    ids = select_for_budget(bridges, budget)
    selected_rows = bridges[bridges["structure_id"].isin(ids)].copy()
    selected_rows = selected_rows.sort_values(["hazard", "score"], ascending=[False, False])
    cost = int(selected_rows["insp_cost"].sum())
    caught = int(round(selected_rows["p_worse"].sum())) if not selected_rows.empty else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Bridges chosen", f"{len(ids)}")
    c2.metric("Total modeled cost", f"${cost:,}")
    c3.metric("Expected catches", f"{caught}")

    columns = ["structure_id", "county", "worst_rating", "score", "insp_cost", "hazard", "carries", "crosses", "reason"]
    st.dataframe(selected_rows[columns], width="stretch", hide_index=True)

with tab_back:
    try:
        backtest = load_real_backtest()
    except Exception as error:
        st.error("The real backtest artifact could not be loaded.")
        st.exception(error)
    else:
        at_top_budget = backtest[backtest["budget"] == backtest["budget"].max()]
        ours = int(at_top_budget.loc[at_top_budget["method"] == "ours", "caught"].sum())
        worst_first = int(at_top_budget.loc[at_top_budget["method"] == "worst_first", "caught"].sum())
        lift = f"{(ours / worst_first):.1f}x" if worst_first else "n/a"
        st.markdown(
            f'<div class="headline"><span class="accent">{lift}</span> '
            f'more declines caught than worst first</div>'
            f'<div class="headline-sub">at the top budget, ours caught {ours} '
            f'vs {worst_first} for worst first</div>',
            unsafe_allow_html=True,
        )
        chart = backtest.pivot_table(index="budget", columns="method", values="caught")
        st.line_chart(chart, width="stretch")

with tab_detail:
    options = bridges.sort_values("score", ascending=False)["structure_id"].tolist()
    structure_id = st.selectbox("Bridge", options, index=0 if options else None)
    if structure_id:
        row = bridges[bridges["structure_id"] == structure_id].iloc[0]
        inspection_history = history(panel, structure_id).set_index("year")[["deck", "super", "sub"]]
        a, b = st.columns([2, 1])
        with a:
            st.line_chart(inspection_history, width="stretch")
        with b:
            st.markdown(
                f'<div class="card"><b>{row.get("carries", "")}</b>'
                f'<div class="headline-sub">crosses {row.get("crosses", "")}<br>'
                f'{row.get("county", "")} county</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="headline-sub">{row.get("reason", "")}</div>',
                unsafe_allow_html=True,
            )
            st.link_button("Open in FHWA InfoBridge", INFOBRIDGE)

with tab_hazard:
    st.caption("Underwater or fracture-critical bridges are shown first in the selected list.")
    selected_rows = bridges[bridges["structure_id"].isin(chosen_ids)].copy()
    selected_rows = selected_rows.sort_values(["hazard", "score"], ascending=[False, False])
    selected_rows["hazard_kind"] = selected_rows.apply(
        lambda row: ", ".join(
            label
            for label, flagged in (
                ("underwater", row["underwater"]),
                ("fracture critical", row["fracture_critical"]),
            )
            if flagged
        )
        or "none",
        axis=1,
    )
    columns = ["structure_id", "county", "hazard_kind", "worst_rating", "score", "carries", "crosses"]
    st.dataframe(selected_rows[columns], width="stretch", hide_index=True)

with tab_export:
    selected_rows = bridges[bridges["structure_id"].isin(chosen_ids)].copy()
    selected_rows = selected_rows.sort_values(["county", "hazard", "score"], ascending=[True, False, False])
    columns = ["county", "structure_id", "worst_rating", "score", "insp_cost", "hazard", "carries", "crosses", "reason"]
    csv = selected_rows[columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download chosen list (CSV, grouped by county)",
        csv,
        file_name="chosen_bridges.csv",
        mime="text/csv",
    )
    st.dataframe(selected_rows[columns], width="stretch", hide_index=True)

with tab_methods:
    st.markdown(
        """
### Methods

Data source: Federal Highway Administration National Bridge Inventory (NBI),
Massachusetts, years 2012, 2016, 2018, and 2022.

The app fits the repository's real decline model on the approved 2012 to 2016
training interval, then scores the latest model-eligible bridge snapshot. NBI
does not publish a usable per-bridge inspection price, so `insp_cost` is a
modeled drone-flight planning estimate, not a vendor quote. The NBI
`improvement_cost` field is a repair-cost field and is not used for inspection
selection.

The historical chart is read from the repository's real 2018 to 2022 backtest
artifact. It compares the model selection with worst-condition-first and
oldest-first baselines on the same eligible bridge pool. Rebuilt bridges and
not-applicable condition records are excluded rather than being assigned an
invented rating.
"""
    )
