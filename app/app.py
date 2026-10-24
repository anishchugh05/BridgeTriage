# Inspection planner front end. Front end only, reads Person A data and the
# B and C functions. Falls back to stubs when the real code is not present.
import os
import pandas as pd
import pydeck as pdk
import streamlit as st

from data import load_panel, latest_year, counties, history, using_fake
from style import inject_css

# Real code first, stubs on any import failure. No edits needed when B and C land.
try:
    from model.score import score
    from planner.pick import pick, backtest
    USING_STUBS = False
except Exception:
    from stubs import score, pick, backtest
    USING_STUBS = True

st.set_page_config(page_title="MA Bridge Inspection Planner", layout="wide")
inject_css()

INFOBRIDGE = "https://infobridge.fhwa.dot.gov/Data"


@st.cache_data(show_spinner=False)
def scored_latest():
    panel = load_panel()
    cur = latest_year(panel)
    sc = score(cur)
    merged = cur.merge(sc, on="structure_id", how="left")
    merged["hazard"] = merged["fracture_critical"].fillna(False) | \
        merged["underwater"].fillna(False)
    return panel, merged


panel, bridges = scored_latest()
year = int(panel["year"].max())

# ---- Header: headline number and map come first ----
total = len(bridges)
hazards = int(bridges["hazard"].sum())
demo_budget = 500_000
chosen_ids = pick(bridges[["structure_id", "p_worse", "consequence",
                           "insp_cost", "score", "reason"]], demo_budget)
chosen = bridges[bridges["structure_id"].isin(chosen_ids)].copy()
caught_est = int(round(chosen["p_worse"].sum())) if not chosen.empty else 0

left, right = st.columns([2, 1])
with left:
    st.markdown(
        f'<div class="headline"><span class="accent">{caught_est}</span> '
        f'likely declines caught</div>'
        f'<div class="headline-sub">at a {demo_budget:,} dollar budget, '
        f'{len(chosen_ids)} of {total} {year} bridges chosen</div>',
        unsafe_allow_html=True,
    )
with right:
    src = "fake sample" if using_fake() else "real NBI panel"
    mode = "stub model" if USING_STUBS else "live model"
    st.markdown(
        f'<div class="card"><span class="pill">{mode}</span>'
        f'<div class="headline-sub">data: {src}<br>'
        f'{hazards} hazard bridges flagged</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

tab_map, tab_budget, tab_back, tab_detail, tab_hazard, tab_export, tab_methods = \
    st.tabs(["Map", "Budget", "Backtest", "Bridge detail", "Hazards",
             "Export", "Methods"])


def color_for(row, chosen_set):
    if row["structure_id"] in chosen_set:
        return [229, 72, 77]        # red, chosen
    s = row.get("score")
    if pd.isna(s):
        return [160, 160, 160]
    # gray to orange ramp by score
    hi = bridges["score"].max() or 1
    t = min(1.0, float(s) / float(hi))
    return [int(120 + 109 * t), int(120 - 48 * t), int(120 - 43 * t)]


# ---- Tab 1: Map ----
with tab_map:
    pickable = ["All"] + counties(bridges)
    sel = st.selectbox("County", pickable, index=0)
    view = bridges if sel == "All" else bridges[bridges["county"] == sel]
    chosen_set = set(chosen_ids)
    m = view.copy()
    m["fill"] = m.apply(lambda r: color_for(r, chosen_set), axis=1)
    m["radius"] = (m["adt"].fillna(1000).clip(lower=100) ** 0.5) * 3
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=m,
        get_position=["lon", "lat"],
        get_fill_color="fill",
        get_radius="radius",
        pickable=True,
        opacity=0.75,
    )
    mid = pdk.ViewState(latitude=42.2, longitude=-71.8, zoom=7.2)
    tip = {"text": "{carries}\ncrosses {crosses}\n{county}\nscore {score}"}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=mid, tooltip=tip))
    st.caption("Red is chosen for inspection. Warmer means higher risk. "
               "Dot size is traffic.")


# ---- Tab 2: Budget picker ----
with tab_budget:
    budget = st.slider("Inspection budget (dollars)", 100_000, 3_000_000,
                       demo_budget, 50_000)
    ids = pick(bridges[["structure_id", "p_worse", "consequence",
                        "insp_cost", "score", "reason"]], budget)
    sel_rows = bridges[bridges["structure_id"].isin(ids)].copy()
    # hazards first
    sel_rows = sel_rows.sort_values(["hazard", "score"], ascending=[False, False])
    cost = int(sel_rows["insp_cost"].sum())
    caught = int(round(sel_rows["p_worse"].sum())) if not sel_rows.empty else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Bridges chosen", f"{len(ids)}")
    c2.metric("Total cost", f"{cost:,}")
    c3.metric("Expected caught", f"{caught}")
    st.toast(f"Picked {len(ids)} bridges")

    show = sel_rows[["structure_id", "county", "worst_rating", "score",
                     "insp_cost", "hazard", "carries", "crosses", "reason"]]
    st.dataframe(show, use_container_width=True, hide_index=True)


# ---- Tab 3: Backtest ----
with tab_back:
    bt = backtest()
    at_budget = bt[bt["budget"] == bt["budget"].max()]
    ours = int(at_budget[at_budget["method"] == "ours"]["caught"].sum())
    wf = int(at_budget[at_budget["method"] == "worst_first"]["caught"].sum())
    lift = f"{(ours / wf):.1f}x" if wf else "n/a"
    st.markdown(
        f'<div class="headline"><span class="accent">{lift}</span> '
        f'more declines caught than worst first</div>'
        f'<div class="headline-sub">at the top budget, ours caught {ours} '
        f'vs {wf} for worst first</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    chart = bt.pivot_table(index="budget", columns="method", values="caught")
    chart = chart[["ours", "worst_first", "oldest_first"]]
    st.line_chart(chart, use_container_width=True)


# ---- Tab 4: Bridge detail ----
with tab_detail:
    options = bridges.sort_values("score", ascending=False)["structure_id"].tolist()
    sid = st.selectbox("Bridge", options, index=0 if options else None)
    if sid:
        row = bridges[bridges["structure_id"] == sid].iloc[0]
        hist = history(panel, sid).set_index("year")[["deck", "super", "sub"]]
        a, b = st.columns([2, 1])
        with a:
            st.line_chart(hist, use_container_width=True)
        with b:
            st.markdown(f'<div class="card"><b>{row.get("carries","")}</b>'
                        f'<div class="headline-sub">crosses '
                        f'{row.get("crosses","")}<br>{row.get("county","")} county'
                        f'</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="headline-sub">{row.get("reason","")}</div>',
                        unsafe_allow_html=True)
            st.link_button("Open in FHWA InfoBridge", INFOBRIDGE)


# ---- Tab 5: Hazards ----
with tab_hazard:
    st.caption("Underwater or fracture critical bridges are hazardous to "
               "inspect and are shown first in the chosen list.")
    ids = pick(bridges[["structure_id", "p_worse", "consequence",
                        "insp_cost", "score", "reason"]], demo_budget)
    sel_rows = bridges[bridges["structure_id"].isin(ids)].copy()
    sel_rows = sel_rows.sort_values(["hazard", "score"], ascending=[False, False])
    sel_rows["hazard_kind"] = sel_rows.apply(
        lambda r: ", ".join(
            [k for k, v in (("underwater", r["underwater"]),
                            ("fracture critical", r["fracture_critical"])) if v])
        or "none", axis=1)
    show = sel_rows[["structure_id", "county", "hazard_kind", "worst_rating",
                     "score", "carries", "crosses"]]
    if int(sel_rows["hazard"].sum()) == 0:
        st.info("No hazard bridges in the current data. If the panel shows zero "
                "fracture critical and underwater bridges, the loader may be "
                "reading NBI items 092A and 092B as plain Y instead of the Y plus "
                "months form (for example Y24). That is a Person A loader fix.")
    st.dataframe(show, use_container_width=True, hide_index=True)


# ---- Tab 6: Export ----
with tab_export:
    ids = pick(bridges[["structure_id", "p_worse", "consequence",
                        "insp_cost", "score", "reason"]], demo_budget)
    sel_rows = bridges[bridges["structure_id"].isin(ids)].copy()
    sel_rows = sel_rows.sort_values(["county", "hazard", "score"],
                                    ascending=[True, False, False])
    cols = ["county", "structure_id", "worst_rating", "score", "insp_cost",
            "hazard", "carries", "crosses", "reason"]
    csv = sel_rows[cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download chosen list (CSV, grouped by county)", csv,
                       file_name="chosen_bridges.csv", mime="text/csv")
    st.dataframe(sel_rows[cols], use_container_width=True, hide_index=True)


# ---- Tab 7: Methods ----
with tab_methods:
    st.markdown("""
### Methods

Data source: Federal Highway Administration National Bridge Inventory (NBI),
Massachusetts, years 2012, 2016, 2018, 2022.

Inspection cost is an assumed estimate. NBI does not publish a per bridge
inspection cost, so we use a modeled estimate, not a real quote. The
improvement cost field in NBI is repair cost, not inspection cost.

Schema note: NBI is being replaced by the Specification for the National Bridge
Inventory (SNBI). Item numbers and some fields change under SNBI, so a loader
built on NBI item numbers will need a mapping update when the data moves to
SNBI.

We match bridges across years by structure number and look for a drop, meaning
the worst of the deck, superstructure, and substructure ratings falls by one or
more between two appearances. We exclude rebuilt bridges because an improvement
is not natural decline.
""")