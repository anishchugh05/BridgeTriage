# Design - Inspection planner front end

## Files

- app/app.py: main Streamlit app, tabs, layout, wiring.
- app/style.py: one function inject_css() with a single st.markdown block.
- app/data.py: load_panel() cached, helpers for the 2022 slice and history.
- app/stubs.py: stub score(), pick(), backtest() matching B and C contracts.

## Import with fallback

    try:
        from model.score import score
        from planner.pick import pick, backtest
    except Exception:
        from stubs import score, pick, backtest

Broad except so any import problem falls back to stubs cleanly.

## Data layer

- load_panel reads bridges/panel.parquet else bridges/fake.parquet, relative to
  the repo root resolved from this file location.
- latest_year slice (2022) drives the map and picker.
- history(structure_id) returns the four year ratings for the detail chart.

## Stubs (match the real contracts)

- score(table) -> DataFrame: structure_id, p_worse, consequence, insp_cost,
  score, reason. Deterministic from the row so the demo is stable.
- pick(scores, budget) -> list of structure_id chosen greedily by score per cost
  until the budget runs out.
- backtest() -> DataFrame: budget, method, caught for methods ours, worst_first,
  oldest_first, ours on top.

## Tabs and layout

Headline number and map first. Tabs via st.tabs. Hazard bridges (underwater or
fracture_critical) sort first in the chosen list. Export groups by county.

## Design system

Single CSS block: ease out cubic-bezier(0.23, 1, 0.32, 1), animations under
300ms, press scale 0.97, entrance from scale(0.95) opacity 0, 40ms stagger
capped at 8, tabular-nums, soft layered shadows, red accent, reduced motion
disables animation. Headline counts up under 600ms.

## Deploy

Relative parquet paths. requirements.txt includes streamlit, pydeck, pandas,
pyarrow. Main file app/app.py.