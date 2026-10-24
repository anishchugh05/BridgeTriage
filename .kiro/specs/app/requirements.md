# Requirements - Inspection planner front end

## Goal

A Streamlit app in app/ that shows which Massachusetts bridges to inspect first
under a budget, and proves the strategy with a backtest. Front end only. Do not
edit model/, planner/, or bridges/.

## Data

- Read bridges/panel.parquet if present, else bridges/fake.parquet.
- Load from relative paths so it works after deploy. Cache with st.cache_data.
- 21 column schema from bridges/README.md.

## Real code with stub fallback

- B provides score(table) returning structure_id, p_worse, consequence,
  insp_cost, score, reason.
- C provides pick(scores, budget) returning chosen structure_ids, and a backtest
  returning budget, method, caught with methods ours, worst_first, oldest_first.
- app/stubs.py holds fake versions with the exact signatures and columns.
- The app imports the real functions in a try block and falls back to stubs, so
  switching to real code needs no edits.

## App tabs

1. Map: pydeck map of 2022 bridges colored by score, sized by adt, county
   filter, chosen bridges highlighted red.
2. Budget picker: slider reruns pick() live, shows bridges chosen, total cost,
   expected caught.
3. Backtest: large headline number (ours vs worst_first at the same budget) and
   a line chart of caught vs budget for all three methods.
4. Bridge detail: pick a bridge, see ratings across 2012, 2016, 2018, 2022, the
   one line reason, carries and crosses, and a link to FHWA InfoBridge.
5. Hazard flag: underwater or fracture_critical bridges marked hazardous and
   shown first in the chosen list.
6. Export: download the chosen list as CSV grouped by county.
7. Methods: data source, inspection cost is an assumed estimate, short NBI to
   SNBI schema note.

The headline number and the map come first on the page.

## Design

- Follow .kiro/steering/frontend-design.md (Emil Kowalski style).
- All CSS in one st.markdown block in app/style.py.

## Deployment

- Deployable on Streamlit Community Cloud, main file app/app.py.
- requirements.txt has streamlit, pydeck, pandas, pyarrow.

## Constraints

- No em dashes or en dashes in code, comments, or UI copy.
- Keep code short and plain. Commit small, push to person-a-app, never gitignore
  .kiro.